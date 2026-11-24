// CUDA kernels for bf16ref's BFR8 blocks, compiled with nvcc in bf16ref_gpu.cu, or for the CPU emulator
// gpu/cuemu.h.
//
// Decoding: a parse kernel reads each block's header, tables and sub-stream index, builds the block's
// decoding tables in global memory and writes one descriptor per sub-stream.  The decode kernels give each
// sub-stream to a group of L lanes of a warp (L = 32: one sub-stream per warp; L = 16: two); the L lanes
// hold the L rANS states and decode L consecutive weights per step.  The states that need a word after a
// step take the next words of the sub-stream in lane order (a ballot and a population count give each lane
// its word).  Standalone, the table of a block is copied to shared memory (SM = 1) or read from global
// memory through the L1 cache (SM = 0), and the 5 raw bits of weight i are at bit 5 i of the raw bits.
// Given the reference, the tables are read from global memory, and the raw bits of a step are located by a
// prefix sum of their lengths across the lanes.  With one sub-stream per block and L = 16 this decodes BFR7.
//
// Encoding: sample histogram, choice of the 3 extra bits (standalone), symbols and their histogram, table
// normalization (bf16ref's normalize, made parallel where the result cannot change), then each sub-stream is
// coded by L lanes walking it backwards; the words a step emits are placed by a ballot, as in decoding.
// Words go to a scratch area per sub-stream and are assembled into blocks afterwards.
//
// Every byte-addressed read goes through ld32u (two aligned 32-bit loads and a funnel shift), since BFR7
// blocks are not aligned; buffers are padded by 16 bytes.  Errors set bits in *err.

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef long long i64;

#define RL (1u << 16)
#define PB 14
#define PTOT (1u << PB)
#define PM (PTOT - 1)
#define NSA 2048
#define PBR 13
#define PTR (1u << PBR)
#define PMR (PTR - 1)
#define MB 2
#define NMAG ((1 << (MB + 1)) + (15 - MB - 1) * (1 << MB))
#define NSD (4 * NMAG)
#define NCX 257          // context tables per block given the reference (256 + the zeroed one)

__device__ __forceinline__ u32 ld32u(const u8 *__restrict__ b, u64 a) {
    u64 q = a & ~3ull; u32 lo = __ldg((const u32 *)(b + q)), hi = __ldg((const u32 *)(b + q + 4));
    return __funnelshift_r(lo, hi, (u32)(a & 3) * 8);
}
__device__ __forceinline__ u32 getbits(const u8 *__restrict__ b, u64 bit, u32 k) {
    u64 q = (bit >> 5) << 2; u32 lo = __ldg((const u32 *)(b + q)), hi = __ldg((const u32 *)(b + q + 4));
    return __funnelshift_r(lo, hi, (u32)(bit & 31)) & ((1u << k) - 1);
}
#ifndef G8_EMU
__device__ __forceinline__ u32 lanemask_lt() { u32 r; asm("mov.u32 %0, %%lanemask_lt;" : "=r"(r)); return r; }
__device__ __forceinline__ u32 lanemask_gt() { u32 r; asm("mov.u32 %0, %%lanemask_gt;" : "=r"(r)); return r; }
#define DSMEM(T, name) extern __shared__ T name[]
#endif

// ============================== decoding ==============================
// per block (host): off = byte offset of the block in arc, len, n = weights, out = first weight in the output
// (u16 index), base = first weight of the reference (u16 index), sub0 = index of its first sub-stream.
// SW = weights per sub-stream (0: one sub-stream).

extern "C" __global__ void parse_alone(const u8 *__restrict__ arc, const u64 *b_off, const u64 *b_len, const u64 *b_n,
        const u64 *b_out, const u32 *b_sub0, u64 SW, int L, u32 *es, u16 *vs, u16 *rawmap,
        u64 *d_wb, u64 *d_we, u32 *d_m, u64 *d_out, u64 *d_raw, u64 *d_base, u32 *err) {
    __shared__ u16 sym_s[NSA], f_s[NSA]; __shared__ u32 cum_s[NSA + 1]; __shared__ int ok_s;
    int blk = blockIdx.x, tid = threadIdx.x;
    u64 off = b_off[blk], n = b_n[blk];
    u32 c0 = arc[off], c1 = arc[off + 1], c2 = arc[off + 2], nnz = ld32u(arc, off + 4), nw = ld32u(arc, off + 8);
    u64 ng = (n + 7) / 8, tp = off + 12 + 5 * ng, nsub = SW ? (n + SW - 1) / SW : 1, ip = tp + 4ull * nnz, w0 = ip + 4 * (nsub - 1);
    if (!(c0 < c1 && c1 < c2 && c2 < 8) || nnz < 1 || nnz > NSA || w0 + 2ull * nw != off + b_len[blk]) {
        if (tid == 0) atomicOr(err, 1u);
        return;
    }
    for (u32 k = tid; k < nnz; k += blockDim.x) { sym_s[k] = (u16)ld32u(arc, tp + 4 * k); f_s[k] = (u16)ld32u(arc, tp + 4 * k + 2); }
    __syncthreads();
    if (tid == 0) {
        u32 acc = 0; int ok = 1, prev = -1;
        for (u32 k = 0; k < nnz; k++) { cum_s[k] = acc; acc += f_s[k]; if ((int)sym_s[k] <= prev || sym_s[k] >= NSA || !f_s[k]) ok = 0; prev = sym_s[k]; }
        cum_s[nnz] = acc; if (acc != PTOT) ok = 0;
        ok_s = ok; if (!ok) atomicOr(err, 2u);
    }
    __syncthreads();
    if (!ok_s) return;
#define OTHPOS(bit) ((bit) == 7 ? 15 : (bit))
    for (u32 sl = tid; sl < PTOT; sl += blockDim.x) {
        u32 lo = 0, hi = nnz;                  // last k with cum[k] <= sl
        while (hi - lo > 1) { u32 mid = (lo + hi) >> 1; if (cum_s[mid] <= sl) lo = mid; else hi = mid; }
        u32 s = sym_s[lo], f = f_s[lo];
        es[(u64)blk * PTOT + sl] = (f << 16) | (sl - cum_s[lo]);
        vs[(u64)blk * PTOT + sl] = (u16)(((s >> 3) << 7) | ((s & 1) << OTHPOS(c0)) | (((s >> 1) & 1) << OTHPOS(c1)) | (((s >> 2) & 1) << OTHPOS(c2)));
    }
    if (tid < 32) {
        int rawbit[5], ri = 0;
        for (int bit = 0; bit < 8; bit++) if (bit != (int)c0 && bit != (int)c1 && bit != (int)c2) rawbit[ri++] = bit;
        u32 x = 0; for (int k = 0; k < 5; k++) x |= ((tid >> k) & 1) << OTHPOS(rawbit[k]);
        rawmap[blk * 32 + tid] = (u16)x;
    }
#undef OTHPOS
    u64 sw = SW ? SW : n;
    for (u64 s = tid; s < nsub; s += blockDim.x) {
        u64 a = s * sw, m = n - a < sw ? n - a : sw;
        u32 wb = s ? ld32u(arc, ip + 4 * (s - 1)) : 0, we = s + 1 < nsub ? ld32u(arc, ip + 4 * s) : nw;
        if (wb > we || we > nw || we - wb < 2u * L) atomicOr(err, 4u);
        u64 g = b_sub0[blk] + s;
        d_wb[g] = w0 + 2ull * wb; d_we[g] = w0 + 2ull * we; d_m[g] = (u32)m; d_out[g] = b_out[blk] + a;
        d_raw[g] = (off + 12) * 8 + 5 * a; d_base[g] = 0;
    }
}

// one work item = up to cnt sub-streams of one block, taken by the warps of a thread block.  Each group of L
// lanes decodes U sub-streams at once, their steps interleaved, so that the loads of one hide the latency of
// the others (a warp issues in order, and a step is a chain of dependent loads).
template <int L, int SM, int U>
__device__ void dec_alone_t(const u8 *__restrict__ arc, const u32 *it_blk, const u32 *it_s0, const u32 *it_cnt,
        const u32 *__restrict__ es, const u16 *__restrict__ vs, const u16 *__restrict__ rawmap,
        const u64 *d_wb, const u64 *d_we, const u32 *d_m, const u64 *d_out, const u64 *d_raw, u16 *__restrict__ out, u32 *err) {
    DSMEM(u32, sm);
    u32 *es_s = sm; u16 *vs_s = (u16 *)(sm + (SM ? PTOT : 0)); u16 *rm_s = vs_s + (SM ? PTOT : 0);
    u32 item = blockIdx.x, blk = it_blk[item], s0 = it_s0[item], cnt = it_cnt[item];
    const u32 *esb = es + (u64)blk * PTOT; const u16 *vsb = vs + (u64)blk * PTOT;
    if (SM) {
        const uint4 *a = (const uint4 *)esb, *b = (const uint4 *)vsb;
        for (int k = threadIdx.x; k < PTOT / 4; k += blockDim.x) ((uint4 *)es_s)[k] = a[k];
        for (int k = threadIdx.x; k < PTOT / 8; k += blockDim.x) ((uint4 *)vs_s)[k] = b[k];
    }
    if (threadIdx.x < 32) rm_s[threadIdx.x] = rawmap[blk * 32 + threadIdx.x];
    __syncthreads();
    const u32 *E = SM ? es_s : esb; const u16 *V = SM ? vs_s : vsb;
    const int G = 32 / L; int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
    int g = lane / L, gl = lane % L; u32 gm = L == 32 ? 0xFFFFFFFFu : (0xFFFFu << (16 * g)), lt = lanemask_lt();
    for (u32 base = warp * G * U; base < cnt; base += nwarp * G * U) {
        u32 m[U], x[U]; u64 p[U], pe[U], o[U], rb[U]; bool valid[U];
        u32 mloc = 0;
#pragma unroll
        for (int u = 0; u < U; u++) {
            u32 si = base + g + u * G; valid[u] = si < cnt; u32 s = s0 + si;
            m[u] = valid[u] ? d_m[s] : 0; x[u] = RL; p[u] = pe[u] = o[u] = rb[u] = 0;
            if (valid[u]) { p[u] = d_wb[s]; pe[u] = d_we[s]; x[u] = ld32u(arc, p[u] + 4 * gl); p[u] += 4 * L; o[u] = d_out[s]; rb[u] = d_raw[s]; }
            mloc = max(mloc, m[u]);
        }
        u32 mmax = __reduce_max_sync(0xFFFFFFFFu, mloc);
        for (u32 k0 = 0; k0 < mmax; k0 += L) {
            u32 y[U], v[U], r5[U]; bool act[U], need[U];
#pragma unroll
            for (int u = 0; u < U; u++) {
                act[u] = k0 + gl < m[u]; y[u] = v[u] = r5[u] = 0;
                if (act[u]) { u32 sl = x[u] & PM, e = E[sl]; v[u] = V[sl]; y[u] = (e >> 16) * (x[u] >> PB) + (e & 0xFFFF);
                              r5[u] = getbits(arc, rb[u] + 5ull * (k0 + gl), 5); }
                need[u] = act[u] && y[u] < RL;
            }
#pragma unroll
            for (int u = 0; u < U; u++) {
                u32 bal = __ballot_sync(0xFFFFFFFFu, need[u]) & gm;
                if (need[u]) { u64 a = p[u] + 2 * __popc(bal & lt); if (a + 2 > pe[u]) a = pe[u] - 2; y[u] = (y[u] << 16) | (ld32u(arc, a) & 0xFFFF); }
                p[u] += 2 * __popc(bal);
            }
#pragma unroll
            for (int u = 0; u < U; u++)
                if (act[u]) { x[u] = y[u]; out[o[u] + k0 + gl] = (u16)(v[u] | rm_s[r5[u]]); }
        }
#pragma unroll
        for (int u = 0; u < U; u++) if (valid[u] && (x[u] != RL || p[u] != pe[u])) atomicOr(err, 8u);
    }
}
#define DEC_ALONE(L, SM, U) extern "C" __global__ void dec_alone_##L##_##SM##_u##U(const u8 *__restrict__ arc, const u32 *it_blk, const u32 *it_s0, \
        const u32 *it_cnt, const u32 *__restrict__ es, const u16 *__restrict__ vs, const u16 *__restrict__ rawmap, const u64 *d_wb, \
        const u64 *d_we, const u32 *d_m, const u64 *d_out, const u64 *d_raw, u16 *__restrict__ out, u32 *err) { \
    dec_alone_t<L, SM, U>(arc, it_blk, it_s0, it_cnt, es, vs, rawmap, d_wb, d_we, d_m, d_out, d_raw, out, err); }
DEC_ALONE(16, 0, 1) DEC_ALONE(16, 1, 1) DEC_ALONE(32, 0, 1) DEC_ALONE(32, 1, 1)
DEC_ALONE(16, 0, 2) DEC_ALONE(16, 1, 2) DEC_ALONE(32, 0, 2) DEC_ALONE(32, 1, 2)
DEC_ALONE(16, 0, 4) DEC_ALONE(16, 1, 4) DEC_ALONE(32, 0, 4) DEC_ALONE(32, 1, 4)

// given the reference: per block NCX context tables: slot -> symbol (u8, PTR each) and symbol -> entry (u64):
// bits 0..15 start, 16..31 f, 32..47 A (t = b + A + raw mod 2^16), 48..51 k raw bits; cbm: context -> table
extern "C" __global__ void parse_ref(const u8 *__restrict__ arc, const u64 *b_off, const u64 *b_len, const u64 *b_n,
        const u64 *b_out, const u64 *b_base, const u32 *b_sub0, u64 SW, int L, u8 *slot, u64 *se, u16 *cbm,
        u64 *d_wb, u64 *d_we, u32 *d_m, u64 *d_out, u64 *d_raw, u64 *d_base, u32 *err) {
    __shared__ u64 co_s[256]; __shared__ u32 nz_s[256], cx_s[256]; __shared__ u16 f_s[NSD]; __shared__ u32 cum_s[NSD + 1];
    __shared__ u64 ip_s; __shared__ int ok_s;
    int blk = blockIdx.x, tid = threadIdx.x;
    u64 off = b_off[blk], n = b_n[blk], end = off + b_len[blk];
    u32 rawlen = ld32u(arc, off), nw = ld32u(arc, off + 4);
    u64 nsub = SW ? (n + SW - 1) / SW : 1;
    u32 nact = ld32u(arc, off + 8 + rawlen) & 0xFFFF;
    if (tid == 0) {
        int ok = nact >= 1 && nact <= 256 && 8ull + rawlen + 2 * (u64)nw + 2 <= b_len[blk];
        u64 r = off + 8 + rawlen + 2; int prev = -1;
        for (u32 a = 0; ok && a < nact; a++) {
            if (r + 2 > end) { ok = 0; break; }
            u32 cx = arc[r], nz = arc[r + 1] + 1u;
            if ((int)cx <= prev || nz > NSD) { ok = 0; break; }
            prev = (int)cx; cx_s[a] = cx; nz_s[a] = nz; co_s[a] = r + 2; r += 2 + 3ull * nz;
        }
        ip_s = r;
        if (ok && r + 8 * (nsub - 1) + 2ull * nw != end) ok = 0;
        ok_s = ok; if (!ok) atomicOr(err, 16u);
    }
    __syncthreads();
    if (!ok_s) return;
    for (int c = tid; c < 256; c += blockDim.x) cbm[blk * 256 + c] = (u16)nact;
    __syncthreads();
    for (u32 a = tid; a < nact; a += blockDim.x) cbm[blk * 256 + cx_s[a]] = (u16)a;
    u64 sb = (u64)blk * NCX;
    for (u32 a = 0; a <= nact; a++) {
        for (int s = tid; s < NSD; s += blockDim.x) f_s[s] = 0;
        __syncthreads();
        if (a < nact) for (u32 k = tid; k < nz_s[a]; k += blockDim.x) { u64 q = co_s[a] + 3 * k; u32 s = arc[q]; if (s < NSD) f_s[s] = (u16)(ld32u(arc, q + 1) & 0xFFFF); }
        __syncthreads();
        if (tid == 0) {
            u32 acc = 0; for (int s = 0; s < NSD; s++) { cum_s[s] = acc; acc += f_s[s]; }
            cum_s[NSD] = acc; if (a < nact && acc != PTR) atomicOr(err, 32u);
        }
        __syncthreads();
        for (int s = tid; s < 256; s += blockDim.x) {
            u64 e = 0;
            if (s < NSD && a < nact) {
                u32 flip = s / (2 * NMAG), neg = (s / NMAG) & 1, m = s % NMAG, base, k;
                if (m < (1u << (MB + 1))) { base = m; k = 0; }
                else { u32 Lb = (m - (1u << (MB + 1))) / (1u << MB) + MB + 2, top = (m - (1u << (MB + 1))) % (1u << MB);
                       k = Lb - 1 - MB; base = ((1u << MB) | top) << k; }
                u32 A = neg ? (0u - base - ((1u << k) - 1)) & 0xFFFF : base;
                A = (A + (flip ? 0x8000u : 0)) & 0xFFFF;
                e = cum_s[s] | ((u64)f_s[s] << 16) | ((u64)A << 32) | ((u64)k << 48);
            }
            se[(sb + a) * 256 + s] = e;
        }
        for (u32 sl = tid; sl < PTR; sl += blockDim.x) {
            u32 r = 0;
            if (a < nact) { u32 lo = 0, hi = NSD; while (hi - lo > 1) { u32 mid = (lo + hi) >> 1; if (cum_s[mid] <= sl) lo = mid; else hi = mid; } r = lo; }
            slot[(sb + a) * PTR + sl] = (u8)r;
        }
        __syncthreads();
    }
    u64 ip = ip_s, w0 = ip + 8 * (nsub - 1), sw = SW ? SW : n;
    for (u64 s = tid; s < nsub; s += blockDim.x) {
        u64 a = s * sw, m = n - a < sw ? n - a : sw;
        u32 wb = s ? ld32u(arc, ip + 8 * (s - 1)) : 0, we = s + 1 < nsub ? ld32u(arc, ip + 8 * s) : nw;
        u32 rb = s ? ld32u(arc, ip + 8 * (s - 1) + 4) : 0;
        if (wb > we || we > nw || we - wb < 2u * L) atomicOr(err, 4u);
        u64 g = b_sub0[blk] + s;
        d_wb[g] = w0 + 2ull * wb; d_we[g] = w0 + 2ull * we; d_m[g] = (u32)m; d_out[g] = b_out[blk] + a;
        d_raw[g] = (off + 8) * 8 + rb; d_base[g] = b_base[blk] + a;
    }
}

template <int L, int U>
__device__ void dec_ref_t(const u8 *__restrict__ arc, const u32 *it_blk, const u32 *it_s0, const u32 *it_cnt,
        const u8 *__restrict__ slot, const u64 *__restrict__ se, const u16 *__restrict__ cbm, const u16 *__restrict__ bref,
        const u64 *d_wb, const u64 *d_we, const u32 *d_m, const u64 *d_out, const u64 *d_raw, const u64 *d_base, u16 *__restrict__ out, u32 *err) {
    __shared__ u16 cb_s[256];
    u32 item = blockIdx.x, blk = it_blk[item], s0 = it_s0[item], cnt = it_cnt[item];
    for (int c = threadIdx.x; c < 256; c += blockDim.x) cb_s[c] = cbm[blk * 256 + c];
    __syncthreads();
    const u8 *SL = slot + (u64)blk * NCX * PTR; const u64 *SE = se + (u64)blk * NCX * 256;
    const int G = 32 / L; int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
    int g = lane / L, gl = lane % L; u32 gm = L == 32 ? 0xFFFFFFFFu : (0xFFFFu << (16 * g)), lt = lanemask_lt();
    for (u32 base = warp * G * U; base < cnt; base += nwarp * G * U) {
        u32 m[U], x[U]; u64 p[U], pe[U], o[U], rb[U], bo[U]; bool valid[U];
        u32 mloc = 0;
#pragma unroll
        for (int u = 0; u < U; u++) {
            u32 si = base + g + u * G; valid[u] = si < cnt; u32 s = s0 + si;
            m[u] = valid[u] ? d_m[s] : 0; x[u] = RL; p[u] = pe[u] = o[u] = rb[u] = bo[u] = 0;
            if (valid[u]) { p[u] = d_wb[s]; pe[u] = d_we[s]; x[u] = ld32u(arc, p[u] + 4 * gl); p[u] += 4 * L; o[u] = d_out[s]; rb[u] = d_raw[s]; bo[u] = d_base[s]; }
            mloc = max(mloc, m[u]);
        }
        u32 mmax = __reduce_max_sync(0xFFFFFFFFu, mloc);
        for (u32 k0 = 0; k0 < mmax; k0 += L) {
            u32 y[U], k[U], bb[U]; u64 e[U]; bool act[U], need[U];
#pragma unroll
            for (int u = 0; u < U; u++) { act[u] = k0 + gl < m[u]; bb[u] = act[u] ? bref[bo[u] + k0 + gl] : 0; }
#pragma unroll
            for (int u = 0; u < U; u++) {
                y[u] = k[u] = 0; e[u] = 0;
                if (act[u]) {
                    u32 a = cb_s[(bb[u] >> 7) & 0xFF], sl = x[u] & PMR;
                    e[u] = SE[(u64)a * 256 + SL[(u64)a * PTR + sl]];
                    y[u] = (u32)((e[u] >> 16) & 0xFFFF) * (x[u] >> PBR) + sl - (u32)(e[u] & 0xFFFF);
                    k[u] = (u32)(e[u] >> 48) & 15;
                }
                need[u] = act[u] && y[u] < RL;
            }
#pragma unroll
            for (int u = 0; u < U; u++) {
                u32 bal = __ballot_sync(0xFFFFFFFFu, need[u]) & gm;
                if (need[u]) { u64 a = p[u] + 2 * __popc(bal & lt); if (a + 2 > pe[u]) a = pe[u] - 2; y[u] = (y[u] << 16) | (ld32u(arc, a) & 0xFFFF); }
                p[u] += 2 * __popc(bal);
                u32 inc = k[u];                               // prefix sum of the raw bit counts within the group
#pragma unroll
                for (int d = 1; d < L; d <<= 1) { u32 t = __shfl_up_sync(0xFFFFFFFFu, inc, d, L); if (gl >= d) inc += t; }
                u32 tot = __shfl_sync(0xFFFFFFFFu, inc, L - 1, L);
                if (act[u]) {
                    x[u] = y[u];
                    u32 rv = k[u] ? getbits(arc, rb[u] + inc - k[u], k[u]) : 0;
                    out[o[u] + k0 + gl] = (u16)(bb[u] + (u32)(e[u] >> 32) + rv);
                }
                rb[u] += tot;
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++) if (valid[u] && (x[u] != RL || p[u] != pe[u])) atomicOr(err, 64u);
    }
}
#define DEC_REF(L, U) extern "C" __global__ void dec_ref_##L##_u##U(const u8 *__restrict__ arc, const u32 *it_blk, const u32 *it_s0, const u32 *it_cnt, \
        const u8 *__restrict__ slot, const u64 *__restrict__ se, const u16 *__restrict__ cbm, const u16 *__restrict__ bref, \
        const u64 *d_wb, const u64 *d_we, const u32 *d_m, const u64 *d_out, const u64 *d_raw, const u64 *d_base, u16 *__restrict__ out, u32 *err) { \
    dec_ref_t<L, U>(arc, it_blk, it_s0, it_cnt, slot, se, cbm, bref, d_wb, d_we, d_m, d_out, d_raw, d_base, out, err); }
DEC_REF(16, 1) DEC_REF(32, 1) DEC_REF(16, 2) DEC_REF(32, 2) DEC_REF(16, 4) DEC_REF(32, 4)

// ============================== decoding, v2 (L = 32) ==============================
// The same archives, decoded with no global-memory load on the chain of dependent operations of a step.  Each
// byte stream a warp reads (the sub-stream's words, its raw bits, and given the reference the reference
// weights) goes through a ring in shared memory: RNC chunks of RCB bytes (one u64 per lane), filled two
// chunks ahead of the reader by coalesced loads held in registers (s0, s1), so a load has the time of two
// chunks of reading to arrive.  A step reads at most 64 bytes of a stream, less than a chunk, so the reader
// enters at most one new chunk per step; entering chunk c+1 stores chunk c+3 (staged) into the slot of chunk
// c-1 and stages chunk c+5.  Standalone, the decoding table is smaller than v1's (44 KiB instead of 96 KiB),
// so more warps fit on an SM: slot -> symbol index (u16), and per symbol index f << 16 | cum (u32) and the
// bits it gives the weight (u16).
#define RCB 256
#define RNC 4
#define RQ (RNC * 32)                    // u64 per ring
struct Ring {
    u64 *buf; const u8 *src; u64 lim, gb; u32 ch; u64 s0, s1;
    __device__ __forceinline__ u64 ld(u32 j, int lane) const {
        u64 a = gb + (u64)j * RCB + 8 * lane;
        return a + 8 <= lim ? __ldg((const u64 *)(src + a)) : 0ull;
    }
    // the reader's first byte is at addr; returns its logical offset (bytes from gb)
    __device__ __forceinline__ u32 init(const u8 *s, u64 l, u64 addr, u64 *b, int lane) {
        buf = b; src = s; lim = l; gb = addr & ~7ull; ch = 0;
#pragma unroll
        for (int j = 0; j < 3; j++) buf[j * 32 + lane] = ld(j, lane);
        s0 = ld(3, lane); s1 = ld(4, lane);
        __syncwarp();
        return (u32)(addr - gb);
    }
    // 16-bit little-endian value at logical byte b.  Blocks are not 2-byte aligned, so b can be odd; its parity
    // is the same for every read of a stream, so the branch does not diverge.
    __device__ __forceinline__ u32 u16at(u32 b) const {
        if (!(b & 1)) return ((const u16 *)buf)[(b >> 1) & (RQ * 4 - 1)];
        const u8 *c = (const u8 *)buf; return c[b & (RQ * 8 - 1)] | ((u32)c[(b + 1) & (RQ * 8 - 1)] << 8);
    }
    __device__ __forceinline__ u32 w32(u32 w) const { return ((const u32 *)buf)[w & (RQ * 2 - 1)]; }
    __device__ __forceinline__ u32 bits(u32 q, u32 k) const { return __funnelshift_r(w32(q >> 5), w32((q >> 5) + 1), q & 31) & ((1u << k) - 1); }
    // pos: the reader's logical byte offset after a step (warp-uniform)
    template <bool NL = false> __device__ __forceinline__ void adv(u32 pos, int lane) {
        if ((pos >> 8) != ch) { buf[((ch + 3) & (RNC - 1)) * 32 + lane] = s0; s0 = s1; s1 = NL ? (u64)ch : ld(ch + 5, lane); ch++; __syncwarp(); }
    }
};

extern "C" __global__ void parse_alone2(const u8 *__restrict__ arc, const u64 *b_off, const u64 *b_len, const u64 *b_n,
        const u64 *b_out, const u32 *b_sub0, u64 SW, u16 *ss, u32 *fc, u16 *vv, u16 *rawmap,
        u64 *d_wb, u64 *d_we, u32 *d_m, u64 *d_out, u64 *d_raw, u32 *err) {
    __shared__ u16 sym_s[NSA], f_s[NSA]; __shared__ u32 cum_s[NSA + 1]; __shared__ int ok_s;
    int blk = blockIdx.x, tid = threadIdx.x;
    u64 off = b_off[blk], n = b_n[blk];
    u32 c0 = arc[off], c1 = arc[off + 1], c2 = arc[off + 2], nnz = ld32u(arc, off + 4), nw = ld32u(arc, off + 8);
    u64 ng = (n + 7) / 8, tp = off + 12 + 5 * ng, nsub = SW ? (n + SW - 1) / SW : 1, ip = tp + 4ull * nnz, w0 = ip + 4 * (nsub - 1);
    if (!(c0 < c1 && c1 < c2 && c2 < 8) || nnz < 1 || nnz > NSA || w0 + 2ull * nw != off + b_len[blk]) {
        if (tid == 0) atomicOr(err, 1u);
        return;
    }
    for (u32 k = tid; k < nnz; k += blockDim.x) { sym_s[k] = (u16)ld32u(arc, tp + 4 * k); f_s[k] = (u16)ld32u(arc, tp + 4 * k + 2); }
    __syncthreads();
    if (tid == 0) {
        u32 acc = 0; int ok = 1, prev = -1;
        for (u32 k = 0; k < nnz; k++) { cum_s[k] = acc; acc += f_s[k]; if ((int)sym_s[k] <= prev || sym_s[k] >= NSA || !f_s[k]) ok = 0; prev = sym_s[k]; }
        cum_s[nnz] = acc; if (acc != PTOT) ok = 0;
        ok_s = ok; if (!ok) atomicOr(err, 2u);
    }
    __syncthreads();
    if (!ok_s) return;
#define OTHPOS(bit) ((bit) == 7 ? 15 : (bit))
    for (u32 sl = tid; sl < PTOT; sl += blockDim.x) {
        u32 lo = 0, hi = nnz;
        while (hi - lo > 1) { u32 mid = (lo + hi) >> 1; if (cum_s[mid] <= sl) lo = mid; else hi = mid; }
        ss[(u64)blk * PTOT + sl] = (u16)lo;
    }
    for (u32 k = tid; k < NSA; k += blockDim.x) {
        u32 s = k < nnz ? sym_s[k] : 0;
        fc[(u64)blk * NSA + k] = k < nnz ? ((u32)f_s[k] << 16) | cum_s[k] : 0;
        vv[(u64)blk * NSA + k] = (u16)(((s >> 3) << 7) | ((s & 1) << OTHPOS(c0)) | (((s >> 1) & 1) << OTHPOS(c1)) | (((s >> 2) & 1) << OTHPOS(c2)));
    }
    if (tid < 32) {
        int rawbit[5], ri = 0;
        for (int bit = 0; bit < 8; bit++) if (bit != (int)c0 && bit != (int)c1 && bit != (int)c2) rawbit[ri++] = bit;
        u32 x = 0; for (int k = 0; k < 5; k++) x |= ((tid >> k) & 1) << OTHPOS(rawbit[k]);
        rawmap[blk * 32 + tid] = (u16)x;
    }
#undef OTHPOS
    u64 sw = SW ? SW : n;
    for (u64 s = tid; s < nsub; s += blockDim.x) {
        u64 a = s * sw, m = n - a < sw ? n - a : sw;
        u32 wb = s ? ld32u(arc, ip + 4 * (s - 1)) : 0, we = s + 1 < nsub ? ld32u(arc, ip + 4 * s) : nw;
        if (wb > we || we > nw || we - wb < 64u) atomicOr(err, 4u);
        u64 g = b_sub0[blk] + s;
        d_wb[g] = w0 + 2ull * wb; d_we[g] = w0 + 2ull * we; d_m[g] = (u32)m; d_out[g] = b_out[blk] + a;
        d_raw[g] = (off + 12) * 8 + 5 * a;
    }
}

#define SMA2 (PTOT * 2 + NSA * 4 + NSA * 2 + 64)     // bytes of the standalone v2 table in shared memory
// ABL (ablations, for timing only; the output is wrong): 1 no output stores, 2 no ring refill loads, 4 no table
// reads, 8 no raw-bit reads, 16 no renormalization reads (and given the reference 32: no reference-weight reads)
template <int U, int ABL = 0>
__device__ void dec2_alone_t(const u8 *__restrict__ arc, u64 alen, const u32 *it_blk, const u32 *it_s0, const u32 *it_cnt,
        const u16 *__restrict__ ss, const u32 *__restrict__ fc, const u16 *__restrict__ vv, const u16 *__restrict__ rawmap,
        const u64 *d_wb, const u64 *d_we, const u32 *d_m, const u64 *d_out, const u64 *d_raw, u16 *__restrict__ out, u32 *err) {
    DSMEM(u64, smq);
    u16 *ss_s = (u16 *)smq; u32 *fc_s = (u32 *)(ss_s + PTOT); u16 *vv_s = (u16 *)(fc_s + NSA); u16 *rm_s = vv_s + NSA;
    u64 *rings = smq + SMA2 / 8;
    u32 item = blockIdx.x, blk = it_blk[item], s0 = it_s0[item], cnt = it_cnt[item];
    {
        const uint4 *a = (const uint4 *)(ss + (u64)blk * PTOT), *b = (const uint4 *)(fc + (u64)blk * NSA), *c = (const uint4 *)(vv + (u64)blk * NSA);
        for (int k = threadIdx.x; k < PTOT / 8; k += blockDim.x) ((uint4 *)ss_s)[k] = a[k];
        for (int k = threadIdx.x; k < NSA / 4; k += blockDim.x) ((uint4 *)fc_s)[k] = b[k];
        for (int k = threadIdx.x; k < NSA / 8; k += blockDim.x) ((uint4 *)vv_s)[k] = c[k];
        if (threadIdx.x < 32) rm_s[threadIdx.x] = rawmap[blk * 32 + threadIdx.x];
    }
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5; u32 lt = lanemask_lt();
    u64 *wr = rings + (u64)warp * U * 2 * RQ;
    for (u32 base = warp * U; base < cnt; base += nwarp * U) {
        Ring W[U], Rr[U]; u32 x[U], m[U], pw[U], pe[U], pr[U]; u16 *op[U]; bool valid[U]; u32 mmax = 0;
#pragma unroll
        for (int u = 0; u < U; u++) {
            u32 si = base + u; valid[u] = si < cnt; u32 s = s0 + si;             // warp-uniform
            m[u] = 0; x[u] = RL; pw[u] = pe[u] = pr[u] = 0; op[u] = out;
            if (valid[u]) {
                m[u] = d_m[s]; u64 wb = d_wb[s], we = d_we[s], rb = d_raw[s];
                pw[u] = W[u].init(arc, alen, wb, wr + (2 * u) * RQ, lane); pe[u] = pw[u] + (u32)(we - wb);
                x[u] = W[u].u16at(pw[u] + 4 * lane) | (W[u].u16at(pw[u] + 4 * lane + 2) << 16); pw[u] += 128; W[u].adv(pw[u], lane);
                pr[u] = Rr[u].init(arc, alen, rb >> 3, wr + (2 * u + 1) * RQ, lane) * 8 + (u32)(rb & 7);
                op[u] = out + d_out[s] + lane;
                mmax = max(mmax, m[u]);
            }
        }
        u32 acc = 0;
        for (u32 k0 = 0; k0 < mmax; k0 += 32) {
            u32 y[U], v[U], r5[U]; bool act[U], need[U];
#pragma unroll
            for (int u = 0; u < U; u++) {
                act[u] = k0 + lane < m[u]; y[u] = v[u] = r5[u] = 0;
                if (act[u]) {
                    u32 sl = x[u] & PM, k, e;
                    if (ABL & 4) { k = sl & (NSA - 1); e = (((sl >> 3) | 1) << 16) | (sl & 7); v[u] = x[u] & 0xFFFF; }
                    else { k = ss_s[sl]; e = fc_s[k]; v[u] = vv_s[k]; }
                    y[u] = (e >> 16) * (x[u] >> PB) + sl - (e & 0xFFFF);
                    r5[u] = (ABL & 8) ? (x[u] >> 3) & 31 : Rr[u].bits(pr[u] + 5 * lane, 5);
                }
                need[u] = act[u] && y[u] < RL;
            }
#pragma unroll
            for (int u = 0; u < U; u++) {
                if (!valid[u]) continue;                                           // warp-uniform
                u32 bal = __ballot_sync(0xFFFFFFFFu, need[u]);
                if (need[u]) y[u] = (y[u] << 16) | ((ABL & 16) ? (pw[u] & 0xFFFF) : W[u].u16at(pw[u] + 2 * __popc(bal & lt)));
                pw[u] += 2 * __popc(bal);
                if (act[u]) { x[u] = y[u]; u32 o = v[u] | rm_s[r5[u]]; if (ABL & 1) acc += o; else op[u][k0] = (u16)o; }
                pr[u] += 160;
                W[u].template adv<(ABL & 2) != 0>(pw[u], lane); Rr[u].template adv<(ABL & 2) != 0>(pr[u] >> 3, lane);
            }
        }
        if ((ABL & 1) && acc == 0x9E3779B9u) out[0] = (u16)acc;            // keeps the work of the loop
#pragma unroll
        for (int u = 0; u < U; u++) if (valid[u] && (x[u] != RL || pw[u] != pe[u])) atomicOr(err, 8u);
    }
}
#define DEC2_ALONE(U) extern "C" __global__ void dec2_alone_u##U(const u8 *__restrict__ arc, u64 alen, const u32 *it_blk, const u32 *it_s0, \
        const u32 *it_cnt, const u16 *__restrict__ ss, const u32 *__restrict__ fc, const u16 *__restrict__ vv, const u16 *__restrict__ rawmap, \
        const u64 *d_wb, const u64 *d_we, const u32 *d_m, const u64 *d_out, const u64 *d_raw, u16 *__restrict__ out, u32 *err) { \
    dec2_alone_t<U>(arc, alen, it_blk, it_s0, it_cnt, ss, fc, vv, rawmap, d_wb, d_we, d_m, d_out, d_raw, out, err); }
DEC2_ALONE(1) DEC2_ALONE(2)
#define DEC2_ALONE_A(A) extern "C" __global__ void dec2_alone_a##A(const u8 *__restrict__ arc, u64 alen, const u32 *it_blk, const u32 *it_s0, \
        const u32 *it_cnt, const u16 *__restrict__ ss, const u32 *__restrict__ fc, const u16 *__restrict__ vv, const u16 *__restrict__ rawmap, \
        const u64 *d_wb, const u64 *d_we, const u32 *d_m, const u64 *d_out, const u64 *d_raw, u16 *__restrict__ out, u32 *err) { \
    dec2_alone_t<1, A>(arc, alen, it_blk, it_s0, it_cnt, ss, fc, vv, rawmap, d_wb, d_we, d_m, d_out, d_raw, out, err); }
DEC2_ALONE_A(1) DEC2_ALONE_A(2) DEC2_ALONE_A(4) DEC2_ALONE_A(8) DEC2_ALONE_A(16) DEC2_ALONE_A(31)

// given the reference (tables of parse_ref, in global memory, read through L1): three rings per sub-stream
// (words, raw bits, reference weights)
template <int U>
__device__ void dec2_ref_t(const u8 *__restrict__ arc, u64 alen, const u32 *it_blk, const u32 *it_s0, const u32 *it_cnt,
        const u8 *__restrict__ slot, const u64 *__restrict__ se, const u16 *__restrict__ cbm, const u16 *__restrict__ bref, u64 blen,
        const u64 *d_wb, const u64 *d_we, const u32 *d_m, const u64 *d_out, const u64 *d_raw, const u64 *d_base, u16 *__restrict__ out, u32 *err) {
    __shared__ u16 cb_s[256]; DSMEM(u64, rings);
    u32 item = blockIdx.x, blk = it_blk[item], s0 = it_s0[item], cnt = it_cnt[item];
    for (int c = threadIdx.x; c < 256; c += blockDim.x) cb_s[c] = cbm[blk * 256 + c];
    __syncthreads();
    const u8 *SL = slot + (u64)blk * NCX * PTR; const u64 *SE = se + (u64)blk * NCX * 256;
    const u8 *b8 = (const u8 *)bref;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5; u32 lt = lanemask_lt();
    u64 *wr = rings + (u64)warp * U * 3 * RQ;
    for (u32 base = warp * U; base < cnt; base += nwarp * U) {
        Ring W[U], Rr[U], B[U]; u32 x[U], m[U], pw[U], pe[U], pr[U], pb[U]; u16 *op[U]; bool valid[U]; u32 mmax = 0;
#pragma unroll
        for (int u = 0; u < U; u++) {
            u32 si = base + u; valid[u] = si < cnt; u32 s = s0 + si;
            m[u] = 0; x[u] = RL; pw[u] = pe[u] = pr[u] = pb[u] = 0; op[u] = out;
            if (valid[u]) {
                m[u] = d_m[s]; u64 wb = d_wb[s], we = d_we[s], rb = d_raw[s];
                pw[u] = W[u].init(arc, alen, wb, wr + (3 * u) * RQ, lane); pe[u] = pw[u] + (u32)(we - wb);
                x[u] = W[u].u16at(pw[u] + 4 * lane) | (W[u].u16at(pw[u] + 4 * lane + 2) << 16); pw[u] += 128; W[u].adv(pw[u], lane);
                pr[u] = Rr[u].init(arc, alen, rb >> 3, wr + (3 * u + 1) * RQ, lane) * 8 + (u32)(rb & 7);
                pb[u] = B[u].init(b8, blen, 2 * d_base[s], wr + (3 * u + 2) * RQ, lane);
                op[u] = out + d_out[s] + lane;
                mmax = max(mmax, m[u]);
            }
        }
        for (u32 k0 = 0; k0 < mmax; k0 += 32) {
            u32 y[U], k[U], bb[U], A[U]; bool act[U], need[U];
#pragma unroll
            for (int u = 0; u < U; u++) {
                act[u] = k0 + lane < m[u]; y[u] = k[u] = A[u] = 0; bb[u] = valid[u] ? B[u].u16at(pb[u] + 2 * lane) : 0;
                if (act[u]) {
                    u32 a = cb_s[(bb[u] >> 7) & 0xFF], sl = x[u] & PMR;
                    u64 e = __ldg(&SE[a * 256 + __ldg(&SL[a * PTR + sl])]);
                    y[u] = (u32)((e >> 16) & 0xFFFF) * (x[u] >> PBR) + sl - (u32)(e & 0xFFFF);
                    k[u] = (u32)(e >> 48) & 15; A[u] = (u32)(e >> 32);
                }
                need[u] = act[u] && y[u] < RL;
            }
#pragma unroll
            for (int u = 0; u < U; u++) {
                if (!valid[u]) continue;
                u32 bal = __ballot_sync(0xFFFFFFFFu, need[u]);
                if (need[u]) y[u] = (y[u] << 16) | W[u].u16at(pw[u] + 2 * __popc(bal & lt));
                pw[u] += 2 * __popc(bal);
                u32 inc = k[u];
#pragma unroll
                for (int d = 1; d < 32; d <<= 1) { u32 t = __shfl_up_sync(0xFFFFFFFFu, inc, d); if (lane >= d) inc += t; }
                u32 tot = __shfl_sync(0xFFFFFFFFu, inc, 31);
                if (act[u]) {
                    x[u] = y[u];
                    u32 rv = k[u] ? Rr[u].bits(pr[u] + inc - k[u], k[u]) : 0;
                    op[u][k0] = (u16)(bb[u] + A[u] + rv);
                }
                pr[u] += tot; pb[u] += 64;
                W[u].adv(pw[u], lane); Rr[u].adv(pr[u] >> 3, lane); B[u].adv(pb[u], lane);
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++) if (valid[u] && (x[u] != RL || pw[u] != pe[u])) atomicOr(err, 64u);
    }
}
#define DEC2_REF(U) extern "C" __global__ void dec2_ref_u##U(const u8 *__restrict__ arc, u64 alen, const u32 *it_blk, const u32 *it_s0, const u32 *it_cnt, \
        const u8 *__restrict__ slot, const u64 *__restrict__ se, const u16 *__restrict__ cbm, const u16 *__restrict__ bref, u64 blen, \
        const u64 *d_wb, const u64 *d_we, const u32 *d_m, const u64 *d_out, const u64 *d_raw, const u64 *d_base, u16 *__restrict__ out, u32 *err) { \
    dec2_ref_t<U>(arc, alen, it_blk, it_s0, it_cnt, slot, se, cbm, bref, blen, d_wb, d_we, d_m, d_out, d_raw, d_base, out, err); }
DEC2_REF(1) DEC2_REF(2)

// ============================== decoding, v3 (L = 32, U = 1) ==============================
// The same archives and the rings of v2, with smaller decoding tables, so that a step makes fewer random reads
// of shared memory (v2 is bound by them and by instruction issue, not by latency: U = 2 and longer work items
// were slower).
//   Standalone: two u32 per weight instead of three reads: T1[slot] = k << 14 | (slot - cum_k) and
//   T2[k] = (f_k - 1) << 16 | (the bits symbol k gives the weight).
//   Given the reference: v2 gathers from 8 KiB slot tables in global memory, one per context, scattered over
//   up to 32 cache lines per warp and table.  v3 packs a block's tables into one image of ~20 KiB (Qwen3-14B:
//   ~26 contexts, ~1350 symbols per block) that sits in shared memory: cb[256] (u32, base exponent -> context a
//   << 24 | index of its first symbol), per symbol E (u64, as v2's se entries) and CUM (u16; a sentinel PTR
//   after each context's symbols), and per context 256 buckets of 32 slots SB (u8, the symbol holding the
//   bucket's first slot).  A slot's symbol is SB's, or a later one (0.11 more on average; a short loop over CUM).
//   The raw-bit offsets of the lanes (a prefix sum of k < 16) take 4 ballots instead of 5 shuffles.
extern "C" __global__ void parse_alone3(const u8 *__restrict__ arc, const u64 *b_off, const u64 *b_len, const u64 *b_n,
        const u64 *b_out, const u32 *b_sub0, u64 SW, u32 *t1, u32 *t2, u16 *rawmap,
        u64 *d_wb, u64 *d_we, u32 *d_m, u64 *d_out, u64 *d_raw, u32 *err) {
    __shared__ u16 sym_s[NSA], f_s[NSA]; __shared__ u32 cum_s[NSA + 1]; __shared__ int ok_s;
    int blk = blockIdx.x, tid = threadIdx.x;
    u64 off = b_off[blk], n = b_n[blk];
    u32 c0 = arc[off], c1 = arc[off + 1], c2 = arc[off + 2], nnz = ld32u(arc, off + 4), nw = ld32u(arc, off + 8);
    u64 ng = (n + 7) / 8, tp = off + 12 + 5 * ng, nsub = SW ? (n + SW - 1) / SW : 1, ip = tp + 4ull * nnz, w0 = ip + 4 * (nsub - 1);
    if (!(c0 < c1 && c1 < c2 && c2 < 8) || nnz < 1 || nnz > NSA || w0 + 2ull * nw != off + b_len[blk]) {
        if (tid == 0) atomicOr(err, 1u);
        return;
    }
    for (u32 k = tid; k < nnz; k += blockDim.x) { sym_s[k] = (u16)ld32u(arc, tp + 4 * k); f_s[k] = (u16)ld32u(arc, tp + 4 * k + 2); }
    __syncthreads();
    if (tid == 0) {
        u32 acc = 0; int ok = 1, prev = -1;
        for (u32 k = 0; k < nnz; k++) { cum_s[k] = acc; acc += f_s[k]; if ((int)sym_s[k] <= prev || sym_s[k] >= NSA || !f_s[k]) ok = 0; prev = sym_s[k]; }
        cum_s[nnz] = acc; if (acc != PTOT) ok = 0;
        ok_s = ok; if (!ok) atomicOr(err, 2u);
    }
    __syncthreads();
    if (!ok_s) return;
#define OTHPOS(bit) ((bit) == 7 ? 15 : (bit))
    for (u32 sl = tid; sl < PTOT; sl += blockDim.x) {
        u32 lo = 0, hi = nnz;
        while (hi - lo > 1) { u32 mid = (lo + hi) >> 1; if (cum_s[mid] <= sl) lo = mid; else hi = mid; }
        t1[(u64)blk * PTOT + sl] = (lo << 14) | (sl - cum_s[lo]);
    }
    for (u32 k = tid; k < NSA; k += blockDim.x) {
        u32 s = k < nnz ? sym_s[k] : 0;
        u32 v = ((s >> 3) << 7) | ((s & 1) << OTHPOS(c0)) | (((s >> 1) & 1) << OTHPOS(c1)) | (((s >> 2) & 1) << OTHPOS(c2));
        t2[(u64)blk * NSA + k] = k < nnz ? ((u32)(f_s[k] - 1) << 16) | v : 0;
    }
    if (tid < 32) {
        int rawbit[5], ri = 0;
        for (int bit = 0; bit < 8; bit++) if (bit != (int)c0 && bit != (int)c1 && bit != (int)c2) rawbit[ri++] = bit;
        u32 x = 0; for (int k = 0; k < 5; k++) x |= ((tid >> k) & 1) << OTHPOS(rawbit[k]);
        rawmap[blk * 32 + tid] = (u16)x;
    }
#undef OTHPOS
    u64 sw = SW ? SW : n;
    for (u64 s = tid; s < nsub; s += blockDim.x) {
        u64 a = s * sw, m = n - a < sw ? n - a : sw;
        u32 wb = s ? ld32u(arc, ip + 4 * (s - 1)) : 0, we = s + 1 < nsub ? ld32u(arc, ip + 4 * s) : nw;
        if (wb > we || we > nw || we - wb < 64u) atomicOr(err, 4u);
        u64 g = b_sub0[blk] + s;
        d_wb[g] = w0 + 2ull * wb; d_we[g] = w0 + 2ull * we; d_m[g] = (u32)m; d_out[g] = b_out[blk] + a;
        d_raw[g] = (off + 12) * 8 + 5 * a;
    }
}

#define SMA3 (PTOT * 4 + NSA * 4 + 64)     // bytes of the standalone v3 table in shared memory
extern "C" __global__ void dec3_alone(const u8 *__restrict__ arc, u64 alen, const u32 *it_blk, const u32 *it_s0, const u32 *it_cnt,
        const u32 *__restrict__ t1, const u32 *__restrict__ t2, const u16 *__restrict__ rawmap,
        const u64 *d_wb, const u64 *d_we, const u32 *d_m, const u64 *d_out, const u64 *d_raw, u16 *__restrict__ out, u32 *err) {
    DSMEM(u64, smq);
    u32 *t1_s = (u32 *)smq, *t2_s = t1_s + PTOT; u16 *rm_s = (u16 *)(t2_s + NSA);
    u64 *rings = smq + SMA3 / 8;
    u32 item = blockIdx.x, blk = it_blk[item], s0 = it_s0[item], cnt = it_cnt[item];
    {
        const uint4 *a = (const uint4 *)(t1 + (u64)blk * PTOT), *b = (const uint4 *)(t2 + (u64)blk * NSA);
        for (int k = threadIdx.x; k < PTOT / 4; k += blockDim.x) ((uint4 *)t1_s)[k] = a[k];
        for (int k = threadIdx.x; k < NSA / 4; k += blockDim.x) ((uint4 *)t2_s)[k] = b[k];
        if (threadIdx.x < 32) rm_s[threadIdx.x] = rawmap[blk * 32 + threadIdx.x];
    }
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5; u32 lt = lanemask_lt();
    u64 *wr = rings + (u64)warp * 2 * RQ;
    for (u32 si = warp; si < cnt; si += nwarp) {
        Ring W, Rr; u32 s = s0 + si;
        u32 m = d_m[s]; u64 wb = d_wb[s], we = d_we[s], rb = d_raw[s];
        u32 pw = W.init(arc, alen, wb, wr, lane), pe = pw + (u32)(we - wb);
        u32 x = W.u16at(pw + 4 * lane) | (W.u16at(pw + 4 * lane + 2) << 16); pw += 128; W.adv(pw, lane);
        u32 pr = Rr.init(arc, alen, rb >> 3, wr + RQ, lane) * 8 + (u32)(rb & 7);
        u16 *op = out + d_out[s] + lane;
        for (u32 k0 = 0; k0 < m; k0 += 32) {
            bool act = k0 + lane < m; u32 y = 0, v = 0, r5 = 0;
            if (act) {
                u32 sl = x & PM, e1 = t1_s[sl], e2 = t2_s[e1 >> 14];
                y = ((e2 >> 16) + 1) * (x >> PB) + (e1 & 0x3FFF); v = e2 & 0xFFFF;
                r5 = Rr.bits(pr + 5 * lane, 5);
            }
            bool need = act && y < RL;
            u32 bal = __ballot_sync(0xFFFFFFFFu, need);
            if (need) y = (y << 16) | W.u16at(pw + 2 * __popc(bal & lt));
            pw += 2 * __popc(bal);
            if (act) { x = y; op[k0] = (u16)(v | rm_s[r5]); }
            pr += 160;
            W.adv(pw, lane); Rr.adv(pr >> 3, lane);
        }
        if (x != RL || pw != pe) atomicOr(err, 8u);
    }
}

// given the reference: sizes of the v3 table images (one thread per block; the same checks as parse_ref)
#define CB3 1024                          // bytes of cb at the start of an image
extern "C" __global__ void size_ref3(const u8 *__restrict__ arc, const u64 *b_off, const u64 *b_len, u32 nb, u64 *tsz, u32 *tnt, u32 *err) {
    u32 blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= nb) return;
    u64 off = b_off[blk], end = off + b_len[blk];
    u32 rawlen = ld32u(arc, off);
    // the fixed header, the raw area and the context count must lie in the block before nact is loaded
    if (10ull + rawlen > b_len[blk]) { atomicOr(err, 16u); tsz[blk] = 16; tnt[blk] = 0; return; }
    u32 nact = ld32u(arc, off + 8 + rawlen) & 0xFFFF;
    u64 r = off + 8 + rawlen + 2, ntot = 0;
    if (nact < 1 || nact > 256) { atomicOr(err, 16u); tsz[blk] = 16; tnt[blk] = 0; return; }
    for (u32 a = 0; a < nact; a++) {
        if (r + 2 > end) { atomicOr(err, 16u); tsz[blk] = 16; tnt[blk] = 0; return; }
        u32 nz = arc[r + 1] + 1u; ntot += nz + 1; r += 2 + 3ull * nz;
    }
    tsz[blk] = (CB3 + 10 * ntot + 256ull * nact + 15) & ~15ull; tnt[blk] = (u32)ntot;
}

// the images (thread 0 walks the contexts; then one warp per context: scans over the 256 symbols in s order)
extern "C" __global__ void parse_ref3(const u8 *__restrict__ arc, const u64 *b_off, const u64 *b_len, const u64 *b_n,
        const u64 *b_out, const u64 *b_base, const u32 *b_sub0, u64 SW, int L, u8 *tab, const u64 *tofs, const u64 *tsz, const u32 *tnt,
        u64 *d_wb, u64 *d_we, u32 *d_m, u64 *d_out, u64 *d_raw, u64 *d_base, u32 *err) {
    __shared__ u64 co_s[256]; __shared__ u32 nz_s[256], cx_s[256], bs_s[256]; __shared__ u64 ip_s; __shared__ u32 ntot_s; __shared__ int ok_s;
    __shared__ u16 fw_s[8][256], cw_s[8][260];
    int blk = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    u64 off = b_off[blk], n = b_n[blk], end = off + b_len[blk];
    u32 rawlen = ld32u(arc, off), nw = ld32u(arc, off + 4);
    u64 nsub = SW ? (n + SW - 1) / SW : 1;
    // as in size_ref3: nact is loaded only when the header, the raw area and the count lie in the block
    u32 nact = 10ull + rawlen <= b_len[blk] ? ld32u(arc, off + 8 + rawlen) & 0xFFFF : 0;
    if (tid == 0) {
        int ok = nact >= 1 && nact <= 256 && 8ull + rawlen + 2 * (u64)nw + 2 <= b_len[blk];
        u64 r = off + 8 + rawlen + 2; int prev = -1; u32 bs = 0;
        for (u32 a = 0; ok && a < nact; a++) {
            if (r + 2 > end) { ok = 0; break; }
            u32 cx = arc[r], nz = arc[r + 1] + 1u;
            if ((int)cx <= prev || nz > NSD) { ok = 0; break; }
            prev = (int)cx; cx_s[a] = cx; nz_s[a] = nz; co_s[a] = r + 2; bs_s[a] = bs; bs += nz + 1; r += 2 + 3ull * nz;
        }
        ip_s = r; ntot_s = bs;
        if (ok && r + 8 * (nsub - 1) + 2ull * nw != end) ok = 0;
        if (ok && (CB3 + 10ull * bs + 256ull * nact + 15) / 16 * 16 != tsz[blk] || bs != tnt[blk]) ok = 0;
        ok_s = ok; if (!ok) atomicOr(err, 16u);
    }
    __syncthreads();
    if (!ok_s) return;
    u8 *img = tab + tofs[blk]; u32 *cb = (u32 *)img; u64 *E = (u64 *)(img + CB3); u32 ntot = ntot_s;
    u16 *CUM = (u16 *)(img + CB3 + 8ull * ntot); u8 *SB = img + CB3 + 10ull * ntot;
    for (int c = tid; c < 256; c += blockDim.x) cb[c] = 0;                    // unused exponents: context 0
    __syncthreads();
    for (u32 a = tid; a < nact; a += blockDim.x) cb[cx_s[a]] = (a << 24) | bs_s[a];
    for (u32 a = warp; a < nact; a += blockDim.x >> 5) {
        u16 *fw = fw_s[warp], *cw = cw_s[warp];
        for (int s = lane; s < 256; s += 32) fw[s] = 0;
        __syncwarp();
        for (u32 k = lane; k < nz_s[a]; k += 32) { u64 q = co_s[a] + 3 * k; u32 s = arc[q]; if (s < NSD) fw[s] = (u16)(ld32u(arc, q + 1) & 0xFFFF); }
        __syncwarp();
        u32 carry = 0, cnt = 0, bs = bs_s[a];
        for (int c = 0; c < 256 / 32; c++) {
            u32 s = c * 32 + lane, f = fw[s], inc = f;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) { u32 t = __shfl_up_sync(0xFFFFFFFFu, inc, d); if (lane >= d) inc += t; }
            u32 cum = carry + inc - f, bal = __ballot_sync(0xFFFFFFFFu, f > 0), j = cnt + __popc(bal & lanemask_lt());
            if (f) {
                u32 flip = s / (2 * NMAG), neg = (s / NMAG) & 1, m = s % NMAG, base, k;
                if (m < (1u << (MB + 1))) { base = m; k = 0; }
                else { u32 Lb = (m - (1u << (MB + 1))) / (1u << MB) + MB + 2, top = (m - (1u << (MB + 1))) % (1u << MB);
                       k = Lb - 1 - MB; base = ((1u << MB) | top) << k; }
                u32 A = neg ? (0u - base - ((1u << k) - 1)) & 0xFFFF : base;
                A = (A + (flip ? 0x8000u : 0)) & 0xFFFF;
                E[bs + j] = cum | ((u64)f << 16) | ((u64)A << 32) | ((u64)k << 48); CUM[bs + j] = (u16)cum; cw[j] = (u16)cum;
            }
            carry += __shfl_sync(0xFFFFFFFFu, inc, 31); cnt += __popc(bal);
        }
        if (lane == 0) { E[bs + cnt] = 0; CUM[bs + cnt] = (u16)PTR; cw[cnt] = (u16)PTR; if (carry != PTR || !cnt) atomicOr(err, 32u); }
        for (u32 j = cnt + 1 + lane; j <= nz_s[a]; j += 32) { E[bs + j] = 0; CUM[bs + j] = (u16)PTR; }
        __syncwarp();
        for (int b = lane; b < 256; b += 32) {
            u32 sl = b * 32, lo = 0, hi = cnt;                                       // the last j < cnt with cw[j] <= sl
            while (hi - lo > 1) { u32 mid = (lo + hi) >> 1; if (cw[mid] <= sl) lo = mid; else hi = mid; }
            SB[a * 256 + b] = (u8)lo;
        }
        __syncwarp();
    }
    u64 ip = ip_s, w0 = ip + 8 * (nsub - 1), sw = SW ? SW : n;
    for (u64 s = tid; s < nsub; s += blockDim.x) {
        u64 a = s * sw, m = n - a < sw ? n - a : sw;
        u32 wb = s ? ld32u(arc, ip + 8 * (s - 1)) : 0, we = s + 1 < nsub ? ld32u(arc, ip + 8 * s) : nw;
        u32 rb = s ? ld32u(arc, ip + 8 * (s - 1) + 4) : 0;
        if (wb > we || we > nw || we - wb < 2u * L || rb > 8ull * rawlen) atomicOr(err, 4u);
        u64 g = b_sub0[blk] + s;
        d_wb[g] = w0 + 2ull * wb; d_we[g] = w0 + 2ull * we; d_m[g] = (u32)m; d_out[g] = b_out[blk] + a;
        d_raw[g] = (off + 8) * 8 + rb; d_base[g] = b_base[blk] + a;
    }
}

// one sub-stream per warp; img: the block's image, in shared memory when it fits (SH), else in global memory
template <bool SH, int ABL = 0>
__device__ __forceinline__ void dec3_ref_w(const u8 *img, u32 ntot, const u8 *__restrict__ arc, u64 alen, const u8 *b8, u64 blen, u32 s,
        u64 *wr, int lane, u32 lt, const u64 *d_wb, const u64 *d_we, const u32 *d_m, const u64 *d_out, const u64 *d_raw,
        const u64 *d_base, u16 *__restrict__ out, u32 *err) {
    const u32 *cb = (const u32 *)img; const u64 *E = (const u64 *)(img + CB3);
    const u16 *CUM = (const u16 *)(img + CB3 + 8 * ntot); const u8 *SB = img + CB3 + 10 * ntot;
    Ring W, Rr, B;
    u32 m = d_m[s]; u64 wb = d_wb[s], we = d_we[s], rb = d_raw[s];
    u32 pw = W.init(arc, alen, wb, wr, lane), pe = pw + (u32)(we - wb);
    u32 x = W.u16at(pw + 4 * lane) | (W.u16at(pw + 4 * lane + 2) << 16); pw += 128; W.adv(pw, lane);
    u32 pr = Rr.init(arc, alen, rb >> 3, wr + RQ, lane) * 8 + (u32)(rb & 7);
    u32 pb = B.init(b8, blen, 2 * d_base[s], wr + 2 * RQ, lane);
    u16 *op = out + d_out[s] + lane;
    u32 acc = 0;
    for (u32 k0 = 0; k0 < m; k0 += 32) {
        bool act = k0 + lane < m; u32 y = 0, k = 0, A = 0, bb = (ABL & 32) ? (x ^ k0) & 0xFFFF : B.u16at(pb + 2 * lane);
        if (act) {
            u32 sl = x & PMR; u64 e;
            if (ABL & 4) e = ((u64)((sl >> 3) | 1) << 16) | (sl & 7) | ((u64)(bb & 3) << 48) | ((u64)bb << 32);
            else {
                u32 c = cb[(bb >> 7) & 0xFF], j = (c & 0xFFFFFF) + SB[((c >> 24) << 8) | (sl >> 5)];
                while (CUM[j + 1] <= sl) j++;
                e = E[j];
            }
            y = (u32)((e >> 16) & 0xFFFF) * (x >> PBR) + sl - (u32)(e & 0xFFFF);
            k = (u32)(e >> 48) & 15; A = (u32)(e >> 32) & 0xFFFF;
        }
        bool need = act && y < RL;
        u32 bal = __ballot_sync(0xFFFFFFFFu, need);
        if (need) y = (y << 16) | ((ABL & 16) ? (pw & 0xFFFF) : W.u16at(pw + 2 * __popc(bal & lt)));
        pw += 2 * __popc(bal);
        if (ABL & 8) k = 0;
        u32 b0 = __ballot_sync(0xFFFFFFFFu, k & 1), b1 = __ballot_sync(0xFFFFFFFFu, k & 2), b2 = __ballot_sync(0xFFFFFFFFu, k & 4),
            b3 = __ballot_sync(0xFFFFFFFFu, k & 8);
        u32 ex = __popc(b0 & lt) + 2 * __popc(b1 & lt) + 4 * __popc(b2 & lt) + 8 * __popc(b3 & lt);
        u32 tot = __popc(b0) + 2 * __popc(b1) + 4 * __popc(b2) + 8 * __popc(b3);
        if (act) {
            x = y;
            u32 rv = k ? Rr.bits(pr + ex, k) : 0;
            u32 o = bb + A + rv; if (ABL & 1) acc += o; else op[k0] = (u16)o;
        }
        pr += tot; pb += 64;
        constexpr bool NL = (ABL & 2) != 0;
        W.template adv<NL>(pw, lane); Rr.template adv<NL>(pr >> 3, lane); B.template adv<NL>(pb, lane);
    }
    if ((ABL & 1) && acc == 0x9E3779B9u) out[0] = (u16)acc;
    if (x != RL || pw != pe) atomicOr(err, 64u);
}
// smt: bytes of dynamic shared memory for an image (the rings follow); a block whose image is larger reads it
// from global memory
template <int ABL>
__device__ void dec3_ref_t(const u8 *__restrict__ arc, u64 alen, const u32 *it_blk, const u32 *it_s0, const u32 *it_cnt,
        const u8 *__restrict__ tab, const u64 *tofs, const u64 *tsz, const u32 *tnt, u32 smt, const u16 *__restrict__ bref, u64 blen,
        const u64 *d_wb, const u64 *d_we, const u32 *d_m, const u64 *d_out, const u64 *d_raw, const u64 *d_base, u16 *__restrict__ out, u32 *err) {
    DSMEM(u64, smq);
    u32 item = blockIdx.x, blk = it_blk[item], s0 = it_s0[item], cnt = it_cnt[item], nt = tnt[blk];
    u64 sz = tsz[blk]; const u8 *g = tab + tofs[blk]; bool sh = sz <= smt;
    if (sh) for (u32 k = threadIdx.x; k < sz / 16; k += blockDim.x) ((uint4 *)smq)[k] = ((const uint4 *)g)[k];
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5; u32 lt = lanemask_lt();
    u64 *wr = smq + smt / 8 + (u64)warp * 3 * RQ;
    const u8 *b8 = (const u8 *)bref;
    for (u32 si = warp; si < cnt; si += nwarp) {
        if (sh) dec3_ref_w<true, ABL>((const u8 *)smq, nt, arc, alen, b8, blen, s0 + si, wr, lane, lt, d_wb, d_we, d_m, d_out, d_raw, d_base, out, err);
        else dec3_ref_w<false, ABL>(g, nt, arc, alen, b8, blen, s0 + si, wr, lane, lt, d_wb, d_we, d_m, d_out, d_raw, d_base, out, err);
    }
}
#define DEC3_REF(N, A) extern "C" __global__ void N(const u8 *__restrict__ arc, u64 alen, const u32 *it_blk, const u32 *it_s0, const u32 *it_cnt, \
        const u8 *__restrict__ tab, const u64 *tofs, const u64 *tsz, const u32 *tnt, u32 smt, const u16 *__restrict__ bref, u64 blen, \
        const u64 *d_wb, const u64 *d_we, const u32 *d_m, const u64 *d_out, const u64 *d_raw, const u64 *d_base, u16 *__restrict__ out, u32 *err) { \
    dec3_ref_t<A>(arc, alen, it_blk, it_s0, it_cnt, tab, tofs, tsz, tnt, smt, bref, blen, d_wb, d_we, d_m, d_out, d_raw, d_base, out, err); }
DEC3_REF(dec3_ref, 0) DEC3_REF(dec3_ref_a1, 1) DEC3_REF(dec3_ref_a2, 2) DEC3_REF(dec3_ref_a4, 4) DEC3_REF(dec3_ref_a8, 8)
DEC3_REF(dec3_ref_a16, 16) DEC3_REF(dec3_ref_a32, 32) DEC3_REF(dec3_ref_a63, 63)

// ============================== encoding ==============================
// chunks (host): ch_blk, ch_a (first weight within the block), ch_n.  Per block: b_in (u16 index of its first
// weight in the input), b_base (same in the reference), b_k (index of its first weight in the key arrays).

__device__ __forceinline__ u32 oth(u32 w) { return (w & 0x7F) | ((w >> 8) & 0x80); }

// sample histogram (every 8th weight of a block), hs: 65536 per block
// shared-memory histograms of 16-bit counters, two per 32-bit word (a chunk adds fewer than 65536 to any counter)
__device__ __forceinline__ void hist16_add(u32 *h, u32 k, u32 v) { atomicAdd(&h[k >> 1], v << (16 * (k & 1))); }
__device__ void hist16_flush(u32 *h, u32 nbins, u32 *dst) {
    for (u32 w = threadIdx.x; w < nbins / 2; w += blockDim.x) {
        u32 x = h[w];
        if (x & 0xFFFF) atomicAdd(&dst[2 * w], x & 0xFFFF);
        if (x >> 16) atomicAdd(&dst[2 * w + 1], x >> 16);
    }
}

extern "C" __global__ void enc_sample(const u16 *__restrict__ t, const u32 *ch_blk, const u64 *ch_a, const u32 *ch_n, const u64 *b_in, u32 *hs) {
    DSMEM(u32, h);                                          // 65536 counters (128 KiB)
    u32 c = blockIdx.x, blk = ch_blk[c]; u64 a = ch_a[c]; u32 n = ch_n[c];   // chunks start at multiples of 32 weights
    for (u32 w = threadIdx.x; w < 32768; w += blockDim.x) h[w] = 0;
    __syncthreads();
    u32 wbase = threadIdx.x & ~31u;
    for (u64 j = 0; a + 8 * (j * blockDim.x + wbase) < a + n; j++) {        // warp-uniform loop
        u64 i = a + 8 * (j * blockDim.x + threadIdx.x); bool act = i < a + n;
        u32 w = act ? t[b_in[blk] + i] : 0;
        u32 peers = __match_any_sync(0xFFFFFFFFu, act ? w : 0x10000u);
        if (act && (__ffs(peers) - 1) == (int)(threadIdx.x & 31)) hist16_add(h, w, (u32)__popc(peers));
    }
    __syncthreads();
    hist16_flush(h, 65536, hs + (u64)blk * 65536);
}

// choice of the 3 bits per block, exactly as bf16ref's enc_alone: 56 candidates; maximize the sum over the
// sample's exponents of c log2 c over the 8 cells (in the same integer fixed point as bf16ref.c's clogc_fx, so
// that CPU and GPU choose the same bits); on an exact tie the first candidate in loop order wins
#define LG_FRAC 24
__device__ __forceinline__ u64 log2_fx(u32 c) {
    int L = 31 - __clz((int)c);
    u64 m = (u64)c << (31 - L), f = 0;
    for (int k = 0; k < LG_FRAC; k++) { m = (m * m) >> 31; f <<= 1; if (m >= (1ull << 32)) { m >>= 1; f |= 1; } }
    return ((u64)L << LG_FRAC) | f;
}
extern "C" __global__ void enc_choose(const u32 *__restrict__ hs, u8 *sel, u8 *rawt, u8 *bcs) {
    __shared__ u32 ho[256]; __shared__ u64 hsum[56]; __shared__ int any_s;
    int blk = blockIdx.x, tid = threadIdx.x;               // 256 threads
    int a = 0, b = 0, d = 0;
    if (tid < 56) { int c = 0; for (int x = 0; x < 8; x++) for (int y = x + 1; y < 8; y++) for (int z = y + 1; z < 8; z++) { if (c == tid) { a = x; b = y; d = z; } c++; } }
    u64 h = 0;
    const u32 *H = hs + (u64)blk * 65536;
    for (int e = 0; e < 256; e++) {
        if (tid == 0) any_s = 0;
        __syncthreads();
        u32 v = tid; u32 cval = H[((v & 0x80) << 8) | (e << 7) | (v & 0x7F)];
        ho[tid] = cval; if (cval) any_s = 1;
        __syncthreads();
        if (any_s && tid < 56) {
            u32 cc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int w = 0; w < 256; w++) cc[((w >> a) & 1) | (((w >> b) & 1) << 1) | (((w >> d) & 1) << 2)] += ho[w];
            for (int i = 0; i < 8; i++) if (cc[i] > 1) h += (u64)cc[i] * log2_fx(cc[i]);
        }
        __syncthreads();
    }
    if (tid < 56) hsum[tid] = h;
    __syncthreads();
    __shared__ int bc[3];
    if (tid == 0) {
        u64 best = 0; int have = 0, c = 0;
        bc[0] = 4; bc[1] = 5; bc[2] = 6;
        for (int x = 0; x < 8; x++) for (int y = x + 1; y < 8; y++) for (int z = y + 1; z < 8; z++) {
            if (!have || hsum[c] > best) { best = hsum[c]; have = 1; bc[0] = x; bc[1] = y; bc[2] = z; }
            c++;
        }
        bcs[blk * 4] = (u8)bc[0]; bcs[blk * 4 + 1] = (u8)bc[1]; bcs[blk * 4 + 2] = (u8)bc[2]; bcs[blk * 4 + 3] = 0;
    }
    __syncthreads();
    {
        int v = tid, sv = 0, r = 0, ri = 0;
        for (int bit = 0; bit < 8; bit++) {
            int x = (v >> bit) & 1;
            if (bit == bc[0]) sv |= x; else if (bit == bc[1]) sv |= x << 1; else if (bit == bc[2]) sv |= x << 2; else r |= x << ri++;
        }
        sel[blk * 256 + v] = (u8)sv; rawt[blk * 256 + v] = (u8)r;
    }
}

// symbols (keys), their histogram (2048 per block) and the raw bits (5 per weight, at bit 5 i of the block's
// raw area in rawbuf, which starts at b_raw[blk]); chunks start at multiples of 8 weights of their block
extern "C" __global__ void enc_sym_alone(const u16 *__restrict__ t, const u32 *ch_blk, const u64 *ch_a, const u32 *ch_n,
        const u64 *b_in, const u64 *b_k, const u64 *b_raw, const u8 *__restrict__ sel, const u8 *__restrict__ rawt,
        u16 *key, u32 *hist, u8 *rawbuf) {
    __shared__ u32 h_s[NSA]; __shared__ u8 sel_s[256], raw_s[256];
    u32 c = blockIdx.x, blk = ch_blk[c]; u64 a = ch_a[c]; u32 n = ch_n[c];
    for (int k = threadIdx.x; k < NSA; k += blockDim.x) h_s[k] = 0;
    for (int k = threadIdx.x; k < 256; k += blockDim.x) { sel_s[k] = sel[blk * 256 + k]; raw_s[k] = rawt[blk * 256 + k]; }
    __syncthreads();
    const u16 *T = t + b_in[blk]; u16 *K = key + b_k[blk]; u8 *R = rawbuf + b_raw[blk];
    for (u64 g = a / 8 + threadIdx.x; 8 * g < a + n; g += blockDim.x) {
        u64 v = 0; u64 i0 = 8 * g; u32 cnt = (u32)min((u64)8, a + n - i0);
        for (u32 j = 0; j < cnt; j++) {
            u32 w = T[i0 + j], o = oth(w), s = ((w >> 4) & 0x7F8) | sel_s[o];
            K[i0 + j] = (u16)s; atomicAdd(&h_s[s], 1u); v |= (u64)raw_s[o] << (5 * j);
        }
        u8 *q = R + 5 * g; q[0] = (u8)v; q[1] = (u8)(v >> 8); q[2] = (u8)(v >> 16); q[3] = (u8)(v >> 24); q[4] = (u8)(v >> 32);
    }
    __syncthreads();
    for (int k = threadIdx.x; k < NSA; k += blockDim.x) if (h_s[k]) atomicAdd(&hist[(u64)blk * NSA + k], h_s[k]);
}

// bf16ref's normalize, for one table of nsym counts, by one thread
__device__ void normalize1(const u32 *cnt, u16 *freq, int n, u64 total, u32 ptot) {
    u32 sum = 0;
    for (int s = 0; s < n; s++) {
        u32 f = 0;
        if (cnt[s]) { f = (u32)(((u64)cnt[s] * ptot + total / 2) / total); if (!f) f = 1; }
        freq[s] = (u16)f; sum += f;
    }
    while (sum != ptot) {
        int best = -1;
        for (int s = 0; s < n; s++) if (freq[s] && (best < 0 || freq[s] > freq[best])) best = s;
        if (sum < ptot) { freq[best] += (u16)(ptot - sum); sum = ptot; }
        else { u32 d = sum - ptot, room = freq[best] - 1u, take = d < room ? d : room;
               if (!take) return;
               freq[best] -= (u16)take; sum -= take; }
    }
}

// standalone tables: frequencies, encoder entries (f << 16 | start) and the table bytes (nnz x (u16 sym, u16 f))
extern "C" __global__ void enc_norm_alone(const u32 *hist, const u64 *b_n, u32 *etab, u8 *tabbytes, u32 *nnz_out) {
    __shared__ u16 f_s[NSA];
    int blk = blockIdx.x;
    if (threadIdx.x == 0) {
        const u32 *h = hist + (u64)blk * NSA;
        normalize1(h, f_s, NSA, b_n[blk], PTOT);
        u32 acc = 0, nz = 0; u8 *q = tabbytes + (u64)blk * NSA * 4;
        for (int s = 0; s < NSA; s++) {
            etab[(u64)blk * NSA + s] = ((u32)f_s[s] << 16) | acc; acc += f_s[s];
            if (f_s[s]) { q[0] = (u8)s; q[1] = (u8)(s >> 8); q[2] = (u8)f_s[s]; q[3] = (u8)(f_s[s] >> 8); q += 4; nz++; }
        }
        nnz_out[blk] = nz;
    }
}

// given the reference: keys (context x 224 + symbol), raw value and length (rk = rv | k << 12), histogram
// (256 x 224 per block) and the raw bit count of each sub-stream (sbits); sub-streams of SW weights (0: one; else a multiple of 32)
extern "C" __global__ void enc_sym_ref(const u16 *__restrict__ t, const u16 *__restrict__ bref, const u32 *ch_blk, const u64 *ch_a,
        const u32 *ch_n, const u64 *b_in, const u64 *b_base, const u64 *b_k, const u32 *b_sub0, u64 SW,
        u16 *key, u16 *rk, u32 *hist, u32 *sbits) {
    DSMEM(u32, h);                                          // 256 x 224 counters (112 KiB); chunks have fewer than 65536 weights
    u32 c = blockIdx.x, blk = ch_blk[c]; u64 a = ch_a[c]; u32 n = ch_n[c];
    const u16 *T = t + b_in[blk], *B = bref + b_base[blk];
    for (u32 w = threadIdx.x; w < 256 * NSD / 2; w += blockDim.x) h[w] = 0;
    __syncthreads();
    u32 wbase = threadIdx.x & ~31u;
    for (u64 j = 0; a + j * blockDim.x + wbase < a + n; j++) {               // warp-uniform loop
        u64 i = a + j * blockDim.x + threadIdx.x; bool act = i < a + n;
        u32 tw = act ? T[i] : 0, bw = act ? B[i] : 0;
        int d = (int)(tw & 0x7FFF) - (int)(bw & 0x7FFF), sg = d >> 31;
        u32 av = (u32)((d ^ sg) - sg), neg = (u32)sg & 1, flip = (tw ^ bw) >> 15;
        int Lb = 32 - __clz(av | 1), v = Lb - 1 - MB; u32 sh = (u32)(v & ~(v >> 31));
        u32 m = Lb <= MB + 1 ? av : (1u << (MB + 1)) + (u32)(Lb - MB - 2) * (1u << MB) + ((av >> sh) & ((1u << MB) - 1));
        u32 s = (flip * 2 + neg) * NMAG + m, mk = (1u << sh) - 1, cx = (bw >> 7) & 0xFF, kk = cx * NSD + s;
        if (act) { key[b_k[blk] + i] = (u16)kk; rk[b_k[blk] + i] = (u16)(((av & mk) ^ (mk & (u32)sg)) | (sh << 12)); }
        u32 peers = __match_any_sync(0xFFFFFFFFu, act ? kk : 0xFFFFu);
        if (act && (__ffs(peers) - 1) == (int)(threadIdx.x & 31)) hist16_add(h, kk, (u32)__popc(peers));
        u32 tot = __reduce_add_sync(0xFFFFFFFFu, act ? sh : 0u);           // the warp's 32 weights are in one sub-stream
        u64 i0 = a + j * blockDim.x + wbase;
        if ((threadIdx.x & 31) == 0) atomicAdd(&sbits[b_sub0[blk] + (u32)(SW ? i0 / SW : 0)], tot);
    }
    __syncthreads();
    hist16_flush(h, 256 * NSD, hist + (u64)blk * 256 * NSD);
}

// tables given the reference: one thread per context normalizes it (bf16ref's normalize, 224 symbols),
// then the table bytes are laid out (u16 nact, then per active context u8 context, u8 nz - 1, nz x (u8 s, u16 f))
extern "C" __global__ void enc_norm_ref(const u32 *hist, u16 *fbuf, u32 *etab, u8 *tabbytes, u32 *tablen) {
    __shared__ u32 nz_s[256], off_s[256];
    int blk = blockIdx.x, cx = threadIdx.x;               // 256 threads
    const u32 *h = hist + ((u64)blk * 256 + cx) * NSD; u16 *f = fbuf + ((u64)blk * 256 + cx) * NSD;
    u64 ct = 0; for (int s = 0; s < NSD; s++) ct += h[s];
    u32 nz = 0;
    if (ct) {
        normalize1(h, f, NSD, ct, PTR);
        u32 acc = 0;
        for (int s = 0; s < NSD; s++) { etab[((u64)blk * 256 + cx) * NSD + s] = ((u32)f[s] << 16) | acc; acc += f[s]; nz += f[s] != 0; }
    }
    nz_s[cx] = nz;
    __syncthreads();
    if (cx == 0) {
        u32 o = 2, na = 0;
        for (int c = 0; c < 256; c++) { off_s[c] = o; if (nz_s[c]) { o += 2 + 3 * nz_s[c]; na++; } }
        u8 *q = tabbytes + (u64)blk * (2 + 256 * (2 + 3 * NSD)); q[0] = (u8)na; q[1] = (u8)(na >> 8);
        tablen[blk] = o;
    }
    __syncthreads();
    if (nz) {
        u8 *q = tabbytes + (u64)blk * (2 + 256 * (2 + 3 * NSD)) + off_s[cx];
        q[0] = (u8)cx; q[1] = (u8)(nz - 1); q += 2;
        for (int s = 0; s < NSD; s++) if (f[s]) { q[0] = (u8)s; q[1] = (u8)f[s]; q[2] = (u8)(f[s] >> 8); q += 3; }
    }
}

// raw bits given the reference, one warp per sub-stream: each step of 32 weights places its raw values with a
// prefix sum of their lengths and ORs them into the block's raw area (rawbuf words, zeroed; bit 0 of the block's
// area at bit 8 * b_raw[blk]; b_raw is a multiple of 4); words shared by neighbouring lanes are merged first
extern "C" __global__ void enc_raw_ref(const u16 *__restrict__ rk, const u32 *s_blk, const u64 *s_a, const u32 *s_m,
        const u64 *s_rb, const u64 *b_k, const u64 *b_raw, u32 *rawbuf, u32 nsubs) {
    u32 s = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); int lane = threadIdx.x & 31;
    if (s >= nsubs) return;
    u32 blk = s_blk[s], m = s_m[s]; const u16 *R = rk + b_k[blk] + s_a[s];
    u64 bit = 8 * b_raw[blk] + s_rb[s];
    for (u32 k0 = 0; k0 < m; k0 += 32) {
        u32 v = 0, k = 0;
        if (k0 + lane < m) { u32 x = R[k0 + lane]; v = x & 0xFFF; k = x >> 12; }
        u32 inc = k;
        for (int d = 1; d < 32; d <<= 1) { u32 t = __shfl_up_sync(0xFFFFFFFFu, inc, d); if (lane >= d) inc += t; }
        u32 tot = __shfl_sync(0xFFFFFFFFu, inc, 31);
        u64 b0 = bit + inc - k; u64 w = b0 >> 5; u32 sh = (u32)(b0 & 31);
        u32 lo = k ? v << sh : 0, hi = (k && sh + k > 32) ? v >> (32 - sh) : 0;
        // merge runs of lanes writing the same word (positions increase with the lane): segmented OR to the run's first lane
        for (int pass = 0; pass < 2; pass++) {
            u64 ww = w + pass; u32 val = pass ? hi : lo;
            for (int d = 1; d < 32; d <<= 1) {
                u32 tv = __shfl_down_sync(0xFFFFFFFFu, val, d); u64 tw = __shfl_down_sync(0xFFFFFFFFu, ww, d);
                if (lane + d < 32 && tw == ww) val |= tv;
            }
            u64 pw = __shfl_up_sync(0xFFFFFFFFu, ww, 1);
            if ((lane == 0 || pw != ww) && val) atomicOr(&rawbuf[ww], val);
        }
        bit += tot;
    }
}

// rANS coding of the sub-streams; L lanes per sub-stream, U sub-streams per lane group interleaved; etab per block
// (f << 16 | start), ntab entries per block.  Sub-stream s writes backwards from scr + s_scr[s] + s_cap[s]
// (u16 words); s_nw = its words, s_w0 = where they start.
template <int L, int SMT, int P, int U>
__device__ void enc_rans_t(const u16 *__restrict__ key, const u32 *it_blk, const u32 *it_s0, const u32 *it_cnt, const u32 *__restrict__ etab,
        u32 ntab, const u64 *s_a, const u32 *s_m, const u64 *s_scr, const u32 *s_cap, const u64 *b_k, u16 *scr, u32 *s_nw, u64 *s_w0) {
    DSMEM(u32, tab_s);
    u32 item = blockIdx.x, blk = it_blk[item], s0 = it_s0[item], cnt = it_cnt[item];
    const u32 *et = etab + (u64)blk * ntab;
    if (SMT) { for (u32 k = threadIdx.x; k < ntab; k += blockDim.x) tab_s[k] = et[k]; __syncthreads(); }
    const u32 *E = SMT ? tab_s : et;
    const int G = 32 / L; int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
    int g = lane / L, gl = lane % L; u32 gm = L == 32 ? 0xFFFFFFFFu : (0xFFFFu << (16 * g)), gt = lanemask_gt();
    for (u32 base = warp * G * U; base < cnt; base += nwarp * G * U) {
        u32 m[U], x[U]; u64 p[U]; const u16 *K[U]; bool valid[U]; u32 sidx[U];
        u32 mloc = 0;
#pragma unroll
        for (int u = 0; u < U; u++) {
            u32 si = base + g + u * G; valid[u] = si < cnt; u32 s = s0 + si; sidx[u] = s;
            m[u] = valid[u] ? s_m[s] : 0; x[u] = RL; p[u] = 0; K[u] = key;
            if (valid[u]) { p[u] = s_scr[s] + s_cap[s]; K[u] = key + b_k[blk] + s_a[s]; }
            mloc = max(mloc, m[u]);
        }
        u32 mmax = __reduce_max_sync(0xFFFFFFFFu, mloc);
        // steps from the last to the first; step k0 covers weights k0 .. k0 + L - 1 of the sub-stream
        u32 kl = mmax ? ((mmax - 1) / L) * L : 0;
        for (i64 k0 = kl; k0 >= 0 && mmax; k0 -= L) {
            u32 f[U], st[U]; bool act[U], emit[U];
#pragma unroll
            for (int u = 0; u < U; u++) {
                act[u] = (u32)k0 + gl < m[u]; f[u] = 1; st[u] = 0; emit[u] = false;
                if (act[u]) { u32 e = E[K[u][k0 + gl]]; f[u] = e >> 16; st[u] = e & 0xFFFF; emit[u] = (u64)x[u] >= ((u64)f[u] << (32 - P)); }
            }
#pragma unroll
            for (int u = 0; u < U; u++) {
                u32 bal = __ballot_sync(0xFFFFFFFFu, act[u] && emit[u]) & gm;
                if (act[u] && emit[u]) { scr[p[u] - 1 - __popc(bal & gt)] = (u16)x[u]; x[u] >>= 16; }
                p[u] -= __popc(bal);
                if (act[u]) { u32 q = x[u] / f[u]; x[u] = (q << P) + (x[u] - q * f[u]) + st[u]; }
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++)
            if (valid[u]) {
                u32 s = sidx[u]; p[u] -= 2 * L;
                scr[p[u] + 2 * gl] = (u16)x[u]; scr[p[u] + 2 * gl + 1] = (u16)(x[u] >> 16);
                if (gl == 0) { s_nw[s] = (u32)(s_scr[s] + s_cap[s] - p[u]); s_w0[s] = p[u]; }
            }
    }
}
#define ENC_RANS(L, SMT, P, U) extern "C" __global__ void enc_rans_##L##_##SMT##_##P##_u##U(const u16 *__restrict__ key, const u32 *it_blk, \
        const u32 *it_s0, const u32 *it_cnt, const u32 *__restrict__ etab, u32 ntab, const u64 *s_a, const u32 *s_m, const u64 *s_scr, \
        const u32 *s_cap, const u64 *b_k, u16 *scr, u32 *s_nw, u64 *s_w0) { \
    enc_rans_t<L, SMT, P, U>(key, it_blk, it_s0, it_cnt, etab, ntab, s_a, s_m, s_scr, s_cap, b_k, scr, s_nw, s_w0); }
ENC_RANS(16, 1, 14, 1) ENC_RANS(32, 1, 14, 1) ENC_RANS(16, 0, 13, 1) ENC_RANS(32, 0, 13, 1)
ENC_RANS(16, 1, 14, 2) ENC_RANS(32, 1, 14, 2) ENC_RANS(16, 0, 13, 2) ENC_RANS(32, 0, 13, 2)
ENC_RANS(16, 1, 14, 4) ENC_RANS(32, 1, 14, 4) ENC_RANS(16, 0, 13, 4) ENC_RANS(32, 0, 13, 4)

// assembling the blocks.  Block header, raw bits, tables and index: one thread block per block.
//   standalone: hdr = c0 c1 c2 0 | nnz | nw; given the reference: rawlen | nw;  then raw (rawlen bytes from
//   rawsrc + b_raw), table bytes (tablen from tabsrc + blk * tabstride), index ((nsub - 1) entries: u32 word
//   offset, and given the reference u32 raw bit offset), then the words (asm_words).
extern "C" __global__ void asm_head(int kind, u8 *arc, const u64 *b_dst, const u64 *b_rawlen, const u64 *b_raw, const u8 *rawsrc,
        const u32 *tablen, const u8 *tabsrc, u64 tabstride, const u8 *bcs, const u32 *nnz, const u32 *b_sub0, const u32 *b_nsub,
        const u64 *s_woff, const u64 *s_rb, const u64 *b_nw) {
    int blk = blockIdx.x; u8 *o = arc + b_dst[blk];
    u64 rl = b_rawlen[blk], hl = kind == 1 ? 12 : 8;
    u32 nw = (u32)b_nw[blk];
    if (threadIdx.x == 0) {
        if (kind == 1) { o[0] = bcs[blk * 4]; o[1] = bcs[blk * 4 + 1]; o[2] = bcs[blk * 4 + 2]; o[3] = 0;
                         u32 z = nnz[blk]; for (int i = 0; i < 4; i++) { o[4 + i] = (u8)(z >> (8 * i)); o[8 + i] = (u8)(nw >> (8 * i)); } }
        else { for (int i = 0; i < 4; i++) { o[i] = (u8)(rl >> (8 * i)); o[4 + i] = (u8)(nw >> (8 * i)); } }
    }
    u64 tl = tablen[blk]; const u8 *ts = tabsrc + blk * tabstride; u8 *to = o + hl + rl;
    for (u64 i = threadIdx.x; i < tl; i += blockDim.x) to[i] = ts[i];
    u8 *io = to + tl; u32 ns = b_nsub[blk], es = kind == 1 ? 4 : 8;
    for (u32 j = threadIdx.x; j + 1 < ns; j += blockDim.x) {
        u32 s = b_sub0[blk] + j + 1; u32 wo = (u32)s_woff[s];
        for (int i = 0; i < 4; i++) io[es * j + i] = (u8)(wo >> (8 * i));
        if (kind != 1) { u32 rb = (u32)s_rb[s]; for (int i = 0; i < 4; i++) io[es * j + 4 + i] = (u8)(rb >> (8 * i)); }
    }
}
// raw bits of the blocks to their place (after the block header, hl bytes): chunk c is bytes [r_off, r_off + r_len) of
// block r_blk's raw area, cut to the block's actual raw length
extern "C" __global__ void asm_raw(u8 *arc, const u64 *b_dst, u32 hl, const u64 *b_rawlen, const u64 *b_raw, const u8 *rawsrc,
        const u32 *r_blk, const u64 *r_off, const u32 *r_len) {
    u32 c = blockIdx.x, blk = r_blk[c]; u64 a = r_off[c], rl = b_rawlen[blk];
    if (a >= rl) return;
    u64 n = min((u64)r_len[c], rl - a);
    u8 *o = arc + b_dst[blk] + hl + a; const u8 *src = rawsrc + b_raw[blk] + a;
    for (u64 i = threadIdx.x; i < n; i += blockDim.x) o[i] = src[i];
}
// words of each sub-stream to their place: one warp per sub-stream; s_dst = byte offset in arc
extern "C" __global__ void asm_words(u8 *arc, const u16 *scr, const u64 *s_w0, const u32 *s_nw, const u64 *s_dst, u32 nsubs) {
    u32 s = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); int lane = threadIdx.x & 31;
    if (s >= nsubs) return;
    const u16 *w = scr + s_w0[s]; u32 n = s_nw[s]; u8 *o = arc + s_dst[s];
    if ((s_dst[s] & 1) == 0) { u16 *o2 = (u16 *)o; for (u32 i = lane; i < n; i += 32) o2[i] = w[i]; }
    else for (u32 i = lane; i < n; i += 32) { u16 v = w[i]; o[2 * i] = (u8)v; o[2 * i + 1] = (u8)(v >> 8); }
}
