/* GPU coding of bf16ref's BFR8 blocks with 32 lanes (bf16ref.c --gpu).
 *
 * The kernels (gpu/kernels.cuh) come from the G8 prototype in the bf16ref repository; this file is its host
 * driver (prototype/gpu/g8gpu.py) in C++: the layout of sub-streams, work items and chunks, and the encode and
 * decode pipelines.  It writes and reads the same blocks as bf16ref.c's CPU encoder and decoder, byte for byte.
 *
 * Built with nvcc (make CUDA=1), or with -DBFR_GPU_EMU as plain C++20 on the CPU emulator (gpu/cuemu.h), which
 * runs the same kernels one CUDA thread per OS thread, to test this code without a GPU (small inputs only).
 */
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef BFR_GPU_EMU
#include "gpu/cuemu.h"
#else
#include <cuda_runtime.h>
#include "gpu/kernels.cuh"
#endif

namespace {
typedef std::vector<u64> V64;
struct GpuError : std::runtime_error { using std::runtime_error::runtime_error; };
int g_hthreads = 1;    /* host threads for the copies between the staging buffers and host memory (bfr_gpu_threads) */

#ifdef BFR_GPU_EMU
void *dalloc(size_t n) { void *p = calloc(n + 16, 1); if (!p) throw GpuError("out of host memory (emulator)"); return p; }
void dfree(void *p) { free(p); }
void h2d(void *d, const void *h, size_t n) { if (n) memcpy(d, h, n); }
void d2h(void *h, const void *d, size_t n) { if (n) memcpy(h, d, n); }
void dzero(void *d, size_t n) { memset(d, 0, n); }
void dsync() {}
void d2h_big(void *h, const void *d, size_t n) { d2h(h, d, n); }
void h2d_big(void *d, const void *h, size_t n) { h2d(d, h, n); }
#define LAUNCH(k, grid, block, smem, ...) do { if (grid) emu_run(k, (unsigned)(grid), (unsigned)(block), (size_t)(smem), __VA_ARGS__); } while (0)
#else
void ck(cudaError_t e, const char *what) { if (e != cudaSuccess) throw GpuError(std::string(what) + ": " + cudaGetErrorString(e)); }
void *dalloc(size_t n) { void *p = nullptr; ck(cudaMalloc(&p, n + 16), "cudaMalloc"); ck(cudaMemset(p, 0, 16), "cudaMemset"); return p; }
void dfree(void *p) { if (p) cudaFree(p); }
void h2d(void *d, const void *h, size_t n) { if (n) ck(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice), "copy to the GPU"); }
void d2h(void *h, const void *d, size_t n) { if (n) ck(cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost), "copy from the GPU"); }
void dzero(void *d, size_t n) { ck(cudaMemset(d, 0, n), "cudaMemset"); }
void dsync() { ck(cudaDeviceSynchronize(), "kernel"); }
/* Large copies between the GPU and ordinary (pageable) host memory, such as the mapped input file and the
 * output buffer, run at a few GB/s when done in one cudaMemcpy.  These go through two pinned staging buffers:
 * the GPU copies chunk k + 1 into one while the CPU copies chunk k out of the other. */
const size_t STG = (size_t)64 << 20;
/* memcpy on up to g_hthreads threads, at least 4 MiB each.  The large host buffers (the output, the archive) are
 * fresh memory, and one thread copying into them is held to about 1 GB/s by the page faults of the first touch. */
void par_memcpy(void *d, const void *s, size_t n) {
    size_t t = std::min((size_t)g_hthreads, n >> 22);
    if (t <= 1) { memcpy(d, s, n); return; }
    std::vector<std::thread> th; size_t per = (n + t - 1) / t;
    for (size_t a = 0; a < n; a += per) { size_t m = std::min(per, n - a); th.emplace_back([=] { memcpy((char *)d + a, (const char *)s + a, m); }); }
    for (auto &x : th) x.join();
}
struct Staging {
    void *b[2] = {nullptr, nullptr}; cudaEvent_t ev[2]; cudaStream_t st;
    Staging() { for (int i = 0; i < 2; i++) { ck(cudaMallocHost(&b[i], STG), "cudaMallocHost"); ck(cudaEventCreate(&ev[i]), "event"); }
                ck(cudaStreamCreate(&st), "stream"); }
    ~Staging() { for (int i = 0; i < 2; i++) { cudaFreeHost(b[i]); cudaEventDestroy(ev[i]); } cudaStreamDestroy(st); }
};
Staging &staging() { static Staging s; return s; }
void d2h_big(void *h, const void *d, size_t n) {
    if (n < 2 * STG) { d2h(h, d, n); return; }
    Staging &S = staging(); size_t nc = (n + STG - 1) / STG;
    auto issue = [&](size_t k) { size_t a = k * STG, m = std::min(STG, n - a);
        ck(cudaMemcpyAsync(S.b[k & 1], (const char *)d + a, m, cudaMemcpyDeviceToHost, S.st), "copy from the GPU"); ck(cudaEventRecord(S.ev[k & 1], S.st), "event"); };
    issue(0);
    for (size_t k = 0; k < nc; k++) {
        if (k + 1 < nc) issue(k + 1);
        ck(cudaEventSynchronize(S.ev[k & 1]), "copy from the GPU");
        size_t a = k * STG, m = std::min(STG, n - a); par_memcpy((char *)h + a, S.b[k & 1], m);
    }
}
void h2d_big(void *d, const void *h, size_t n) {
    if (n < 2 * STG) { h2d(d, h, n); return; }
    Staging &S = staging(); size_t nc = (n + STG - 1) / STG;
    for (size_t k = 0; k < nc; k++) {
        size_t a = k * STG, m = std::min(STG, n - a);
        if (k >= 2) ck(cudaEventSynchronize(S.ev[k & 1]), "copy to the GPU");      /* the buffer's previous copy is done */
        par_memcpy(S.b[k & 1], (const char *)h + a, m);
        ck(cudaMemcpyAsync((char *)d + a, S.b[k & 1], m, cudaMemcpyHostToDevice, S.st), "copy to the GPU"); ck(cudaEventRecord(S.ev[k & 1], S.st), "event");
    }
    ck(cudaStreamSynchronize(S.st), "copy to the GPU");
}
#define LAUNCH(k, grid, block, smem, ...) do { if (grid) {                                                     \
        if ((size_t)(smem) > 48 * 1024) ck(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(smem)), #k); \
        k<<<(unsigned)(grid), (unsigned)(block), (size_t)(smem)>>>(__VA_ARGS__); ck(cudaGetLastError(), #k); } } while (0)
#endif

/* device array, freed with its owner */
template <class T> struct DA {
    T *p = nullptr; size_t n = 0;
    DA() {}
    explicit DA(size_t n_, bool zero = false) : n(n_) { p = (T *)dalloc(n * sizeof(T)); if (zero) dzero(p, n * sizeof(T) + 16); }
    template <class S> explicit DA(const std::vector<S> &v) : n(v.size()) {
        std::vector<T> t(v.begin(), v.end()); p = (T *)dalloc(n * sizeof(T)); h2d(p, t.data(), n * sizeof(T));
    }
    DA(const DA &) = delete; DA &operator=(const DA &) = delete;
    DA(DA &&o) noexcept : p(o.p), n(o.n) { o.p = nullptr; }
    DA &operator=(DA &&o) noexcept { if (this != &o) { dfree(p); p = o.p; n = o.n; o.p = nullptr; } return *this; }
    ~DA() { dfree(p); }
    std::vector<T> get() const { std::vector<T> v(n); d2h(v.data(), p, n * sizeof(T)); return v; }
};

const int L = 32;                       /* lanes: one sub-stream per warp */
const u64 CH = 1 << 17, CHR = 65504, RAWCH = 1 << 18;
const u32 TABSTRIDE_REF = 2 + 256 * (2 + 3 * NSD);
const int EWARPS = 8, DWARPS = 16;      /* warps per thread block, encoding and decoding (as in the prototype's runs) */
const u32 SMT = 48 * 1024;              /* shared memory for a block's table image given the reference (v3) */

/* BFR_GPU_TIMING=1: time of each stage on stderr (after a device synchronization) */
double g_t0 = 0; int g_timing = -1;
double tnow() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void stage(const char *what) {
    if (g_timing < 0) { const char *e = getenv("BFR_GPU_TIMING"); g_timing = e && *e && *e != '0'; }
    if (!g_timing) return;
    dsync(); double t = tnow(); fprintf(stderr, "bf16ref gpu: %-28s %8.3f s\n", what, g_t0 ? t - g_t0 : 0.0); g_t0 = t;
}

V64 excl(const V64 &a) { V64 r(a.size()); u64 s = 0; for (size_t i = 0; i < a.size(); i++) { r[i] = s; s += a[i]; } return r; }

/* chunks of n[b] units of size `size`: (block, start, count) */
struct Chunks {
    DA<u32> blk; DA<u64> a; DA<u32> cnt; size_t k = 0;
    Chunks(const V64 &n, u64 size) {
        std::vector<u32> cb, cn; V64 ca;
        for (size_t b = 0; b < n.size(); b++) for (u64 a0 = 0; a0 < n[b]; a0 += size) { cb.push_back((u32)b); ca.push_back(a0); cn.push_back((u32)std::min(size, n[b] - a0)); }
        k = cb.size(); blk = DA<u32>(cb); a = DA<u64>(ca); cnt = DA<u32>(cn);
    }
};

/* sub-streams of SW weights (0: one per block) and work items of spi sub-streams of one block */
struct Layout {
    size_t nb, ns, nit; V64 n, nsub, sub0, s_blk, s_a, s_m, s_cap;
    DA<u64> b_n, b_in, b_base, b_out, b_k; DA<u32> d_sub0, d_nsub, d_s_blk, d_s_m, d_s_cap; DA<u64> d_s_a, d_s_scr;
    DA<u32> it_blk, it_s0, it_cnt;
    u64 N = 0, scr_total = 0;
    Layout(const V64 &n_, const V64 &tin, const V64 &bin, const V64 &out, u64 SW, int spi) : nb(n_.size()), n(n_) {
        for (size_t b = 0; b < nb; b++) { u64 sw = SW ? SW : std::max<u64>(n[b], 1); nsub.push_back(std::max<u64>(1, (n[b] + sw - 1) / sw)); }
        sub0 = excl(nsub); ns = 0; for (u64 x : nsub) ns += x;
        std::vector<u32> ib, is0, ic;
        for (size_t b = 0; b < nb; b++) {
            u64 sw = SW ? SW : std::max<u64>(n[b], 1);
            for (u64 j = 0; j < nsub[b]; j++) {
                u64 a = j * sw, m = std::min(sw, n[b] - a);
                s_blk.push_back(b); s_a.push_back(a); s_m.push_back(m); s_cap.push_back(m + 2 * L + 2);
            }
            for (u64 k = 0; k < nsub[b]; k += spi) { ib.push_back((u32)b); is0.push_back((u32)(sub0[b] + k)); ic.push_back((u32)std::min<u64>(spi, nsub[b] - k)); }
            N += n[b];
        }
        nit = ib.size();
        for (u64 c : s_cap) scr_total += c;
        b_n = DA<u64>(n); b_in = DA<u64>(tin); b_base = DA<u64>(bin); b_out = DA<u64>(out); b_k = DA<u64>(excl(n));
        d_sub0 = DA<u32>(sub0); d_nsub = DA<u32>(nsub); d_s_blk = DA<u32>(s_blk); d_s_m = DA<u32>(s_m); d_s_cap = DA<u32>(s_cap);
        d_s_a = DA<u64>(s_a); d_s_scr = DA<u64>(excl(s_cap));
        it_blk = DA<u32>(ib); it_s0 = DA<u32>(is0); it_cnt = DA<u32>(ic);
    }
    /* per-block sums of a per-sub-stream array, and each sub-stream's exclusive prefix within its block */
    void segsum(const std::vector<u32> &v, V64 &bsum, V64 &pre) const {
        bsum.assign(nb, 0); pre.assign(ns, 0);
        for (size_t s = 0; s < ns; s++) { pre[s] = bsum[s_blk[s]]; bsum[s_blk[s]] += v[s]; }
    }
};

/* encodes the blocks of one kind; returns the blocks one after another in arc, block b at dst[b], blen[b] bytes */
void encode_kind(int kind, const u16 *t16, const u16 *b16, const V64 &n, const V64 &tin, const V64 &bin, u64 SW,
                 std::unique_ptr<u8[]> &arc, V64 &dst, V64 &blen) {
    Layout lay(n, tin, bin, tin, SW, EWARPS * (32 / L));
    size_t nb = lay.nb, ns = lay.ns; const int T = 256;
    V64 rawlen(nb), tablen(nb), s_rb(ns, 0), b_rawsrc_h; u64 hl, ie, tabstride;
    DA<u16> key(lay.N + 32); DA<u32> etab(kind == 1 ? nb * NSA : nb * 256 * NSD);
    DA<u8> tabb(kind == 1 ? nb * NSA * 4 : nb * (u64)TABSTRIDE_REF), bcs(nb * 4, true); DA<u32> nnz(nb, true);
    DA<u8> rawsrc; u32 ntab;
    if (kind == 1) {
        Chunks ch(n, CH);
        DA<u32> hs(nb * 65536, true);
        LAUNCH(enc_sample, ch.k, T, 65536 * 2, t16, ch.blk.p, ch.a.p, ch.cnt.p, lay.b_in.p, hs.p);
        DA<u8> sel(nb * 256), rawt(nb * 256);
        LAUNCH(enc_choose, nb, T, 0, hs.p, sel.p, rawt.p, bcs.p);
        V64 ra(nb); for (size_t b = 0; b < nb; b++) ra[b] = 5 * ((n[b] + 7) / 8) + 8;
        b_rawsrc_h = excl(ra); u64 tot = 0; for (u64 x : ra) tot += x;
        DA<u64> b_raw(b_rawsrc_h); DA<u32> hist(nb * NSA, true); rawsrc = DA<u8>(tot + 16, true);
        LAUNCH(enc_sym_alone, ch.k, T, 0, t16, ch.blk.p, ch.a.p, ch.cnt.p, lay.b_in.p, lay.b_k.p, b_raw.p, sel.p, rawt.p, key.p, hist.p, rawsrc.p);
        LAUNCH(enc_norm_alone, nb, 32, 0, hist.p, lay.b_n.p, etab.p, tabb.p, nnz.p);
        std::vector<u32> nz = nnz.get();
        for (size_t b = 0; b < nb; b++) { rawlen[b] = 5 * ((n[b] + 7) / 8); tablen[b] = 4 * (u64)nz[b]; }
        hl = 12; ie = 4; tabstride = NSA * 4; ntab = NSA;
    } else {
        Chunks cr(n, CHR);
        DA<u32> hist(nb * 256 * NSD, true), sbits(ns, true); DA<u16> rk(lay.N + 32);
        LAUNCH(enc_sym_ref, cr.k, T, 256 * NSD * 2, t16, b16, cr.blk.p, cr.a.p, cr.cnt.p, lay.b_in.p, lay.b_base.p, lay.b_k.p, lay.d_sub0.p, SW, key.p, rk.p, hist.p, sbits.p);
        DA<u16> fbuf(nb * 256 * NSD); DA<u32> tl32(nb);
        LAUNCH(enc_norm_ref, nb, T, 0, hist.p, fbuf.p, etab.p, tabb.p, tl32.p);
        V64 bits; lay.segsum(sbits.get(), bits, s_rb);
        V64 rr(nb); for (size_t b = 0; b < nb; b++) rr[b] = ((n[b] * 12 + 7) / 8 + 8 + 3) / 4 * 4;
        b_rawsrc_h = excl(rr); u64 tot = 0; for (u64 x : rr) tot += x;
        DA<u64> b_raw(b_rawsrc_h), d_s_rb(s_rb); rawsrc = DA<u8>(tot + 16, true);
        LAUNCH(enc_raw_ref, (ns + 7) / 8, T, 0, rk.p, lay.d_s_blk.p, lay.d_s_a.p, lay.d_s_m.p, d_s_rb.p, lay.b_k.p, b_raw.p, (u32 *)rawsrc.p, (u32)ns);
        std::vector<u32> tl = tl32.get();
        for (size_t b = 0; b < nb; b++) { rawlen[b] = (bits[b] + 7) / 8; tablen[b] = tl[b]; }
        hl = 8; ie = 8; tabstride = TABSTRIDE_REF; ntab = 256 * NSD;
    }
    DA<u16> scr(lay.scr_total + 16); DA<u32> s_nw(ns); DA<u64> s_w0(ns);
    if (kind == 1) LAUNCH(enc_rans_32_1_14_u1, lay.nit, EWARPS * 32, NSA * 4, key.p, lay.it_blk.p, lay.it_s0.p, lay.it_cnt.p, etab.p, ntab,
                          lay.d_s_a.p, lay.d_s_m.p, lay.d_s_scr.p, lay.d_s_cap.p, lay.b_k.p, scr.p, s_nw.p, s_w0.p);
    else LAUNCH(enc_rans_32_0_13_u1, lay.nit, EWARPS * 32, 0, key.p, lay.it_blk.p, lay.it_s0.p, lay.it_cnt.p, etab.p, ntab,
                lay.d_s_a.p, lay.d_s_m.p, lay.d_s_scr.p, lay.d_s_cap.p, lay.b_k.p, scr.p, s_nw.p, s_w0.p);
    std::vector<u32> nwv = s_nw.get();
    V64 nw, woff; lay.segsum(nwv, nw, woff);
    blen.assign(nb, 0);
    for (size_t b = 0; b < nb; b++) blen[b] = hl + rawlen[b] + tablen[b] + ie * (lay.nsub[b] - 1) + 2 * nw[b];
    dst = excl(blen); u64 total = nb ? dst[nb - 1] + blen[nb - 1] : 0;
    DA<u8> d_arc(total + 16, true);
    {
        DA<u64> d_dst(dst), d_rawlen(rawlen), d_brs(b_rawsrc_h), d_woff(woff), d_srb(s_rb), d_nw(nw); DA<u32> d_tablen(tablen);
        LAUNCH(asm_head, nb, T, 0, kind, d_arc.p, d_dst.p, d_rawlen.p, d_brs.p, rawsrc.p, d_tablen.p, tabb.p, tabstride, bcs.p, nnz.p,
               lay.d_sub0.p, lay.d_nsub.p, d_woff.p, d_srb.p, d_nw.p);
        Chunks rc(rawlen, RAWCH);
        LAUNCH(asm_raw, rc.k, T, 0, d_arc.p, d_dst.p, (u32)hl, d_rawlen.p, d_brs.p, rawsrc.p, rc.blk.p, rc.a.p, rc.cnt.p);
        V64 s_dst(ns);
        for (size_t s = 0; s < ns; s++) { size_t b = lay.s_blk[s]; s_dst[s] = dst[b] + hl + rawlen[b] + tablen[b] + ie * (lay.nsub[b] - 1) + 2 * woff[s]; }
        DA<u64> d_sdst(s_dst);
        LAUNCH(asm_words, (ns + 7) / 8, T, 0, d_arc.p, scr.p, s_w0.p, s_nw.p, d_sdst.p, (u32)ns);
        dsync();
    }
    stage("  encode kernels");
    arc.reset(new u8[total + 1]); d2h_big(arc.get(), d_arc.p, total);     /* not zero-filled: every byte is copied */
}

/* decodes the blocks of one kind (block b at aoff[b], alen[b] bytes of arc) into out16 (u16 index out[b]) */
void decode_kind(int kind, const u8 *arc, u64 arclen, const V64 &aoff, const V64 &alen, const V64 &n, const V64 &out, const V64 &bin,
                 u64 SW, const u16 *b16, u64 b16bytes, u16 *out16) {
    Layout lay(n, out, bin, out, SW, DWARPS * (32 / L));
    size_t nb = lay.nb, ns = lay.ns; const int T = 256;
    DA<u64> offs(aoff), lens(alen), d_wb(ns), d_we(ns), d_out(ns), d_raw(ns), d_base(ns); DA<u32> d_m(ns), err(1, true);
    if (kind == 1) {
        DA<u32> t1(nb * PTOT), t2(nb * NSA); DA<u16> rawmap(nb * 32);
        LAUNCH(parse_alone3, nb, T, 0, arc, offs.p, lens.p, lay.b_n.p, lay.b_out.p, lay.d_sub0.p, SW, t1.p, t2.p, rawmap.p,
               d_wb.p, d_we.p, d_m.p, d_out.p, d_raw.p, err.p);
        stage("  parse alone");
        LAUNCH(dec3_alone, lay.nit, DWARPS * 32, PTOT * 4 + NSA * 4 + 64 + DWARPS * 2 * 1024, arc, arclen, lay.it_blk.p, lay.it_s0.p, lay.it_cnt.p,
               t1.p, t2.p, rawmap.p, d_wb.p, d_we.p, d_m.p, d_out.p, d_raw.p, out16, err.p);
        dsync(); stage("  dec3_alone kernel");
    } else {
        DA<u64> tsz(nb); DA<u32> tnt(nb);
        LAUNCH(size_ref3, (nb + 63) / 64, 64, 0, arc, offs.p, lens.p, (u32)nb, tsz.p, tnt.p, err.p);
        if (err.get()[0]) throw GpuError("corrupt archive (GPU parse)");
        V64 tofs_h = excl(tsz.get()); u64 ttot = nb ? tofs_h[nb - 1] + tsz.get()[nb - 1] : 0;
        DA<u64> tofs(tofs_h); DA<u8> tab(ttot + 16, true);
        LAUNCH(parse_ref3, nb, T, 0, arc, offs.p, lens.p, lay.b_n.p, lay.b_out.p, lay.b_base.p, lay.d_sub0.p, SW, (int)L, tab.p, tofs.p, tsz.p, tnt.p,
               d_wb.p, d_we.p, d_m.p, d_out.p, d_raw.p, d_base.p, err.p);
        stage("  parse given ref");
        LAUNCH(dec3_ref, lay.nit, DWARPS * 32, SMT + DWARPS * 3 * 1024, arc, arclen, lay.it_blk.p, lay.it_s0.p, lay.it_cnt.p, tab.p, tofs.p, tsz.p, tnt.p,
               SMT, b16, b16bytes, d_wb.p, d_we.p, d_m.p, d_out.p, d_raw.p, d_base.p, out16, err.p);
        dsync(); stage("  dec3_ref kernel");
    }
    if (err.get()[0]) throw GpuError("corrupt archive (GPU decode, error bits " + std::to_string(err.get()[0]) + ")");
}
}  // namespace

/* ---------------- interface for bf16ref.c (see bfr_gpu.h) ---------------- */
extern "C" {
#include "bfr_gpu.h"

void bfr_gpu_threads(int threads) { g_hthreads = threads < 1 ? 1 : threads; }

int bfr_gpu_encode(const uint8_t *in, size_t ilen, const uint8_t *ref, size_t rlen, size_t nseg, const int *kind, const uint64_t *off,
                   const uint64_t *len, const uint64_t *roff, uint64_t sw, uint8_t **body, uint64_t *outlen, void **keep, char *msg, size_t msglen) {
    try {
        g_t0 = 0; stage("start");
        DA<u8> d_in(ilen + 16), d_ref(rlen + 16);
        h2d_big(d_in.p, in, ilen); if (ref) h2d_big(d_ref.p, ref, rlen);
        dzero(d_in.p + ilen, 16); dzero(d_ref.p + rlen, 16);
        stage("allocate, copy in");
        std::vector<std::unique_ptr<u8[]>> *arcs = new std::vector<std::unique_ptr<u8[]>>(2);
        *keep = arcs;
        for (int k = 1; k <= 2; k++) {
            V64 n, tin, bin; std::vector<size_t> idx;
            for (size_t i = 0; i < nseg; i++) if (kind[i] == k) {
                if ((off[i] | len[i] | roff[i]) & 1) throw GpuError("a bfloat16 segment at an odd offset");
                n.push_back(len[i] / 2); tin.push_back(off[i] / 2); bin.push_back(roff[i] / 2); idx.push_back(i);
            }
            if (n.empty()) continue;
            V64 dst, blen;
            encode_kind(k, (const u16 *)d_in.p, (const u16 *)d_ref.p, n, tin, bin, sw, (*arcs)[k - 1], dst, blen);
            stage(k == 1 ? "encode alone, copy out" : "encode given ref, copy out");
            for (size_t j = 0; j < idx.size(); j++) { body[idx[j]] = (*arcs)[k - 1].get() + dst[j]; outlen[idx[j]] = blen[j]; }
        }
        return 0;
    } catch (const std::exception &e) { snprintf(msg, msglen, "%s", e.what()); return 1; }
}
void bfr_gpu_free(void *keep) { delete (std::vector<std::unique_ptr<u8[]>> *)keep; }

int bfr_gpu_decode(const uint8_t *arc, size_t alen, const uint8_t *ref, size_t rlen, size_t nseg, const int *kind, const uint64_t *off,
                   const uint64_t *len, const uint64_t *roff, const uint64_t *ain, const uint64_t *ainlen, uint64_t sw, uint8_t *out, size_t olen,
                   char *msg, size_t msglen) {
    try {
        g_t0 = 0; stage("start");
        DA<u8> d_arc(alen + 16), d_ref(rlen + 16), d_out(olen + 16, true);
        h2d_big(d_arc.p, arc, alen); if (ref) h2d_big(d_ref.p, ref, rlen);
        dzero(d_arc.p + alen, 16); dzero(d_ref.p + rlen, 16);     /* the kernels read whole aligned u64 words up to the length they are given */
        stage("allocate, copy in");
        for (int k = 1; k <= 2; k++) {
            V64 n, o, bin, ao, al;
            for (size_t i = 0; i < nseg; i++) if (kind[i] == k) {
                if ((off[i] | len[i] | roff[i]) & 1) throw GpuError("a bfloat16 segment at an odd offset");
                n.push_back(len[i] / 2); o.push_back(off[i] / 2); bin.push_back(roff[i] / 2); ao.push_back(ain[i]); al.push_back(ainlen[i]);
            }
            if (n.empty()) continue;
            decode_kind(k, d_arc.p, alen + 16, ao, al, n, o, bin, sw, (const u16 *)d_ref.p, rlen + 16, (u16 *)d_out.p);
            stage(k == 1 ? "decode alone" : "decode given ref");
        }
        d2h_big(out, d_out.p, olen);
        stage("copy out");
        return 0;
    } catch (const std::exception &e) { snprintf(msg, msglen, "%s", e.what()); return 1; }
}
}
