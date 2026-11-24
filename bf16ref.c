/* bf16ref: fast lossless coder for bfloat16 files and safetensors models, alone or given a reference
 * (a fine-tuned model given its base model).  Writes archive format BFR7; also decodes BFR6.
 *
 * Standalone (BFR7): each weight is one 11-bit symbol, its 8 exponent bits plus 3 of its other 8 bits
 * (sign and mantissa), chosen per block as the 3 that minimize the symbol's entropy on a 1/8 sample.
 * The other 5 bits are stored raw, 8 weights per 5 bytes.
 * Given a reference (BFR7): each 16-bit weight t is coded given the reference weight b at the same
 * position.  With j = |t| - |b| in units in the last place, the symbol is the sign-flip bit, the sign of
 * j, the bit length of |j| and the 2 bits after its leading one; the remaining low bits of |j| are stored
 * raw (complemented when j < 0, so that decoding always adds).  224 symbols, no escapes.  The context is
 * the exponent of b.
 * Symbols are coded with static frequency tables, one per context and per block, stored in the archive
 * (precision 14 bits standalone, 13 bits given a reference), and a 16-way interleaved 32-bit rANS coder
 * whose states share one word stream.
 *
 * The archive is a list of segments that cover the input in order: raw bytes, bfloat16 coded alone, or
 * bfloat16 coded given a range of the reference.  By default the blocks are written in the BFR8 layout (256 KiB
 * sub-streams of 32 lanes, see "BFR8 decoders" below); -s 0 -l 16 writes BFR7.  A safetensors file keeps its header raw; each BF16
 * tensor is coded in blocks of at most block_MiB, given the reference tensor of the same name, dtype and
 * size when the reference is a safetensors file that has one, and alone otherwise; other tensors and
 * any gaps are raw.  Any other file is a sequence of bfloat16 blocks (given the reference at the same
 * position, which must then have the same length).  A segment that would not shrink is stored raw.
 * Segments are independent and are coded in parallel; encoded blocks are written in order as they
 * finish, and decoded blocks are written to their place in the output with pwrite.  With a reference, a
 * fingerprint of it (its length and 3 MiB sampled from it) is stored so that decoding with a different
 * file is refused.  Each segment's record holds a CRC32C-based check of its original bytes, verified
 * before the decoded segment is written, which catches corruption and a wrong reference that the
 * sampled fingerprint misses.  The decoder checks every stream against its bounds every 4096 weights, so a corrupt
 * or truncated archive fails with an error instead of reading out of bounds.
 *
 *   bf16ref c IN OUT [--ref REF] [-t threads] [-b block_MiB]
 *   bf16ref d IN OUT [--ref REF] [-t threads]
 *
 * Build: clang -O3 -march=native -o bf16ref bf16ref.c -lpthread
 *        (-DBF16REF_PORTABLE uses the portable C refill on ARM too; other CPUs always use it)
 * Assumes a little-endian CPU.
 */
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#if !defined(BF16REF_PORTABLE) && defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#define CRC_HW 1
#define CRC8B(c, w) __crc32cd((c), (w))
#define CRC1B(c, b) __crc32cb((c), (b))
#elif !defined(BF16REF_PORTABLE) && defined(__x86_64__) && defined(__SSE4_2__)
#include <nmmintrin.h>
#define CRC_HW 1
#define CRC8B(c, w) ((uint32_t)_mm_crc32_u64((c), (w)))
#define CRC1B(c, b) _mm_crc32_u8((c), (b))
#endif

#ifdef BF16REF_GPU
#include "bfr_gpu.h"               /* --gpu: BFR8 blocks with 32 lanes coded on the GPU (bf16ref_gpu.cu; make CUDA=1) */
#endif

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "bf16ref assumes a little-endian CPU"
#endif

static const char MAGIC7[4] = {'B', 'F', 'R', '7'}, MAGIC6[4] = {'B', 'F', 'R', '6'}, MAGIC8[4] = {'B', 'F', 'R', '8'};
/* BFR8 layout of the archive being written or read: each coded block's words are cut into sub-streams of
 * g_sw weights (0: one sub-stream per block), each with g_lanes interleaved states.  0 and 16 is BFR7. */
static size_t g_sw = 0;
static int g_lanes = 16;
#define BFR7_LAYOUT() (g_sw == 0 && g_lanes == 16)
static const char *g_outpath;          /* removed if we fail */

__attribute__((noreturn)) static void die(const char *m) {
    fprintf(stderr, "bf16ref: %s\n", m);
    if (g_outpath) unlink(g_outpath);
    _exit(1);
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("out of memory"); return p; }
static void *xcalloc(size_t n, size_t s) { void *p = calloc(n ? n : 1, s ? s : 1); if (!p) die("out of memory"); return p; }
static inline uint16_t ld16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static inline uint32_t ld32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* bounded little-endian reader for headers and tables: reading past the end sets bad */
typedef struct { const uint8_t *p, *e; int bad; } Rd;
static inline uint32_t rd8(Rd *r) { if (r->e - r->p < 1) { r->bad = 1; return 0; } return *r->p++; }
static inline uint32_t rd16(Rd *r) { if (r->e - r->p < 2) { r->bad = 1; return 0; } uint32_t v = ld16(r->p); r->p += 2; return v; }
static inline uint32_t rd32(Rd *r) { if (r->e - r->p < 4) { r->bad = 1; return 0; } uint32_t v = ld32(r->p); r->p += 4; return v; }

#define RL (1u << 16)          /* rANS lower bound; state in [RL, RL << 16) */
#define NS 16                  /* interleaved rANS states (BFR7) */
#define CH 4096                /* weights between stream bound checks; a multiple of NS */
#define PB 14                  /* standalone table precision */
#define PTOT (1u << PB)
#define PM (PTOT - 1)
#define NSA 2048               /* standalone alphabet: 8 exponent bits + 3 chosen bits */
#define PBR 13                 /* given-the-reference table precision */
#define PTR (1u << PBR)
#define PMR (PTR - 1)
#define MB 2                   /* bits after the leading one of |j| kept in the symbol */
#define NMAG ((1 << (MB + 1)) + (15 - MB - 1) * (1 << MB))   /* 56 magnitude symbols */
#define NSD (4 * NMAG)         /* 224: flip x sign of j x magnitude */
#define RAWB ((CH / 4) * 7 + 16)   /* raw bytes the delta decoder can read in CH weights, with read-ahead */

#define FS(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8) M(9) M(10) M(11) M(12) M(13) M(14) M(15)
#define FSR(M) M(15) M(14) M(13) M(12) M(11) M(10) M(9) M(8) M(7) M(6) M(5) M(4) M(3) M(2) M(1) M(0)
#define XINIT(q) uint32_t x##q = RL;
#define XPTR(q) &x##q,
#define XLOAD(q) uint32_t x##q = ld32(p + 4 * q);
#define XBAD(q) || x##q != RL
#define XVAL(q) x##q,

/* decoder refill: X = y < RL ? (y << 16) | next word : y.  Written as a conditional select: compilers
 * turn the plain C into an unpredictable branch, which costs 3-4x on these kernels.  p is a byte pointer. */
#if defined(__aarch64__) && !defined(BF16REF_PORTABLE)
#define RENORM(X, Y) do { uint32_t y__ = (Y), t_ = (y__ << 16) | ld16(p), r_;                         \
        __asm__("cmp %w[y], #16, lsl #12\n\tcsel %w[x], %w[t], %w[y], lo\n\tcset %w[r], lo"          \
                : [x] "=&r"(X), [r] "=&r"(r_) : [y] "r"(y__), [t] "r"(t_) : "cc"); p += r_ << 1; } while (0)
#else
#define RENORM(X, Y) do { uint32_t y__ = (Y), t_ = (y__ << 16) | ld16(p), m_ = 0u - (uint32_t)(y__ < RL); \
        (X) = (t_ & m_) | (y__ & ~m_); p += m_ & 2; } while (0)
#endif

/* Scale counts to a total of ptot, keeping every seen symbol at frequency >= 1 (needs ptot >= n). */
static void normalize(const uint32_t *cnt, uint16_t *freq, int n, uint64_t total, uint32_t ptot) {
    uint32_t sum = 0;
    for (int s = 0; s < n; s++) {
        uint32_t f = 0;
        if (cnt[s]) { f = (uint32_t)(((uint64_t)cnt[s] * ptot + total / 2) / total); if (!f) f = 1; }
        freq[s] = (uint16_t)f; sum += f;
    }
    while (sum != ptot) {             /* move the rounding error onto the most frequent symbols */
        int best = -1;
        for (int s = 0; s < n; s++) if (freq[s] && (best < 0 || freq[s] > freq[best])) best = s;
        if (sum < ptot) { freq[best] += (uint16_t)(ptot - sum); sum = ptot; }
        else { uint32_t d = sum - ptot, room = freq[best] - 1u, take = d < room ? d : room;
               if (!take) die("cannot normalize table");
               freq[best] -= (uint16_t)take; sum -= take; }
    }
}

/* Encoder entry: x' = x + bias + q * cmpl with q = floor(x / f) = umulh(x, M), M = ceil(2^64 / f), exact
 * for x < 2^32 (the error x (M - 2^64/f) / 2^64 < 2^-32 is below 1/f).  For f = 1, M = 2^64 - 1 gives
 * q = x - 1 and the bias absorbs it (as in ryg_rans), so there is no branch.  Renormalize (emit 16 bits)
 * when x > xm1 = f * 2^(32 - pb) - 1; written with xm1 so that f = 2^pb never renormalizes. */
typedef struct { uint64_t M; uint32_t xm1; uint16_t cmpl, bias; } Enc;
static void mkenc(const uint16_t *f, int n, Enc *e, uint32_t pb) {
    uint32_t acc = 0, ptot = 1u << pb;
    for (int s = 0; s < n; s++) {
        e[s].xm1 = (uint32_t)(((uint64_t)f[s] << (32 - pb)) - 1);
        e[s].cmpl = (uint16_t)(ptot - f[s]);
        if (f[s] <= 1) { e[s].M = UINT64_MAX; e[s].bias = (uint16_t)(acc + ptot - 1); }
        else { e[s].M = UINT64_MAX / f[s] + 1; e[s].bias = (uint16_t)acc; }
        acc += f[s];
    }
}
/* one encoding step; p points one past the next free word of a stream written backwards */
#define ENC1(X, EX) do { const Enc E_ = (EX); uint32_t r_ = (X) > E_.xm1;                              \
        *(p - 1) = (uint16_t)(X); p -= r_; (X) >>= r_ << 4;                                          \
        uint32_t q_ = (uint32_t)(((__uint128_t)(X) * E_.M) >> 64);                                   \
        (X) = (X) + E_.bias + q_ * E_.cmpl; } while (0)

/* log2(c) for c >= 1 in fixed point with LG_FRAC fractional bits, in integer arithmetic only (the bits of the
 * fraction by repeated squaring, each truncated), so that every CPU, compiler and GPU computes the same value.
 * The choice of the 3 coded bits depends on it, and with floating-point log2 two math libraries could round
 * exactly tied entropies differently and write different archives for the same input. */
#define LG_FRAC 24
static inline uint64_t log2_fx(uint32_t c) {
    int L = 31 - __builtin_clz(c);
    uint64_t m = (uint64_t)c << (31 - L), f = 0;                   /* m in [2^31, 2^32): Q1.31 in [1, 2) */
    for (int k = 0; k < LG_FRAC; k++) {
        m = (m * m) >> 31; f <<= 1;
        if (m >= ((uint64_t)1 << 32)) { m >>= 1; f |= 1; }
    }
    return ((uint64_t)L << LG_FRAC) | f;
}
/* sum of c log2 c over the 8 cells, fixed point: the entropy of the cells is (T log2 T - this) / T bits per
 * sample, with T their total, which does not depend on the choice of bits; so the best choice maximizes it */
static inline uint64_t clogc_fx(const uint32_t *c) {
    uint64_t s = 0;
    for (int i = 0; i < 8; i++) if (c[i] > 1) s += (uint64_t)c[i] * log2_fx(c[i]);
    return s;
}
/* "other" byte of a weight: mantissa (bits 0..6) and sign (bit 7) */
static inline uint32_t oth(uint16_t w) { return (w & 0x7F) | ((w >> 8) & 0x80); }

/* BFR8: code a block's weights (key[k] = table entry of weight k) as sub-streams of g_sw weights, each with
 * g_lanes interleaved states and its own words, coded backwards as BFR7's single stream is (weight i of a
 * sub-stream uses state i mod g_lanes; the final states are written first, 2 words each).  Writes the index
 * (for each sub-stream after the first: its word offset, and given rb its raw-bit offset) and then the words
 * of the sub-streams in order; returns the bytes written.  With g_sw = 0 and g_lanes = 16 the words are
 * BFR7's. */
static size_t enc_streams8(uint16_t *wend, const uint32_t *key, const Enc *E, size_t n, const uint64_t *rb, uint8_t *o, uint32_t *nw_out) {
    int L = g_lanes; size_t sw = g_sw ? g_sw : (n ? n : 1), ns = n ? (n + sw - 1) / sw : 1;
    size_t isz = (ns - 1) * (rb ? 8 : 4);
    uint8_t *idx = o, *wo = o + isz; uint32_t nw = 0;
    for (size_t s = 0; s < ns; s++) {
        size_t a = s * sw, m = n - a < sw ? n - a : sw;
        const uint32_t *ky = key + a;
        uint16_t *p = wend; uint32_t x[32];
        for (int j = 0; j < L; j++) x[j] = RL;
        size_t full = m / (size_t)L, rem = m % (size_t)L;
        for (size_t j = rem; j-- > 0;) ENC1(x[j], E[ky[full * L + j]]);
        for (size_t st = full; st-- > 0;) { const uint32_t *kk = ky + st * L; for (int j = L; j-- > 0;) ENC1(x[j], E[kk[j]]); }
        for (int j = L - 1; j >= 0; j--) { *--p = (uint16_t)(x[j] >> 16); *--p = (uint16_t)x[j]; }
        size_t w = (size_t)(wend - p);
        if (s) { uint32_t v = nw; memcpy(idx, &v, 4); idx += 4; if (rb) { uint32_t r = (uint32_t)rb[s]; memcpy(idx, &r, 4); idx += 4; } }
        memcpy(wo + 2 * (size_t)nw, p, 2 * w); nw += (uint32_t)w;
    }
    *nw_out = nw;
    return isz + 2 * (size_t)nw;
}

/* ---------- per-thread buffers, allocated once for the largest block ---------- */
typedef struct {
    size_t cap;                       /* weights */
    /* encoder */
    uint16_t *sy, *wb, *tcopy, *bcopy;
    uint32_t *key;                    /* BFR8: table entry of each weight */
    uint32_t *hs, *ho, *h4, *hd;      /* sample histogram, [exp][other], 4 x NSA, 2 x 256 x NSD */
    uint16_t *fd; Enc *ed;            /* delta frequencies and encoder entries, 256 x NSD */
    /* decoder */
    uint16_t *dbuf;
    uint32_t *es; uint16_t *vs;       /* standalone: per slot (f << 16 | offset) and value bits */
    uint8_t *slot; uint64_t *se;      /* delta: per context slot -> symbol, symbol -> entry */
    uint8_t *wtail, *rtail;           /* padded tail copies of the word and raw streams */
} TC;

static size_t enc_cap(size_t n) {     /* bytes an encoded block of n weights can take, either kind */
    size_t a = 12 + 5 * ((n + 7) / 8) + 8 + 4 * NSA + 2 * (n + 2 * NS) + 16;
    size_t d = 8 + (n * 12 + 7) / 8 + 16 + 2 + 256 * (2 + 3 * NSD) + 2 * (n + 2 * NS) + 16;
    size_t sub = (n / 256 + 2) * (8 + 4 * 32);   /* BFR8: index entries and the states of every sub-stream */
    return (a > d ? a : d) + sub;
}
static void tc_init(TC *c, size_t cap, int enc) {
    memset(c, 0, sizeof *c); c->cap = cap;
    if (enc) {
        c->sy = xmalloc(cap * 2 + 32); c->wb = xmalloc((cap + 2 * 32 + 16) * 2); c->key = xmalloc(cap * 4 + 4);
        c->tcopy = xmalloc(cap * 2 + 2); c->bcopy = xmalloc(cap * 2 + 2);
        c->hs = xmalloc(65536 * 4); c->ho = xmalloc(256 * 256 * 4); c->h4 = xmalloc(4 * NSA * 4);
        c->hd = xmalloc((size_t)2 * 256 * NSD * 4); c->fd = xmalloc((size_t)256 * NSD * 2);
        c->ed = xmalloc((size_t)256 * NSD * sizeof(Enc));
    } else {
        c->dbuf = xmalloc(cap * 2 + 32); c->bcopy = xmalloc(cap * 2 + 2);
        c->es = xmalloc(PTOT * 4); c->vs = xmalloc(PTOT * 2);
        c->slot = xmalloc((size_t)257 * PTR); c->se = xmalloc((size_t)257 * 256 * 8);
        c->wtail = xmalloc(2 * (2 * CH + 16)); c->rtail = xmalloc(2 * RAWB + 32);
    }
}
static void tc_free(TC *c) {
    free(c->sy); free(c->wb); free(c->key); free(c->tcopy); free(c->bcopy); free(c->hs); free(c->ho); free(c->h4);
    free(c->hd); free(c->fd); free(c->ed); free(c->dbuf); free(c->es); free(c->vs); free(c->slot);
    free(c->se); free(c->wtail); free(c->rtail);
}

/* ======================= standalone (BFR7 kind 1) =======================
 * block: u8 c0 c1 c2 0 | u32 nnz | u32 nw | raw: 5 bytes per 8 weights | nnz x (u16 symbol, u16 freq)
 *        | nw words: the 16 initial states (2 words each), then the shared stream */
static size_t enc_alone(TC *c, const uint16_t *restrict t, size_t n, uint8_t *restrict o) {
    uint32_t *restrict hs = c->hs, (*ho)[256] = (uint32_t (*)[256])c->ho;
    memset(hs, 0, 65536 * 4);
    for (size_t i = 0; i < n; i += 8) hs[t[i]]++;        /* the bit choice uses a 1/8 sample */
    int act[256], na = 0;
    for (int e = 0; e < 256; e++) {
        uint32_t any = 0;
        for (int v = 0; v < 256; v++) { ho[e][v] = hs[((v & 0x80) << 8) | (e << 7) | (v & 0x7F)]; any |= ho[e][v]; }
        if (any) act[na++] = e;
    }
    /* the 3 bits that minimize the sample's entropy of the symbol (per exponent), in integer arithmetic; on an
     * exact tie the first choice in this loop order wins, so the choice is the same on every machine */
    int bc[3] = {4, 5, 6}; uint64_t best = 0; int have = 0;
    for (int a = 0; a < 8; a++) for (int b = a + 1; b < 8; b++) for (int d = b + 1; d < 8; d++) {
        uint64_t s = 0;
        for (int k = 0; k < na; k++) {
            uint32_t cc[8] = {0}; int e = act[k];
            for (int v = 0; v < 256; v++) cc[((v >> a) & 1) | (((v >> b) & 1) << 1) | (((v >> d) & 1) << 2)] += ho[e][v];
            s += clogc_fx(cc);
        }
        if (!have || s > best) { best = s; have = 1; bc[0] = a; bc[1] = b; bc[2] = d; }
    }
    uint8_t SEL[256], RAW[256];
    for (int v = 0; v < 256; v++) {
        int sv = 0, r = 0, ri = 0;
        for (int bit = 0; bit < 8; bit++) {
            int x = (v >> bit) & 1;
            if (bit == bc[0]) sv |= x; else if (bit == bc[1]) sv |= x << 1; else if (bit == bc[2]) sv |= x << 2;
            else r |= x << ri++;
        }
        SEL[v] = (uint8_t)sv; RAW[v] = (uint8_t)r;
    }
    size_t ng = (n + 7) / 8, full = n / 8;
    o[0] = (uint8_t)bc[0]; o[1] = (uint8_t)bc[1]; o[2] = (uint8_t)bc[2]; o[3] = 0;
    /* fused pass: symbol of each weight, its histogram (four copies), raw fields 8 per 5 bytes */
    uint16_t *restrict sy = c->sy; uint32_t (*restrict h4)[NSA] = (uint32_t (*)[NSA])c->h4;
    memset(h4, 0, 4 * NSA * 4);
    uint8_t *restrict rp = o + 12;
#define FU(j) uint32_t o##j = oth(w[j]); uint32_t s##j = ((w[j] >> 4) & 0x7F8) | SEL[o##j];              \
              sy[8 * g + j] = (uint16_t)s##j; h4[(j) & 3][s##j]++;
    for (size_t g = 0; g < full; g++) {
        const uint16_t *w = t + 8 * g;
        FU(0) FU(1) FU(2) FU(3) FU(4) FU(5) FU(6) FU(7)
        uint64_t v = (uint64_t)RAW[o0] | (uint64_t)RAW[o1] << 5 | (uint64_t)RAW[o2] << 10 | (uint64_t)RAW[o3] << 15 |
                     (uint64_t)RAW[o4] << 20 | (uint64_t)RAW[o5] << 25 | (uint64_t)RAW[o6] << 30 | (uint64_t)RAW[o7] << 35;
        memcpy(rp, &v, 8); rp += 5;
    }
#undef FU
    if (ng > full) {
        uint64_t v = 0;
        for (size_t k = 8 * full; k < n; k++) {
            uint32_t ok = oth(t[k]), sk = ((t[k] >> 4) & 0x7F8) | SEL[ok];
            sy[k] = (uint16_t)sk; h4[0][sk]++; v |= (uint64_t)RAW[ok] << (5 * (k % 8));
        }
        memcpy(rp, &v, 8); rp += 5;
    }
    uint8_t *q = rp;
    uint32_t cnt[NSA]; uint16_t f[NSA]; uint32_t nnz = 0;
    for (int s = 0; s < NSA; s++) { cnt[s] = h4[0][s] + h4[1][s] + h4[2][s] + h4[3][s]; nnz += cnt[s] != 0; }
    normalize(cnt, f, NSA, n, PTOT);
    Enc E[NSA]; mkenc(f, NSA, E, PB);
    for (int s = 0; s < NSA; s++) if (f[s]) { uint16_t a = (uint16_t)s, b = f[s]; memcpy(q, &a, 2); memcpy(q + 2, &b, 2); q += 4; }
    if (!BFR7_LAYOUT()) {                     /* BFR8: index after the table, then the sub-streams */
        uint32_t *key = c->key, nw8;
        for (size_t k = 0; k < n; k++) key[k] = sy[k];
        q += enc_streams8(c->wb + n + 2 * 32 + 16, key, E, n, NULL, q, &nw8);
        memcpy(o + 4, &nnz, 4); memcpy(o + 8, &nw8, 4);
        return (size_t)(q - o);
    }
    uint16_t *wend = c->wb + n + 2 * NS + 16, *p = wend;
    FS(XINIT)
    size_t k = n;
    while (k % NS) { k--; uint32_t *xs[NS] = {FS(XPTR)}; uint32_t X = *xs[k % NS]; ENC1(X, E[sy[k]]); *xs[k % NS] = X; }
#define EQA(q) ENC1(x##q, E[sy[k - NS + q]]);
    for (; k; k -= NS) { FSR(EQA) }
#undef EQA
    uint32_t xf[NS] = {FS(XVAL)};
    for (int s = NS - 1; s >= 0; s--) { *--p = (uint16_t)(xf[s] >> 16); *--p = (uint16_t)xf[s]; }
    uint32_t nw = (uint32_t)(wend - p);
    memcpy(q, p, 2 * (size_t)nw); q += 2 * (size_t)nw;
    memcpy(o + 4, &nnz, 4); memcpy(o + 8, &nw, 4);
    return (size_t)(q - o);
}

/* switch a stream to its padded tail copy once `need` bytes or fewer remain; once on the copy, a stream
 * more than `slack` bytes past its real end is corrupt (the raw-bit reader loads up to 8 bytes ahead of
 * the bits it has used; the word reader does not run ahead) */
#define STREAM_CHECK(P, END, TAILBUF, ONTAIL, CONS0, ORIG, NEED, SLACK) do {                                   \
        if (!(ONTAIL)) {                                                                               \
            if ((size_t)((END) - (P)) <= (size_t)(NEED)) {                                             \
                size_t R_ = (size_t)((END) - (P));                                                     \
                memcpy((TAILBUF), (P), R_); memset((TAILBUF) + R_, 0, (size_t)(NEED) + (SLACK) + 16);  \
                (CONS0) = (size_t)((P) - (ORIG)); (P) = (TAILBUF); (END) = (TAILBUF) + R_; (ONTAIL) = 1; \
            }                                                                                          \
        } else if ((P) > (END) + (SLACK)) die("corrupt block (stream overrun)");                      \
    } while (0)

static void dec_alone(TC *c, const uint8_t *restrict in, size_t len, size_t n, uint16_t *restrict t) {
    Rd r = {in, in + len, 0};
    int c0 = rd8(&r), c1 = rd8(&r), c2 = rd8(&r); rd8(&r);
    uint32_t nnz = rd32(&r), nw = rd32(&r);
    size_t ng = (n + 7) / 8;
    if (r.bad || !(c0 < c1 && c1 < c2 && c2 < 8) || nnz < 1 || nnz > NSA || nw < 2 * NS) die("corrupt block (header)");
    if ((uint64_t)12 + 5 * (uint64_t)ng + 4 * (uint64_t)nnz + 2 * (uint64_t)nw != len) die("corrupt block (length)");
    const uint8_t *rp = in + 12;
    r.p = in + 12 + 5 * ng;
    uint16_t f[NSA]; memset(f, 0, sizeof f);
    uint32_t acc = 0; int prev = -1;
    for (uint32_t k = 0; k < nnz; k++) {
        int s = (int)rd16(&r); uint32_t v = rd16(&r);
        if (s <= prev || s >= NSA || v < 1 || v > PTOT) die("corrupt block (table)");
        f[s] = (uint16_t)v; acc += v; prev = s;
    }
    if (acc != PTOT) die("corrupt block (table)");
    /* value bits of each symbol and of each raw field */
    uint16_t RAWMAP[32];
    int rawbit[5], ri = 0;
    for (int bit = 0; bit < 8; bit++) if (bit != c0 && bit != c1 && bit != c2) rawbit[ri++] = bit;
#define OTHPOS(bit) ((bit) == 7 ? 15 : (bit))
    for (int v = 0; v < 32; v++) { uint16_t x = 0; for (int k = 0; k < 5; k++) x |= ((v >> k) & 1) << OTHPOS(rawbit[k]); RAWMAP[v] = x; }
    uint32_t *restrict es = c->es; uint16_t *restrict vs = c->vs;
    acc = 0;
    for (int s = 0; s < NSA; s++) {
        if (!f[s]) continue;
        uint16_t val = (uint16_t)(((s >> 3) << 7) | ((s & 1) << OTHPOS(c0)) | (((s >> 1) & 1) << OTHPOS(c1)) | (((s >> 2) & 1) << OTHPOS(c2)));
        for (uint32_t z = 0; z < f[s]; z++) { es[acc + z] = ((uint32_t)f[s] << 16) | z; vs[acc + z] = val; }
        acc += f[s];
    }
#undef OTHPOS
#define DECA(X) ({ uint32_t sl_ = (X) & PM; uint32_t e_ = es[sl_]; uint32_t v_ = vs[sl_];             \
        uint32_t y_ = (e_ >> 16) * ((X) >> PB) + (e_ & 0xFFFF);                                       \
        RENORM(X, y_); v_; })
    const uint8_t *w0 = r.p, *p = w0, *pend = w0 + 2 * (size_t)nw;
    FS(XLOAD)
    p += 4 * NS;
    int ontail = 0; size_t cons0 = 0;
    size_t i = 0;
#define OQ(q) { const uint64_t v = vv[(q) >> 3]; uint32_t v_ = DECA(x##q); t[i + (q)] = (uint16_t)(v_ | RAWMAP[(v >> (5 * ((q) & 7))) & 31]); }
    while (i < n) {
        size_t cend = n - i > CH ? i + CH : n;
        STREAM_CHECK(p, pend, c->wtail, ontail, cons0, w0, 2 * CH + 2, 0);
        for (; i + NS <= cend; i += NS) {
            uint64_t vv[2]; memcpy(&vv[0], rp, 8); memcpy(&vv[1], rp + 5, 8); rp += 10;
            FS(OQ)
        }
        if (i < cend) {                       /* fewer than NS weights left in the block */
            uint32_t *xs[NS] = {FS(XPTR)};
            for (size_t j = 0; i + j < cend; j++) {
                uint64_t v; memcpy(&v, rp + 5 * (j / 8), 8);
                uint32_t X = *xs[j]; uint32_t v_ = DECA(X); t[i + j] = (uint16_t)(v_ | RAWMAP[(v >> (5 * (j & 7))) & 31]); *xs[j] = X;
            }
            i = cend;
        }
    }
#undef OQ
#undef DECA
    size_t used = ontail ? cons0 + (size_t)(p - c->wtail) : (size_t)(p - w0);
    if (0 FS(XBAD) || used != 2 * (size_t)nw) die("corrupt block (final state)");
}

/* ======================= given the reference (BFR7 kind 2) =======================
 * block: u32 rawlen | u32 nw | rawlen bytes of raw bits (LSB first) | u16 nact |
 *        nact x (u8 context, u8 nz - 1, nz x (u8 symbol, u16 freq)) | nw words: 16 initial states, stream */
static size_t enc_ref(TC *c, const uint16_t *restrict t, const uint16_t *restrict b, size_t n, uint8_t *restrict o) {
    uint32_t *restrict h = c->hd; uint16_t *restrict sy = c->sy;
    memset(h, 0, (size_t)2 * 256 * NSD * 4);
    uint8_t *restrict rp = o + 8; uint64_t acc = 0; uint32_t na = 0;
    /* one pass: symbol, histogram (two copies), raw bits written LSB first; for negative j the raw bits
     * are complemented so that the decoder always adds them */
#define SYM1(i) do { uint32_t tw_ = t[i], bw_ = b[i];                                                \
        int32_t d_ = (int32_t)(tw_ & 0x7FFF) - (int32_t)(bw_ & 0x7FFF), sg_ = d_ >> 31;                 \
        uint32_t a_ = (uint32_t)((d_ ^ sg_) - sg_), neg_ = (uint32_t)sg_ & 1, flip_ = (tw_ ^ bw_) >> 15;\
        int32_t L_ = 32 - __builtin_clz(a_ | 1), v_ = L_ - 1 - MB; uint32_t sh_ = (uint32_t)(v_ & ~(v_ >> 31)); \
        uint32_t big_ = (1u << (MB + 1)) + (uint32_t)(L_ - MB - 2) * (1u << MB) + ((a_ >> sh_) & ((1u << MB) - 1)); \
        uint32_t small_ = 0u - (uint32_t)(L_ <= MB + 1), m_ = big_ + ((a_ - big_) & small_);           \
        uint32_t s_ = (flip_ * 2 + neg_) * NMAG + m_, mk_ = (1u << sh_) - 1;                           \
        sy[i] = (uint16_t)s_; h[((i) & 1) * 256 * NSD + ((bw_ >> 7) & 0xFF) * NSD + s_]++;              \
        acc |= (uint64_t)((a_ & mk_) ^ (mk_ & (uint32_t)sg_)) << na; na += sh_; } while (0)
#define FLUSH() do { memcpy(rp, &acc, 8); rp += na >> 3; acc >>= na & 56; na &= 7; } while (0)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) { SYM1(i); SYM1(i + 1); SYM1(i + 2); SYM1(i + 3); FLUSH(); }
    for (; i < n; i++) { SYM1(i); FLUSH(); }
#undef SYM1
#undef FLUSH
    memcpy(rp, &acc, 8); rp += (na + 7) >> 3;
    uint32_t rawlen = (uint32_t)(rp - (o + 8));
    for (size_t z = 0; z < (size_t)256 * NSD; z++) h[z] += h[256 * NSD + z];
    uint16_t *restrict f = c->fd; Enc *restrict E = c->ed;
    uint8_t *q = rp; uint8_t *qn = q; q += 2; uint16_t nact = 0;
    for (int cx = 0; cx < 256; cx++) {
        uint64_t ct = 0;
        for (int s = 0; s < NSD; s++) ct += h[cx * NSD + s];
        if (!ct) continue;
        normalize(h + cx * NSD, f + cx * NSD, NSD, ct, PTR); mkenc(f + cx * NSD, NSD, E + cx * NSD, PBR); nact++;
        int nz = 0; for (int s = 0; s < NSD; s++) nz += f[cx * NSD + s] != 0;
        q[0] = (uint8_t)cx; q[1] = (uint8_t)(nz - 1); q += 2;
        for (int s = 0; s < NSD; s++) if (f[cx * NSD + s]) { uint16_t v = f[cx * NSD + s]; q[0] = (uint8_t)s; memcpy(q + 1, &v, 2); q += 3; }
    }
    memcpy(qn, &nact, 2);
    if (!BFR7_LAYOUT()) {                     /* BFR8: index (word and raw-bit offsets) after the tables */
        uint32_t *key = c->key, nw8; size_t sw = g_sw ? g_sw : (n ? n : 1), ns = n ? (n + sw - 1) / sw : 1;
        uint64_t *rb = xmalloc(ns * 8), bits = 0;
        for (size_t k = 0; k < n; k++) {
            if (k % sw == 0) rb[k / sw] = bits;
            uint32_t m = sy[k] % NMAG; bits += m < (1u << (MB + 1)) ? 0 : (m - (1u << (MB + 1))) / (1u << MB) + 1;
            key[k] = ((b[k] >> 7) & 0xFF) * NSD + sy[k];
        }
        if (n == 0) rb[0] = 0;
        q += enc_streams8(c->wb + n + 2 * 32 + 16, key, E, n, rb, q, &nw8);
        free(rb);
        memcpy(o, &rawlen, 4); memcpy(o + 4, &nw8, 4);
        return (size_t)(q - o);
    }
    uint16_t *wend = c->wb + n + 2 * NS + 16, *p = wend;
    FS(XINIT)
#define SYR(k) E[((b[k] >> 7) & 0xFF) * NSD + sy[k]]
    size_t k = n;
    while (k % NS) { k--; uint32_t *xs[NS] = {FS(XPTR)}; uint32_t X = *xs[k % NS]; ENC1(X, SYR(k)); *xs[k % NS] = X; }
#define EQR(q) ENC1(x##q, SYR(k - NS + q));
    for (; k; k -= NS) { FSR(EQR) }
#undef EQR
#undef SYR
    uint32_t xf[NS] = {FS(XVAL)};
    for (int s = NS - 1; s >= 0; s--) { *--p = (uint16_t)(xf[s] >> 16); *--p = (uint16_t)xf[s]; }
    uint32_t nw = (uint32_t)(wend - p);
    memcpy(q, p, 2 * (size_t)nw); q += 2 * (size_t)nw;
    memcpy(o, &rawlen, 4); memcpy(o + 4, &nw, 4);
    return (size_t)(q - o);
}

static void dec_ref(TC *c, const uint8_t *restrict in, size_t len, size_t n, const uint16_t *restrict b, uint16_t *restrict t) {
    Rd r = {in, in + len, 0};
    uint32_t rawlen = rd32(&r), nw = rd32(&r);
    if (r.bad || nw < 2 * NS || (uint64_t)rawlen + 2 * (uint64_t)nw + 10 > len) die("corrupt block (header)");
    const uint8_t *raw0 = in + 8, *rawend = raw0 + rawlen;
    r.p = rawend; r.e = in + len - 2 * (size_t)nw;
    uint32_t nact = rd16(&r);
    if (r.bad || nact < 1 || nact > 256) die("corrupt block (table)");
    /* per active context: slot -> symbol (u8) and symbol -> entry:
     *   bits 0..15 start, 16..31 f, 32..47 A (t = b + A + raw mod 2^16), 48..51 k raw bits, 52..63 mask.
     * Contexts not in the table map to a zeroed table (index nact): a wrong reference or a corrupt block
     * then decodes garbage within bounds and fails the final check. */
    uint8_t *restrict slot = c->slot; uint64_t *restrict se = c->se;
    uint32_t cb[256];
    for (int x = 0; x < 256; x++) cb[x] = nact;
    memset(slot + (size_t)nact * PTR, 0, PTR); memset(se + (size_t)nact * 256, 0, 256 * 8);
    int prevc = -1;
    for (uint32_t a = 0; a < nact; a++) {
        int cx = (int)rd8(&r), nz = (int)rd8(&r) + 1;
        if (r.bad || cx <= prevc || nz > NSD) die("corrupt block (table)");
        prevc = cx; cb[cx] = a;
        uint16_t f[NSD]; memset(f, 0, sizeof f);
        int prev = -1;
        for (int k = 0; k < nz; k++) {
            int s = (int)rd8(&r); uint32_t v = rd16(&r);
            if (r.bad || s <= prev || s >= NSD || v < 1 || v > PTR) die("corrupt block (table)");
            f[s] = (uint16_t)v; prev = s;
        }
        uint32_t acc = 0;
        for (int s = 0; s < NSD; s++) {
            uint32_t flip = s / (2 * NMAG), neg = (s / NMAG) & 1, m = s % NMAG, base, k;
            if (m < (1u << (MB + 1))) { base = m; k = 0; }
            else { uint32_t L = (m - (1u << (MB + 1))) / (1u << MB) + MB + 2, top = (m - (1u << (MB + 1))) % (1u << MB);
                   k = L - 1 - MB; base = ((1u << MB) | top) << k; }
            uint16_t A = neg ? (uint16_t)(0u - base - ((1u << k) - 1)) : (uint16_t)base;
            A = (uint16_t)(A + (flip ? 0x8000u : 0));          /* flipping the sign bit = adding 0x8000 */
            if (acc + f[s] > PTR) die("corrupt block (table)");
            se[(size_t)a * 256 + s] = acc | ((uint64_t)f[s] << 16) | ((uint64_t)A << 32) | ((uint64_t)k << 48) | ((uint64_t)((1u << k) - 1) << 52);
            memset(slot + (size_t)a * PTR + acc, s, f[s]);
            acc += f[s];
        }
        if (acc != PTR) die("corrupt block (table)");
    }
    if (r.p != r.e) die("corrupt block (table size)");
    const uint8_t *w0 = r.e, *p = w0, *pend = w0 + 2 * (size_t)nw;
    FS(XLOAD)
    p += 4 * NS;
    const uint8_t *raw = raw0;
    uint64_t buf = 0; uint32_t cnt = 0;
    int wtl = 0, rtl = 0; size_t wcons0 = 0, rcons0 = 0;
#define REFILL() do { uint64_t w_; memcpy(&w_, raw, 8); buf |= w_ << cnt; raw += (63 - cnt) >> 3; cnt |= 56; } while (0)
#define DECR(X, i) do { uint32_t bb_ = b[i]; uint32_t a_ = cb[(bb_ >> 7) & 0xFF];                       \
        uint32_t sl_ = (X) & PMR; uint64_t e_ = se[(size_t)a_ * 256 + slot[(size_t)a_ * PTR + sl_]];   \
        uint32_t y_ = (uint32_t)((e_ >> 16) & 0xFFFF) * ((X) >> PBR) + sl_ - (uint32_t)(e_ & 0xFFFF); \
        RENORM(X, y_);                                                                                 \
        uint32_t k_ = (uint32_t)(e_ >> 48) & 15, rv_ = (uint32_t)(buf & (e_ >> 52)); buf >>= k_; cnt -= k_; \
        t[i] = (uint16_t)(bb_ + (uint32_t)(e_ >> 32) + rv_); } while (0)
#define DQ4(a, b_, c_, d) REFILL(); DECR(x##a, i + a); DECR(x##b_, i + b_); DECR(x##c_, i + c_); DECR(x##d, i + d);
    const uint8_t *rawe = rawend;
    size_t i = 0;
    while (i < n) {
        size_t cend = n - i > CH ? i + CH : n;
        STREAM_CHECK(p, pend, c->wtail, wtl, wcons0, w0, 2 * CH + 2, 0);
        STREAM_CHECK(raw, rawe, c->rtail, rtl, rcons0, raw0, RAWB, 8);
        for (; i + NS <= cend; i += NS) { DQ4(0, 1, 2, 3) DQ4(4, 5, 6, 7) DQ4(8, 9, 10, 11) DQ4(12, 13, 14, 15) }
        if (i < cend) {
            uint32_t *xs[NS] = {FS(XPTR)};
            for (size_t j = 0; i + j < cend; j++) { if (j % 4 == 0) REFILL(); uint32_t X = *xs[j]; DECR(X, i + j); *xs[j] = X; }
            i = cend;
        }
    }
#undef DQ4
#undef DECR
#undef REFILL
    size_t used = wtl ? wcons0 + (size_t)(p - c->wtail) : (size_t)(p - w0);
    uint64_t rbytes = rtl ? rcons0 + (size_t)(raw - c->rtail) : (size_t)(raw - raw0);
    uint64_t bits = 8 * rbytes - cnt;
    if (0 FS(XBAD) || used != 2 * (size_t)nw || (bits + 7) / 8 != rawlen) die("corrupt block (final state)");
}

/* ======================= BFR8 decoders: BFR7 blocks whose words are cut into sub-streams =======================
 * standalone:  as BFR7, with (ns - 1) x u32 word offset between the table and the words
 * given ref:   as BFR7, with (ns - 1) x (u32 word offset, u32 raw-bit offset) between the tables and the words
 * Each sub-stream holds its g_lanes initial states (2 words each) and then its words. */
#define FS32(M) FS(M) M(16) M(17) M(18) M(19) M(20) M(21) M(22) M(23) M(24) M(25) M(26) M(27) M(28) M(29) M(30) M(31)
#define K_L 16
#define K_FS FS
#define K_DQ DQ4(0, 1, 2, 3) DQ4(4, 5, 6, 7) DQ4(8, 9, 10, 11) DQ4(12, 13, 14, 15)
#define K_SUF 16
#include "bfr8_kern.h"
#undef K_L
#undef K_FS
#undef K_DQ
#undef K_SUF
#define K_L 32
#define K_FS FS32
#define K_DQ DQ4(0, 1, 2, 3) DQ4(4, 5, 6, 7) DQ4(8, 9, 10, 11) DQ4(12, 13, 14, 15) \
             DQ4(16, 17, 18, 19) DQ4(20, 21, 22, 23) DQ4(24, 25, 26, 27) DQ4(28, 29, 30, 31)
#define K_SUF 32
#include "bfr8_kern.h"
#undef K_L
#undef K_FS
#undef K_DQ
#undef K_SUF

/* reads the index: ns - 1 entries of `esz` bytes at ix; word offsets must increase by at least 2 L words */
static void sub_bounds(const uint8_t *ix, size_t ns, size_t esz, uint32_t nw, int L, uint32_t *wo, uint64_t *rb, uint64_t rbits) {
    wo[0] = 0; if (rb) rb[0] = 0;
    for (size_t s = 1; s < ns; s++) {
        wo[s] = ld32(ix + (s - 1) * esz);
        if (rb) rb[s] = ld32(ix + (s - 1) * esz + 4);
        if (wo[s] < wo[s - 1] + 2 * (uint32_t)L || (rb && (rb[s] < rb[s - 1] || rb[s] > rbits))) die("corrupt block (index)");
    }
    wo[ns] = nw; if (rb) rb[ns] = rbits;
    if (wo[ns] < wo[ns - 1] + 2 * (uint32_t)L) die("corrupt block (index)");
}

static void dec_alone8(TC *c, const uint8_t *restrict in, size_t len, size_t n, uint16_t *restrict t) {
    int L = g_lanes; size_t sw = g_sw ? g_sw : (n ? n : 1), ns = n ? (n + sw - 1) / sw : 1;
    Rd r = {in, in + len, 0};
    int c0 = rd8(&r), c1 = rd8(&r), c2 = rd8(&r); rd8(&r);
    uint32_t nnz = rd32(&r), nw = rd32(&r);
    size_t ng = (n + 7) / 8;
    if (r.bad || !(c0 < c1 && c1 < c2 && c2 < 8) || nnz < 1 || nnz > NSA) die("corrupt block (header)");
    if ((uint64_t)12 + 5 * (uint64_t)ng + 4 * (uint64_t)nnz + 4 * (uint64_t)(ns - 1) + 2 * (uint64_t)nw != len) die("corrupt block (length)");
    const uint8_t *rp = in + 12;
    r.p = in + 12 + 5 * ng;
    uint16_t f[NSA]; memset(f, 0, sizeof f);
    uint32_t acc = 0; int prev = -1;
    for (uint32_t k = 0; k < nnz; k++) {
        int s = (int)rd16(&r); uint32_t v = rd16(&r);
        if (s <= prev || s >= NSA || v < 1 || v > PTOT) die("corrupt block (table)");
        f[s] = (uint16_t)v; acc += v; prev = s;
    }
    if (acc != PTOT) die("corrupt block (table)");
    uint16_t RAWMAP[32];
    int rawbit[5], ri = 0;
    for (int bit = 0; bit < 8; bit++) if (bit != c0 && bit != c1 && bit != c2) rawbit[ri++] = bit;
#define OTHPOS(bit) ((bit) == 7 ? 15 : (bit))
    for (int v = 0; v < 32; v++) { uint16_t x = 0; for (int k = 0; k < 5; k++) x |= ((v >> k) & 1) << OTHPOS(rawbit[k]); RAWMAP[v] = x; }
    uint32_t *restrict es = c->es; uint16_t *restrict vs = c->vs;
    acc = 0;
    for (int s = 0; s < NSA; s++) {
        if (!f[s]) continue;
        uint16_t val = (uint16_t)(((s >> 3) << 7) | ((s & 1) << OTHPOS(c0)) | (((s >> 1) & 1) << OTHPOS(c1)) | (((s >> 2) & 1) << OTHPOS(c2)));
        for (uint32_t z = 0; z < f[s]; z++) { es[acc + z] = ((uint32_t)f[s] << 16) | z; vs[acc + z] = val; }
        acc += f[s];
    }
#undef OTHPOS
    const uint8_t *ix = r.p, *w0 = ix + 4 * (ns - 1);
    uint32_t *wo = xmalloc((ns + 1) * 4);
    sub_bounds(ix, ns, 4, nw, L, wo, NULL, 0);
    for (size_t s = 0; s < ns; s++) {
        size_t a = s * sw, m = n - a < sw ? n - a : sw;
        const uint8_t *ws = w0 + 2 * (size_t)wo[s], *we = w0 + 2 * (size_t)wo[s + 1];
        if (L == 32) dsub_alone32(c, ws, we, rp + 5 * (a / 8), m, es, vs, RAWMAP, t + a);
        else dsub_alone16(c, ws, we, rp + 5 * (a / 8), m, es, vs, RAWMAP, t + a);
    }
    free(wo);
}

static void dec_ref8(TC *c, const uint8_t *restrict in, size_t len, size_t n, const uint16_t *restrict b, uint16_t *restrict t) {
    int L = g_lanes; size_t sw = g_sw ? g_sw : (n ? n : 1), ns = n ? (n + sw - 1) / sw : 1;
    Rd r = {in, in + len, 0};
    uint32_t rawlen = rd32(&r), nw = rd32(&r);
    if (r.bad || (uint64_t)rawlen + 8 * (uint64_t)(ns - 1) + 2 * (uint64_t)nw + 10 > len) die("corrupt block (header)");
    const uint8_t *raw0 = in + 8, *rawend = raw0 + rawlen;
    r.p = rawend; r.e = in + len - 2 * (size_t)nw - 8 * (ns - 1);
    uint32_t nact = rd16(&r);
    if (r.bad || nact < 1 || nact > 256) die("corrupt block (table)");
    uint8_t *restrict slot = c->slot; uint64_t *restrict se = c->se;
    uint32_t cb[256];
    for (int x = 0; x < 256; x++) cb[x] = nact;
    memset(slot + (size_t)nact * PTR, 0, PTR); memset(se + (size_t)nact * 256, 0, 256 * 8);
    int prevc = -1;
    for (uint32_t a = 0; a < nact; a++) {
        int cx = (int)rd8(&r), nz = (int)rd8(&r) + 1;
        if (r.bad || cx <= prevc || nz > NSD) die("corrupt block (table)");
        prevc = cx; cb[cx] = a;
        uint16_t f[NSD]; memset(f, 0, sizeof f);
        int prev = -1;
        for (int k = 0; k < nz; k++) {
            int s = (int)rd8(&r); uint32_t v = rd16(&r);
            if (r.bad || s <= prev || s >= NSD || v < 1 || v > PTR) die("corrupt block (table)");
            f[s] = (uint16_t)v; prev = s;
        }
        uint32_t acc = 0;
        for (int s = 0; s < NSD; s++) {
            uint32_t flip = s / (2 * NMAG), neg = (s / NMAG) & 1, m = s % NMAG, base, k;
            if (m < (1u << (MB + 1))) { base = m; k = 0; }
            else { uint32_t Lb = (m - (1u << (MB + 1))) / (1u << MB) + MB + 2, top = (m - (1u << (MB + 1))) % (1u << MB);
                   k = Lb - 1 - MB; base = ((1u << MB) | top) << k; }
            uint16_t A = neg ? (uint16_t)(0u - base - ((1u << k) - 1)) : (uint16_t)base;
            A = (uint16_t)(A + (flip ? 0x8000u : 0));
            if (acc + f[s] > PTR) die("corrupt block (table)");
            se[(size_t)a * 256 + s] = acc | ((uint64_t)f[s] << 16) | ((uint64_t)A << 32) | ((uint64_t)k << 48) | ((uint64_t)((1u << k) - 1) << 52);
            memset(slot + (size_t)a * PTR + acc, s, f[s]);
            acc += f[s];
        }
        if (acc != PTR) die("corrupt block (table)");
    }
    if (r.p != r.e) die("corrupt block (table size)");
    const uint8_t *ix = r.e, *w0 = ix + 8 * (ns - 1);
    uint32_t *wo = xmalloc((ns + 1) * 4); uint64_t *rb = xmalloc((ns + 1) * 8);
    sub_bounds(ix, ns, 8, nw, L, wo, rb, 8 * (uint64_t)rawlen);
    for (size_t s = 0; s < ns; s++) {
        size_t a = s * sw, m = n - a < sw ? n - a : sw; int last = s + 1 == ns;
        const uint8_t *ws = w0 + 2 * (size_t)wo[s], *we = w0 + 2 * (size_t)wo[s + 1];
        if (L == 32) dsub_ref32(c, ws, we, raw0, rawend, rb[s], rb[s + 1], last, m, cb, slot, se, b + a, t + a);
        else dsub_ref16(c, ws, we, raw0, rawend, rb[s], rb[s + 1], last, m, cb, slot, se, b + a, t + a);
    }
    free(wo); free(rb);
}

/* ======================= BFR6 decoder (kept so that BFR6 archives still decode) ======================= */
#define K6 255
#define NJ6 (2 * K6 + 1)
#define ESC6 (2 * NJ6)
#define NSYM6 (ESC6 + 1)
#define MAXCTX6 1024
#define CTXEXP6 256
#define NS6 8
static inline int ctx6(uint16_t b, int cb) { return cb == 8 ? (b >> 7) & 0xFF : (b >> 5) & 0x3FF; }
static inline uint16_t unsym6(int s, uint16_t b) {
    int flip = s >= NJ6, j = s - flip * NJ6 - K6;
    return (uint16_t)(((b ^ (flip << 15)) & 0x8000) | ((b & 0x7FFF) + j));
}
/* one rANS step; an entry packs the symbol (bits 32..47), its frequency (16..31) and the slot's offset
 * from the symbol's start (0..15).  Missing contexts map to a zeroed table (index nact). */
#define DEC6(xs, p, a) ({ uint64_t e_ = lut[(size_t)(a) * PTOT + ((xs) & (PTOT - 1))];             \
                      uint32_t y_ = (uint32_t)((e_ >> 16) & 0xFFFF) * ((xs) >> PB) + (uint32_t)(e_ & 0xFFFF); \
                      uint32_t r_ = y_ < RL;                                                       \
                      (xs) = (y_ << (r_ << 4)) | (*(p) & (0u - r_)); (p) += r_;                   \
                      (int)(e_ >> 32); })
static void dec6_block(const uint8_t *in, size_t len, size_t n, int mode, const uint16_t *b, uint16_t *t) {
    Rd r = {in, in + len, 0};
    uint32_t tabbytes = rd32(&r), nesc = rd32(&r), nw[NS6]; uint64_t nwt = 0;
    for (int k = 0; k < NS6; k++) { nw[k] = rd32(&r); nwt += nw[k]; if (nw[k] < 2) r.bad = 1; }
    if (r.bad || (uint64_t)8 + 4 * NS6 + tabbytes + 2 * (uint64_t)nesc + 2 * nwt + 4 != len) die("corrupt block (header)");
    const uint8_t *tab = r.p; r.e = tab + tabbytes;
    uint32_t nact = rd16(&r);
    if (r.bad || nact > MAXCTX6 + 1) die("corrupt block (table)");
    int map[MAXCTX6]; for (int x = 0; x < MAXCTX6; x++) map[x] = (int)nact;
    uint64_t *lut = xcalloc((size_t)(nact + 1) * PTOT, 8);
    uint16_t *f = xmalloc(NSYM6 * 2);
    for (uint32_t a = 0; a < nact; a++) {
        uint32_t cx = rd16(&r), nnz = rd16(&r);
        if (r.bad || cx >= MAXCTX6) die("corrupt block (table)");
        map[cx] = (int)a;
        uint64_t *l = lut + (size_t)a * PTOT; uint32_t acc = 0;
        memset(f, 0, NSYM6 * 2);
        for (uint32_t k = 0; k < nnz; k++) { uint32_t s = rd16(&r), v = rd16(&r); if (r.bad || s >= NSYM6) die("corrupt block (table)"); f[s] = (uint16_t)(v ? v : PTOT); }
        for (int s = 0; s < NSYM6; s++) {
            if (acc + f[s] > PTOT) die("corrupt block (table)");
            for (uint32_t k = 0; k < f[s]; k++) l[acc + k] = ((uint64_t)s << 32) | ((uint64_t)f[s] << 16) | k;
            acc += f[s];
        }
        if (acc != PTOT) die("corrupt block (table)");
    }
    free(f);
    if (r.p != r.e) die("corrupt block (table size)");
    const uint8_t *esc = r.e;
    /* the eight word streams are copied into one buffer padded by the most words any stream can read
     * (one per symbol it codes), so that a corrupt block cannot read past it */
    size_t m = mode ? n : 2 * n, pad = m / NS6 + 8;
    uint16_t *ws = xcalloc(nwt + pad, 2);
    memcpy(ws, esc + 2 * (size_t)nesc, 2 * nwt);
    const uint16_t *pp[NS6], *pend[NS6], *w = ws;
    for (int k = 0; k < NS6; k++) { pp[k] = w; pend[k] = w + nw[k]; w = pend[k]; }
    uint32_t x0, x1, x2, x3, x4, x5, x6, x7;
#define LD6(q) x##q = (uint32_t)pp[q][0] | ((uint32_t)pp[q][1] << 16); pp[q] += 2;
    LD6(0) LD6(1) LD6(2) LD6(3) LD6(4) LD6(5) LD6(6) LD6(7)
#undef LD6
    const uint16_t *p0 = pp[0], *p1 = pp[1], *p2 = pp[2], *p3 = pp[3], *p4 = pp[4], *p5 = pp[5], *p6 = pp[6], *p7 = pp[7];
    size_t e = 0;
#define ESCV() (e < nesc ? ld16(esc + 2 * e++) : (e++, (uint16_t)0))
#define REF1(xs, ps, i, md) do { int s_ = DEC6(xs, ps, map[ctx6(b[i], md)]);                       \
                             t[i] = s_ == ESC6 ? ESCV() : unsym6(s_, b[i]); } while (0)
#define ALONE1(xe, pe, xm, pm, i) do { int ex_ = DEC6(xe, pe, aexp); int sm_ = DEC6(xm, pm, map[ex_ & (MAXCTX6 - 1)]); \
                             t[i] = (uint16_t)(((sm_ >> 7) << 15) | (ex_ << 7) | (sm_ & 0x7F)); } while (0)
#define REFLOOP(MODE) do { const int mode_ = (MODE); size_t i = 0;                                \
        for (; i + 8 <= n; i += 8) {                                                                \
            REF1(x0, p0, i, mode_); REF1(x1, p1, i + 1, mode_); REF1(x2, p2, i + 2, mode_); REF1(x3, p3, i + 3, mode_); \
            REF1(x4, p4, i + 4, mode_); REF1(x5, p5, i + 5, mode_); REF1(x6, p6, i + 6, mode_); REF1(x7, p7, i + 7, mode_); } \
        if (i < n) REF1(x0, p0, i, mode_);                                                          \
        if (i + 1 < n) REF1(x1, p1, i + 1, mode_);                                                  \
        if (i + 2 < n) REF1(x2, p2, i + 2, mode_);                                                  \
        if (i + 3 < n) REF1(x3, p3, i + 3, mode_);                                                  \
        if (i + 4 < n) REF1(x4, p4, i + 4, mode_);                                                  \
        if (i + 5 < n) REF1(x5, p5, i + 5, mode_);                                                  \
        if (i + 6 < n) REF1(x6, p6, i + 6, mode_); } while (0)
    if (mode == 8) REFLOOP(8);
    else if (mode == 10) REFLOOP(10);
    else {
        int aexp = map[CTXEXP6];
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            ALONE1(x0, p0, x1, p1, i); ALONE1(x2, p2, x3, p3, i + 1);
            ALONE1(x4, p4, x5, p5, i + 2); ALONE1(x6, p6, x7, p7, i + 3);
        }
        if (i < n) ALONE1(x0, p0, x1, p1, i);
        if (i + 1 < n) ALONE1(x2, p2, x3, p3, i + 1);
        if (i + 2 < n) ALONE1(x4, p4, x5, p5, i + 2);
    }
#undef REF1
#undef REFLOOP
#undef ALONE1
#undef ESCV
    int bad = e != nesc || x0 != RL || x1 != RL || x2 != RL || x3 != RL || x4 != RL || x5 != RL || x6 != RL || x7 != RL ||
              p0 != pend[0] || p1 != pend[1] || p2 != pend[2] || p3 != pend[3] ||
              p4 != pend[4] || p5 != pend[5] || p6 != pend[6] || p7 != pend[7];
    free(ws); free(lut);
    if (bad) die("corrupt block (final state)");
}

/* ---------- hashes ---------- */
static inline uint64_t mix64(uint64_t h, uint64_t w) { h = (h ^ w) * 0xFF51AFD7ED558CCDull; return h ^ (h >> 29); }
static uint64_t hash16(const uint16_t *p, size_t n) {    /* BFR6's reference hash; not cryptographic */
    uint64_t h[4] = {0x9E3779B97F4A7C15ull ^ n, 0xBF58476D1CE4E5B9ull, 0x94D049BB133111EBull, 0xD6E8FEB86659FD93ull};
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        for (int l = 0; l < 4; l++) {
            uint64_t w; memcpy(&w, p + i + 4 * l, 8);
            h[l] = (h[l] ^ w) * 0xFF51AFD7ED558CCDull; h[l] ^= h[l] >> 29;
        }
    uint64_t r = h[0];
    for (int l = 1; l < 4; l++) { r = (r ^ h[l]) * 0xC4CEB9FE1A85EC53ull; r ^= r >> 29; }
    for (; i < n; i++) { r = (r ^ p[i]) * 0xC4CEB9FE1A85EC53ull; r ^= r >> 29; }
    return r;
}
static const uint16_t *aligned16(const uint8_t *p, size_t len, uint16_t *tmp) {
    if (!((uintptr_t)p & 1)) return (const uint16_t *)p;
    memcpy(tmp, p, len); return tmp;
}
/* BFR6: hash of the whole reference, hash16 of each 64 MiB chunk in parallel, then combined */
typedef struct { const uint8_t *p; size_t len, nch, next; uint64_t *h; pthread_mutex_t mu; } HashJob;
static void *hash_worker(void *arg) {
    HashJob *J = arg; uint16_t *tmp = NULL;
    for (;;) {
        pthread_mutex_lock(&J->mu); size_t k = J->next++; pthread_mutex_unlock(&J->mu);
        if (k >= J->nch) { free(tmp); return NULL; }
        size_t a = k << 26, n = J->len - a < ((size_t)1 << 26) ? J->len - a : (size_t)1 << 26;
        if (((uintptr_t)(J->p + a) & 1) && !tmp) tmp = xmalloc((size_t)1 << 26);
        const uint16_t *w = aligned16(J->p + a, n, tmp);
        J->h[k] = hash16(w, n / 2) ^ (n & 1 ? (uint64_t)J->p[a + n - 1] << 56 : 0);
    }
}
static uint64_t file_hash6(const uint8_t *p, size_t len, int threads) {
    HashJob J = {p, len, (len + ((size_t)1 << 26) - 1) >> 26, 0, NULL, PTHREAD_MUTEX_INITIALIZER};
    J.h = xcalloc(J.nch, 8);
    if ((size_t)threads > J.nch) threads = (int)(J.nch ? J.nch : 1);
    pthread_t th[256];
    for (int i = 0; i < threads; i++) pthread_create(&th[i], NULL, hash_worker, &J);
    for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
    uint64_t r = 0x9E3779B97F4A7C15ull ^ len;
    for (size_t k = 0; k < J.nch; k++) { r = (r ^ J.h[k]) * 0xC4CEB9FE1A85EC53ull; r ^= r >> 29; }
    free(J.h);
    return r;
}
/* BFR7: fingerprint of the reference: its length, its first and last MiB (a safetensors header is at
 * the start) and 256 evenly spaced 4 KiB samples.  It tells models apart, not corrupt copies of one;
 * a wrong reference inside a coded block also changes the contexts and nearly always fails the block's
 * final-state check. */
static uint64_t hash_bytes(const uint8_t *p, size_t n, uint64_t h) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, p + i, 8); h = mix64(h, w); }
    for (; i < n; i++) h = mix64(h, p[i]);
    return mix64(h, n);
}
static uint64_t ref_fingerprint(const uint8_t *p, size_t len) {
    const size_t E = (size_t)1 << 20, S = 4096; const int NSAMP = 256;
    uint64_t h = mix64(0x9E3779B97F4A7C15ull, len);
    size_t head = len < E ? len : E;
    h = hash_bytes(p, head, h);
    if (len > head) { size_t tl = len - head < E ? len - head : E; h = hash_bytes(p + len - tl, tl, h); }
    if (len > S) for (int k = 0; k < NSAMP; k++) h = hash_bytes(p + (len - S) / (NSAMP - 1) * (size_t)k, S, h);
    return h;
}

/* Check of a segment's original bytes, stored in its record and verified before a decoded segment is
 * written: four CRC32C (Castagnoli) chains over the 8-byte words of the segment taken in turn (word i
 * goes to chain i mod 4, so that the chains run in parallel), the trailing bytes into chain 0, and the
 * four 32-bit values combined into 64 bits.  It catches corrupt archives, including bits stored raw, and
 * a wrong reference that the sampled fingerprint misses.  Hardware CRC32C on ARMv8 (+crc) and x86
 * (SSE4.2), a table otherwise; both give the same value. */
#ifndef CRC_HW
static uint32_t crc_tab[256];
static pthread_once_t crc_once = PTHREAD_ONCE_INIT;
static void crc_init(void) {
    for (uint32_t i = 0; i < 256; i++) { uint32_t c = i; for (int k = 0; k < 8; k++) c = c & 1 ? (c >> 1) ^ 0x82F63B78u : c >> 1; crc_tab[i] = c; }
}
static inline uint32_t CRC1B(uint32_t c, uint8_t b) { return (c >> 8) ^ crc_tab[(c ^ b) & 0xFF]; }
static inline uint32_t CRC8B(uint32_t c, uint64_t w) { for (int k = 0; k < 8; k++) c = CRC1B(c, (uint8_t)(w >> (8 * k))); return c; }
#endif
static uint64_t seg_check(const uint8_t *p, size_t n) {
#ifndef CRC_HW
    pthread_once(&crc_once, crc_init);
#endif
    uint32_t c0 = ~0u, c1 = ~1u, c2 = ~2u, c3 = ~3u;
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        uint64_t w[4]; memcpy(w, p + i, 32);
        c0 = CRC8B(c0, w[0]); c1 = CRC8B(c1, w[1]); c2 = CRC8B(c2, w[2]); c3 = CRC8B(c3, w[3]);
    }
    for (; i < n; i++) c0 = CRC1B(c0, p[i]);
    return (((uint64_t)c0 | (uint64_t)c1 << 32) ^ (((uint64_t)c2 | (uint64_t)c3 << 32) * 0x9E3779B97F4A7C15ull)) ^ n;
}

/* ---------- segments ---------- */
#define K_RAW 255              /* segment kinds: K_RAW; BFR7: 1 = alone, 2 = given the reference; BFR6: 0, 8, 10 */
#define K_ALONE 1
#define K_REF 2
typedef struct {
    int kind;
    uint64_t off, len, roff;          /* position in the file, length in bytes, position in the reference */
    uint64_t outlen;                  /* encoded length */
    uint64_t check;                   /* seg_check of the original bytes (BFR7) */
    const uint8_t *in;                /* encoded segment, when decoding */
} Block;

static void write_all(int fd, const void *p, size_t n) {
    const uint8_t *q = p;
    while (n) { size_t c = n < ((size_t)1 << 30) ? n : (size_t)1 << 30; ssize_t w = write(fd, q, c); if (w <= 0) die("write failed"); q += w; n -= (size_t)w; }
}
static void pwrite_all(int fd, const void *p, size_t n, uint64_t off) {
    const uint8_t *q = p;
    while (n) { size_t c = n < ((size_t)1 << 30) ? n : (size_t)1 << 30; ssize_t w = pwrite(fd, q, c, (off_t)off); if (w <= 0) die("write failed"); q += w; n -= (size_t)w; off += (uint64_t)w; }
}

/* encoding: workers take segments in order and code them with their own buffers; whichever worker
 * finishes the next segment to be written writes it and any finished ones after it.  At most `window`
 * segments are in flight, so at most that many output buffers exist. */
typedef struct {
    Block *blk; size_t nblk, maxn;
    const uint8_t *in, *ref;
    int fd;
    size_t next_take, next_write, window;
    uint8_t **pool; size_t npool; uint8_t **out; char *done; int writing;
    pthread_mutex_t mu; pthread_cond_t cv;
} EJob;

static size_t encode_segment(TC *c, Block *B, const uint8_t *in, const uint8_t *ref, uint8_t *o) {
    size_t n = B->len / 2;
    const uint16_t *t = aligned16(in + B->off, B->len, c->tcopy);
    if (B->kind == K_ALONE) return enc_alone(c, t, n, o);
    const uint16_t *b = aligned16(ref + B->roff, B->len, c->bcopy);
    return enc_ref(c, t, b, n, o);
}
static void *enc_worker(void *arg) {
    EJob *J = arg; TC c; tc_init(&c, J->maxn, 1);
    size_t cap = enc_cap(J->maxn);
    pthread_mutex_lock(&J->mu);
    for (;;) {
        while (J->next_take < J->nblk && J->next_take >= J->next_write + J->window) pthread_cond_wait(&J->cv, &J->mu);
        if (J->next_take >= J->nblk) break;
        size_t k = J->next_take++; Block *B = &J->blk[k]; uint8_t *buf = NULL;
        if (B->kind != K_RAW) { buf = J->npool ? J->pool[--J->npool] : NULL; }
        pthread_mutex_unlock(&J->mu);
        if (B->kind != K_RAW) {
            if (!buf) buf = xmalloc(cap);
            B->outlen = encode_segment(&c, B, J->in, J->ref, buf);
            if (B->outlen > cap) die("internal error: block buffer overrun");
            if (B->outlen >= B->len) B->kind = K_RAW;      /* not smaller: store raw */
        }
        if (B->kind == K_RAW) B->outlen = B->len;
        B->check = seg_check(J->in + B->off, B->len);
        pthread_mutex_lock(&J->mu);
        J->out[k] = buf; J->done[k] = 1;
        if (!J->writing) {
            J->writing = 1;
            while (J->next_write < J->nblk && J->done[J->next_write]) {
                Block *W = &J->blk[J->next_write]; uint8_t *ob = J->out[J->next_write];
                pthread_mutex_unlock(&J->mu);
                write_all(J->fd, W->kind == K_RAW ? J->in + W->off : ob, W->outlen);
                pthread_mutex_lock(&J->mu);
                if (ob) J->pool[J->npool++] = ob;
                J->out[J->next_write] = NULL; J->next_write++;
                pthread_cond_broadcast(&J->cv);
            }
            J->writing = 0;
        }
    }
    pthread_mutex_unlock(&J->mu);
    tc_free(&c);
    return NULL;
}

/* decoding: workers take segments, decode into their own buffer and pwrite it into place */
typedef struct {
    Block *blk; size_t nblk, maxn, next; int v6;
    const uint8_t *ref; int fd;
    pthread_mutex_t mu, wmu;
} DJob;
/* Write a decoded segment at its place in the output.  Segments start and end anywhere, so the first and
 * last page of a segment can be shared with its neighbours.  On macOS (APFS), two threads writing disjoint
 * parts of one page of a freshly sized output file at the same time occasionally lost one of the writes:
 * the output kept zeros there, after every segment's check had passed in memory (seen on 4 of 28 shards
 * decoded under memory pressure).  So the parts of a segment outside the 64 KiB-aligned span it covers are
 * written under one lock, and only the aligned span, whose pages no other segment touches, in parallel. */
#define WALIGN ((uint64_t)1 << 16)
static void pwrite_edges(int fd, pthread_mutex_t *mu, const uint8_t *p, uint64_t n, uint64_t off) {
    uint64_t a = (off + WALIGN - 1) & ~(WALIGN - 1), e = (off + n) & ~(WALIGN - 1);
    if (a >= e) { pthread_mutex_lock(mu); pwrite_all(fd, p, n, off); pthread_mutex_unlock(mu); return; }
    if (a > off) { pthread_mutex_lock(mu); pwrite_all(fd, p, a - off, off); pthread_mutex_unlock(mu); }
    pwrite_all(fd, p + (a - off), e - a, a);
    if (off + n > e) { pthread_mutex_lock(mu); pwrite_all(fd, p + (e - off), off + n - e, e); pthread_mutex_unlock(mu); }
}
static void pwrite_seg(DJob *J, const uint8_t *p, uint64_t n, uint64_t off) { pwrite_edges(J->fd, &J->wmu, p, n, off); }

/* Writes n pieces (src[i], len[i] bytes, at off[i] in the file) on up to `threads` threads, with the edges of
 * each piece under one lock as in pwrite_seg.  Used by --gpu, whose output is ready all at once. */
typedef struct { int fd; const uint8_t *const *src; const uint64_t *len, *off; size_t n, next; pthread_mutex_t mu, wmu; } WJob;
static void *write_worker(void *arg) {
    WJob *J = arg;
    for (;;) {
        pthread_mutex_lock(&J->mu); size_t k = J->next++; pthread_mutex_unlock(&J->mu);
        if (k >= J->n) break;
        if (J->len[k]) pwrite_edges(J->fd, &J->wmu, J->src[k], J->len[k], J->off[k]);
    }
    return NULL;
}
static int clamp_threads(int threads, size_t jobs);
static void write_par(int fd, const uint8_t *const *src, const uint64_t *len, const uint64_t *off, size_t n, int threads) {
    WJob J = {fd, src, len, off, n, 0, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER};
    int th = clamp_threads(threads, n);
    pthread_t tid[256];
    for (int i = 0; i < th; i++) pthread_create(&tid[i], NULL, write_worker, &J);
    for (int i = 0; i < th; i++) pthread_join(tid[i], NULL);
}
/* one buffer of n bytes written at off in pieces of 64 MiB */
static void write_buf_par(int fd, const uint8_t *p, uint64_t n, uint64_t off, int threads) {
    const uint64_t P = (uint64_t)64 << 20; size_t np = (size_t)((n + P - 1) / P);
    const uint8_t **src = xcalloc(np ? np : 1, sizeof *src); uint64_t *ln = xcalloc(np ? np : 1, 8), *of = xcalloc(np ? np : 1, 8);
    for (size_t i = 0; i < np; i++) { src[i] = p + i * P; of[i] = off + i * P; ln[i] = n - i * P < P ? n - i * P : P; }
    write_par(fd, src, ln, of, np, threads);
    free(src); free(ln); free(of);
}
static void *dec_worker(void *arg) {
    DJob *J = arg; TC c; tc_init(&c, J->maxn, 0);
    for (;;) {
        pthread_mutex_lock(&J->mu); size_t k = J->next++; pthread_mutex_unlock(&J->mu);
        if (k >= J->nblk) break;
        Block *B = &J->blk[k];
        if (B->kind == K_RAW) {
            if (!J->v6 && seg_check(B->in, B->len) != B->check) die("checksum mismatch: corrupt archive");
            pwrite_seg(J, B->in, B->len, B->off); continue;
        }
        size_t n = B->len / 2;
        const uint16_t *b = NULL;
        if (B->kind == K_REF || B->kind == 8 || B->kind == 10) b = aligned16(J->ref + B->roff, B->len, c.bcopy);
        if (J->v6) dec6_block(B->in, B->outlen, n, B->kind, b, c.dbuf);
        else if (B->kind == K_ALONE) { if (BFR7_LAYOUT()) dec_alone(&c, B->in, B->outlen, n, c.dbuf); else dec_alone8(&c, B->in, B->outlen, n, c.dbuf); }
        else { if (BFR7_LAYOUT()) dec_ref(&c, B->in, B->outlen, n, b, c.dbuf); else dec_ref8(&c, B->in, B->outlen, n, b, c.dbuf); }
        if (!J->v6 && seg_check((const uint8_t *)c.dbuf, B->len) != B->check)
            die(B->kind == K_REF ? "checksum mismatch: corrupt archive or wrong reference" : "checksum mismatch: corrupt archive");
        pwrite_seg(J, (const uint8_t *)c.dbuf, B->len, B->off);
    }
    tc_free(&c);
    return NULL;
}

/* ck[k] = seg_check of segment k's bytes at base + off, on `threads` threads (the --gpu paths, where the checks
 * are the only per-byte work left on the CPU) */
typedef struct { const uint8_t *base; const Block *b; size_t n, next; uint64_t *ck; pthread_mutex_t mu; } CJob;
static void *check_worker(void *arg) {
    CJob *J = arg;
    for (;;) {
        pthread_mutex_lock(&J->mu); size_t k = J->next++; pthread_mutex_unlock(&J->mu);
        if (k >= J->n) break;
        J->ck[k] = seg_check(J->base + J->b[k].off, J->b[k].len);
    }
    return NULL;
}
static void check_all(const uint8_t *base, const Block *b, size_t n, int threads, uint64_t *ck) {
    CJob J = {base, b, n, 0, ck, PTHREAD_MUTEX_INITIALIZER};
    if (threads < 1) threads = 1;
    if ((size_t)threads > n) threads = (int)(n ? n : 1);
    pthread_t tid[256];
    for (int i = 0; i < threads; i++) pthread_create(&tid[i], NULL, check_worker, &J);
    for (int i = 0; i < threads; i++) pthread_join(tid[i], NULL);
}

static int clamp_threads(int threads, size_t jobs) {
    if (threads < 1) threads = 1;
    if (threads > 256) threads = 256;
    if ((size_t)threads > jobs) threads = (int)(jobs ? jobs : 1);
    return threads;
}

/* ---------- safetensors header: 8-byte little-endian length N, then N bytes of JSON ---------- */
typedef struct { const char *name; size_t nlen; int bf16; uint64_t b, e; } Tensor;
typedef struct { const char *s, *e; } Cur;
static void ws(Cur *c) { while (c->s < c->e && (*c->s == ' ' || *c->s == '\t' || *c->s == '\n' || *c->s == '\r')) c->s++; }
static int jstr(Cur *c, const char **v, size_t *n) {     /* raw contents of a JSON string */
    ws(c); if (c->s >= c->e || *c->s != '"') return 0;
    const char *a = ++c->s;
    while (c->s < c->e && *c->s != '"') c->s += *c->s == '\\' ? 2 : 1;
    if (c->s >= c->e) return 0;
    *v = a; *n = (size_t)(c->s - a); c->s++; return 1;
}
static int jchar(Cur *c, char ch) { ws(c); if (c->s < c->e && *c->s == ch) { c->s++; return 1; } return 0; }
static int juint(Cur *c, uint64_t *v) {
    ws(c); if (c->s >= c->e || *c->s < '0' || *c->s > '9') return 0;
    uint64_t x = 0;
    while (c->s < c->e && *c->s >= '0' && *c->s <= '9') { if (x > (UINT64_MAX - 9) / 10) return 0; x = x * 10 + (uint64_t)(*c->s++ - '0'); }
    *v = x; return 1;
}
static int jskip(Cur *c, int depth) {                    /* skip any JSON value */
    const char *v; size_t n;
    ws(c); if (c->s >= c->e || depth > 64) return 0;
    if (*c->s == '"') return jstr(c, &v, &n);
    if (*c->s == '{' || *c->s == '[') {
        char close = *c->s == '{' ? '}' : ']'; c->s++;
        if (jchar(c, close)) return 1;
        for (;;) {
            if (close == '}' && !(jstr(c, &v, &n) && jchar(c, ':'))) return 0;
            if (!jskip(c, depth + 1)) return 0;
            if (jchar(c, close)) return 1;
            if (!jchar(c, ',')) return 0;
        }
    }
    while (c->s < c->e && *c->s != ',' && *c->s != '}' && *c->s != ']' && *c->s != ' ' && *c->s != '\n') c->s++;
    return 1;
}
/* Returns the tensors sorted by position, or 0 if the file is not a well-formed safetensors file. */
static int parse_safetensors(const uint8_t *p, size_t len, Tensor **out, size_t *nt, uint64_t *data_start) {
    *out = NULL; *nt = 0;
    if (len < 10) return 0;
    uint64_t N; memcpy(&N, p, 8);
    if (N < 2 || N > len - 8 || p[8] != '{') return 0;
    Cur c = {(const char *)p + 8, (const char *)p + 8 + N};
    size_t cap = 64, n = 0; Tensor *T = xmalloc(cap * sizeof(Tensor));
    uint64_t dlen = len - 8 - N;
    if (!jchar(&c, '{')) goto bad;
    if (!jchar(&c, '}')) for (;;) {
        const char *key; size_t kl;
        if (!jstr(&c, &key, &kl) || !jchar(&c, ':')) goto bad;
        if (kl == 12 && !memcmp(key, "__metadata__", 12)) { if (!jskip(&c, 0)) goto bad; }
        else {
            Tensor t = {key, kl, 0, 0, 0}; int have = 0;
            if (!jchar(&c, '{')) goto bad;
            if (!jchar(&c, '}')) for (;;) {
                const char *f, *v; size_t fl, vl;
                if (!jstr(&c, &f, &fl) || !jchar(&c, ':')) goto bad;
                if (fl == 5 && !memcmp(f, "dtype", 5)) { if (!jstr(&c, &v, &vl)) goto bad; t.bf16 = vl == 4 && !memcmp(v, "BF16", 4); }
                else if (fl == 12 && !memcmp(f, "data_offsets", 12)) {
                    if (!(jchar(&c, '[') && juint(&c, &t.b) && jchar(&c, ',') && juint(&c, &t.e) && jchar(&c, ']'))) goto bad;
                    have = 1;
                } else if (!jskip(&c, 1)) goto bad;
                if (jchar(&c, '}')) break;
                if (!jchar(&c, ',')) goto bad;
            }
            if (!have || t.b > t.e || t.e > dlen) goto bad;
            t.b += 8 + N; t.e += 8 + N;
            if (n == cap) { T = realloc(T, (cap *= 2) * sizeof(Tensor)); if (!T) die("out of memory"); }
            T[n++] = t;
        }
        if (jchar(&c, '}')) break;
        if (!jchar(&c, ',')) goto bad;
    }
    for (size_t i = 1; i < n; i++)                        /* insertion sort by position */
        for (size_t j = i; j > 0 && T[j].b < T[j - 1].b; j--) { Tensor x = T[j]; T[j] = T[j - 1]; T[j - 1] = x; }
    for (size_t i = 1; i < n; i++) if (T[i].b < T[i - 1].e) goto bad;   /* overlapping tensors */
    *out = T; *nt = n; *data_start = 8 + N; return 1;
bad:
    free(T); return 0;
}

/* ---------- segment list ---------- */
typedef struct { Block *b; size_t n, cap; } Segs;
static void seg_add(Segs *S, int kind, uint64_t off, uint64_t len, uint64_t roff) {
    if (!len) return;
    if (kind == K_RAW && S->n && S->b[S->n - 1].kind == K_RAW && S->b[S->n - 1].off + S->b[S->n - 1].len == off) {
        S->b[S->n - 1].len += len; return;            /* merge adjacent raw ranges */
    }
    if (S->n == S->cap) { S->b = realloc(S->b, (S->cap = S->cap ? 2 * S->cap : 64) * sizeof(Block)); if (!S->b) die("out of memory"); }
    Block B; memset(&B, 0, sizeof B); B.kind = kind; B.off = off; B.len = len; B.roff = roff;
    S->b[S->n++] = B;
}
static void seg_add_bf16(Segs *S, int kind, uint64_t off, uint64_t len, uint64_t roff, uint64_t bsz) {
    uint64_t even = len & ~(uint64_t)1;
    for (uint64_t a = 0; a < even; a += bsz) {
        uint64_t l = even - a < bsz ? even - a : bsz;
        seg_add(S, kind, off + a, l, roff + a);
    }
    seg_add(S, K_RAW, off + even, len - even, 0);
}

static const uint8_t *map_file(const char *path, size_t *len) {
    int fd = open(path, O_RDONLY); if (fd < 0) die("cannot open input");
    struct stat st; if (fstat(fd, &st)) die("cannot stat input");
    *len = (size_t)st.st_size;
    if (!*len) { close(fd); return (const uint8_t *)""; }
    void *m = mmap(NULL, *len, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    if (m == MAP_FAILED) die("mmap failed");
    return m;
}

/* BFR7 header (48 bytes): magic, PB, PBR, safetensors flag, reference flag, total length, reference
 * length, reference fingerprint, number of segments, block size in MiB, 8 spare.
 * BFR6 header (40 bytes): magic, PB, safetensors flag, reference flag, spare, total, reference length,
 * reference hash, number of segments, 4 spare.
 * Segment record: kind, 7 spare, offset, length, reference offset, encoded length (40 bytes, BFR6),
 * then the check of the original bytes (48 bytes, BFR7). */
#define HDR7 48
#define HDR6 40
#define SEGREC6 40
#define SEGREC7 48
#define MAXSEG ((uint64_t)1 << 30)     /* largest coded segment the decoder accepts */

#ifdef CMX_UNIFIED                 /* built into the cmx program (cmx.c), which dispatches to it */
#define main bf16ref_main
#endif
int main(int argc, char **argv) {
    if (argc < 4 || (argv[1][0] != 'c' && argv[1][0] != 'd') || argv[1][1]) {
        fprintf(stderr, "usage: bf16ref c IN OUT [--ref REF] [-t threads] [-b block_MiB] [-s substream_KiB] [-l lanes] [--gpu]\n"
                        "       bf16ref d IN OUT [--ref REF] [-t threads] [--gpu]\n"
                        "  -s, -l  BFR8 layout: each block's words cut into sub-streams of substream_KiB of weights\n"
                        "          with 16 or 32 interleaved states each (default -s 256 -l 32); -s 0 -l 16 writes BFR7\n"
                        "  --gpu   code on the GPU (BFR8 with 32 lanes; a build with make CUDA=1)\n");
        return 2;
    }
    /* default layout: BFR8, 256 KiB sub-streams of 32 lanes (GPU-decodable, +0.04 pp; as fast to decode on the CPU as
     * BFR7 at 16 threads); -s 0 -l 16 writes BFR7 */
    int threads = 1, gpu = 0, layout_given = 0; size_t bmib = 64, skib = 256; const char *refpath = NULL;
    g_lanes = 32;
    for (int i = 4; i < argc; i += 2) {
        if (!strcmp(argv[i], "--gpu")) { gpu = 1; i--; continue; }
        if (i + 1 >= argc) die("option needs a value");
        if (!strcmp(argv[i], "--ref")) refpath = argv[i + 1];
        else if (!strcmp(argv[i], "-t")) threads = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-b")) bmib = (size_t)atol(argv[i + 1]);
        else if (!strcmp(argv[i], "-s")) { skib = (size_t)atol(argv[i + 1]); layout_given = 1; }
        else if (!strcmp(argv[i], "-l")) { g_lanes = atoi(argv[i + 1]); layout_given = 1; }
        else die("unknown option");
    }
    if (bmib < 1 || bmib > 1024) die("-b must be 1..1024");
    if (g_lanes != 16 && g_lanes != 32) die("-l must be 16 or 32");
    if (skib > bmib * 1024) die("-s must be at most the block size");
    g_sw = skib * 512;                                     /* KiB of bfloat16 -> weights */
    if (!BFR7_LAYOUT() && bmib > 256) die("BFR8 blocks are at most 256 MiB (-s 0 -l 16 writes BFR7)");
    if (threads < 1) threads = 1;
    if (threads > 256) threads = 256;
#ifndef BF16REF_GPU
    if (gpu) die("--gpu: this build has no GPU support (build with make CUDA=1)");
#endif
    if (gpu && argv[1][0] == 'c' && g_lanes != 32) die("--gpu writes BFR8 with 32 lanes (-l 32)");
    double t0 = now();
    size_t ilen, rlen = 0;
    const uint8_t *in = map_file(argv[2], &ilen), *ref = refpath ? map_file(refpath, &rlen) : NULL;
    int fd = open(argv[3], O_WRONLY | O_CREAT | O_TRUNC, 0644); if (fd < 0) die("cannot open output");
    g_outpath = argv[3];
    if (argv[1][0] == 'c') {
        Segs S = {0}; uint64_t bsz = (uint64_t)bmib << 20;
        Tensor *T, *R = NULL; size_t nt, nr = 0; uint64_t ds, rds;
        int st = parse_safetensors(in, ilen, &T, &nt, &ds), paired = 0, alone = 0;
        if (st) {
            int rst = refpath && parse_safetensors(ref, rlen, &R, &nr, &rds);
            if (refpath && !rst) fprintf(stderr, "bf16ref: the reference is not a safetensors file; tensors are coded alone\n");
            uint64_t pos = ds;
            seg_add(&S, K_RAW, 0, ds, 0);                 /* header */
            for (size_t i = 0; i < nt; i++) {
                seg_add(&S, K_RAW, pos, T[i].b - pos, 0);
                uint64_t len = T[i].e - T[i].b;
                if (T[i].bf16) {
                    size_t r = nr;
                    for (size_t k = 0; k < nr; k++)
                        if (R[k].nlen == T[i].nlen && !memcmp(R[k].name, T[i].name, T[i].nlen) && R[k].bf16 && R[k].e - R[k].b == len) { r = k; break; }
                    if (r < nr) { seg_add_bf16(&S, K_REF, T[i].b, len, R[r].b, bsz); paired++; }
                    else { seg_add_bf16(&S, K_ALONE, T[i].b, len, 0, bsz); alone++; }
                } else seg_add(&S, K_RAW, T[i].b, len, 0);
                pos = T[i].e;
            }
            seg_add(&S, K_RAW, pos, ilen - pos, 0);
        } else {
            if (refpath && rlen != ilen) die("reference length differs from input length (and the input is not a safetensors file)");
            seg_add_bf16(&S, refpath ? K_REF : K_ALONE, 0, ilen, 0, bsz);
        }
        uint64_t fp = refpath ? ref_fingerprint(ref, rlen) : 0;
        size_t maxn = 1;
        for (size_t k = 0; k < S.n; k++) if (S.b[k].kind != K_RAW && S.b[k].len / 2 > maxn) maxn = S.b[k].len / 2;
        size_t hsz = HDR7 + S.n * SEGREC7;
        uint8_t *hdr = xcalloc(hsz, 1);
        write_all(fd, hdr, hsz);                           /* placeholder; the real header is written last */
        double t1 = now();
        EJob J; memset(&J, 0, sizeof J);
        if (gpu) {
#ifdef BF16REF_GPU
            /* every coded segment on the GPU, then the checks and the writes here, in order */
            int *kd = xcalloc(S.n, sizeof(int)); uint64_t *of = xcalloc(S.n, 8), *ln = xcalloc(S.n, 8), *ro = xcalloc(S.n, 8), *ol = xcalloc(S.n, 8);
            uint8_t **bd = xcalloc(S.n, sizeof(uint8_t *)); void *keep = NULL; char msg[256];
            for (size_t k = 0; k < S.n; k++) { kd[k] = S.b[k].kind; of[k] = S.b[k].off; ln[k] = S.b[k].len; ro[k] = S.b[k].roff; }
            bfr_gpu_threads(threads);
            if (bfr_gpu_encode(in, ilen, ref, rlen, S.n, kd, of, ln, ro, g_sw, bd, ol, &keep, msg, sizeof msg)) {
                fprintf(stderr, "bf16ref: GPU: %s\n", msg); die("GPU encoding failed");
            }
            uint64_t *ck = xcalloc(S.n, 8), *wl = xcalloc(S.n, 8), *wo = xcalloc(S.n, 8), pos = hsz;
            const uint8_t **ws = xcalloc(S.n, sizeof *ws);
            check_all(in, S.b, S.n, threads, ck);
            for (size_t k = 0; k < S.n; k++) {
                Block *B = &S.b[k];
                if (B->kind != K_RAW && ol[k] >= B->len) B->kind = K_RAW;      /* not smaller: store raw */
                B->outlen = B->kind == K_RAW ? B->len : ol[k];
                B->check = ck[k];
                ws[k] = B->kind == K_RAW ? in + B->off : bd[k]; wl[k] = B->outlen; wo[k] = pos; pos += B->outlen;
            }
            write_par(fd, ws, wl, wo, S.n, threads);
            free(ck); free(wl); free(wo); free(ws);
            bfr_gpu_free(keep); free(kd); free(of); free(ln); free(ro); free(ol); free(bd);
#endif
        } else {
        int th = clamp_threads(threads, S.n);
        J.blk = S.b; J.nblk = S.n; J.maxn = maxn; J.in = in; J.ref = ref; J.fd = fd;
        J.window = 2 * (size_t)th; J.pool = xcalloc(J.window + 1, sizeof(uint8_t *));
        J.out = xcalloc(S.n, sizeof(uint8_t *)); J.done = xcalloc(S.n, 1);
        pthread_mutex_init(&J.mu, NULL); pthread_cond_init(&J.cv, NULL);
        pthread_t tid[256];
        for (int i = 0; i < th; i++) pthread_create(&tid[i], NULL, enc_worker, &J);
        for (int i = 0; i < th; i++) pthread_join(tid[i], NULL);
        for (size_t i = 0; i < J.npool; i++) free(J.pool[i]);
        }
        double t2 = now();
        memcpy(hdr, BFR7_LAYOUT() ? MAGIC7 : MAGIC8, 4); hdr[4] = PB; hdr[5] = PBR; hdr[6] = (uint8_t)st; hdr[7] = refpath != NULL;
        if (!BFR7_LAYOUT()) { uint32_t sw32 = (uint32_t)g_sw; memcpy(hdr + 40, &sw32, 4); hdr[44] = (uint8_t)g_lanes; }
        uint64_t v[3] = {ilen, rlen, fp}; uint32_t u[2] = {(uint32_t)S.n, (uint32_t)bmib};
        memcpy(hdr + 8, v, 24); memcpy(hdr + 32, u, 8);
        uint64_t by[3][2] = {{0}}, total = hsz;            /* bytes in and out by kind: raw, alone, given the reference */
        for (size_t k = 0; k < S.n; k++) {
            Block *B = &S.b[k]; uint8_t *rec = hdr + HDR7 + k * SEGREC7;
            rec[0] = (uint8_t)B->kind;
            uint64_t w[5] = {B->off, B->len, B->roff, B->outlen, B->check}; memcpy(rec + 8, w, 40);
            int g = B->kind == K_RAW ? 0 : B->kind == K_ALONE ? 1 : 2; by[g][0] += B->len; by[g][1] += B->outlen;
            total += B->outlen;
        }
        pwrite_all(fd, hdr, hsz, 0);
        if (close(fd)) die("write failed");
        double t3 = now();
        fprintf(stderr, "bf16ref: %zu -> %llu bytes (%.4f%%), %s, %s, %zu segments, %.2f s (setup %.2f, code+write %.2f, header %.2f)\n",
                ilen, (unsigned long long)total, ilen ? 100.0 * total / ilen : 0.0, BFR7_LAYOUT() ? "BFR7" : "BFR8",
                st ? "safetensors" : "raw bfloat16", S.n,
                t3 - t0, t1 - t0, t2 - t1, t3 - t2);
        if (st) fprintf(stderr, "bf16ref:   %zu tensors: %d BF16 given the reference, %d BF16 alone, %zu other\n",
                        nt, paired, alone, nt - (size_t)paired - (size_t)alone);
        fprintf(stderr, "bf16ref:   raw %llu -> %llu, alone %llu -> %llu, given the reference %llu -> %llu bytes\n",
                (unsigned long long)by[0][0], (unsigned long long)by[0][1], (unsigned long long)by[1][0],
                (unsigned long long)by[1][1], (unsigned long long)by[2][0], (unsigned long long)by[2][1]);
        free(T); free(R); free(S.b); free(hdr); free(J.pool); free(J.out); free(J.done);
    } else {
        int v6;
        int v8 = ilen >= HDR7 && !memcmp(in, MAGIC8, 4);
        if (v8) {                                          /* BFR8: BFR7 with the sub-stream layout in the header */
            uint32_t sw32; memcpy(&sw32, in + 40, 4); g_sw = sw32; g_lanes = in[44];
            if ((g_lanes != 16 && g_lanes != 32) || (g_sw && g_sw % 512) || BFR7_LAYOUT() || in[45] || in[46] || in[47])
                die("corrupt archive (BFR8 layout)");
            v6 = 0;
        }
        else if (ilen >= HDR7 && !memcmp(in, MAGIC7, 4)) { v6 = 0; g_sw = 0; g_lanes = 16; }
        else if (ilen >= HDR6 && !memcmp(in, MAGIC6, 4)) v6 = 1;
        else die("not a bf16ref archive");
        size_t hdr = v6 ? HDR6 : HDR7, SEGREC = v6 ? SEGREC6 : SEGREC7;
        if (v6 ? in[4] != PB : (in[4] != PB || in[5] != PBR)) die("unsupported precision");
        int hasref = v6 ? in[6] : in[7];
        uint64_t v[3]; uint32_t ns; memcpy(v, in + 8, 24); memcpy(&ns, in + 32, 4);
        uint64_t olen = v[0], arl = v[1], rh = v[2];
        if ((uint64_t)ns > (ilen - hdr) / SEGREC) die("corrupt archive (segment count)");
        Block *blk = xcalloc(ns, sizeof(Block));
        const uint8_t *body = in + hdr + (size_t)ns * SEGREC, *end = in + ilen;
        uint64_t pos = 0; int needref = 0; size_t maxn = 1;
        for (uint32_t k = 0; k < ns; k++) {
            const uint8_t *rec = in + hdr + (size_t)k * SEGREC; uint64_t w[5] = {0}; memcpy(w, rec + 8, SEGREC - 8);
            Block *B = &blk[k];
            B->kind = rec[0]; B->off = w[0]; B->len = w[1]; B->roff = w[2]; B->outlen = w[3]; B->check = w[4];
            int isref = v6 ? (B->kind == 8 || B->kind == 10) : B->kind == K_REF;
            int ok = B->kind == K_RAW || (v6 ? (B->kind == 0 || isref) : (B->kind == K_ALONE || isref));
            if (!ok) die("corrupt archive (segment kind)");
            if (B->off != pos || B->len > olen - pos || B->outlen > (uint64_t)(end - body)) die("corrupt archive (segment bounds)");
            if (B->kind != K_RAW && ((B->len & 1) || B->len > MAXSEG)) die("corrupt archive (segment length)");
            if (B->kind == K_RAW && B->outlen != B->len) die("corrupt archive (raw segment)");
            if (isref) { needref = 1; if (B->roff > arl || B->len > arl - B->roff) die("corrupt archive (reference range)"); }
            if (B->kind != K_RAW && B->len / 2 > maxn) maxn = B->len / 2;
            B->in = body; body += B->outlen; pos += B->len;
        }
        if (pos != olen || body != end) die("corrupt archive (length)");
        if (hasref && !refpath) die("archive was made with --ref; give the same reference");
        if (!hasref && refpath) die("archive was made without a reference");
        if (needref && !hasref) die("corrupt archive (reference flag)");
        if (refpath) {
            if (rlen != arl) die("wrong reference file (length differs)");
            if ((v6 ? file_hash6(ref, rlen, threads) : ref_fingerprint(ref, rlen)) != rh) die("wrong reference file");
        }
        if (gpu) {
#ifdef BF16REF_GPU
            /* the coded segments on the GPU, the raw ones and every check here, then one write */
            if (v6 || g_lanes != 32) die("--gpu decodes BFR8 archives with 32 lanes; decode this one without --gpu");
            double t1 = now();
            uint8_t *out = xmalloc(olen + 16);
            int *kd = xcalloc(ns, sizeof(int)); uint64_t *of = xcalloc(ns, 8), *ln = xcalloc(ns, 8), *ro = xcalloc(ns, 8), *ai = xcalloc(ns, 8), *al = xcalloc(ns, 8);
            char msg[256];
            for (uint32_t k = 0; k < ns; k++) { kd[k] = blk[k].kind; of[k] = blk[k].off; ln[k] = blk[k].len; ro[k] = blk[k].roff;
                                                ai[k] = (uint64_t)(blk[k].in - in); al[k] = blk[k].outlen; }
            bfr_gpu_threads(threads);
            if (bfr_gpu_decode(in, ilen, ref, rlen, ns, kd, of, ln, ro, ai, al, g_sw, out, olen, msg, sizeof msg)) {
                fprintf(stderr, "bf16ref: GPU: %s\n", msg); die("GPU decoding failed");
            }
            for (uint32_t k = 0; k < ns; k++) if (blk[k].kind == K_RAW) memcpy(out + blk[k].off, blk[k].in, blk[k].len);
            uint64_t *ck = xcalloc(ns, 8);
            check_all(out, blk, ns, threads, ck);
            for (uint32_t k = 0; k < ns; k++)
                if (ck[k] != blk[k].check)
                    die(blk[k].kind == K_REF ? "checksum mismatch: corrupt archive or wrong reference" : "checksum mismatch: corrupt archive");
            free(ck);
            if (ftruncate(fd, (off_t)olen)) die("cannot size output");
            write_buf_par(fd, out, olen, 0, threads);
            if (close(fd)) die("write failed");
            double t2 = now();
            fprintf(stderr, "bf16ref: decoded %llu bytes from BFR8 on the GPU, %.2f s (setup %.2f, decode+check+write %.2f)\n",
                    (unsigned long long)olen, t2 - t0, t1 - t0, t2 - t1);
            free(out); free(kd); free(of); free(ln); free(ro); free(ai); free(al); free(blk);
            return 0;
#endif
        }
        if (ftruncate(fd, (off_t)olen)) die("cannot size output");
        double t1 = now();
        int th = clamp_threads(threads, ns);
        DJob J; memset(&J, 0, sizeof J);
        J.blk = blk; J.nblk = ns; J.maxn = maxn; J.v6 = v6; J.ref = ref; J.fd = fd;
        pthread_mutex_init(&J.mu, NULL); pthread_mutex_init(&J.wmu, NULL);
        pthread_t tid[256];
        for (int i = 0; i < th; i++) pthread_create(&tid[i], NULL, dec_worker, &J);
        for (int i = 0; i < th; i++) pthread_join(tid[i], NULL);
        if (close(fd)) die("write failed");
        double t2 = now();
        fprintf(stderr, "bf16ref: decoded %llu bytes from %s, %.2f s (setup %.2f, decode+write %.2f)\n",
                (unsigned long long)olen, v6 ? "BFR6" : v8 ? "BFR8" : "BFR7", t2 - t0, t1 - t0, t2 - t1);
        free(blk);
    }
    return 0;
}
