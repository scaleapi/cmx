/* BFR8 sub-stream decoders of bf16ref.c, included once per lane count with
 *   K_L   lanes (interleaved rANS states of one sub-stream): 16 or 32
 *   K_FS  macro applying its argument to 0 .. K_L - 1
 *   K_DQ  the given-the-reference decode of K_L weights, in groups of 4 (DQ4 below)
 *   K_SUF name suffix
 * Each function decodes one sub-stream of m weights whose words are [w0, pend), exactly as dec_alone /
 * dec_ref decode a BFR7 block (which is one sub-stream of 16 lanes), and dies unless the sub-stream uses all of
 * its words and its final states are the encoder's initial ones. */
#define KCAT_(a, b) a##b
#define KCAT(a, b) KCAT_(a, b)

/* standalone: rp = the sub-stream's raw fields (5 bytes per 8 weights; m is a multiple of 8 except in the
 * block's last sub-stream) */
static void KCAT(dsub_alone, K_SUF)(TC *c, const uint8_t *w0, const uint8_t *pend, const uint8_t *rp, size_t m,
                                    const uint32_t *restrict es, const uint16_t *restrict vs, const uint16_t *RAWMAP,
                                    uint16_t *restrict t) {
    const size_t wlen = (size_t)(pend - w0);          /* pend moves to the tail copy near the end */
    if (wlen < 4 * K_L) die("corrupt block (sub-stream)");
#define DECA(X) ({ uint32_t sl_ = (X) & PM; uint32_t e_ = es[sl_]; uint32_t v_ = vs[sl_];             \
        uint32_t y_ = (e_ >> 16) * ((X) >> PB) + (e_ & 0xFFFF);                                       \
        RENORM(X, y_); v_; })
    const uint8_t *p = w0;
    K_FS(XLOAD)
    p += 4 * K_L;
    int ontail = 0; size_t cons0 = 0, i = 0;
#define OQ(q) { const uint64_t v = vv[(q) >> 3]; uint32_t v_ = DECA(x##q); t[i + (q)] = (uint16_t)(v_ | RAWMAP[(v >> (5 * ((q) & 7))) & 31]); }
    while (i < m) {
        size_t cend = m - i > CH ? i + CH : m;
        STREAM_CHECK(p, pend, c->wtail, ontail, cons0, w0, 2 * CH + 2, 0);
        for (; i + K_L <= cend; i += K_L) {
            uint64_t vv[K_L / 8];
            for (int g = 0; g < K_L / 8; g++) memcpy(&vv[g], rp + 5 * g, 8);
            rp += 5 * (K_L / 8);
            K_FS(OQ)
        }
        if (i < cend) {                       /* fewer than K_L weights left in the sub-stream */
            uint32_t *xs[K_L] = {K_FS(XPTR)};
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
    if (0 K_FS(XBAD) || used != wlen) die("corrupt block (final state)");
}

/* given the reference: the sub-stream's raw bits start at bit rb0 of the block's raw bits [raw0, rawend), and
 * must end at bit rb1 (for the block's last sub-stream, within the last byte: rb1 = 8 * rawlen) */
static void KCAT(dsub_ref, K_SUF)(TC *c, const uint8_t *w0, const uint8_t *pend, const uint8_t *raw0,
                                  const uint8_t *rawend, uint64_t rb0, uint64_t rb1, int last, size_t m,
                                  const uint32_t *cb, const uint8_t *restrict slot, const uint64_t *restrict se,
                                  const uint16_t *restrict b, uint16_t *restrict t) {
    const size_t wlen = (size_t)(pend - w0);          /* pend moves to the tail copy near the end */
    if (wlen < 4 * K_L) die("corrupt block (sub-stream)");
    const uint8_t *p = w0;
    K_FS(XLOAD)
    p += 4 * K_L;
    /* raw reader positioned at bit rb0: whole bytes skipped, the rest dropped after the first refill */
    const uint8_t *rbase = raw0 + rb0 / 8, *raw = rbase, *rawe = rawend;
    uint64_t buf = 0; uint32_t cnt = 0, skip = (uint32_t)(rb0 & 7);
    int wtl = 0, rtl = 0; size_t wcons0 = 0, rcons0 = 0;
#define REFILL() do { uint64_t w_; memcpy(&w_, raw, 8); buf |= w_ << cnt; raw += (63 - cnt) >> 3; cnt |= 56; } while (0)
#define DECR(X, i) do { uint32_t bb_ = b[i]; uint32_t a_ = cb[(bb_ >> 7) & 0xFF];                       \
        uint32_t sl_ = (X) & PMR; uint64_t e_ = se[(size_t)a_ * 256 + slot[(size_t)a_ * PTR + sl_]];   \
        uint32_t y_ = (uint32_t)((e_ >> 16) & 0xFFFF) * ((X) >> PBR) + sl_ - (uint32_t)(e_ & 0xFFFF); \
        RENORM(X, y_);                                                                                 \
        uint32_t k_ = (uint32_t)(e_ >> 48) & 15, rv_ = (uint32_t)(buf & (e_ >> 52)); buf >>= k_; cnt -= k_; \
        t[i] = (uint16_t)(bb_ + (uint32_t)(e_ >> 32) + rv_); } while (0)
#define DQ4(a, b_, c_, d) REFILL(); DECR(x##a, i + a); DECR(x##b_, i + b_); DECR(x##c_, i + c_); DECR(x##d, i + d);
    size_t i = 0; int first = 1;
    while (i < m) {
        size_t cend = m - i > CH ? i + CH : m;
        STREAM_CHECK(p, pend, c->wtail, wtl, wcons0, w0, 2 * CH + 2, 0);
        STREAM_CHECK(raw, rawe, c->rtail, rtl, rcons0, rbase, RAWB, 8);
        if (first) { REFILL(); buf >>= skip; cnt -= skip; first = 0; }
        for (; i + K_L <= cend; i += K_L) { K_DQ }
        if (i < cend) {
            uint32_t *xs[K_L] = {K_FS(XPTR)};
            for (size_t j = 0; i + j < cend; j++) { if (j % 4 == 0) REFILL(); uint32_t X = *xs[j]; DECR(X, i + j); *xs[j] = X; }
            i = cend;
        }
    }
#undef DQ4
#undef DECR
#undef REFILL
    size_t used = wtl ? wcons0 + (size_t)(p - c->wtail) : (size_t)(p - w0);
    uint64_t rbytes = rtl ? rcons0 + (size_t)(raw - c->rtail) : (size_t)(raw - rbase);
    uint64_t bit = 8 * (rb0 / 8) + 8 * rbytes - cnt;
    if (m == 0) bit = rb0;
    if (0 K_FS(XBAD) || used != wlen || (last ? (bit + 7) / 8 != rb1 / 8 : bit != rb1))
        die("corrupt block (final state)");
}
#undef KCAT
#undef KCAT_
