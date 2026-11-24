/*
 * cmxc: a context-mixing lossless compressor.
 * Copyright (c) 2026 Scale AI. Released under the MIT License (see LICENSE).
 *
 *   cmxc c <in> <out>    compress
 *   cmxc d <in> <out>    decompress
 *
 * The model is an online mixture of experts: hashed byte contexts (orders 0-11), match models,
 * word, sparse, indirect and record contexts, a multi-scale ladder of coarsened summaries and
 * approximate retrieval, combined by nine first-level mixers, a second-level mixer and four
 * adaptive probability maps, driving a binary arithmetic coder. Table-backed experts store a
 * nibble's predictions in one cache line. Every model-shaping setting is written to the archive
 * header, so the decoder needs no options. See README.md for settings and build notes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <time.h>
#include <pthread.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAVE_NEON 1
#endif

/* ================================================================= coder ==
 * Carryless binary arithmetic coder.  The design is the well-known carryless
 * coder of the PAQ/lpaq family; this is an independent implementation with a
 * 16-bit probability.  p is P(bit==1) scaled to 1..65535.
 *
 *   xmid = x1 + floor((x2-x1) * p / 2^16)      computed without overflow
 *   bit==1 -> [x1, xmid]        bit==0 -> [xmid+1, x2]
 *
 * Proof obligations, both hold for p in [1,65535] and x1 < x2:
 *   xmid >= x1  (the two products are non-negative)
 *   xmid <  x2  (with r = a*2^16+b, the sum equals r - a - [b>0] < r)
 * After the renormalisation loop x1 and x2 differ in their top byte, hence
 * x1 < x2, so the invariant is re-established every step.
 *
 * flush() writes all four bytes of x1, so the decoder's window reconstructs
 * exactly x1 (which is inside the final interval) and any read past the end of
 * the block returns 0, which only appends zero bytes below x1 -- still inside
 * every nested interval.  No EOF convention to get wrong.
 * ========================================================================= */
typedef struct {
    uint32_t x1, x2, x;
    uint8_t *buf; size_t cap, n;              /* encode: growable output   */
    const uint8_t *in; size_t inlen, inpos;   /* decode: input block       */
} Coder;

static void cod_grow(Coder *C){
    size_t nc = C->cap ? C->cap*2 : 65536;
    uint8_t *nb = (uint8_t*)realloc(C->buf, nc);
    if(!nb){ fprintf(stderr,"cmxc: out of memory in coder\n"); exit(1); }
    C->buf = nb; C->cap = nc;
}
static inline void cod_put(Coder *C, uint8_t b){
    if(C->n >= C->cap) cod_grow(C);
    C->buf[C->n++] = b;
}
static inline uint32_t cod_get(Coder *C){
    return C->inpos < C->inlen ? (uint32_t)C->in[C->inpos++] : 0u;
}
static void enc_init(Coder *C){ C->x1=0; C->x2=0xFFFFFFFFu; C->x=0; }
static void dec_init(Coder *C){
    C->x1=0; C->x2=0xFFFFFFFFu; C->x=0;
    for(int i=0;i<4;i++) C->x = (C->x<<8) | cod_get(C);
}
static inline uint32_t cod_mid(uint32_t x1, uint32_t x2, uint32_t p){
    uint32_t r = x2 - x1;
    return x1 + (r>>16)*p + (((r & 0xFFFFu) * p) >> 16);
}
static inline void enc_bit(Coder *C, int bit, uint32_t p){
    uint32_t xmid = cod_mid(C->x1, C->x2, p);
    if(bit) C->x2 = xmid; else C->x1 = xmid + 1;
    while(((C->x1 ^ C->x2) & 0xFF000000u) == 0){
        cod_put(C, (uint8_t)(C->x2>>24));
        C->x1 <<= 8; C->x2 = (C->x2<<8) | 255u;
    }
}
static inline int dec_bit(Coder *C, uint32_t p){
    uint32_t xmid = cod_mid(C->x1, C->x2, p);
    int bit = (C->x <= xmid);
    if(bit) C->x2 = xmid; else C->x1 = xmid + 1;
    while(((C->x1 ^ C->x2) & 0xFF000000u) == 0){
        C->x1 <<= 8; C->x2 = (C->x2<<8) | 255u;
        C->x = (C->x<<8) | cod_get(C);
    }
    return bit;
}
static void enc_flush(Coder *C){
    for(int i=0;i<4;i++){ cod_put(C,(uint8_t)(C->x1>>24)); C->x1 <<= 8; }
}
/* the single quantiser -- called from the one shared model path */
static inline uint32_t quant_p(float pr){
    /* Clamped in float before the conversion, so a non-finite pr (impossible
       for a valid archive) still gives a p in [1,65535] instead of undefined
       behaviour.  For every finite pr the result equals truncate-then-clamp. */
    float s = pr * 65536.0f;
    if(!(s >= 1.0f)) s = 1.0f;
    if(s > 65535.0f) s = 65535.0f;
    return (uint32_t)s;
}

/* ------------------------------------------------------------ sizes ------ */
/* The number of hashed byte orders is a runtime value (`nord`).  NORDMAX only
 * bounds the per-Model table array and the stack arrays in code_block(); the
 * live count is `nord`, which is recorded in the container header so the
 * decoder never re-derives it. */
#define NORDMAX 12                      /* hashed byte orders 0..nord-1      */
#define NMATCH  3
#define NWORDM  7                       /* W0 W1 W3 WCASE BC BC2 BC3         */
#define MAXEX   32                      /* sparse / indirect models          */
#define NREC    7                       /* record + stride models            */
#define MAXIN   128                     /* upper bound on mixer inputs       */

/* ---------------- coarse-scale ladder (CMXS_MASK) --------------------------
 * The multi-scale ladder evaluated in paper §4, with the contexts,
 * constants and state machine used there.  Seven coarse rungs:
 *   0 W2    2 complete words + partial      4 SPOS position within sentence
 *   1 W4    4 complete words + partial      5 LPOS position within line
 *   2 W8    8 complete words + p[-1]        6 NEST markup/bracket nesting
 *   3 SHAPE word-length pattern
 * Rungs already present in the base model (W0/W1/word-trigram/WCASE/BC1-3)
 * are not duplicated.  scmask==0 gives the base model exactly.
 *
 * CAUSALITY (this is a CODEC, not a model): every rung reads only `p[-1]` and
 * the ladder's own scalar state, and that state is advanced from `sym` AFTER
 * the byte has been coded.  The decoder therefore has every input at the time
 * it needs it.  All of the scalar state lives in code_block() locals, so it is
 * automatically per-block and cannot leak across a T>1 block boundary.        */
#define NSC     7
/* ---------------- fuzzy / approximate retrieval (CMX_FUZZ) -----------------
 * Approximate retrieval as evaluated in paper §4.
 *   bit0 EN_TOL  tolerant match: advances THROUGH mismatches, re-seeks only
 *                when the miss density over the last 8 bytes reaches FZT
 *   bit1 EN_LSH  bit-sampling LSH over FZR of the last FZW bytes, FZL tables,
 *                candidates VERIFIED by counting agreement over the last FZK
 * CAUSALITY: both index tables are written with (i+1) at the END of byte i and
 * read BEFORE that write, so a retrieved position is always <= i.  The tolerant
 * pointer carries `tptr<i` and the LSH candidates carry `fz_cpos[q]<i` guards
 * at use time.  The similarity scan reads D[c-1-j] for j<FZK<=32 and p[-j];
 * with c>=1 the lowest address touched is D[-32], inside the 64 zero bytes of
 * pre-context every block buffer carries.  Nothing reads at or beyond the
 * byte being coded. */
#define MAXL    8
#define MAXK    32
#define SIMB    (MAXK+1)
/* The FZ sub-parameters are FIXED at the values evaluated in the paper.  They are NOT
 * env-tunable here on purpose: FZW/FZR/FZL determine the LSH bit-position
 * permutation, which both sides must generate identically, and freezing them
 * keeps that out of the container. */
#define FZK     24
#define FZR     4
#define FZL     8
#define FZW     24
#define FZT     4
#define FZTORD  12

#define LENB    32
#define RLENB   32
#define HDEPTH  12
#define HSZ     ((1<<(HDEPTH+1))-1)
#define MHIST   3
#define NMH     (1<<MHIST)
#define CNTCAP  40
#define CNTSC   0.12f
#define FASTR   14.0f
#define APM_MINRATE_V 0.010f
#define APM_N0_V 4.0f
#define IN_SCALE 0.25f
#define L2_SCALE 0.25f

/* base (2 MB-tuned) table widths, shrunk by MEMSH for short blocks.
 * SCB0 (ladder), FTB0 (tolerant-match index) and FLB0 (LSH index) participate
 * in the auto-shrink exactly like every other table, so a small input or a
 * high thread count shrinks them too.  All eight+three widths are written into
 * the header, so the decoder never recomputes any of them. */
enum { TB0_BASE=22, WB0_BASE=25, MB0_BASE=24, RB0_BASE=22, XB0_BASE=23, IB0_BASE=22,
       S2B0_BASE=22, S4B0_BASE=22, SCB0_BASE=22, FTB0_BASE=22, FLB0_BASE=20, WIDTH_MIN=14 };
static int TB0=TB0_BASE, WB0=WB0_BASE, MB0=MB0_BASE, RB0=RB0_BASE, XB0=XB0_BASE, IB0=IB0_BASE,
           S2B0=S2B0_BASE, S4B0=S4B0_BASE;
static int SCB0=SCB0_BASE, FTB0=FTB0_BASE, FLB0=FLB0_BASE;
static int MEMSH=0;
static uint32_t TSIZE,TMASK,WSIZE,WMASK,MSIZE,MMASK,RSIZE,RMASK;
static uint32_t XSIZE,XMASK,ISIZE,IMASK,S2SIZE,S2MASK,S4SIZE,S4MASK;
static uint32_t SCSIZE,SCMASK,TTSIZE,TTMASK,LSIZE,LMASKB;
/* nibble-row set masks: a Ctr table of SIZE cells holds SIZE/32
   2-way sets of 128 B; a Slot table holds TSIZE/16 rows (NIBO 1,2) or
   TSIZE/32 2-way sets of two adjacent 128-B Slot8 rows (NIBO 3). */
static uint32_t WROWM,SCROWM,XROWM,RROWM,TROWM,TSETM;

static const int MORD[NMATCH] = {6, 12, 24};
/* Mixer learning rate.  0.07 is the best value of a 0.04..0.16 sweep on the
   tuning set at NM=103, for both table layouts, and held-out files agree
   within 0.002% (protocol: paper §3).  Header-carried. */
static float LR = 0.07f;

/* --------------------------------------------------------- run config ---- */
static int EN_STATE=1, EN_WORD=1, EN_SPARSE=1, OPT_REC=1, OPT_E8=2;
static int SSE_MODE=1, NL1=9, MIX2CTX=3, VERB=0;
static float LR2_MUL=0.10f, WINIT=0.3f, SSEW3=0.65f;
static int PFETCH=1, USE_SIMD=1, PFD=1;
/* nibble-row layout switches.  Header-carried (offset 80, 81).  NIB selects
   the cache-line nibble-row layout for the Ctr tables (the row of the low
   nibble is addressed at b==3, from bits both sides already have); NIBO
   selects the layout of the order-model tables. */
/* Defaults NIB=1 NIBO=3.  On the Silesia corpus NIBO=3 is 0.18% larger than
   NIBO=0 (35,838,365 vs 35,773,030 bytes) but about 19% faster, so NIBO=3 is
   the default and NIBO=0 is the max-compression mode (CMX_NIBO=0 at encode
   time; header-carried).  See paper §8. */
static int NIB=1, NIBO=3;

/* set once in main, read-only afterwards */
static int do_rec=0, R=0, R2=0, USE_NN=0;
static int I_P,I_DUAL,I_HIST,I_CNT,I_MATCH,I_RUN,I_WORD,I_SPARSE,I_REC,I_NN;
static int I_SCALE,I_TOL,I_FZ;
static int NM,NIN,NINP,NIN2;
static int nex=0, exid[MAXEX];

/* --- byte-order count, multi-scale ladder and approximate retrieval (all
 * header-carried).  The defaults are the best values measured on the tuning
 * and held-out sets:
 *   nord 12     best of a 7..12 sweep (gains shrink with each added order)
 *   scmask 127  all seven ladder rungs
 *   LPOSCAP 127 best of 63/127/255/511
 *   FUZZ 3      tolerant match + LSH
 * CMX_NORD=7 CMXS=0 CMX_FUZZ=0 gives the base model without these three. */
static int nord    = 12;                /* hashed byte orders 0..nord-1      */
static int scmask  = 127;               /* bitmask of the NSC ladder rungs   */
static int scn     = 0, scid[NSC];      /* derived from scmask               */
static int FUZZ    = 3;                 /* bit0 tolerant match, bit1 LSH     */
static int EN_TOL  = 0, EN_LSH = 0;     /* derived from FUZZ                 */
static int LPOSCAP = 127;               /* ladder rung 5 line-position cap   */
/* LSH bit-position permutation.  Depends ONLY on FZW/FZR/FZL, which are
   compile-time constants, so encoder and decoder generate it identically. */
static int lpos[MAXL][MAXK];
static void init_lpos(void){
    uint32_t rs0=0x12345678u;
    for(int t=0;t<FZL;t++){
        int perm[MAXK]; for(int j=0;j<FZW;j++) perm[j]=j;
        for(int j=0;j<FZW-1;j++){
            rs0=rs0*1664525u+1013904223u;
            int q=j+(int)((rs0>>8)%(uint32_t)(FZW-j));
            int tm=perm[j]; perm[j]=perm[q]; perm[q]=tm;
        }
        for(int j=0;j<FZR;j++) lpos[t][j]=perm[j];
    }
}
static inline uint32_t sc_lg(uint32_t v){ uint32_t b=0; while(v){b++;v>>=1;} return b; }

/* ------------------------------------------------------------ math ------- */
static float STRETCH[4096];
static uint8_t BCLS[256];
/* exact reciprocal tables: the model's adaptation rates depend only on a
   uint8 counter, so ~65 scalar FDIVs per bit become L1 table loads. */
static float RATE[256], RATEF[256], RATEA[256], RCPN[128];
static void init_rates(void){
    for(int n=0;n<256;n++){
        float r=1.f/((float)n+1.6f); if(r<0.0025f)r=0.0025f; RATE[n]=r;
        float f=1.f/((float)n+1.6f); if(f<1.f/FASTR)f=1.f/FASTR; RATEF[n]=f;
        float a=1.f/((float)n+APM_N0_V); if(a<APM_MINRATE_V) a=APM_MINRATE_V; RATEA[n]=a;
    }
    for(int k=0;k<128;k++) RCPN[k]=1.f/((float)k+0.7f);
}
static inline float squashf(float x){ if(x>18.f)x=18.f; if(x<-18.f)x=-18.f; return 1.f/(1.f+expf(-x)); }

/* 2^f, |f|<=0.5, degree-7 minimax(=Taylor) -- |abs err| < 6e-9 */
#define E2C1 0.69314718055994531f
#define E2C2 0.24022650695910071f
#define E2C3 0.05550410866482158f
#define E2C4 0.00961812910762848f
#define E2C5 0.00133335581464284f
#define E2C6 0.00015403530393381f
#define E2C7 0.00001525273380406f
#define LOG2E 1.44269504088896341f

static inline float fexp2s(float x){
    float r = rintf(x);
    float f = x - r;
    float p = E2C7;
    p = p*f + E2C6; p = p*f + E2C5; p = p*f + E2C4;
    p = p*f + E2C3; p = p*f + E2C2; p = p*f + E2C1; p = p*f + 1.f;
    union { float f; uint32_t u; } s;
    int32_t n = (int32_t)r; if(n<-126) n=-126; if(n>126) n=126;
    s.u = (uint32_t)(n+127) << 23;
    return p * s.f;
}
/* logistic 1/(1+e^-x) via fexp2s, clamped to +-18 */
static inline float sig1(float x){
    if(x> 18.f) x= 18.f;
    if(!(x>=-18.f)) x=-18.f;            /* also maps NaN into range */
    return 1.f/(1.f+fexp2s(-x*LOG2E));
}
#ifdef HAVE_NEON
static inline float32x4_t fexp2v(float32x4_t x){
    float32x4_t r = vrndnq_f32(x);
    float32x4_t f = vsubq_f32(x,r);
    float32x4_t p = vdupq_n_f32(E2C7);
    p = vfmaq_f32(vdupq_n_f32(E2C6), p, f);
    p = vfmaq_f32(vdupq_n_f32(E2C5), p, f);
    p = vfmaq_f32(vdupq_n_f32(E2C4), p, f);
    p = vfmaq_f32(vdupq_n_f32(E2C3), p, f);
    p = vfmaq_f32(vdupq_n_f32(E2C2), p, f);
    p = vfmaq_f32(vdupq_n_f32(E2C1), p, f);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, f);
    int32x4_t n = vcvtnq_s32_f32(r);
    n = vmaxq_s32(vminq_s32(n,vdupq_n_s32(126)),vdupq_n_s32(-126));
    int32x4_t e = vshlq_n_s32(vaddq_s32(n,vdupq_n_s32(127)),23);
    return vmulq_f32(p, vreinterpretq_f32_s32(e));
}
static inline float32x4_t sig1v(float32x4_t x){
    x = vminq_f32(vmaxq_f32(x,vdupq_n_f32(-18.f)),vdupq_n_f32(18.f));
    float32x4_t e = fexp2v(vmulq_n_f32(x,-LOG2E));
    return vdivq_f32(vdupq_n_f32(1.f), vaddq_f32(vdupq_n_f32(1.f), e));
}
#endif

/* log2 via atanh series, |rel err| ~1e-8, x>0 */
static inline float flog2(float x){
    union { float f; uint32_t u; } v; v.f = x;
    int e = (int)(v.u >> 23) - 127;
    v.u = (v.u & 0x007FFFFFu) | 0x3F800000u;
    float m = v.f;
    if(m > 1.41421356f){ m *= 0.5f; e++; }
    float t  = (m - 1.f)/(m + 1.f);
    float t2 = t*t;
    float s  = t*(2.88539008177793f + t2*(0.96179669392598f
             + t2*(0.57707801635559f + t2*(0.41219858310970f
             + t2*0.32059889797533f))));
    return (float)e + s;
}

static inline float stretch_p(float p){
    if(p<1e-6f)p=1e-6f; if(p>1.f-1e-6f)p=1.f-1e-6f; return logf(p/(1.f-p));
}
static inline float fstretch(float p){
    /* float clamp first (NaN -> 0): same index as before for every finite p */
    float s=p*4096.f; if(!(s>=0.f))s=0.f; if(s>4095.f)s=4095.f; return STRETCH[(int)s];
}
static inline uint32_t mix32w(uint32_t h){ h^=h>>16; h*=0x7FEB352Du; h^=h>>15; h*=0x846CA68Bu; h^=h>>16; return h; }
static inline uint32_t mix32s(uint32_t h){ h^=h>>15; h*=0x2C1B3C6Du; h^=h>>12; return h; }
static inline uint32_t hb(uint32_t seed,uint32_t a){ return mix32s(seed*0x9E3779B1u+a*0x85EBCA77u); }
static inline uint32_t hashn(const uint8_t*p,int k,uint32_t seed){
    uint32_t h=seed*0x9E3779B1u;
    for(int j=0;j<k;j++) h=h*0x85EBCA77u+p[-1-j];
    h^=h>>15; return h;
}

#define PF0(a) __builtin_prefetch((const void*)(a),0,3)
#define PF1(a) __builtin_prefetch((const void*)(a),1,3)

/* --------------------------------------------------------- counters ------ */
typedef struct { uint16_t p; uint8_t n; } Ctr;
static inline void ctr_update(Ctr *c,int bit){
    int n=c->n; float rate=RATE[n];
    c->p=(uint16_t)(c->p+((bit?65535.f:0.f)-c->p)*rate);
    if(n<255)c->n=n+1;
}
typedef struct { uint16_t p, pf, hist; uint8_t chk, n, n0, n1; } Slot;
static inline void slot_reset(Slot*s,uint8_t chk){
    s->p=32768; s->pf=32768; s->hist=0; s->chk=chk; s->n=0; s->n0=0; s->n1=0;
}
static inline void slot_update_plain(Slot*s,int bit){
    int n=s->n; float rate=RATE[n];
    s->p=(uint16_t)(s->p+((bit?65535.f:0.f)-s->p)*rate);
    if(n<255)s->n=n+1;
}
static inline void slot_update_rich(Slot*s,int bit){
    int n=s->n;
    uint8_t *a = bit? &s->n1 : &s->n0, *b = bit? &s->n0 : &s->n1;
    if(*a<CNTCAP)(*a)++;
    if(*b>2) *b=(uint8_t)((*b>>1)+1); else if(*b) (*b)--;
    float rate=RATE[n];
    float tgt=bit?65535.f:0.f;
    s->p=(uint16_t)(s->p+(tgt-s->p)*rate);
    float rf=RATEF[n];
    s->pf=(uint16_t)(s->pf+(tgt-s->pf)*rf);
    if(n<255)s->n=n+1;
    s->hist=(uint16_t)((s->hist<<1)|bit);
}
static inline Slot* slot_find2(Slot*T,uint32_t h){
    uint8_t chk=(uint8_t)(h>>24);
    Slot*s=&T[(h&TMASK)&~1u];
    if(s[0].chk==chk) return &s[0];
    if(s[1].chk==chk) return &s[1];
    Slot*v = (s[0].n<=s[1].n)? &s[0] : &s[1];
    slot_reset(v,chk);
    return v;
}

/* ===================== optional probe instrumentation ====================
 * -DPROBE only.  Counts one probe per TABLE ADDRESS GENERATED, by family,
 * counted as cache lines (a 160-B row counts 2, the second way of a NIBO=3
 * set counts only when way 0 missed).  Global
 * counters: meaningful at T=1 only.  Never compiled into the shipped binary. */
#ifdef PROBE
enum { PC_ORD, PC_WORD, PC_SCALE, PC_SPARSE, PC_REC,
       PC_MTAB, PC_IHIST, PC_LTAB, PC_TTAB,
       PC_IND, PC_SMALL, PC_SSE, PC_MIXROW, PC_NC };
static unsigned long long PC[PC_NC];
static const char *PCNAME[PC_NC] = {
  "order(Slot x nord)", "word(Ctr x7)", "coarse-scale(Ctr x7)",
  "sparse/indirect(Ctr x nex)", "record(Ctr x7)",
  "mtab(byte x3)", "ihist(byte)", "ltab(byte x8)", "ttab(byte)",
  "indhist(Ctr, direct-indexed)", "small-rel(mrel/rrel/trel/frel/fens)",
  "SSE apmB+apmD (hashed, 85-B rows)", "mixer rows W1 (NL1/bit)" };
static const int PC_BIG[PC_NC] = {1,1,1,1,1, 1,1,1,1, 0,0,0,0};
#define PCI(x)    (PC[x]++)
#define PCN(x,n)  (PC[x]+=(unsigned long long)(n))
#else
#define PCI(x)    ((void)0)
#define PCN(x,n)  ((void)0)
#endif

/* ---------------- CACHE-LINE NIBBLE ROWS -------------------------
 * The cache-line layout described in paper §8.  Adaptation rates
 * come from the exact RATE[]/RATEF[] tables, 1/(n+1.6) clamped.
 *
 * Ctr rows: RSET Ctr == one 128-byte line == a 2-way set of two RWAY-Ctr rows.
 *   way[0]     header: p = checksum (always odd, so a zeroed row never hits)
 *                      n = saturating usage count, used for replacement
 *   way[1..15] the 4-bit nibble tree, cell index = partial nibble context nc.
 * Tables in this layout are ZERO-initialised (rows self-initialise on first
 * touch), which is what lets them come straight from calloc().           */
#define RWAY 16
#define RSET 32
_Static_assert(sizeof(Ctr)==4, "nibble rows assume a 4-byte Ctr (32 per 128-B line)");
static inline void crow_reset(Ctr*w, uint16_t chk){
    w[0].p=chk; w[0].n=0;
    for(int t=1;t<RWAY;t++){ w[t].p=32768; w[t].n=0; }
}
static inline uint16_t row_chk(uint32_t h){ return (uint16_t)(((h*0x9E3779B1u)>>16) | 1u); }
static inline Ctr* crow_find(Ctr*T, uint32_t setmask, uint32_t h){
    Ctr*r = T + ((size_t)(h & setmask) << 5);
    uint16_t chk = row_chk(h);
    if(r[0].p==chk){      if(r[0].n<255) r[0].n++;       return r; }
    if(r[RWAY].p==chk){   if(r[RWAY].n<255) r[RWAY].n++; return r+RWAY; }
    Ctr*v = (r[0].n <= r[RWAY].n) ? r : r+RWAY;   /* replace the less-used way */
    crow_reset(v, chk);
    return v;
}
static inline Ctr* crow_addr(Ctr*T, uint32_t setmask, uint32_t h){
    return T + ((size_t)(h & setmask) << 5);
}

/* Slot8: the 8-byte order-model cell.  16 x 8 B = 128 B = ONE line.  `chk`
   is dropped (the row header carries it) and n0/n1 are packed into one byte
   as two 4-bit saturating counts (cap 15 instead of CNTCAP=40).  The packing
   was measured to cost 0.002%. */
typedef struct { uint16_t p, pf, hist; uint8_t n, n01; } Slot8;
_Static_assert(sizeof(Slot8)==8, "Slot8 must be 8 bytes (16 per 128-B line)");
#define S8CAP 15
static inline void slot8_reset(Slot8*s){ s->p=32768; s->pf=32768; s->hist=0; s->n=0; s->n01=0; }
static inline void slot8_update_rich(Slot8*s,int bit){
    int n=s->n;
    unsigned q0=(unsigned)(s->n01>>4), q1=(unsigned)(s->n01&15u);
    unsigned *a = bit? &q1 : &q0, *b = bit? &q0 : &q1;
    if(*a<S8CAP)(*a)++;
    if(*b>2) *b=((*b)>>1)+1; else if(*b) (*b)--;
    s->n01=(uint8_t)((q0<<4)|q1);
    float rate=RATE[n];
    float tgt=bit?65535.f:0.f;
    s->p=(uint16_t)(s->p+(tgt-s->p)*rate);
    float rf=RATEF[n];
    s->pf=(uint16_t)(s->pf+(tgt-s->pf)*rf);
    if(n<255)s->n=n+1;
    s->hist=(uint16_t)((s->hist<<1)|bit);
}
static inline void slot8_update_plain(Slot8*s,int bit){
    int n=s->n; float rate=RATE[n];
    s->p=(uint16_t)(s->p+((bit?65535.f:0.f)-s->p)*rate);
    if(n<255)s->n=n+1;
}
/* NIBO=1: 16 x 10-B Slot = 160 B (2 lines), direct-mapped, row checksum */
static inline Slot* srow_find(Slot*T, uint32_t rowmask, uint32_t h){
    Slot*r = T + ((size_t)(h & rowmask) << 4);
    uint16_t chk = row_chk(h);
    if(r[0].p==chk){ if(r[0].n<255) r[0].n++; return r; }
    r[0].p=chk; r[0].n=0;
    for(int t=1;t<16;t++) slot_reset(&r[t],0);
    return r;
}
/* NIBO=2: 16 x 8-B Slot8 = 128 B (1 line), direct-mapped */
static inline Slot8* s8row_find(Slot8*T, uint32_t rowmask, uint32_t h){
    Slot8*r = T + ((size_t)(h & rowmask) << 4);
    uint16_t chk = row_chk(h);
    if(r[0].p==chk){ if(r[0].n<255) r[0].n++; return r; }
    r[0].p=chk; r[0].n=0;
    for(int t=1;t<16;t++) slot8_reset(&r[t]);
    return r;
}
/* NIBO=3: 2-way set of two Slot8 rows in ADJACENT lines; way 1 is read only
   when way 0 misses. */
static inline Slot8* s8row_find2(Slot8*T, uint32_t setmask, uint32_t h){
    Slot8*r = T + ((size_t)(h & setmask) << 5);
    uint16_t chk = row_chk(h);
    if(r[0].p==chk){ if(r[0].n<255) r[0].n++; return r; }
    Slot8*r2 = r + 16;
    PCI(PC_ORD);
    if(r2[0].p==chk){ if(r2[0].n<255) r2[0].n++; return r2; }
    Slot8*v = (r[0].n <= r2[0].n) ? r : r2;
    v[0].p=chk; v[0].n=0;
    for(int t=1;t<16;t++) slot8_reset(&v[t]);
    return v;
}

/* ------------------------------------------------------------ word ------- */
static inline int bcls(uint8_t c){
    if(c>='a'&&c<='z') return 1;
    if(c>='A'&&c<='Z') return 2;
    if(c>='0'&&c<='9') return 3;
    if(c==' '||c=='\t') return 4;
    if(c=='\n'||c=='\r') return 5;
    if(c>=128) return 7;
    if(c>32) return 6;
    return 0;
}
static inline int alnum(uint8_t c){
    return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9');
}
static inline uint32_t fold(uint8_t c){ return (c>='A'&&c<='Z') ? (uint32_t)(c+32) : (uint32_t)c; }

static int ex_kind(int id){ return id<20 ? 0 : (id<40 ? 1 : 2); }
static const int DEF_EX[] = {
    41,40,44,42,43,25,22,20,24,23,29,26,28,27,
    15,14,12,11,5,19,18,10,3,4,7,9,
};

/* ------------------------------------------- APM chain (interpolating) --- */
#define NB   17
#define SLIM 8.0f
#define APM_MINRATE 0.010f
#define APM_N0 4.0f
typedef struct __attribute__((packed)) { float t; uint8_t n; } ACell;   /* 5 B */
typedef struct { ACell *c; uint32_t nctx; } APM;

static void apm_init(APM*a, uint32_t nctx){
    a->nctx = nctx;
    a->c = (ACell*)malloc((size_t)nctx*NB*sizeof(ACell));
    if(!a->c){ fprintf(stderr,"cmxc: oom (apm)\n"); exit(1); }
    float v[NB];
    for(int i=0;i<NB;i++){
        float s = ((float)i - (NB-1)*0.5f) * (2.f*SLIM/(NB-1));
        v[i] = squashf(s);
    }
    ACell *q = a->c;
    for(uint32_t c=0;c<nctx;c++){
        for(int i=0;i<NB;i++){ q->t=v[i]; q->n=0; q++; }
    }
}
static inline float apm_p(APM*a, float p, uint32_t cx, uint32_t*ib, float*wb){
    float s=fstretch(p);
    if(s<-SLIM)s=-SLIM; if(s>SLIM)s=SLIM;
    float x=(s+SLIM)*((NB-1)/(2.f*SLIM));
    int idx=(int)x; if(idx<0)idx=0; if(idx>NB-2)idx=NB-2;
    float w=x-(float)idx;
    uint32_t b=cx*NB+(uint32_t)idx;
    *ib=b; *wb=w;
    return a->c[b].t*(1.f-w)+a->c[b+1].t*w;
}
static inline void apm_upd(APM*a, uint32_t b, float w, int bit){
    float tgt = bit?1.f:0.f;
    float ws[2]; ws[0]=1.f-w; ws[1]=w;
    for(int k=0;k<2;k++){
        ACell *e = a->c + b + k;
        int n=e->n;
        float r=RATEA[n]*ws[k];
        e->t += (tgt-e->t)*r;
        if(n<255) e->n=(uint8_t)(n+1);
    }
}
#define S4ORD  4
#define MLENB  8
#define BSSE   0.75f
#define NS2    5
#define LR2SSE 0.02f
static inline int lenbucket(int L){ int b=0; while(L>1 && b<MLENB-1){ L>>=1; b++; } return b; }

/* ------------------------------------------------ APM pair (33 buckets) --- */
#define APMCTX 1024
#define APM_W  0.40f
#define APM2_W 0.40f
#define APM_RATE 0.008f
static inline float apm_pp(uint16_t*T,float st,int cx,int*idx){
    if(!(st>=-8.f))st=-8.f; if(st>8.f)st=8.f;   /* NaN -> -8 */
    float w=(st+8.f)*2.f; int i=(int)w; if(i>31)i=31; float fr=w-i;
    uint16_t*t=T+cx*33+i;
    *idx=cx*33+i+(fr>=0.5f?1:0);
    return (t[0]*(1.f-fr)+t[1]*fr)/65536.f;
}

/* --------------------------------------------------------- selectors ----- */
enum { SEL_PREV=0, SEL_C0, SEL_MATCH, SEL_TYPE, SEL_H1C0, SEL_H2C0, SEL_ORDEST,
       SEL_H3C0, SEL_ORDMB, SEL_MSLB, NSELK };
static const uint32_t SELSIZE[NSELK] = {
    256, 256, 128, 2048, 4096, 16384, 64, 16384, 512, 16384
};
static const int BANK[9] = { SEL_PREV, SEL_C0, SEL_MATCH, SEL_TYPE, SEL_H1C0,
                             SEL_H2C0, SEL_ORDEST, SEL_H3C0, SEL_MSLB };
#define SEL_L2 SEL_ORDMB

/* ------------------------------------------------- E8E9 / BCJ (filter) ---
 * `e8p`: the classic x86 E8/E9 call/jump transform (relative operand ->
 * absolute), in a form that is invertible by construction:
 *   - the predicate reads ONLY d[i], the opcode byte, which the operation at i
 *     never writes (the operation at i writes d[i+1..i+4]);
 *   - forward scans i DESCENDING, inverse scans i ASCENDING;
 *   - there is no 4-byte skip, so every opcode position is an independent op.
 * Write F_i for the operation at position i.  Forward computes F_0 o ... o F_N
 * (because descending application applies F_N first).  Its inverse is therefore
 * F_N^-1 o ... o F_0^-1, i.e. the ascending order used below.  The only ops that
 * write d[i] are F_j for j in [i-4,i-1]; in BOTH orders those are strictly later
 * than the step that reads d[i].  So both directions evaluate an identical
 * predicate on an identical byte and the pair is an exact inverse for EVERY
 * input.
 *
 * Unlike filters that also require the operand's high byte to be 00/FF, e8p
 * rewrites every E8/E9 byte, including stray ones; that is the price of
 * provable invertibility, and it is why the filter is chosen per file by
 * measurement (see the E8E9 selection probe below).
 */
static void e8p_forward(uint8_t*d,size_t n){
    if(n<5) return;
    for(size_t i = n-5+1; i-- > 0; ){
        if((d[i]&0xFEu)!=0xE8u) continue;
        uint32_t rel = (uint32_t)d[i+1] | ((uint32_t)d[i+2]<<8)
                     | ((uint32_t)d[i+3]<<16) | ((uint32_t)d[i+4]<<24);
        uint32_t ab = rel + (uint32_t)(i+5);
        d[i+1]=(uint8_t)(ab>>24); d[i+2]=(uint8_t)(ab>>16);
        d[i+3]=(uint8_t)(ab>>8);  d[i+4]=(uint8_t)ab;
    }
}
static void e8p_backward(uint8_t*d,size_t n){
    if(n<5) return;
    for(size_t i=0;i+5<=n;i++){
        if((d[i]&0xFEu)!=0xE8u) continue;
        uint32_t ab = ((uint32_t)d[i+1]<<24)|((uint32_t)d[i+2]<<16)
                    | ((uint32_t)d[i+3]<<8) |((uint32_t)d[i+4]);
        uint32_t rel = ab - (uint32_t)(i+5);
        d[i+1]=(uint8_t)rel;       d[i+2]=(uint8_t)(rel>>8);
        d[i+3]=(uint8_t)(rel>>16); d[i+4]=(uint8_t)(rel>>24);
    }
}
/* The detector counts screened E8/E9 call sites (operand high byte 00/FF);
   above the 0.0025 threshold the encoder measures whether e8p helps. */
static double e8e9_score(const uint8_t*d,size_t n){
    if(n<64) return 0.0;
    size_t cnt=0;
    for(size_t i=0;i+5<=n;i++)
        if((d[i]&0xFEu)==0xE8u && (d[i+4]==0x00u||d[i+4]==0xFFu)) cnt++;
    return (double)cnt/(double)n;
}

/* ----------------------------------------- record / stride detection ----- */
#define MAXLAG  1024
#define MINLAG  2
static int R2OUT=0;
static int detect_record(const uint8_t*d,size_t n,double*sc,double*rel_chance,double*rel_med){
    *sc=0; *rel_chance=0; *rel_med=0;
    if(n < 4*MAXLAG) return 0;
    size_t start = MAXLAG+64;
    size_t N = n-start; if(N > (1u<<17)) N = (1u<<17);
    static double s[MAXLAG+1];
    for(int L=MINLAG;L<=MAXLAG;L++){
        const uint8_t*a=d+start, *b=d+start-L;
        size_t c=0;
        for(size_t k=0;k<N;k++) c += (a[k]==b[k]);
        s[L]=(double)c/(double)N;
    }
    size_t freq[256]={0};
    for(size_t k=0;k<N;k++) freq[d[start+k]]++;
    double chance=0; for(int v=0;v<256;v++){ double p=(double)freq[v]/(double)N; chance+=p*p; }
    int best=MINLAG; for(int L=MINLAG;L<=MAXLAG;L++) if(s[L]>s[best]) best=L;
    for(int L=MINLAG;L<best;L++)
        if(best%L==0 && s[L]>=0.97*s[best]){ best=L; break; }
    static double tmp[MAXLAG+1]; int m=0;
    for(int L=MINLAG;L<=MAXLAG;L++) tmp[m++]=s[L];
    for(int i=1;i<m;i++){ double v=tmp[i]; int j=i-1; while(j>=0&&tmp[j]>v){tmp[j+1]=tmp[j];j--;} tmp[j+1]=v; }
    double med=tmp[m/2];
    { int W=24; double bestr=0; int bl=0;
      for(int L=MINLAG;L<=MAXLAG;L++){
        int a=L-W<MINLAG?MINLAG:L-W, b2=L+W>MAXLAG?MAXLAG:L+W;
        double sum=0; int c2=0;
        for(int q=a;q<=b2;q++){ sum+=s[q]; c2++; }
        double r=s[L]-sum/c2;
        if(L!=best && r>bestr){ bestr=r; bl=L; }
      }
      R2OUT = (bestr>=0.02 && bl>=MINLAG)? bl : 0;
    }
    *sc=s[best];
    *rel_chance = chance>0? s[best]/chance : 99.0;
    *rel_med    = med>1e-9? s[best]/med : 99.0;
    return best;
}

/* ============================================================== model ==== */
typedef struct {
    /* big tables (owned) */
    Slot     *tab[NORDMAX];        /* NIBO 0 (base) and 1 (10-B Slot rows)  */
    Slot8    *o8tab[NORDMAX];      /* NIBO 2,3: 8-B Slot8 rows              */
    Ctr      *ind;
    uint32_t *mtab[NMATCH];
    Ctr      *wtab[NWORDM];
    Ctr      *xtab[MAXEX];
    uint32_t *ihist[MAXEX];
    Ctr      *rtab[NREC];
    Ctr      *stab[NSC];            /* coarse-scale ladder                   */
    uint32_t *ttab;                 /* tolerant-match index                  */
    uint32_t *ltab[MAXL];           /* LSH index tables                      */
    float    *W1[9];
    float    *W2;
    uint16_t *apmt, *apm2t;
    APM       apmA,apmB,apmC,apmD,apmE;

    /* small state */
    Ctr      mrel[NMATCH][LENB][8][NMH];
    Ctr      rrel[RLENB][8][NMH];
    Ctr      trel[LENB][8][8];                  /* tolerant match            */
    Ctr      frel[SIMB][MAXL+1][8][3];          /* LSH best candidate        */
    Ctr      fens[MAXL+1][MAXL+1][8];           /* LSH vote ensemble         */
    float    w2lin[256][NS2+2];
    uint32_t mptr[NMATCH]; int mlen[NMATCH];
    uint8_t  mhis[NMATCH], rhis;

    /* per-byte scratch */
    __attribute__((aligned(64))) float inp[MAXIN];
    double   trie[512];
    uint32_t chs[NORDMAX], whs[NWORDM], xhs[MAXEX], rhs[NREC], shs[NSC];
    uint32_t exii[MAXEX];
    int      recvalid[NREC];

    /* block */
    uint8_t *D;                    /* == blkbuf+64, local index 0..end       */
    size_t   pstart, begin, end;
    double   bits;
    FILE    *pf; float *pbuf; size_t PCH, phave, pnext;

    /* codec */
    uint8_t *blkbuf;               /* owned: 64 zero bytes + prime + payload */
    size_t   blkcap;
    size_t   abs_start;            /* absolute offset of payload in the file  */
    Coder    cod;
    int      enc;
    int      allocated;
} Model;
_Static_assert(sizeof(Model)%64==0, "run_wave aligned_alloc needs sizeof(Model) % 64 == 0");

/* ---------------------------------------------------- sparse contexts ---- */
static void ex_prepare(Model*M,const uint8_t*p){
    uint32_t *xhs=M->xhs, *exii=M->exii; uint32_t **ihist=M->ihist;
    for(int e=0;e<nex;e++){
        int id=exid[e]; uint32_t s=(uint32_t)(id+101), h;
        switch(id){
        case 0:  h=hb(s,(uint32_t)p[-2]|((uint32_t)p[-3]<<8)); break;
        case 1:  h=hb(s,(uint32_t)p[-1]|((uint32_t)p[-3]<<8)); break;
        case 2:  h=hb(s,(uint32_t)p[-1]|((uint32_t)p[-4]<<8)); break;
        case 3:  h=hb(s,(uint32_t)p[-2]|((uint32_t)p[-4]<<8)); break;
        case 4:  h=hb(s,(uint32_t)p[-3]|((uint32_t)p[-4]<<8)); break;
        case 5:  h=hb(s,(uint32_t)(p[-1]&0xF0)|((uint32_t)(p[-2]&0xF0)<<8)); break;
        case 6:  h=hb(s,(uint32_t)(p[-1]&0xF0)); break;
        case 7:  h=hb(s,(uint32_t)p[-2]|((uint32_t)p[-3]<<8)|((uint32_t)p[-4]<<16)); break;
        case 8:  h=hb(s,(uint32_t)p[-1]|((uint32_t)p[-2]<<8)|((uint32_t)p[-4]<<16)); break;
        case 9:  h=hb(s,(uint32_t)p[-1]|((uint32_t)p[-3]<<8)|((uint32_t)p[-5]<<16)); break;
        case 10: h=hb(s,(uint32_t)p[-2]|((uint32_t)p[-4]<<8)|((uint32_t)p[-6]<<16)); break;
        case 11: h=hb(s,(uint32_t)((p[-1]&0xC0)>>6)|((uint32_t)(p[-2]&0xC0)>>4)
                        |((uint32_t)(p[-3]&0xC0)>>2)|((uint32_t)(p[-4]&0xC0))); break;
        case 12: h=hb(s,(uint32_t)p[-1]|((uint32_t)p[-5]<<8)); break;
        case 13: h=mix32s(s*0x9E3779B1u+p[-1]*0x85EBCA77u+p[-2]*0x27D4EB2Fu
                        +p[-3]*0xC2B2AE3Du+p[-5]*0x165667B1u+p[-6]*0x9E3779B9u); break;
        case 14: h=hb(s,(uint32_t)p[-4]|((uint32_t)p[-8]<<8)); break;
        case 15: h=hb(s,(uint32_t)(p[-1]&0xF0)|((uint32_t)p[-2]<<8)|((uint32_t)(p[-3]&0xF0)<<16)); break;
        case 16: h=hb(s,(uint32_t)(p[-1]&0x0F)|((uint32_t)(p[-2]&0x0F)<<4)|((uint32_t)(p[-3]&0x0F)<<8)); break;
        case 17: h=hb(s,(uint32_t)(p[-1]&0xF0)|((uint32_t)(p[-3]&0xF0)<<8)); break;
        case 18: h=hb(s,(uint32_t)(p[-1]>>1)|((uint32_t)(p[-2]>>1)<<7)|((uint32_t)(p[-3]>>1)<<14)); break;
        case 19: h=hb(s,(uint32_t)(p[-2]&0xF0)|((uint32_t)(p[-4]&0xF0)<<8)); break;
        case 20: { uint32_t ii=hb(s,(uint32_t)p[-1])&IMASK; exii[e]=ii;
                   h=mix32s(s*0x9E3779B1u+(ihist[e][ii]&0xFFFFu)*0x85EBCA77u+p[-1]*0x27D4EB2Fu); } break;
        case 21: { uint32_t ii=hb(s,(uint32_t)p[-1]|((uint32_t)p[-2]<<8))&IMASK; exii[e]=ii;
                   h=mix32s(s*0x9E3779B1u+(ihist[e][ii]&0xFFu)*0x85EBCA77u
                            +((uint32_t)p[-1]|((uint32_t)p[-2]<<8))*0x27D4EB2Fu); } break;
        case 22: { uint32_t ii=hb(s,(uint32_t)p[-1])&IMASK; exii[e]=ii;
                   h=mix32s(s*0x9E3779B1u+(ihist[e][ii]&0xFFFFFFu)*0x85EBCA77u+p[-1]*0x27D4EB2Fu); } break;
        case 23: { uint32_t ii=hb(s,(uint32_t)p[-1]|((uint32_t)p[-2]<<8)|((uint32_t)p[-3]<<16))&IMASK;
                   exii[e]=ii;
                   h=mix32s(s*0x9E3779B1u+(ihist[e][ii]&0xFFu)*0x85EBCA77u
                            +((uint32_t)p[-1]|((uint32_t)p[-2]<<8)|((uint32_t)p[-3]<<16))*0x27D4EB2Fu); } break;
        case 24: { uint32_t ii=hb(s,(uint32_t)p[-1]|((uint32_t)p[-2]<<8))&IMASK; exii[e]=ii;
                   h=mix32s(s*0x9E3779B1u+(ihist[e][ii]&0xFFFFu)*0x85EBCA77u
                            +((uint32_t)p[-1]|((uint32_t)p[-2]<<8))*0x27D4EB2Fu); } break;
        case 25: { uint32_t ii=hb(s,(uint32_t)p[-1])&IMASK; exii[e]=ii;
                   h=mix32s(s*0x9E3779B1u+ihist[e][ii]*0x85EBCA77u); } break;
        case 26: { uint32_t c=(uint32_t)p[-2]|((uint32_t)p[-3]<<8);
                   uint32_t ii=hb(s,c)&IMASK; exii[e]=ii;
                   h=mix32s(s*0x9E3779B1u+(ihist[e][ii]&0xFFFFu)*0x85EBCA77u+c*0x27D4EB2Fu); } break;
        case 27: { uint32_t c=(uint32_t)(p[-1]&0xF0)|((uint32_t)p[-2]<<8);
                   uint32_t ii=hb(s,c)&IMASK; exii[e]=ii;
                   h=mix32s(s*0x9E3779B1u+(ihist[e][ii]&0xFFu)*0x85EBCA77u+c*0x27D4EB2Fu); } break;
        case 28: { uint32_t c=hashn(p,4,s);
                   uint32_t ii=c&IMASK; exii[e]=ii;
                   h=mix32s(s*0x9E3779B1u+(ihist[e][ii]&0xFFFFu)*0x85EBCA77u+c*0x27D4EB2Fu); } break;
        case 29: { uint32_t c=(uint32_t)p[-1]|((uint32_t)p[-3]<<8);
                   uint32_t ii=hb(s,c)&IMASK; exii[e]=ii;
                   h=mix32s(s*0x9E3779B1u+(ihist[e][ii]&0xFFFFu)*0x85EBCA77u+c*0x27D4EB2Fu); } break;
        case 40: { uint32_t ii=hb(s,(uint32_t)p[-1]|((uint32_t)p[-2]<<8))&IMASK; exii[e]=ii;
                   uint32_t v=ihist[e][ii]; uint32_t lb=v&0xFF, rc=v>>8; if(rc>15)rc=15;
                   h=mix32s(s*0x9E3779B1u+lb*0x85EBCA77u+rc*0x27D4EB2Fu
                            +((uint32_t)p[-1])*0xC2B2AE3Du); } break;
        case 41: { uint32_t ii=mix32s(s*0x9E3779B1u+p[-1]*0x85EBCA77u+p[-2]*0x27D4EB2Fu
                            +p[-3]*0xC2B2AE3Du+p[-4]*0x165667B1u)&IMASK; exii[e]=ii;
                   uint32_t v=ihist[e][ii]; uint32_t lb=v&0xFF, rc=v>>8; if(rc>15)rc=15;
                   h=mix32s(s*0x9E3779B1u+lb*0x85EBCA77u+rc*0x27D4EB2Fu); } break;
        case 42: { uint32_t ii=hb(s,(uint32_t)p[-1])&IMASK; exii[e]=ii;
                   uint32_t v=ihist[e][ii]; uint32_t lb=v&0xFF, rc=v>>8; if(rc>15)rc=15;
                   h=mix32s(s*0x9E3779B1u+lb*0x85EBCA77u+rc*0x27D4EB2Fu
                            +((uint32_t)p[-1])*0xC2B2AE3Du); } break;
        case 43: { uint32_t c=(uint32_t)p[-2]|((uint32_t)p[-3]<<8);
                   uint32_t ii=hb(s,c)&IMASK; exii[e]=ii;
                   uint32_t v=ihist[e][ii]; uint32_t lb=v&0xFF, rc=v>>8; if(rc>15)rc=15;
                   h=mix32s(s*0x9E3779B1u+lb*0x85EBCA77u+rc*0x27D4EB2Fu+c*0xC2B2AE3Du); } break;
        case 44: { uint32_t ii=hashn(p,6,s)&IMASK; exii[e]=ii;
                   uint32_t v=ihist[e][ii]; uint32_t lb=v&0xFF, rc=v>>8; if(rc>15)rc=15;
                   h=mix32s(s*0x9E3779B1u+lb*0x85EBCA77u+rc*0x27D4EB2Fu); } break;
        default: h=hb(s,0); break;
        }
        xhs[e]=h*0xC2B2AE3Du;
    }
}
static void ex_after_byte(Model*M,uint8_t sym){
    for(int e=0;e<nex;e++){
        int k=ex_kind(exid[e]);
        if(k==1){ uint32_t ii=M->exii[e]; M->ihist[e][ii]=(M->ihist[e][ii]<<8)|sym; }
        else if(k==2){ uint32_t ii=M->exii[e]; uint32_t v=M->ihist[e][ii];
            if((v&0xFF)==sym){ uint32_t rc=v>>8; if(rc<255)rc++; M->ihist[e][ii]=(rc<<8)|sym; }
            else M->ihist[e][ii]=(1u<<8)|sym; }
    }
}

/* --------------------------------------------------------- allocation ---- */
/* a ZEROED, 128-B-aligned table for the nibble-row layouts.  Row
   tables need no initialisation pass: a zero header can never match an odd
   checksum, so a row resets itself on first touch.  Large calloc() blocks
   come from fresh, page-aligned VM on this platform, so the zero fill is
   lazy; the 128-B alignment is CHECKED here, never assumed. */
static void* zalloc128(size_t bytes){
    void*q=calloc(bytes,1);
    if(q && ((uintptr_t)q & 127u)==0) return q;
    free(q); q=NULL;
    if(posix_memalign(&q,128,bytes)){ fprintf(stderr,"cmxc: oom (row table)\n"); exit(1); }
    memset(q,0,bytes); return q;
}
static Ctr* ctab_alloc(size_t n){
    if(NIB) return (Ctr*)zalloc128(sizeof(Ctr)*n);
    Ctr*t=aligned_alloc(64,sizeof(Ctr)*n);
    if(!t){ fprintf(stderr,"cmxc: oom (table)\n"); exit(1); }
    for(size_t i=0;i<n;i++){ t[i].p=32768; t[i].n=0; }
    return t;
}
static void model_alloc(Model*M){
    memset(M,0,sizeof(*M));
    for(int k=0;k<nord;k++){
        if(NIBO>=2)      M->o8tab[k]=(Slot8*)zalloc128(sizeof(Slot8)*(size_t)TSIZE);
        else if(NIBO==1) M->tab[k]=(Slot*)zalloc128(sizeof(Slot)*(size_t)TSIZE);
        else { M->tab[k]=malloc(sizeof(Slot)*(size_t)TSIZE);
               for(uint32_t i=0;i<TSIZE;i++) slot_reset(&M->tab[k][i],0); } }
    if(EN_STATE){
        M->ind=malloc(sizeof(Ctr)*(size_t)nord*8*HSZ);
        for(size_t i=0;i<(size_t)nord*8*HSZ;i++){ M->ind[i].p=32768; M->ind[i].n=0; }
    }
    for(int j=0;j<NMATCH;j++){ M->mtab[j]=calloc((size_t)MSIZE,4); M->mptr[j]=0; M->mlen[j]=0; M->mhis[j]=0;
        for(int b=0;b<LENB;b++) for(int q=0;q<8;q++) for(int u=0;u<NMH;u++){
            M->mrel[j][b][q][u].p=32768; M->mrel[j][b][q][u].n=0; } }
    for(int b=0;b<RLENB;b++) for(int q=0;q<8;q++) for(int u=0;u<NMH;u++){
        M->rrel[b][q][u].p=32768; M->rrel[b][q][u].n=0; }
    M->rhis=0;
    /* the four hashed Ctr families: base per-bit layout (p=32768 fill) or,
       with NIB, zeroed nibble-row tables (ctab_alloc) */
    if(EN_WORD) for(int k=0;k<NWORDM;k++) M->wtab[k]=ctab_alloc((size_t)WSIZE);
    for(int x=0;x<nex;x++){ M->xtab[x]=ctab_alloc((size_t)XSIZE);
        M->ihist[x]= ex_kind(exid[x])? (uint32_t*)calloc((size_t)ISIZE,4) : NULL; }
    if(do_rec) for(int k=0;k<NREC;k++) M->rtab[k]=ctab_alloc((size_t)RSIZE);
    /* ---- ladder rung tables: one per ENABLED rung, per block ---- */
    for(int s=0;s<scn;s++) M->stab[s]=ctab_alloc((size_t)SCSIZE);
    /* ---- fuzzy retrieval: index tables and relation counters, per block ---- */
    if(EN_TOL){
        M->ttab=(uint32_t*)calloc((size_t)TTSIZE,4);
        if(!M->ttab){fprintf(stderr,"cmxc: oom (ttab)\n");exit(1);}
        for(int a=0;a<LENB;a++)for(int q=0;q<8;q++)for(int u=0;u<8;u++){
            M->trel[a][q][u].p=32768; M->trel[a][q][u].n=0; }
    }
    if(EN_LSH){
        for(int t=0;t<FZL;t++){ M->ltab[t]=(uint32_t*)calloc((size_t)LSIZE,4);
            if(!M->ltab[t]){fprintf(stderr,"cmxc: oom (ltab)\n");exit(1);} }
        for(int a=0;a<SIMB;a++)for(int q=0;q<=MAXL;q++)for(int u=0;u<8;u++)for(int d=0;d<3;d++){
            M->frel[a][q][u][d].p=32768; M->frel[a][q][u][d].n=0; }
        for(int a=0;a<=MAXL;a++)for(int q=0;q<=MAXL;q++)for(int u=0;u<8;u++){
            M->fens[a][q][u].p=32768; M->fens[a][q][u].n=0; }
    }

    float w0 = WINIT * (11.0f/(float)NM);
    for(int k=0;k<NL1;k++){
        uint32_t ns=SELSIZE[BANK[k]];
        M->W1[k]=malloc(sizeof(float)*(size_t)ns*NINP);
        for(uint32_t s=0;s<ns;s++) for(int t=0;t<NINP;t++)
            M->W1[k][(size_t)s*NINP+t]=(t<NM)?w0:0.f;
    }
    M->W2=malloc(sizeof(float)*(size_t)SELSIZE[SEL_L2]*NIN2);
    { float v=1.f/(L2_SCALE*NL1);
      for(uint32_t s=0;s<SELSIZE[SEL_L2];s++){
        for(int t=0;t<NL1;t++) M->W2[(size_t)s*NIN2+t]=v; M->W2[(size_t)s*NIN2+NL1]=0.f; } }

    if(SSE_MODE==0 || SSE_MODE==3){
        M->apmt =malloc(sizeof(uint16_t)*APMCTX*33);
        M->apm2t=malloc(sizeof(uint16_t)*APMCTX*33);
        for(int i=0;i<APMCTX*33;i++){ float p=squashf((i%33-16)/2.f);
            M->apmt[i]=(uint16_t)(p*65535.f); M->apm2t[i]=(uint16_t)(p*65535.f); }
    }
    if(SSE_MODE==1 || SSE_MODE==2){
        apm_init(&M->apmA, 256);
        apm_init(&M->apmB, S2SIZE);
        apm_init(&M->apmC, 256u*(2u*MLENB+1u));
        apm_init(&M->apmD, S4SIZE);
        for(int s=0;s<256;s++){ for(int i=0;i<NS2;i++) M->w2lin[s][i]=4.f/NS2; M->w2lin[s][NS2]=0.f; }
    }
    if(SSE_MODE==2 || SSE_MODE==3) apm_init(&M->apmE, 256u<<6);
    for(int t=0;t<MAXIN;t++) M->inp[t]=0.f;
}

/* ------------------------------------------------------- mixer kernels --- */
static inline float mix_dot(const float * restrict W, const float * restrict X, int n){
#ifdef HAVE_NEON
    if(USE_SIMD){
        float32x4_t a0=vdupq_n_f32(0.f),a1=a0,a2=a0,a3=a0;
        int t=0;
        for(; t+15<n; t+=16){
            a0=vfmaq_f32(a0, vld1q_f32(W+t   ), vld1q_f32(X+t   ));
            a1=vfmaq_f32(a1, vld1q_f32(W+t+4 ), vld1q_f32(X+t+4 ));
            a2=vfmaq_f32(a2, vld1q_f32(W+t+8 ), vld1q_f32(X+t+8 ));
            a3=vfmaq_f32(a3, vld1q_f32(W+t+12), vld1q_f32(X+t+12));
        }
        for(; t+3<n; t+=4) a0=vfmaq_f32(a0, vld1q_f32(W+t), vld1q_f32(X+t));
        float32x4_t s=vaddq_f32(vaddq_f32(a0,a1),vaddq_f32(a2,a3));
        return vgetq_lane_f32(s,0)+vgetq_lane_f32(s,1)
              +vgetq_lane_f32(s,2)+vgetq_lane_f32(s,3);
    }
#endif
    { float d0=0,d1=0,d2=0,d3=0; int t=0;
      for(;t+3<n;t+=4){ d0+=W[t]*X[t]; d1+=W[t+1]*X[t+1];
                        d2+=W[t+2]*X[t+2]; d3+=W[t+3]*X[t+3]; }
      float d=(d0+d1)+(d2+d3);
      for(;t<n;t++) d+=W[t]*X[t];
      return d; }
}
static inline void mix_upd(float * restrict W, const float * restrict X, float r, int n){
#ifdef HAVE_NEON
    if(USE_SIMD){
        float32x4_t vr=vdupq_n_f32(-r);
        int t=0;
        for(; t+15<n; t+=16){
            vst1q_f32(W+t,    vfmaq_f32(vld1q_f32(W+t   ), vr, vld1q_f32(X+t   )));
            vst1q_f32(W+t+4,  vfmaq_f32(vld1q_f32(W+t+4 ), vr, vld1q_f32(X+t+4 )));
            vst1q_f32(W+t+8,  vfmaq_f32(vld1q_f32(W+t+8 ), vr, vld1q_f32(X+t+8 )));
            vst1q_f32(W+t+12, vfmaq_f32(vld1q_f32(W+t+12), vr, vld1q_f32(X+t+12)));
        }
        for(; t+3<n; t+=4) vst1q_f32(W+t, vfmaq_f32(vld1q_f32(W+t), vr, vld1q_f32(X+t)));
        for(; t<n; t++) W[t]-=r*X[t];
        return;
    }
#endif
    for(int t=0;t<n;t++) W[t]-=r*X[t];
}

/* ============================================================== engine ====
 * THE shared model.  enc==1 -> read the byte from D and drive the arithmetic
 * encoder; enc==0 -> drive the arithmetic decoder and write the byte into D.
 * Nothing else differs, so both directions execute identical float code.
 * ========================================================================= */
static void code_block(Model *M)
{
    const int enc = M->enc;
    Coder * const C = &M->cod;
    if(enc) enc_init(C); else dec_init(C);

    Slot **tab   = M->tab;
    Ctr   *ind   = M->ind;
    Ctr  **wtab  = M->wtab;
    Ctr  **xtab  = M->xtab;
    Ctr  **rtab  = M->rtab;
    Ctr  **stab  = M->stab;
    uint32_t **mtab = M->mtab;
    uint32_t *mptr = M->mptr;  int *mlen = M->mlen;
    uint8_t  *mhis = M->mhis;
    float *inp = M->inp;
    uint32_t *chs=M->chs, *whs=M->whs, *xhs=M->xhs, *rhs=M->rhs, *shs=M->shs;
    int *recvalid = M->recvalid;
    float **W1 = M->W1, *W2 = M->W2;
    APM *apmA=&M->apmA,*apmB=&M->apmB,*apmC=&M->apmC,*apmD=&M->apmD,*apmE=&M->apmE;
    uint8_t * const D = M->D;
    const size_t begin=M->begin, end=M->end;

    /* per-order hash constants:  hashn(p,k,k+1) = CK[k] + P_k  (mod 2^32) */
    uint32_t CK[NORDMAX], MK[NMATCH];
    for(int k=0;k<nord;k++){ uint32_t v=((uint32_t)k+1u)*0x9E3779B1u;
        for(int j=0;j<k;j++) v*=0x85EBCA77u; CK[k]=v; }
    for(int j=0;j<NMATCH;j++){ uint32_t v=(0xB5u+(uint32_t)j)*0x9E3779B1u;
        for(int q=0;q<MORD[j];q++) v*=0x85EBCA77u; MK[j]=v; }

    uint32_t wh=0, w1=0, w2=0, w3=0, clsh=0; int wcase=0, wlen=0;
    uint8_t rbyte=0; int rlen=0;
    uint32_t prfl=0;
    float lossema=1.0f;
    double bits=0.0;
    double *trie=M->trie;

    /* ---- ladder state: block-local, so T>1 blocks cannot see each other ---- */
    uint32_t swh=0; int swlen=0;
    uint32_t swq[8]={0,0,0,0,0,0,0,0}; uint8_t swlq[4]={0,0,0,0};
    uint32_t s_sentpos=0, s_sentw=0;
    uint32_t s_linepos=0, s_prevlinelen=0;
    uint32_t s_nest=0, s_tagh=0, s_curtag=0; int s_intag=0;
    /* ---- fuzzy-retrieval state: likewise block-local ---- */
    uint32_t *ttab=M->ttab; uint32_t tptr=0; int tlen=0; uint8_t tmis=0;
    uint32_t **ltab=M->ltab;
    int fz_ncand=0; uint32_t fz_cpos[MAXL]; int fz_csim[MAXL];
    for(int q=0;q<MAXL;q++){ fz_cpos[q]=0; fz_csim[q]=0; }

    for(size_t i=M->pstart;i<end;i++){
        uint8_t *p=D+i;
        /* i<begin is the priming region: those bytes are already known to BOTH
           sides (the encoder has the file, the decoder has decoded them), so
           the model is run over them but nothing is coded. */
        const int counting = (i>=begin);
        const int docode   = counting;
        /* sym is known up front only when we are encoding (or priming). */
        uint8_t sym = (enc || !docode) ? *p : 0;

        if(USE_NN){
            if(M->pnext>=M->phave){ M->phave=fread(M->pbuf,sizeof(float)*256,M->PCH,M->pf); M->pnext=0;
                if(!M->phave){fprintf(stderr,"probs exhausted at %zu\n",i);exit(1);} }
            const float*P=M->pbuf+256*M->pnext; M->pnext++;
            double s=0; for(int v=0;v<256;v++){ trie[256+v]=P[v]; s+=P[v]; }
            if(s<=0) for(int v=0;v<256;v++) trie[256+v]=1.0/256.0;
            for(int nd=255;nd>=1;nd--) trie[nd]=trie[2*nd]+trie[2*nd+1];
        }

        /* ---- byte-order hashes, incremental polynomial ---- */
        { uint32_t Pk=0;
          for(int k=0;k<nord;k++){
            uint32_t h = CK[k] + Pk; h^=h>>15;
            chs[k]=h*0xC2B2AE3Du;
            Pk = Pk*0x85EBCA77u + p[-1-k];
          }
        }
        if(nex) ex_prepare(M,p);
#ifdef PROBE
        for(int e=0;e<nex;e++) if(M->ihist[e]) PCI(PC_IHIST);
        PCN(PC_MTAB,NMATCH);
        if(EN_TOL) PCI(PC_TTAB);
        if(EN_LSH) PCN(PC_LTAB,FZL);
#endif

        /* ---- coarse-scale ladder contexts ----
           Reads p[-1] and the ladder's own state only; both are available to
           the decoder at this point, since the state advances from `sym` only
           after the byte has been fully coded. */
        for(int s=0;s<scn;s++){
            uint32_t h; uint32_t b1=(uint32_t)p[-1];
            switch(scid[s]){
            case 0: h=mix32w(swq[1]*0x9E3779B1u + swq[0]*0x85EBCA77u
                             + swh*0x27D4EB2Fu + 0x51u); break;
            case 1: { uint32_t t=0x165667B1u;
                      for(int j=3;j>=0;j--) t=t*0x9E3779B1u+swq[j];
                      h=mix32w(t*0x85EBCA77u + swh*0x27D4EB2Fu); } break;
            case 2: { uint32_t t=0x7FEB352Du;
                      for(int j=7;j>=0;j--) t=t*0x9E3779B1u+swq[j];
                      h=mix32w(t*0x846CA68Bu + b1); } break;
            case 3: { uint32_t t=0x1234567u;
                      for(int j=3;j>=0;j--) t=t*37u+(swlq[j]>15?15:swlq[j]);
                      h=mix32w(t*0x9E3779B1u
                               +(uint32_t)(swlen>15?15:swlen)*0x85EBCA77u+0x9Du); } break;
            case 4: h=mix32w(sc_lg(s_sentpos)*0x9E3779B1u
                             +(s_sentw>31?31u:s_sentw)*0x85EBCA77u
                             + b1*0x27D4EB2Fu + 0xC7u); break;
            /* rung 5 (LPOS): the line position is capped at LPOSCAP
               (default 127, the best of 63/127/255/511). */
            case 5: h=mix32w((s_linepos>(uint32_t)LPOSCAP?(uint32_t)LPOSCAP:s_linepos)*0x9E3779B1u
                             + sc_lg(s_prevlinelen)*0x85EBCA77u
                             + b1*0xC2B2AE3Du + 0xE9u); break;
            default: h=mix32w((s_nest>15?15u:s_nest)*0x9E3779B1u
                             +(uint32_t)s_intag*0x85EBCA77u + s_curtag*0xC2B2AE3Du
                             + b1*0x27D4EB2Fu + 0x13u); break;
            }
            shs[s]=h*0xC2B2AE3Du;
        }

        if(EN_WORD){
            whs[0]=mix32w(wh*0x2545F491u+0x9E3779B1u)*0xC2B2AE3Du;
            whs[1]=mix32w(w1*0xC2B2AE3Du + wh*0x27D4EB2Fu + 0x165667B1u)*0xC2B2AE3Du;
            whs[2]=mix32w(w3*0x27D4EB2Fu + w2*0x9E3779B1u + w1*0x85EBCA77u + 0x51u)*0xC2B2AE3Du;
            whs[3]=mix32w(wh*0x9E3779B1u + (uint32_t)(wcase*0x1000193u) + 0x33u)*0xC2B2AE3Du;
            whs[4]=mix32w((clsh & ((1u<<15)-1u))*0x9E3779B1u + 0x7FEB352Du)*0xC2B2AE3Du;
            whs[5]=mix32w((clsh & ((1u<<30)-1u))*0x846CA68Bu + 0x165667B1u)*0xC2B2AE3Du;
            { uint32_t h3=0x1234567u;
              for(int j=1;j<=32;j++) h3=h3*0x9E3779B1u+(uint32_t)BCLS[p[-j]];
              whs[6]=mix32w(h3)*0xC2B2AE3Du; }
        }

        uint8_t mp[NMATCH]; int mvalid[NMATCH];
        for(int j=0;j<NMATCH;j++){
            mvalid[j] = (mlen[j]>0 && mptr[j]<i);
            mp[j] = mvalid[j] ? D[mptr[j]] : 0;
        }
        int rvalid = EN_STATE && (i>M->pstart);
        uint8_t rp = rvalid ? rbyte : 0;
        int rbucket = rlen<RLENB ? rlen : RLENB-1;

        if(do_rec){
            size_t need = (size_t)2*R;
            int ok1 = i>=(size_t)R, ok2 = i>=need;
            uint8_t bR  = ok1 ? p[-R] : 0;
            uint8_t b2R = ok2 ? p[-2*(ptrdiff_t)R] : 0;
            uint8_t b1  = i>0 ? p[-1] : 0;
            uint32_t col = (uint32_t)(i % (size_t)R);
            recvalid[0]=ok1;  rhs[0]=mix32w(0x11111111u ^ (uint32_t)bR*0x2545F491u)*0xC2B2AE3Du;
            recvalid[1]=ok2;  rhs[1]=mix32w(0x22222222u ^ ((uint32_t)bR<<8) ^ (uint32_t)b2R*0x9E3779B1u)*0xC2B2AE3Du;
            recvalid[2]=ok1;  rhs[2]=mix32w(0x33333333u ^ ((uint32_t)b1<<16) ^ (uint32_t)bR*0x85EBCA77u)*0xC2B2AE3Du;
            recvalid[3]=1;    rhs[3]=mix32w(0x44444444u ^ (col*0xC2B2AE3Du))*0xC2B2AE3Du;
            if(R2>0){
                int o1 = i>=(size_t)R2, o2 = i>=(size_t)2*R2;
                uint8_t cR  = o1 ? p[-R2] : 0;
                uint8_t c2R = o2 ? p[-2*(ptrdiff_t)R2] : 0;
                recvalid[4]=o1; rhs[4]=mix32w(0x55555555u ^ (uint32_t)cR*0x2545F491u)*0xC2B2AE3Du;
                recvalid[5]=o2; rhs[5]=mix32w(0x66666666u ^ ((uint32_t)cR<<8) ^ (uint32_t)c2R*0x9E3779B1u)*0xC2B2AE3Du;
                recvalid[6]=o1; rhs[6]=mix32w(0x77777777u ^ ((uint32_t)b1<<16) ^ (uint32_t)cR*0x85EBCA77u)*0xC2B2AE3Du;
            } else { recvalid[4]=recvalid[5]=recvalid[6]=0;
                     inp[I_REC+4]=inp[I_REC+5]=inp[I_REC+6]=0.f; }
        }

        int lb8=0; { float x=lossema;
            lb8=(x>0.05f)+(x>0.15f)+(x>0.30f)+(x>0.50f)+(x>0.70f)+(x>0.85f)+(x>0.95f); }
        int ms8=0; { int L=0; for(int j=0;j<NMATCH;j++) if(mvalid[j]&&mlen[j]>L) L=mlen[j];
            if(L>0) ms8 = 1+(L>=4)+(L>=8)+(L>=16)+(L>=32)+(L>=64)+(L>=256);
            if(ms8>7) ms8=7; }
        int pcnt=__builtin_popcount(prfl&0xFFFFu);
        int tb = pcnt>=15?3 : pcnt>=10?2 : pcnt>=4?1 : 0;
        uint32_t sPREV = p[-1];
        uint32_t sTYPEbase = (uint32_t)tb*16u + (uint32_t)(p[-1]>>4);
        uint32_t h1 = ((uint32_t)p[-1]+1u)*0x9E3779B1u;
        uint32_t h2 = (((uint32_t)p[-1]<<8 | (uint32_t)p[-2])+1u)*0x85EBCA77u;
        uint32_t h3s= (((uint32_t)p[-1]<<16 | (uint32_t)p[-2]<<8 | (uint32_t)p[-3])+1u)*0xC2B2AE3Du;
        uint32_t sMSLB = (uint32_t)((ms8<<11)|(lb8<<8)|p[-1]) & (SELSIZE[SEL_MSLB]-1);
        uint32_t s4base = hashn(p,S4ORD,0x51u);
        uint32_t s2base = (((uint32_t)p[-1]|((uint32_t)p[-2]<<8))*0x9E3779B1u)>>8;

        /* ---- tolerant-match prediction for this byte --
           tptr indexes a byte STRICTLY BEFORE i (guard below), so D[tptr] is
           known to the decoder. */
        int tvalid=0, tbucket=0, tmisc=0; uint8_t tp=0;
        if(EN_TOL){
            tvalid = (tlen>0 && tptr<i);
            tp = tvalid ? D[tptr] : 0;
            int L=tlen; if(L<16) tbucket=L;
            else { int lg=31-__builtin_clz((unsigned)L); tbucket=12+lg;
                   if(tbucket>LENB-1)tbucket=LENB-1; }
            tmisc=__builtin_popcount((unsigned)(tmis&0x3F)); if(tmisc>7)tmisc=7;
        }
        /* ---- LSH candidates, retrieved at the END of byte i-1 ----
           Each surviving candidate satisfies fz_cpos[q] < i, so every D[] read
           below is of a byte the decoder has already reconstructed. */
        int fncand=0, fbest=-1, fbestsim=-1, fmstate=0;
        uint32_t fcp[MAXL];
        int fvotes=0, fsimb=0; uint8_t fpred=0;
        if(EN_LSH){
            for(int q=0;q<fz_ncand;q++){
                if((size_t)fz_cpos[q]>=i) continue;
                fcp[fncand]=fz_cpos[q];
                if(fz_csim[q]>fbestsim){ fbestsim=fz_csim[q]; fbest=fncand; }
                fncand++;
            }
            int L=0; for(int j=0;j<NMATCH;j++) if(mvalid[j]&&mlen[j]>L) L=mlen[j];
            fmstate = L==0 ? 0 : (L<8 ? 1 : 2);
            if(fbest>=0){
                fpred=D[fcp[fbest]]; fsimb=fbestsim; if(fsimb>MAXK)fsimb=MAXK;
                for(int q=0;q<fncand;q++) if(D[fcp[q]]==fpred) fvotes++;
            }
        }

        /* every ctxbits value this byte will take -- the ENCODER knows the byte,
           so all 8 bits' table addresses are computable up front.  This is the
           1.61x prefetch win and it is *encoder only*: the decoder discovers
           the bits one at a time and simply has no lookahead path.  PFBIT only
           issues __builtin_prefetch, so it cannot touch model state or change
           any probability -- the two sides stay bit-identical. */
        const int can_pf = (enc && PFETCH);
        int cbv[9]; cbv[0]=1;
        if(can_pf) for(int j=0;j<8;j++) cbv[j+1]=(cbv[j]<<1)|((sym>>(7-j))&1);
        else       for(int j=0;j<8;j++) cbv[j+1]=0;

#define PFBIT(J) do{ int _j=(J); if(_j<8){                                       \
    uint32_t _c=(uint32_t)cbv[_j], _n=_c*0x27D4EB2Fu;                            \
    if(!NIBO) for(int k=0;k<nord;k++){ uint32_t h=chs[k]^_n; h^=h>>15;           \
        PF1(&tab[k][(h&TMASK)&~1u]); }                                           \
    if(!NIB){                                                                    \
      for(int s=0;s<scn;s++){ uint32_t h=shs[s]^_n; h^=h>>15;                    \
        PF1(&stab[s][h&SCMASK]); }                                               \
      if(EN_WORD) for(int k=0;k<NWORDM;k++){ uint32_t h=whs[k]^_n; h^=h>>15;     \
        PF1(&wtab[k][h&WMASK]); }                                                \
      for(int x=0;x<nex;x++){ uint32_t h=xhs[x]^_n; h^=h>>15;                    \
        PF1(&xtab[x][h&XMASK]); }                                                \
      if(do_rec) for(int k=0;k<NREC;k++) if(recvalid[k]){                        \
        uint32_t h=rhs[k]^_n; h^=h>>15; PF1(&rtab[k][h&RMASK]); } }              \
    if(SSE_MODE==1||SSE_MODE==2){                                                \
        uint32_t cb2=(s2base+_c*0x2545F491u)&S2MASK;                             \
        const char*q2=(const char*)(apmB->c+(size_t)cb2*NB);                     \
        PF1(q2); PF1(q2+64);                                                     \
        uint32_t cd2=(s4base+_c*0x2545F491u)&S4MASK;                             \
        const char*q4=(const char*)(apmD->c+(size_t)cd2*NB);                     \
        PF1(q4); PF1(q4+64); }                                                   \
    if(NL1>1){ const float*q=W1[1]+(size_t)(_c&255u)*NINP;                       \
        for(int z=0;z<NINP;z+=16) PF1(q+z); }                                    \
    if(NL1>4){ uint32_t h=h1^(_c*0x27D4EB2Fu); h^=h>>15;                         \
        const float*q=W1[4]+(size_t)(h&(SELSIZE[SEL_H1C0]-1))*NINP;              \
        for(int z=0;z<NINP;z+=16) PF1(q+z); }                                    \
    if(NL1>5){ uint32_t h=h2^(_c*0x165667B1u); h^=h>>15;                         \
        const float*q=W1[5]+(size_t)(h&(SELSIZE[SEL_H2C0]-1))*NINP;              \
        for(int z=0;z<NINP;z+=16) PF1(q+z); }                                    \
    if(NL1>7){ uint32_t h=h3s^(_c*0x9E3779B1u); h^=h>>15;                        \
        const float*q=W1[7]+(size_t)(h&(SELSIZE[SEL_H3C0]-1))*NINP;              \
        for(int z=0;z<NINP;z+=16) PF1(q+z); }                                    \
    if(NL1>3){ const float*q=W1[3]+(size_t)(sTYPEbase*8u+(uint32_t)_j)*NINP;     \
        for(int z=0;z<NINP;z+=16) PF1(q+z); }                                    \
  } }while(0)

        if(can_pf) for(int d=1;d<PFD;d++) PFBIT(d);

        /* ---- nibble rows.  The row of nibble q is addressed by
           ctxbits at its first bit: 1 (q=0, b==7) and 16+hi (q=1, b==3).  The
           ENCODER knows hi now, so it can prefetch both nibbles' rows here;
           the decoder issues its prefetches at b==7 / b==3 (rowfetch below),
           which still overlaps every model's miss with every other's. ---- */
        Ctr *wrow[NWORDM], *srow[NSC], *xrow[MAXEX], *rrow[NREC];
        Slot *orow[NORDMAX]; Slot8 *o8row[NORDMAX];
        uint32_t rh_o[NORDMAX], rh_w[NWORDM], rh_s[NSC], rh_x[MAXEX], rh_r[NREC];
        int nc=1;
#define ROWHASH(base,cbv_) ({ uint32_t _h=(base)^(cbv_); _h^=_h>>15; _h; })
#define ROWPF(CB) do{ uint32_t _cb=(CB);                                         \
    if(NIBO==3) for(int k=0;k<nord;k++){ Slot8*_r=M->o8tab[k]                    \
        +((size_t)(ROWHASH(chs[k],_cb)&TSETM)<<5); PF1(_r); PF1(_r+16); }        \
    else if(NIBO==2) for(int k=0;k<nord;k++)                                     \
        PF1(M->o8tab[k]+((size_t)(ROWHASH(chs[k],_cb)&TROWM)<<4));               \
    else if(NIBO==1) for(int k=0;k<nord;k++){ Slot*_r=tab[k]                     \
        +((size_t)(ROWHASH(chs[k],_cb)&TROWM)<<4);                               \
        PF1(_r); PF1((const char*)_r+128); }                                     \
    if(NIB){                                                                     \
      if(EN_WORD) for(int k=0;k<NWORDM;k++)                                      \
        PF1(crow_addr(wtab[k],WROWM,ROWHASH(whs[k],_cb)));                       \
      for(int s=0;s<scn;s++) PF1(crow_addr(stab[s],SCROWM,ROWHASH(shs[s],_cb))); \
      for(int x=0;x<nex;x++) PF1(crow_addr(xtab[x],XROWM,ROWHASH(xhs[x],_cb)));  \
      if(do_rec) for(int k=0;k<NREC;k++) if(recvalid[k])                         \
        PF1(crow_addr(rtab[k],RROWM,ROWHASH(rhs[k],_cb))); }                     \
  }while(0)
        if(can_pf && (NIB||NIBO)){
            ROWPF(1u*0x27D4EB2Fu);
            ROWPF((uint32_t)cbv[4]*0x27D4EB2Fu);
        }

        int ctxbits=1;
        for(int b=7;b>=0;b--){
            const int bidx=7-b;
            int bit;
            uint32_t cb = (uint32_t)ctxbits*0x27D4EB2Fu;

            if(can_pf) PFBIT(bidx+PFD);

            /* ---- one row address per model per NIBBLE ----
               ctxbits here is 1 (b==7) or 16+high nibble (b==3): bits the
               decoder has already decoded.  All row hashes are formed and
               prefetched first, then probed, so the misses overlap. */
            if((NIB||NIBO) && (b==7 || b==3)){
                nc=1;
                if(!can_pf) ROWPF(cb);
                if(NIBO) for(int k=0;k<nord;k++) rh_o[k]=ROWHASH(chs[k],cb);
                if(NIB){
                    if(EN_WORD) for(int k=0;k<NWORDM;k++) rh_w[k]=ROWHASH(whs[k],cb);
                    for(int s=0;s<scn;s++) rh_s[s]=ROWHASH(shs[s],cb);
                    for(int x=0;x<nex;x++) rh_x[x]=ROWHASH(xhs[x],cb);
                    if(do_rec) for(int k=0;k<NREC;k++) rh_r[k]=ROWHASH(rhs[k],cb);
                }
                if(NIBO==3) for(int k=0;k<nord;k++){ PCI(PC_ORD);   /* +1 inside on a way-0 miss */
                    o8row[k]=s8row_find2(M->o8tab[k],TSETM,rh_o[k]); }
                else if(NIBO==2) for(int k=0;k<nord;k++){ PCI(PC_ORD);
                    o8row[k]=s8row_find(M->o8tab[k],TROWM,rh_o[k]); }
                else if(NIBO==1) for(int k=0;k<nord;k++){ PCN(PC_ORD,2);  /* 160 B = 2 lines */
                    orow[k]=srow_find(tab[k],TROWM,rh_o[k]); }
                if(NIB){
                    if(EN_WORD) for(int k=0;k<NWORDM;k++){ PCI(PC_WORD);
                        wrow[k]=crow_find(wtab[k],WROWM,rh_w[k]); }
                    for(int s=0;s<scn;s++){ PCI(PC_SCALE);
                        srow[s]=crow_find(stab[s],SCROWM,rh_s[s]); }
                    for(int x=0;x<nex;x++){ PCI(PC_SPARSE);
                        xrow[x]=crow_find(xtab[x],XROWM,rh_x[x]); }
                    if(do_rec) for(int k=0;k<NREC;k++){
                        if(!recvalid[k]){ rrow[k]=NULL; continue; }
                        PCI(PC_REC); rrow[k]=crow_find(rtab[k],RROWM,rh_r[k]); }
                }
            }

            Slot*cs[NORDMAX]; Slot8*cs8[NORDMAX]; Ctr*ics[NORDMAX];
            int nz=0;
            for(int k=0;k<nord;k++){
                uint16_t sp,spf,shist; int sn,n0,n1;
                if(NIBO>=2){
                    Slot8*s=o8row[k]+nc; cs8[k]=s; cs[k]=NULL;
                    sp=s->p; spf=s->pf; shist=s->hist; sn=s->n;
                    n0=(int)(s->n01>>4); n1=(int)(s->n01&15u);
                } else {
                    Slot*s;
                    if(NIBO==1) s=orow[k]+nc;
                    else { uint32_t h=chs[k]^cb; h^=h>>15; PCI(PC_ORD);
                           s = EN_STATE ? slot_find2(tab[k],h) : &tab[k][h&TMASK]; }
                    cs[k]=s; cs8[k]=NULL;
                    sp=s->p; spf=s->pf; shist=s->hist; sn=s->n; n0=s->n0; n1=s->n1;
                }
                inp[I_P+k]=STRETCH[(sp>>4)&4095]*IN_SCALE;
                if(EN_STATE){
                    inp[I_DUAL+k]=STRETCH[(spf>>4)&4095]*IN_SCALE;
                    int nb = sn<HDEPTH ? sn : HDEPTH;
                    int m  = (1<<nb)-1;
                    Ctr*ic = ind + ((size_t)k*8 + (size_t)(7-b))*HSZ + (size_t)(m + (shist & m));
                    ics[k]=ic; PCI(PC_IND);
                    inp[I_HIST+k]=STRETCH[(ic->p>>4)&4095]*IN_SCALE;
                    inp[I_CNT+k]=fstretch(((float)n1+0.35f)*RCPN[n0+n1])*CNTSC;
                }
                if(sn>=4) nz++;
            }
            /* --- match models --- */
            Ctr *mu[NMATCH]; int mexp[NMATCH];
            int mlenmax=0, mexpbit=0, bestlen=0, bestexp=0, anym=0;
            int nbits=7-b, coded = ctxbits & ((1<<nbits)-1);
            for(int j=0;j<NMATCH;j++){
                mu[j]=NULL; mexp[j]=0; inp[I_MATCH+j]=0.f;
                if(!mvalid[j]) continue;
                if((mp[j]>>(b+1)) != coded) continue;
                int ex=(mp[j]>>b)&1;
                int L=mlen[j], bucket;
                if(L<16) bucket=L;
                else { int lg=31-__builtin_clz((unsigned)L); bucket=12+lg; if(bucket>LENB-1)bucket=LENB-1; }
                Ctr*c = EN_STATE ? &M->mrel[j][bucket][nbits][mhis[j]&(NMH-1)]
                                 : &M->mrel[j][mlen[j]<LENB?mlen[j]:LENB-1][0][0];
                float st=STRETCH[(c->p>>4)&4095];
                inp[I_MATCH+j]=(ex? st : -st)*IN_SCALE; PCI(PC_SMALL);
                mu[j]=c; mexp[j]=ex;
                if(mlen[j]>mlenmax){ mlenmax=mlen[j]; mexpbit=ex; }
                if(mlen[j]>bestlen){ bestlen=mlen[j]; bestexp=ex; }
                anym=1;
            }
            /* --- run model --- */
            Ctr *ru=NULL; int rexp=0;
            if(EN_STATE){
                inp[I_RUN]=0.f;
                if(rvalid && (rp>>(b+1))==coded){
                    rexp=(rp>>b)&1;
                    Ctr*c=&M->rrel[rbucket][nbits][M->rhis&(NMH-1)];
                    float st=STRETCH[(c->p>>4)&4095];
                    inp[I_RUN]=(rexp? st : -st)*IN_SCALE; PCI(PC_SMALL);
                    ru=c;
                }
            }
            /* --- tolerant match --- */
            Ctr*tu=NULL; int texp=0;
            if(EN_TOL){
                inp[I_TOL]=0.f;
                if(tvalid && (tp>>(b+1))==(unsigned)coded){
                    texp=(tp>>b)&1;
                    Ctr*c=&M->trel[tbucket][tmisc][nbits];
                    float st=STRETCH[(c->p>>4)&4095];
                    inp[I_TOL]=(texp? st : -st)*IN_SCALE; PCI(PC_SMALL);
                    tu=c;
                }
            }
            /* --- LSH fuzzy retrieval --- */
            Ctr*fu=NULL; int fexp=0; Ctr*eu=NULL;
            if(EN_LSH){
                inp[I_FZ]=0.f; inp[I_FZ+1]=0.f;
                if(fbest>=0 && (fpred>>(b+1))==(unsigned)coded){
                    fexp=(fpred>>b)&1;
                    int vb=fvotes>MAXL?MAXL:fvotes;
                    Ctr*c=&M->frel[fsimb][vb][nbits][fmstate];
                    float st=STRETCH[(c->p>>4)&4095];
                    inp[I_FZ]=(fexp? st : -st)*IN_SCALE; PCI(PC_SMALL);
                    fu=c;
                }
                int n1=0,n0=0;
                for(int q=0;q<fncand;q++){
                    uint8_t v=D[fcp[q]];
                    if((v>>(b+1))!=(unsigned)coded) continue;
                    if((v>>b)&1) n1++; else n0++;
                }
                if(n1|n0){
                    if(n1>MAXL)n1=MAXL; if(n0>MAXL)n0=MAXL;
                    Ctr*c=&M->fens[n1][n0][nbits];
                    inp[I_FZ+1]=STRETCH[(c->p>>4)&4095]*IN_SCALE; PCI(PC_SMALL);
                    eu=c;
                }
            }
            /* --- coarse-scale ladder; with NIB the
               cell is node nc of the nibble row found at the boundary --- */
            Ctr*ssl[NSC];
            for(int s=0;s<scn;s++){
                if(NIB) ssl[s]=srow[s]+nc;
                else { uint32_t h=shs[s]^cb; h^=h>>15; PCI(PC_SCALE);
                       ssl[s]=&stab[s][h&SCMASK]; }
                inp[I_SCALE+s]=STRETCH[(ssl[s]->p>>4)&4095]*IN_SCALE;
            }
            /* --- word / byte-class, sparse / indirect, record / stride --- */
            Ctr*wsl[NWORDM]; Ctr*xs[MAXEX]; Ctr*rs[NREC];
            if(NIB){
                if(EN_WORD) for(int k=0;k<NWORDM;k++){
                    wsl[k]=wrow[k]+nc;
                    inp[I_WORD+k]=STRETCH[(wsl[k]->p>>4)&4095]*IN_SCALE;
                }
                for(int x=0;x<nex;x++){
                    xs[x]=xrow[x]+nc;
                    inp[I_SPARSE+x]=STRETCH[(xs[x]->p>>4)&4095]*IN_SCALE;
                }
                if(do_rec) for(int k=0;k<NREC;k++){
                    if(!recvalid[k]){ rs[k]=NULL; inp[I_REC+k]=0.f; continue; }
                    rs[k]=rrow[k]+nc;
                    inp[I_REC+k]=STRETCH[(rs[k]->p>>4)&4095]*IN_SCALE;
                }
            } else {
                if(EN_WORD) for(int k=0;k<NWORDM;k++){
                    uint32_t h=whs[k]^cb; h^=h>>15; PCI(PC_WORD);
                    wsl[k]=&wtab[k][h&WMASK];
                    inp[I_WORD+k]=STRETCH[(wsl[k]->p>>4)&4095]*IN_SCALE;
                }
                for(int x=0;x<nex;x++){
                    uint32_t h=xhs[x]^cb; h^=h>>15; PCI(PC_SPARSE);
                    xs[x]=&xtab[x][h&XMASK];
                    inp[I_SPARSE+x]=STRETCH[(xs[x]->p>>4)&4095]*IN_SCALE;
                }
                if(do_rec) for(int k=0;k<NREC;k++){
                    rs[k]=NULL;
                    if(!recvalid[k]){ inp[I_REC+k]=0.f; continue; }
                    uint32_t h=rhs[k]^cb; h^=h>>15; PCI(PC_REC);
                    rs[k]=&rtab[k][h&RMASK];
                    inp[I_REC+k]=STRETCH[(rs[k]->p>>4)&4095]*IN_SCALE;
                }
            }
            if(USE_NN){
                double den=trie[ctxbits];
                float pn = den>0 ? (float)(trie[2*ctxbits+1]/den) : 0.5f;
                inp[I_NN]=stretch_p(pn)*IN_SCALE;
            } else inp[I_NN]=0.f;
            inp[NM]=IN_SCALE;

            /* ---- selectors ---- */
            int lbk = mlenmax==0?0 : mlenmax<2?1 : mlenmax<4?2 : mlenmax<8?3 :
                      mlenmax<16?4 : mlenmax<32?5 : mlenmax<64?6 : 7;
            int bp = 7-b;
            /* SEL_ORDEST is 64 wide (nz*8+bp) and SEL_ORDMB 512 ((nz*8+lbk)*8+bp);
               both require nz<=7.  At nord=7 nz could never exceed 7, so the
               clamp is a no-op for the base model; at nord>7 it is what keeps
               the index in range. */
            int nzs = nz>7 ? 7 : nz;
            uint32_t sv[NSELK];
            sv[SEL_PREV]  = sPREV;
            sv[SEL_C0]    = (uint32_t)ctxbits & 255u;
            sv[SEL_MATCH] = (uint32_t)((lbk*2+mexpbit)*8 + bp);
            sv[SEL_TYPE]  = sTYPEbase*8u + (uint32_t)bp;
            sv[SEL_ORDEST]= (uint32_t)(nzs*8 + bp);
            sv[SEL_ORDMB] = (uint32_t)((nzs*8 + lbk)*8 + bp);
            sv[SEL_MSLB]  = sMSLB;
            { uint32_t h=h1^cb; h^=h>>15; sv[SEL_H1C0]=h&(SELSIZE[SEL_H1C0]-1); }
            { uint32_t h=h2^((uint32_t)ctxbits*0x165667B1u); h^=h>>15;
              sv[SEL_H2C0]=h&(SELSIZE[SEL_H2C0]-1); }
            { uint32_t h=h3s^((uint32_t)ctxbits*0x9E3779B1u); h^=h>>15;
              sv[SEL_H3C0]=h&(SELSIZE[SEL_H3C0]-1); }

            /* ---- level 1 ---- */
            float dot1[12]; float *WK[9];
            dot1[9]=dot1[10]=dot1[11]=0.f; PCN(PC_MIXROW,NL1);
            for(int k=0;k<NL1;k++){
                float*WL=W1[k]+(size_t)sv[BANK[k]]*NINP; WK[k]=WL;
                float d=mix_dot(WL,inp,NINP);
                if(d>18.f)d=18.f; if(d<-18.f)d=-18.f;
                dot1[k]=d;
            }
            /* ---- level 2 ---- */
            float in2[12]; float*W2L; float dotf;
            if(NL1>1){
                for(int k=0;k<NL1;k++) in2[k]=dot1[k]*L2_SCALE;
                in2[NL1]=L2_SCALE;
                W2L=W2+(size_t)sv[SEL_L2]*NIN2;
                float d=0.f; for(int t=0;t<NIN2;t++) d+=W2L[t]*in2[t];
                dotf=d;
            } else { W2L=NULL; dotf=dot1[0]; }
            float pmix=sig1(dotf);
            if(pmix<1e-6f)pmix=1e-6f; if(pmix>1.f-1e-6f)pmix=1.f-1e-6f;

            /* ---- SSE ---- */
            float pr=pmix;
            uint32_t iA=0,iB=0,iC=0,iD=0,iE=0; float wA=0,wB=0,wC=0,wD=0,wE=0;
            float s2v[NS2+2]; float*WS2=NULL; int ns2=0;
            int apm_idx=0, apm2_idx=0;
            if(SSE_MODE==0 || SSE_MODE==3){
                int cx=(((uint32_t)ctxbits&255u)*4u+(uint32_t)tb) & (APMCTX-1);
                float pa=apm_pp(M->apmt,dotf,cx,&apm_idx);
                pr=(1.f-APM_W)*pmix+APM_W*pa;
                int cx2=(uint32_t)((nz*8+lbk)*8+bp) & (APMCTX-1);
                float pa2=apm_pp(M->apm2t,stretch_p(pr),cx2,&apm2_idx);
                pr=(1.f-APM2_W)*pr+APM2_W*pa2;
            } else {
                s2v[ns2++]=fstretch(pmix)*IN_SCALE; PCN(PC_SSE,2);
                { float pa=apm_p(apmA,pr,(uint32_t)ctxbits,&iA,&wA);
                  pr=(1.f-BSSE)*pr+BSSE*pa; s2v[ns2++]=fstretch(pa)*IN_SCALE; }
                { uint32_t cx=(s2base+(uint32_t)ctxbits*0x2545F491u)&S2MASK;
                  float pa=apm_p(apmB,pr,cx,&iB,&wB);
                  pr=(1.f-BSSE)*pr+BSSE*pa; s2v[ns2++]=fstretch(pa)*IN_SCALE; }
                { uint32_t msx = anym ? (uint32_t)(1+2*lenbucket(bestlen)+bestexp) : 0u;
                  uint32_t cx = msx*256u+(uint32_t)ctxbits;
                  float pa=apm_p(apmC,pr,cx,&iC,&wC);
                  pr=(1.f-BSSE)*pr+BSSE*pa; s2v[ns2++]=fstretch(pa)*IN_SCALE; }
                { uint32_t cx=(s4base+(uint32_t)ctxbits*0x2545F491u)&S4MASK;
                  float pa=apm_p(apmD,pr,cx,&iD,&wD);
                  s2v[ns2++]=fstretch(pa)*IN_SCALE; }   /* last stage: feeds the mixer only */
                { int c2 = (MIX2CTX==0)?0:((MIX2CTX==1)?bp:((MIX2CTX==2)?(int)p[-1]:(lb8*8+ms8)));
                  WS2=M->w2lin[c2&255]; s2v[ns2]=IN_SCALE;
                  float d=0.f; for(int t=0;t<=ns2;t++) d+=WS2[t]*s2v[t];
                  pr=sig1(d); }
            }
            float pre=pr;
            if(SSE_MODE==2 || SSE_MODE==3){
                if(pre<1e-6f)pre=1e-6f; if(pre>1.f-1e-6f)pre=1.f-1e-6f;
                uint32_t cx=((uint32_t)ctxbits<<6)|((uint32_t)lb8<<3)|(uint32_t)ms8;
                float pa=apm_p(apmE,pre,cx,&iE,&wE);
                pr=SSEW3*pa+(1.f-SSEW3)*pre;
            }
            if(pr<1e-6f)pr=1e-6f; if(pr>1.f-1e-6f)pr=1.f-1e-6f;

            /* ================= THE ONLY ASYMMETRIC LINE ==================
               pr is identical on both sides, so the quantised p is too. */
            { const uint32_t pq = quant_p(pr);
              if(docode){
                  if(enc){ bit = (sym>>b)&1; enc_bit(C,bit,pq); }
                  else   { bit = dec_bit(C,pq); }
              } else bit = (sym>>b)&1;
            }
            /* ============================================================ */

            { float c=-flog2(bit?pr:1.f-pr);
              if(counting) bits+=(double)c;
              lossema += (c-lossema)*(1.f/48.f); }

            /* ---- SSE updates ---- */
            if(SSE_MODE==0 || SSE_MODE==3){
                float tgt=bit?65535.f:0.f;
                M->apmt[apm_idx]=(uint16_t)(M->apmt[apm_idx]+(tgt-M->apmt[apm_idx])*APM_RATE);
                M->apm2t[apm2_idx]=(uint16_t)(M->apm2t[apm2_idx]+(tgt-M->apm2t[apm2_idx])*APM_RATE);
            } else {
                apm_upd(apmA,iA,wA,bit); apm_upd(apmB,iB,wB,bit);
                apm_upd(apmC,iC,wC,bit); apm_upd(apmD,iD,wD,bit);
                float g2=pre-bit; for(int t=0;t<=ns2;t++) WS2[t]-=LR2SSE*g2*s2v[t];
            }
            if(SSE_MODE==2 || SSE_MODE==3) apm_upd(apmE,iE,wE,bit);

            /* ---- mixer updates ---- */
            if(NL1>1){
                float g2=pmix-bit, lr2=LR*LR2_MUL;
                for(int t=0;t<NIN2;t++) W2L[t]-=lr2*g2*in2[t];
            }
            if(NL1>1){
                float gv[12];
#ifdef HAVE_NEON
                if(USE_SIMD){
                    float32x4_t fb=vdupq_n_f32((float)bit);
                    for(int k=0;k<NL1;k+=4){
                        float32x4_t d=vld1q_f32(dot1+k);
                        vst1q_f32(gv+k, vsubq_f32(sig1v(d),fb));
                    }
                } else
#endif
                { for(int k=0;k<NL1;k++) gv[k]=sig1(dot1[k])-(float)bit; }
                for(int k=0;k<NL1;k++) mix_upd(WK[k],inp,LR*gv[k],NINP);
            } else {
                mix_upd(WK[0],inp,LR*(pmix-(float)bit),NINP);
            }

            /* ---- counter updates ---- */
            for(int k=0;k<nord;k++){
                if(EN_STATE){ ctr_update(ics[k],bit);
                    if(cs8[k]) slot8_update_rich(cs8[k],bit); else slot_update_rich(cs[k],bit); }
                else { if(cs8[k]) slot8_update_plain(cs8[k],bit); else slot_update_plain(cs[k],bit); }
            }
            for(int j=0;j<NMATCH;j++) if(mu[j]) ctr_update(mu[j], bit==mexp[j]);
            if(ru) ctr_update(ru, bit==rexp);
            if(tu) ctr_update(tu, bit==texp);
            if(fu) ctr_update(fu, bit==fexp);
            if(eu) ctr_update(eu, bit);
            for(int s=0;s<scn;s++) ctr_update(ssl[s],bit);
            if(EN_WORD) for(int k=0;k<NWORDM;k++) ctr_update(wsl[k],bit);
            for(int x=0;x<nex;x++) ctr_update(xs[x],bit);
            if(do_rec) for(int k=0;k<NREC;k++) if(rs[k]) ctr_update(rs[k],bit);
            ctxbits=(ctxbits<<1)|bit; nc=(nc<<1)|bit;
        }
#undef PFBIT
#undef ROWPF
#undef ROWHASH

        /* the decoder now knows the byte; publish it before the byte-level
           rolls, which read p[0] (via the match-model polynomial p[1-q]). */
        if(!enc && docode){ sym = (uint8_t)((uint32_t)ctxbits & 255u); *p = sym; }

        /* ---- byte-level state rolls ---- */
        if(EN_STATE){
            for(int j=0;j<NMATCH;j++)
                mhis[j]=(uint8_t)((mhis[j]<<1)|(mvalid[j] && mp[j]==sym));
            M->rhis=(uint8_t)((M->rhis<<1)|(i>M->pstart && rbyte==sym));
        }
        if(nex) ex_after_byte(M,sym);
        /* ---- ladder state roll ----
           Driven entirely by `sym`, which both sides now hold, so the two
           sides advance the ladder in lock-step. */
        if(scn){
            if(alnum(sym)){
                swh = swh*0x2F0FD693u + (fold(sym)+1u); if(!swh) swh=1;
                if(swlen<255) swlen++;
            } else {
                if(swh){
                    for(int j=7;j>0;j--) swq[j]=swq[j-1];
                    for(int j=3;j>0;j--) swlq[j]=swlq[j-1];
                    swq[0]=swh; swlq[0]=(uint8_t)swlen; swh=0; s_sentw++;
                }
                swlen=0;
            }
            s_sentpos++;
            if(sym=='.'||sym=='!'||sym=='?'||sym=='\n'){ s_sentpos=0; s_sentw=0; }
            if(sym=='\n'){ s_prevlinelen=s_linepos; s_linepos=0; } else s_linepos++;
            if(sym=='<'){ s_intag=1; s_tagh=0x11u; }
            else if(sym=='>'){ if(s_intag){ s_curtag=mix32w(s_tagh);
                                            if(s_nest<255)s_nest++; } s_intag=0; }
            else if(s_intag){ if(alnum(sym)) s_tagh=s_tagh*0x9E3779B1u+fold(sym);
                              else if(sym=='/'&&s_nest) s_nest--; }
            else if(sym=='{'||sym=='('||sym=='['){ if(s_nest<255)s_nest++; }
            else if(sym=='}'||sym==')'||sym==']'){ if(s_nest)s_nest--; }
        }
        prfl=(prfl<<1)|((sym>=32&&sym<127)||sym=='\n'||sym=='\t');

        { uint32_t Q=0; int nxt=0;
          uint32_t hj[NMATCH];
          for(int q=1;q<=MORD[NMATCH-1];q++){
            Q = Q*0x85EBCA77u + p[1-q];
            if(nxt<NMATCH && q==MORD[nxt]){ uint32_t h=MK[nxt]+Q; h^=h>>15; hj[nxt]=h&MMASK; nxt++; }
          }
          for(int j=0;j<NMATCH;j++){
            uint32_t h=hj[j];
            if(mvalid[j] && D[mptr[j]]==sym){
                mptr[j]++; if(mlen[j]<1<<20) mlen[j]++;
            } else {
                uint32_t c=mtab[j][h];
                if(c){ mptr[j]=c; mlen[j]=1; } else { mptr[j]=0; mlen[j]=0; }
            }
            mtab[j][h]=(uint32_t)(i+1);
          }
        }
        /* ---- tolerant-match index roll ----
           hashn(p+1,FZTORD,...) reads p[0..1-FZTORD], i.e. the byte just coded
           and FZTORD-1 before it: all known to both sides. */
        if(EN_TOL){
            uint32_t h=hashn(p+1,FZTORD,0x77u) & TTMASK;
            if(tvalid){
                int hit=(tp==sym);
                tmis=(uint8_t)((tmis<<1)|(!hit));
                tptr++;                            /* advance through misses */
                if(tlen<1<<20) tlen++;
                if(__builtin_popcount((unsigned)(tmis&0xFF))>=FZT) tlen=0;
            } else tmis=(uint8_t)(tmis<<1);
            if(tlen==0){
                uint32_t c=ttab[h];
                if(c){ tptr=c; tlen=1; tmis=0; } else { tptr=0; tlen=0; }
            }
            ttab[h]=(uint32_t)(i+1);
        }
        /* ---- LSH retrieval for the NEXT byte ----
           The table slot is READ before it is overwritten with (i+1), and any
           candidate with c>i is dropped, so every retrieved position is <= i.
           The similarity scan reads D[c-1-j] (j<FZK<=32, c>=1) and p[-j]
           (j<FZK), all at or below the byte just coded. */
        if(EN_LSH){
            fz_ncand=0;
            for(int t=0;t<FZL;t++){
                uint32_t h=(uint32_t)(t+1)*0x9E3779B1u;
                for(int j=0;j<FZR;j++) h=h*0x85EBCA77u+p[-lpos[t][j]];
                h^=h>>15; h&=LMASKB;
                uint32_t c=ltab[t][h];
                ltab[t][h]=(uint32_t)(i+1);
                if(!c || (size_t)c>i) continue;
                int dup=0; for(int q=0;q<fz_ncand;q++) if(fz_cpos[q]==c){dup=1;break;}
                if(dup) continue;
                const uint8_t*q1=D+c; int sim=0;
                for(int j=0;j<FZK;j++) sim += (q1[-1-j]==p[-j]);
                fz_cpos[fz_ncand]=c; fz_csim[fz_ncand]=sim; fz_ncand++;
            }
        }
        if(i>M->pstart && sym==rbyte){ if(rlen<(1<<20)) rlen++; } else { rbyte=sym; rlen=1; }

        if(EN_WORD){
            if(alnum(sym)){
                if(sym>='A'&&sym<='Z') wcase |= wlen?2:1;
                wh = wh*0x2F0FD693u + (fold(sym)+1u);
                if(!wh) wh=1;
                if(wlen<255) wlen++;
            } else {
                if(wh){ w3=w2; w2=w1; w1=wh; wh=0; }
                wlen=0; wcase=0;
            }
            clsh=(clsh<<3)|(uint32_t)BCLS[sym];
        }
    }
    M->bits = bits;
    if(enc) enc_flush(C);
}

static void* th_alloc(void*a){ model_alloc((Model*)a); return NULL; }
static void* th_run  (void*a){ code_block((Model*)a);  return NULL; }


/* Format version.  Bumped 1 -> 2 when the E8E9 transform behind header flag
 * bit1 (do_e8) changed from the screened, ascending, skip-4 filter to `e8p`
 * (see the E8E9 section above).  The LAYOUT is byte-for-byte identical and the
 * flag bit still carries exactly one bit of meaning -- "the E8E9 transform was
 * applied" -- so no new field was needed.  What changed is which transform that
 * bit names, and a v1 archive with bit1 set would decode to wrong bytes under a
 * v2 decoder.  The decoder already rejects any version it does not know, so the
 * bump turns that into a clean refusal.  Zero bytes added.
 *
 * Bumped 2 -> 3 when the byte-order count (`nord`), the multi-scale ladder
 * (`scmask`, `SCB0`, `LPOSCAP`) and fuzzy retrieval (`FUZZ`, `FTB0`, `FLB0`)
 * were integrated.  These add 8 bytes to the header and change the model for
 * any archive that uses them; a v2 archive carries none of the fields, so the
 * decoder must refuse it rather than guess.  Eight bytes on a header that is
 * already >= 114 bytes, once per archive. */
/* Bumped 3 -> 4 when the nibble-row table layout was integrated:
 * two header bytes at offset 80 (nib, nibo); exid[] moves to 82.  The layout
 * changes where every hashed counter lives, so a v3 archive must be refused
 * rather than decoded under a guessed layout. */
#define CMXC_VER 4u

/* ==================================================== container format ====
 * Everything the decoder needs to configure an IDENTICAL model lives here.
 * A config mismatch would corrupt silently, so nothing is inferred: the
 * detector outputs (do_e8, do_rec, R, R2), the table bit-widths (which depend
 * on the block count), the SIMD flag (NEON changes the float summation order,
 * hence the results) and every model switch are all recorded explicitly.
 *
 *   0  4  magic "CMXC"
 *   4  2  version
 *   6  2  flags   bit0 use_simd   bit1 do_e8
 *             bit0 use_simd : the mixer kernel the ENCODER actually ran (NEON
 *                 vs scalar; they differ in float summation order).  Since
 *                 format v2 the decoder REFUSES to decode when this says NEON
 *                 and the decoder has no NEON kernel -- see the guard in
 *                 decompress_file().
 *             bit1 do_e8    : the E8E9 transform was applied to the whole
 *                 buffer before block splitting.  One bit is still enough:
 *                 there is exactly one E8E9 transform in the format.  Since
 *                 v2 that transform is `e8p` (provably invertible); in v1 it
 *                 was the screened filter, which is why the version moved.
 *   8  8  orig_len
 *  16  4  nblocks
 *  20  4  hdrlen
 *  24  8  prime
 *  32  4  R          (record model period, 0 if unused)
 *  36  4  R2
 *  40  8  en_state,en_word,en_sparse,do_rec,sse_mode,nl1,mix2ctx,nex
 *  48  8  TB0,WB0,MB0,RB0,XB0,IB0,S2B0,S4B0
 *  56 16  LR, LR2_MUL, WINIT, SSEW3        (f32)
 *  72  8  nord, scmask, FUZZ, SCB0, FTB0, FLB0, LPOSCAP(u16)
 *  80  2  nib, nibo            (nibble-row layout switches)
 *  82 nex exid[]
 *   +  16*nblocks   (compressed_len u64, raw_len u64)
 *   =  hdrlen, then the block streams back to back in block order
 * ========================================================================= */
typedef struct { uint8_t *b; size_t n, cap; } Buf;
static void bput(Buf*B,const void*p,size_t k){
    while(B->n+k > B->cap){
        size_t nc = B->cap? B->cap*2 : 4096;
        uint8_t*nb=(uint8_t*)realloc(B->b,nc);
        if(!nb){fprintf(stderr,"cmxc: oom\n");exit(1);}
        B->b=nb; B->cap=nc;
    }
    memcpy(B->b+B->n,p,k); B->n+=k;
}
static void bu8 (Buf*B,uint32_t v){ uint8_t t=(uint8_t)v; bput(B,&t,1); }
static void bu16(Buf*B,uint32_t v){ uint8_t t[2]={(uint8_t)v,(uint8_t)(v>>8)}; bput(B,t,2); }
static void bu32(Buf*B,uint32_t v){ uint8_t t[4]={(uint8_t)v,(uint8_t)(v>>8),
                                                  (uint8_t)(v>>16),(uint8_t)(v>>24)}; bput(B,t,4); }
static void bu64(Buf*B,uint64_t v){ bu32(B,(uint32_t)v); bu32(B,(uint32_t)(v>>32)); }
static void bf32(Buf*B,float f){ uint32_t u; memcpy(&u,&f,4); bu32(B,u); }

typedef struct { const uint8_t*b; size_t n, p; int bad; } Rdr;
static uint32_t ru8 (Rdr*R){ if(R->p+1>R->n){R->bad=1;return 0;} return R->b[R->p++]; }
static uint32_t ru16(Rdr*R){ uint32_t a=ru8(R),b=ru8(R); return a|(b<<8); }
static uint32_t ru32(Rdr*R){ uint32_t a=ru16(R),b=ru16(R); return a|(b<<16); }
static uint64_t ru64(Rdr*R){ uint64_t a=ru32(R),b=ru32(R); return a|(b<<32); }
static float    rf32(Rdr*R){ uint32_t u=ru32(R); float f; memcpy(&f,&u,4); return f; }

/* ------------------------------------------------- shared configuration --- */
/* derived from the switches; MUST be called identically on both sides */
static void layout_inputs(void){
    /* Input order:
         ... -> I_REC -> I_SCALE -> I_NN -> I_TOL -> I_FZ -> bias
       which reduces to the base layout when scmask==0 and FUZZ==0. */
    I_P=0;
    I_DUAL  = I_P + nord;
    I_HIST  = I_DUAL + (EN_STATE?nord:0);
    I_CNT   = I_HIST + (EN_STATE?nord:0);
    I_MATCH = I_CNT  + (EN_STATE?nord:0);
    I_RUN   = I_MATCH + NMATCH;
    I_WORD  = I_RUN  + (EN_STATE?1:0);
    I_SPARSE= I_WORD + (EN_WORD?NWORDM:0);
    I_REC   = I_SPARSE + (EN_SPARSE?nex:0);
    I_SCALE = I_REC  + (do_rec?NREC:0);
    I_NN    = I_SCALE + scn;
    I_TOL   = I_NN + 1;
    I_FZ    = I_TOL + (EN_TOL?1:0);
    NM      = I_FZ + (EN_LSH?2:0);
    NIN     = NM + 1;
    NINP    = (NIN + 15) & ~15;
    NIN2    = NL1 + 1;
    if(NINP>MAXIN){fprintf(stderr,"cmxc: too many inputs %d\n",NIN);exit(1);}
}
static void set_table_sizes(void){
    TSIZE=1u<<TB0; TMASK=TSIZE-1;
    WSIZE=1u<<WB0; WMASK=WSIZE-1;
    MSIZE=1u<<MB0; MMASK=MSIZE-1;
    RSIZE=1u<<RB0; RMASK=RSIZE-1;
    XSIZE=1u<<XB0; XMASK=XSIZE-1;
    ISIZE=1u<<IB0; IMASK=ISIZE-1;
    S2SIZE=1u<<S2B0; S2MASK=S2SIZE-1;
    S4SIZE=1u<<S4B0; S4MASK=S4SIZE-1;
    SCSIZE=1u<<SCB0; SCMASK=SCSIZE-1;
    TTSIZE=1u<<FTB0; TTMASK=TTSIZE-1;
    LSIZE =1u<<FLB0; LMASKB=LSIZE-1;
    /* nibble-row masks.  Every width is >= 2^14 cells (the MEMSH
       floor), so every table holds >= 512 sets. */
    WROWM=WSIZE/RSET-1u; SCROWM=SCSIZE/RSET-1u; XROWM=XSIZE/RSET-1u; RROWM=RSIZE/RSET-1u;
    TROWM=TSIZE/16u-1u;  TSETM=TSIZE/32u-1u;
}
/* derive scn/scid and EN_TOL/EN_LSH -- MUST be called identically on both
   sides, from the header-carried scmask/FUZZ, before layout_inputs(). */
static void derive_mech(void){
    scn=0;
    for(int k=0;k<NSC;k++) if(scmask&(1<<k)) scid[scn++]=k;
    EN_TOL=(FUZZ&1)!=0; EN_LSH=(FUZZ&2)!=0;
    if(nord<1) nord=1; if(nord>NORDMAX) nord=NORDMAX;
    if(LPOSCAP<1) LPOSCAP=1;
}
static void init_math_tables(void){
    for(int i=0;i<4096;i++){ float p=(i+0.5f)/4096.f; STRETCH[i]=logf(p/(1.f-p)); }
    for(int i=0;i<256;i++) BCLS[i]=(uint8_t)bcls((uint8_t)i);
    init_rates();
    init_lpos();
}
static void model_free(Model*M){
    for(int k=0;k<NORDMAX;k++){ free(M->tab[k]); free(M->o8tab[k]); }
    free(M->ind);
    for(int j=0;j<NMATCH;j++) free(M->mtab[j]);
    for(int k=0;k<NWORDM;k++) free(M->wtab[k]);
    for(int x=0;x<MAXEX;x++){ free(M->xtab[x]); free(M->ihist[x]); }
    for(int k=0;k<NREC;k++) free(M->rtab[k]);
    for(int s=0;s<NSC;s++) free(M->stab[s]);
    free(M->ttab);
    for(int t=0;t<MAXL;t++) free(M->ltab[t]);
    for(int k=0;k<9;k++) free(M->W1[k]);
    free(M->W2); free(M->apmt); free(M->apm2t);
    free(M->apmA.c); free(M->apmB.c); free(M->apmC.c); free(M->apmD.c); free(M->apmE.c);
}

/* ------------------------------------------------------------- blocks ----- */
typedef struct {
    size_t   abs_start, len, prime;
    uint8_t *cbuf; size_t clen;   /* compressed stream (owned)               */
    double   bits;                /* model's own -log2 p sum, for the report  */
} Blk;

/* Run `cnt` blocks starting at B[first] concurrently.  enc==1 fills cbuf/clen
   from file[]; enc==0 fills file[] from cbuf/clen.  Blocks are completely
   independent: each gets a private buffer whose first 64 bytes are zero and
   whose indices start at 0, so no block can observe another block's bytes
   (unless prime>0, in which case the caller must serialise). */
/* Run fn(&MM[t]) for every t, one thread each.  If a thread cannot be
   created (e.g. the process thread limit is reached), that model's work is
   done on the calling thread instead, so every model is always processed. */
static void for_each_model(Model*MM,int cnt,void*(*fn)(void*),
                           pthread_t*th,unsigned char*started)
{
    for(int t=0;t<cnt;t++){
        started[t] = (pthread_create(&th[t],NULL,fn,&MM[t])==0);
        if(!started[t]) fn(&MM[t]);
    }
    for(int t=0;t<cnt;t++)
        if(started[t] && pthread_join(th[t],NULL)){
            fprintf(stderr,"cmxc: pthread_join failed\n"); exit(1);
        }
}

static void run_wave(Blk*B,int first,int cnt,uint8_t*file,int enc)
{
    /* Model has a 64-byte-aligned member, so the array must be 64-byte
       aligned; sizeof(Model) is a multiple of 64, as aligned_alloc requires. */
    Model *MM = (Model*)aligned_alloc(64, sizeof(Model)*(size_t)cnt);
    pthread_t *th = (pthread_t*)malloc(sizeof(pthread_t)*(size_t)cnt);
    unsigned char *started = (unsigned char*)malloc((size_t)cnt);
    if(!MM||!th||!started){fprintf(stderr,"cmxc: oom\n");exit(1);}

    for_each_model(MM,cnt,th_alloc,th,started);

    for(int t=0;t<cnt;t++){
        Blk *b = &B[first+t];
        Model *M = &MM[t];
        size_t need = 64 + b->prime + b->len;
        M->blkbuf = (uint8_t*)calloc(need?need:1,1);
        if(!M->blkbuf){fprintf(stderr,"cmxc: oom (block buffer)\n");exit(1);}
        M->blkcap = need;
        M->D      = M->blkbuf + 64;
        M->pstart = 0;
        M->begin  = b->prime;
        M->end    = b->prime + b->len;
        M->abs_start = b->abs_start;
        M->bits   = 0;
        M->enc    = enc;
        memset(&M->cod,0,sizeof(M->cod));
        if(enc){
            memcpy(M->D, file + b->abs_start - b->prime, b->prime + b->len);
        } else {
            if(b->prime) memcpy(M->D, file + b->abs_start - b->prime, b->prime);
            M->cod.in    = b->cbuf;
            M->cod.inlen = b->clen;
            M->cod.inpos = 0;
        }
    }

    if(cnt==1) code_block(&MM[0]);
    else for_each_model(MM,cnt,th_run,th,started);

    for(int t=0;t<cnt;t++){
        Blk *b = &B[first+t];
        Model *M = &MM[t];
        b->bits = M->bits;
        if(enc){ b->cbuf = M->cod.buf; b->clen = M->cod.n; }
        else   { memcpy(file + b->abs_start, M->D + b->prime, b->len); }
        free(M->blkbuf);
        model_free(M);
    }
    free(MM); free(th); free(started);
}

/* --------------------------------------------------------------- io ------- */
static uint8_t* slurp(const char*path,size_t*outn,size_t lead){
    FILE*f=fopen(path,"rb"); if(!f){perror(path);exit(1);}
    if(fseeko(f,0,SEEK_END)){perror("seek");exit(1);}
    off_t sz=ftello(f); if(sz<0){perror("tell");exit(1);}
    if(fseeko(f,0,SEEK_SET)){perror("seek");exit(1);}
    if((uint64_t)sz > (uint64_t)(SIZE_MAX-64-lead)){fprintf(stderr,"cmxc: %s too large\n",path);exit(1);}
    size_t n=(size_t)sz;
    uint8_t*b=(uint8_t*)calloc(lead+n+64,1);
    if(!b){fprintf(stderr,"cmxc: oom reading %s\n",path);exit(1);}
    if(n && fread(b+lead,1,n,f)!=n){fprintf(stderr,"cmxc: short read %s\n",path);exit(1);}
    fclose(f); *outn=n; return b;
}
static void spew(const char*path,const uint8_t*b,size_t n){
    FILE*f=fopen(path,"wb"); if(!f){perror(path);exit(1);}
    if(n && fwrite(b,1,n,f)!=n){fprintf(stderr,"cmxc: short write %s\n",path);exit(1);}
    if(fclose(f)){perror("close");exit(1);}
}

/* -------------------------------------------------------- env / config ---- */
static int  CFG_THREADS = 8;
static size_t CFG_PRIME = 0;      /* 0 keeps the blocks independent          */
static int  CFG_MEMSH   = -1;
static void read_env(void){
    const char*e;
    if((e=getenv("CMXC_THREADS")) && *e){ CFG_THREADS=atoi(e);
        if(CFG_THREADS<1)CFG_THREADS=1; if(CFG_THREADS>64)CFG_THREADS=64; }
    if((e=getenv("CMXC_PRIME")) && *e)  CFG_PRIME=strtoull(e,0,10);
    if((e=getenv("CMXC_MEM"))   && *e)  CFG_MEMSH=atoi(e);
    if((e=getenv("CMXC_PF"))    && *e)  PFETCH=atoi(e);
    if((e=getenv("CMXC_SIMD"))  && *e)  USE_SIMD=atoi(e);
    if((e=getenv("CMXC_PFD"))   && *e){ PFD=atoi(e); if(PFD<1)PFD=1; if(PFD>4)PFD=4; }
    if((e=getenv("CMXC_VERBOSE"))&& *e) VERB=atoi(e);
    /* model switches (encoder side only -- they are recorded in the header) */
    if((e=getenv("CMX_STATE")) && *e)  EN_STATE=atoi(e);
    if((e=getenv("CMX_WORD"))  && *e)  EN_WORD=atoi(e);
    if((e=getenv("CMX_SPARSE"))&& *e)  EN_SPARSE=atoi(e);
    if((e=getenv("CMX_REC"))   && *e)  OPT_REC=strcmp(e,"auto")?atoi(e):2;
    if((e=getenv("CMX_E8"))    && *e)  OPT_E8=strcmp(e,"auto")?atoi(e):2;
    if((e=getenv("CMX_SSE"))   && *e){ SSE_MODE=atoi(e);       /* 0..3: the APM chains */
        if(SSE_MODE<0)SSE_MODE=0; if(SSE_MODE>3)SSE_MODE=3; }
    if((e=getenv("CMX_BANK"))  && *e){ NL1=atoi(e); if(NL1<1)NL1=1; if(NL1>9)NL1=9; }
    if((e=getenv("CMX_L2LR"))  && *e)  LR2_MUL=atof(e);
    if((e=getenv("CMX_W0"))    && *e)  WINIT=atof(e);
    if((e=getenv("CMX_MIX2CTX"))&&*e)  MIX2CTX=atoi(e);
    if((e=getenv("CMX_SSEW3")) && *e)  SSEW3=atof(e);
    if((e=getenv("CMX_LR"))    && *e)  LR=atof(e);
    /* ---- the three integrated mechanisms (encoder side; header-recorded) -- */
    if((e=getenv("CMX_NORD"))  && *e){ nord=atoi(e);
        if(nord<1)nord=1; if(nord>NORDMAX)nord=NORDMAX; }
    if((e=getenv("CMXS_MASK")) && *e)  scmask=(int)strtol(e,0,0);
    if((e=getenv("CMXS"))      && *e && atoi(e)==0) scmask=0;
    if(scmask<0) scmask=0; if(scmask>=(1<<NSC)) scmask=(1<<NSC)-1;
    if((e=getenv("CMX_FUZZ"))  && *e){ FUZZ=atoi(e);
        if(FUZZ<0)FUZZ=0; if(FUZZ>3)FUZZ=3; }
    if((e=getenv("CMX_LPOSCAP"))&&*e){ LPOSCAP=atoi(e); if(LPOSCAP<1)LPOSCAP=1;
        if(LPOSCAP>65535)LPOSCAP=65535; }
    /* nibble-row layout (encoder side; header-recorded) */
    if((e=getenv("CMX_NIB"))   && *e){ NIB=atoi(e)!=0; }
    if((e=getenv("CMX_NIBO"))  && *e){ NIBO=atoi(e); if(NIBO<0)NIBO=0; if(NIBO>3)NIBO=3; }
    USE_NN=0;                        /* encoder-asymmetric: disabled */
#ifndef HAVE_NEON
    /* No NEON kernel was compiled in, so USE_SIMD cannot be honoured.  Force it
     * to 0 so the header flag records the kernel that was ACTUALLY used rather
     * than the one that was requested.  Without this a non-NEON build would
     * stamp use_simd=1 on an archive it produced with the scalar kernel, and a
     * NEON decoder would then decode it with NEON -- the same silent corruption
     * in the other direction.  On a NEON build this is a no-op. */
    USE_SIMD = 0;
#endif
}

/* ------------------------------------------------- E8E9 selection probe ---
 * WHY A PROBE AND NOT A THRESHOLD.  `e8e9_score` measures the density of
 * screened E8/E9 call sites.  That is a property of the OPCODE STREAM, not a
 * prediction of what the transform does to the model's code length.  Because
 * e8p also rewrites stray E8/E9 bytes, on some executables it is a net loss
 * (about +1.5%) even though the score fires, while on others it is a large
 * gain (about -10%).  A density threshold cannot separate those cases; a
 * measurement can.
 *
 * WHAT THIS DOES.  Code the first 256 KB of the input twice -- once
 * untransformed, once through e8p -- with the real model, and keep whichever
 * the model's own code length says is smaller.  This is the prefix-probe
 * selector evaluated in paper §6.  The decision needs no new header
 * field: it is recorded in the do_e8 flag bit that already exists.
 *
 * COST: two extra short model runs at ENCODE time, only on inputs whose E8/E9
 * density clears the cheap pre-filter.  Decode is untouched.  No format change.
 */
#define E8_PROBE_BYTES 262144u
/* Accept e8p only if it wins by more than 0.1%.  Ties and wins within the
 * probe's noise go to identity; the margins that matter are several percent,
 * so this deadband only suppresses noise. */
#define E8_PROBE_MARGIN 0.001

/* Code one probe arm through the real model, mirroring the full pipeline
   (per-arm record detection, because the real encoder also detects on the
   FILTERED data).  Returns the model's own -log2 p sum in bits. */
static double e8_probe_arm(uint8_t*P,size_t pn)
{
    double rsc=0,rch=0,rmd=0;
    if(OPT_REC){
        R = detect_record(P,pn,&rsc,&rch,&rmd);
        do_rec = (OPT_REC==2) ? (R>=MINLAG && rsc>=0.15 && rch>=1.6 && rmd>=1.15) : 1;
        R2 = R2OUT; if(R2==R) R2=0;
        if(R<MINLAG) do_rec=0;
    }
    if(!do_rec){ R=0; R2=0; }
    layout_inputs();

    Blk pb; memset(&pb,0,sizeof pb);
    pb.abs_start=0; pb.len=pn; pb.prime=0;
    run_wave(&pb,0,1,P,1);
    double bits = pb.bits;
    free(pb.cbuf);
    return bits;
}

/* Returns 1 if e8p should be applied.  Leaves R/R2/do_rec/NM/NINP clobbered --
   the caller re-runs detection and layout on the real buffer afterwards. */
static int e8_probe(const uint8_t*D,size_t want,double*out_id,double*out_e8)
{
    size_t pn = want < (size_t)E8_PROBE_BYTES ? want : (size_t)E8_PROBE_BYTES;
    uint8_t *P = (uint8_t*)malloc(pn?pn:1);
    if(!P){fprintf(stderr,"cmxc: oom (e8 probe)\n");exit(1);}

    memcpy(P,D,pn);
    double bits_id = e8_probe_arm(P,pn);

    /* e8p on a prefix equals e8p on the whole file restricted to that prefix,
       except for the last 4 positions, whose operands run off the end.  At
       256 KB that is 4 bytes in 262,144 -- immaterial to the comparison. */
    memcpy(P,D,pn);
    e8p_forward(P,pn);
    double bits_e8 = e8_probe_arm(P,pn);

    free(P);
    *out_id=bits_id; *out_e8=bits_e8;
    return bits_e8 < bits_id*(1.0-E8_PROBE_MARGIN);
}

/* ============================================================ compress === */
static int compress_file(const char*inp,const char*outp)
{
    size_t want; uint8_t*buf=slurp(inp,&want,64);
    uint8_t*D=buf+64;

    /* ---- choose the block plan first: the table widths depend on it ---- */
    int NT = CFG_THREADS;
    /* Block splitting costs ratio because each block cold-starts its context tables.
     * Measured on dickens: a block of 64 KB codes at 2.5246 bpb, 256 KB at 2.1612,
     * 1 MB at 1.9326, 2 MB at 1.9293 -- i.e. the models need ~1 MB to reach steady
     * state.  The old floor of 32 KB let 8 threads shred a 200 KB input into 25 KB
     * blocks and lose 19.6% ratio.  Keep blocks >= MINBLK and drop threads instead. */
    { size_t minblk = 1u<<20;
      const char *e = getenv("CMXC_MINBLK");
      if(e && *e) minblk = strtoull(e,0,10);
      if(minblk < 4096) minblk = 4096;
      while(NT>1 && want/(size_t)NT < minblk) NT--; }
    if(want==0) NT=0;

    int sh=0;
    if(CFG_MEMSH>=0) sh=CFG_MEMSH;
    else if(NT>1){ int t=NT; while(t>1){ sh++; t>>=1; } sh+=2; }
    MEMSH=sh;
    TB0-=sh; WB0-=sh; MB0-=sh; RB0-=sh; XB0-=sh; IB0-=sh; S2B0-=sh; S4B0-=sh;
    SCB0-=sh; FTB0-=sh; FLB0-=sh;        /* new tables shrink like the rest */
    if(TB0<WIDTH_MIN)TB0=WIDTH_MIN; if(WB0<WIDTH_MIN)WB0=WIDTH_MIN; if(MB0<WIDTH_MIN)MB0=WIDTH_MIN;
    if(RB0<WIDTH_MIN)RB0=WIDTH_MIN; if(XB0<WIDTH_MIN)XB0=WIDTH_MIN; if(IB0<WIDTH_MIN)IB0=WIDTH_MIN;
    if(S2B0<WIDTH_MIN)S2B0=WIDTH_MIN; if(S4B0<WIDTH_MIN)S4B0=WIDTH_MIN; if(SCB0<WIDTH_MIN)SCB0=WIDTH_MIN;
    if(FTB0<WIDTH_MIN)FTB0=WIDTH_MIN; if(FLB0<WIDTH_MIN)FLB0=WIDTH_MIN;
    set_table_sizes();
    derive_mech();

    /* ---- sparse model set ---- */
    if(EN_SPARSE){
        const char*q=getenv("CMX_EX");
        if(q && *q){ if(strcmp(q,"none")==0) nex=0;
            else { while(*q&&nex<MAXEX){ char*end; long v=strtol(q,&end,10);
                     if(end==q)break; exid[nex++]=(int)v; q=end; while(*q==','||*q==' ')q++; } } }
        else { nex=(int)(sizeof(DEF_EX)/sizeof(DEF_EX[0])); if(nex>MAXEX)nex=MAXEX;
               for(int i=0;i<nex;i++) exid[i]=DEF_EX[i]; }
    }

    /* code_block() needs these; the probe below codes real data, so build them
       now rather than after the filter decision.  Both calls are idempotent. */
    init_math_tables();

    /* ---- E8E9 filter (e8p), with a LOUD invertibility assertion ----
     * e8p is invertible by construction, so this check is an
     * assertion, not a gate: it must never fail, and if it ever does we abort
     * rather than silently shipping a filter-less (but still correct) archive.
     * Set CMXC_E8CHECK=0 to skip the assertion (it costs 2n bytes and two
     * extra passes at encode time; it changes no output byte either way). */
    double e8s = e8e9_score(D,want);
    double pbits_id=0, pbits_e8=0; int e8probed=0;
    int do_e8;
    if(OPT_E8==2){
        /* Cheap pre-filter FIRST.  It is a SCREEN, not the decision: a file
         * with essentially no screened E8/E9 call sites has nothing for the
         * filter to gain, so there is no point paying for a probe (mozilla
         * scores 0.000071 and is correctly skipped for free).  It is safe in
         * this role because it can only skip files where the transform has
         * almost nothing to act on.  Everything that clears it is MEASURED. */
        if(e8s < 0.0025 || want < 5) do_e8 = 0;
        else { do_e8 = e8_probe(D,want,&pbits_id,&pbits_e8); e8probed=1; }
    } else {
        do_e8 = (OPT_E8!=0);      /* CMX_E8=0/1 forces, bypassing the probe */
    }
    if(do_e8 && want>=5){
        int e8chk = 1;
        { const char*e=getenv("CMXC_E8CHECK"); if(e&&*e) e8chk=atoi(e); }
        if(e8chk){
            uint8_t*save=(uint8_t*)malloc(want);
            uint8_t*chk =(uint8_t*)malloc(want);
            if(!save||!chk){fprintf(stderr,"cmxc: oom (e8p check)\n");exit(1);}
            memcpy(save,D,want);
            e8p_forward(D,want);
            memcpy(chk,D,want);
            e8p_backward(chk,want);
            if(memcmp(chk,save,want)!=0){
                size_t j=0; while(j<want && chk[j]==save[j]) j++;
                fprintf(stderr,
                  "cmxc: FATAL -- the e8p E8E9 filter is not self-inverse on this input.\n"
                  "      This cannot happen: e8p is invertible by construction (see the\n"
                  "      E8E9 section of cmxc.c).  Its failure means the source or the\n"
                  "      build is broken, not that this file is unusual.  Refusing to\n"
                  "      produce an archive.\n"
                  "      first differing byte at offset %zu of %zu\n", j, want);
                abort();
            }
            free(save); free(chk);
        } else {
            e8p_forward(D,want);
        }
    } else do_e8=0;

    /* ---- record / stride detection (on the FILTERED data) ----------------- */
    double rsc=0,rch=0,rmd=0;
    if(OPT_REC){
        R = detect_record(D,want,&rsc,&rch,&rmd);
        do_rec = (OPT_REC==2) ? (R>=MINLAG && rsc>=0.15 && rch>=1.6 && rmd>=1.15) : 1;
        R2 = R2OUT; if(R2==R) R2=0;
        if(R<MINLAG) do_rec=0;
    }
    if(!do_rec){ R=0; R2=0; }

    layout_inputs();
    init_math_tables();

    /* ---- block table ---- */
    Blk *B = NT? (Blk*)calloc((size_t)NT,sizeof(Blk)) : NULL;
    for(int t=0;t<NT;t++){
        size_t bg = want*(size_t)t/(size_t)NT;
        size_t en = want*(size_t)(t+1)/(size_t)NT;
        B[t].abs_start = bg;
        B[t].len       = en-bg;
        B[t].prime     = CFG_PRIME < bg ? CFG_PRIME : bg;
    }

    if(VERB && e8probed)
        fprintf(stderr,"[cmxc c] e8 probe %u B: id=%.0f bits  e8p=%.0f bits  "
                "(%+.3f%%) -> %s\n", (unsigned)(want<(size_t)E8_PROBE_BYTES?want:(size_t)E8_PROBE_BYTES),
                pbits_id, pbits_e8,
                pbits_id>0? (pbits_e8/pbits_id-1.0)*100.0 : 0.0,
                do_e8?"e8p":"identity");
    if(VERB) fprintf(stderr,"[cmxc c] n=%zu nblk=%d prime=%zu memsh=%d e8=%d(%.5f) "
                     "rec=%d(R=%d R2=%d) sse=%d bank=%d nord=%d scmask=0x%02x fuzz=%d "
                     "lposcap=%d nib=%d nibo=%d NM=%d NINP=%d lr=%.4f simd=%d\n",
                     want,NT,CFG_PRIME,MEMSH,do_e8,e8s,do_rec,R,R2,SSE_MODE,NL1,
                     nord,scmask,FUZZ,LPOSCAP,NIB,NIBO,NM,NINP,LR,USE_SIMD);

#ifdef PROBE
    memset(PC,0,sizeof PC);     /* drop the e8 probe's runs, if any */
#endif
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    if(NT>0) run_wave(B,0,NT,D,1);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double sec=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
#ifdef PROBE
    { double big=0, n=(double)(want?want:1);
      fprintf(stderr,"--- probes per byte (n=%zu T=%d nib=%d nibo=%d nord=%d nex=%d rec=%d) ---\n",
              want,NT,NIB,NIBO,nord,nex,do_rec);
      for(int c=0;c<PC_NC;c++){
          fprintf(stderr,"PROBE %-40s %10.3f %s\n",PCNAME[c],PC[c]/n,PC_BIG[c]?"LARGE":"");
          if(PC_BIG[c]) big+=PC[c]/n; }
      fprintf(stderr,"PROBE %-40s %10.3f\n","TOTAL LARGE",big); }
#endif

    /* ---- header ---- */
    Buf H={0,0,0};
    bput(&H,"CMXC",4);
    bu16(&H,CMXC_VER);
    bu16(&H,(uint32_t)((USE_SIMD?1:0)|(do_e8?2:0)));
    bu64(&H,(uint64_t)want);
    bu32(&H,(uint32_t)NT);
    bu32(&H,0);                       /* hdrlen, patched below (offset 20) */
    bu64(&H,(uint64_t)CFG_PRIME);
    bu32(&H,(uint32_t)R);
    bu32(&H,(uint32_t)R2);
    bu8(&H,EN_STATE); bu8(&H,EN_WORD); bu8(&H,EN_SPARSE); bu8(&H,do_rec);
    bu8(&H,SSE_MODE); bu8(&H,NL1);     bu8(&H,(uint32_t)(MIX2CTX&0xFF)); bu8(&H,nex);
    bu8(&H,TB0); bu8(&H,WB0); bu8(&H,MB0); bu8(&H,RB0);
    bu8(&H,XB0); bu8(&H,IB0); bu8(&H,S2B0); bu8(&H,S4B0);
    bf32(&H,LR); bf32(&H,LR2_MUL); bf32(&H,WINIT); bf32(&H,SSEW3);
    /* the three integrated mechanisms: nothing about them is re-derivable on
       the decode side, so every parameter that shapes the model is written */
    bu8(&H,(uint32_t)nord); bu8(&H,(uint32_t)scmask); bu8(&H,(uint32_t)FUZZ);
    bu8(&H,(uint32_t)SCB0);  bu8(&H,(uint32_t)FTB0);   bu8(&H,(uint32_t)FLB0);
    bu16(&H,(uint32_t)LPOSCAP);
    bu8(&H,(uint32_t)NIB); bu8(&H,(uint32_t)NIBO);      /* layout */
    for(int i=0;i<nex;i++) bu8(&H,(uint32_t)exid[i]);
    for(int t=0;t<NT;t++){ bu64(&H,(uint64_t)B[t].clen); bu64(&H,(uint64_t)B[t].len); }
    uint32_t hdrlen=(uint32_t)H.n;
    H.b[20]=(uint8_t)hdrlen; H.b[21]=(uint8_t)(hdrlen>>8);
    H.b[22]=(uint8_t)(hdrlen>>16); H.b[23]=(uint8_t)(hdrlen>>24);

    size_t total=hdrlen; for(int t=0;t<NT;t++) total+=B[t].clen;
    uint8_t*out=(uint8_t*)malloc(total?total:1);
    if(!out){fprintf(stderr,"cmxc: oom (output)\n");exit(1);}
    memcpy(out,H.b,hdrlen);
    size_t o=hdrlen;
    for(int t=0;t<NT;t++){ if(B[t].clen) memcpy(out+o,B[t].cbuf,B[t].clen); o+=B[t].clen; }
    spew(outp,out,total);

    double predbits=0; for(int t=0;t<NT;t++) predbits+=B[t].bits;
    double payload = (double)(total-hdrlen);
    printf("cmxc c  %-28s %12zu -> %12zu  bpb=%.4f  pred=%.4f  ovh=%+.3f%%  "
           "hdr=%u  %.3f MB/s  T=%d\n",
           inp, want, total,
           want? total*8.0/(double)want : 0.0,
           want? predbits/(double)want  : 0.0,
           (want&&predbits>0)? (payload*8.0/predbits-1.0)*100.0 : 0.0,
           hdrlen, sec>0? want/1e6/sec : 0.0, NT);

    for(int t=0;t<NT;t++) free(B[t].cbuf);
    free(B); free(H.b); free(out); free(buf);
    return 0;
}

/* ========================================================== decompress === */
/* A table width is valid if the encoder could have written it. */
static int width_ok(int b,int base){ return b>=WIDTH_MIN && b<=base; }

static int decompress_file(const char*inp,const char*outp)
{
    size_t cn; uint8_t*cbuf=slurp(inp,&cn,0);
    Blk *B=NULL; int rc=1;
    Rdr Rd={cbuf,cn,0,0};
    if(cn<24 || memcmp(cbuf,"CMXC",4)!=0){fprintf(stderr,"cmxc: not a CMXC file\n");goto fail;}
    Rd.p=4;
    uint32_t ver=ru16(&Rd);
    if(ver!=CMXC_VER){fprintf(stderr,"cmxc: version %u unsupported\n",ver);goto fail;}
    uint32_t flags=ru16(&Rd);
    uint64_t orig=ru64(&Rd);
    uint32_t NT=ru32(&Rd);
    uint32_t hdrlen=ru32(&Rd);
    uint64_t prime=ru64(&Rd);
    R =(int)(int32_t)ru32(&Rd);
    R2=(int)(int32_t)ru32(&Rd);
    EN_STATE=(int)ru8(&Rd); EN_WORD=(int)ru8(&Rd); EN_SPARSE=(int)ru8(&Rd); do_rec=(int)ru8(&Rd);
    SSE_MODE=(int)ru8(&Rd); NL1=(int)ru8(&Rd); MIX2CTX=(int)(int8_t)ru8(&Rd); nex=(int)ru8(&Rd);
    TB0=(int)ru8(&Rd); WB0=(int)ru8(&Rd); MB0=(int)ru8(&Rd); RB0=(int)ru8(&Rd);
    XB0=(int)ru8(&Rd); IB0=(int)ru8(&Rd); S2B0=(int)ru8(&Rd); S4B0=(int)ru8(&Rd);
    LR=rf32(&Rd); LR2_MUL=rf32(&Rd); WINIT=rf32(&Rd); SSEW3=rf32(&Rd);
    nord=(int)ru8(&Rd); scmask=(int)ru8(&Rd); FUZZ=(int)ru8(&Rd);
    SCB0=(int)ru8(&Rd); FTB0=(int)ru8(&Rd);   FLB0=(int)ru8(&Rd);
    LPOSCAP=(int)ru16(&Rd);
    NIB=(int)ru8(&Rd); NIBO=(int)ru8(&Rd);
    if(nex>MAXEX){fprintf(stderr,"cmxc: corrupt header (nex=%d)\n",nex);goto fail;}
    if(nord<1||nord>NORDMAX){fprintf(stderr,"cmxc: corrupt header (nord=%d)\n",nord);goto fail;}
    if(scmask<0||scmask>=(1<<NSC)){fprintf(stderr,"cmxc: corrupt header (scmask=%d)\n",scmask);goto fail;}
    if(FUZZ<0||FUZZ>3){fprintf(stderr,"cmxc: corrupt header (fuzz=%d)\n",FUZZ);goto fail;}
    /* Table widths size every allocation and every hash mask, so each must be
       one the encoder can write: its base width shrunk by MEMSH, never below
       WIDTH_MIN (which also keeps WSIZE/RSET etc. >= 512 rows). */
    if(!width_ok(TB0,TB0_BASE)||!width_ok(WB0,WB0_BASE)||!width_ok(MB0,MB0_BASE)||
       !width_ok(RB0,RB0_BASE)||!width_ok(XB0,XB0_BASE)||!width_ok(IB0,IB0_BASE)||
       !width_ok(S2B0,S2B0_BASE)||!width_ok(S4B0,S4B0_BASE)||!width_ok(SCB0,SCB0_BASE)||
       !width_ok(FTB0,FTB0_BASE)||!width_ok(FLB0,FLB0_BASE)){
        fprintf(stderr,"cmxc: corrupt header (table widths)\n");goto fail;}
    /* record periods: the model divides by R and reads p[-R], p[-2R] */
    if(do_rec ? (R<MINLAG||R>MAXLAG||(R2!=0&&(R2<MINLAG||R2>MAXLAG))) : (R!=0||R2!=0)){
        fprintf(stderr,"cmxc: corrupt header (rec=%d R=%d R2=%d)\n",do_rec,R,R2);goto fail;}
    if(!isfinite(LR)||!isfinite(LR2_MUL)||!isfinite(WINIT)||!isfinite(SSEW3)){
        fprintf(stderr,"cmxc: corrupt header (non-finite parameter)\n");goto fail;}
    /* the encoder writes at most 64 blocks, and none for an empty input */
    if(NT>64u || (NT==0) != (orig==0)){
        fprintf(stderr,"cmxc: corrupt header (nblocks=%u)\n",NT);goto fail;}
    if(LPOSCAP<1){fprintf(stderr,"cmxc: corrupt header (lposcap=%d)\n",LPOSCAP);goto fail;}
    if(NIB<0||NIB>1||NIBO<0||NIBO>3){
        fprintf(stderr,"cmxc: corrupt header (nib=%d nibo=%d)\n",NIB,NIBO);goto fail;}
    for(int i=0;i<nex;i++) exid[i]=(int)ru8(&Rd);
    USE_SIMD = (flags&1)?1:0;
    /* ---- SIMD kernel guard ----
     * A context-mixing decoder must reproduce the encoder's arithmetic exactly.
     * The NEON and scalar mixer kernels sum the same floats in a different
     * order, so their predictions can differ by one ULP, which is enough to flip
     * a coded bit eventually; every byte after that is wrong.  The container has
     * no checksum, so the decoder refuses an archive whose SIMD flag (offset 6,
     * bit 0) does not match this build.  Two NEON builds that differ in float
     * reassociation (another compiler or -O level) still pass this check; only
     * an input checksum would catch that case. */
#ifndef HAVE_NEON
    if(USE_SIMD){
        fprintf(stderr,
          "cmxc: REFUSING TO DECODE -- archive was produced with the NEON mixer\n"
          "      kernel (header flags bit0 = use_simd = 1) but this decoder was\n"
          "      built without NEON.  The two kernels sum the mixer's floats in a\n"
          "      different order, so this build cannot reproduce the encoder's\n"
          "      arithmetic and would emit WRONG BYTES with a success exit code.\n"
          "      Rebuild this decoder with NEON enabled, or re-compress the input\n"
          "      with CMXC_SIMD=0.\n");
        goto fail;
    }
#endif
    int do_e8 = (flags&2)?1:0;
    USE_NN=0;
    if(NL1<1||NL1>9){fprintf(stderr,"cmxc: corrupt header (bank=%d)\n",NL1);goto fail;}
    /* each SSE mode allocates only the APM tables it uses */
    if(SSE_MODE<0||SSE_MODE>3){fprintf(stderr,"cmxc: corrupt header (sse=%d)\n",SSE_MODE);goto fail;}

    if(NT){ B=(Blk*)calloc((size_t)NT,sizeof(Blk));
            if(!B){fprintf(stderr,"cmxc: oom (block table)\n");goto fail;} }
    /* Block table.  All arithmetic is done so that it cannot wrap: every block
       must lie inside the file, the raw lengths must add up to orig exactly,
       and no block may claim more output than its stream can carry.  The last
       bound is a property of the coder: each coded bit keeps at most a
       1-2^-17 fraction of the interval and each output byte widens it 256x,
       so a block of len bytes needs at least len/2^17 - 4 stream bytes. */
    uint64_t sum=0; int badblk=0;
    for(uint32_t t=0;t<NT;t++){
        uint64_t cl=ru64(&Rd), ln=ru64(&Rd);
        if(ln > orig-sum || cl > (uint64_t)cn || ln > ((cl+4u)<<17)) badblk=1;
        if(badblk) break;
        B[t].clen=(size_t)cl; B[t].len=(size_t)ln;
        B[t].abs_start=(size_t)sum;
        sum += ln;
    }
    if(badblk || Rd.bad || Rd.p!=hdrlen || sum!=orig || orig > (uint64_t)(SIZE_MAX-64)){
        fprintf(stderr,"cmxc: corrupt header (hdrlen=%u at=%zu sum=%llu orig=%llu)\n",
                hdrlen,Rd.p,(unsigned long long)sum,(unsigned long long)orig);
        goto fail;
    }
    size_t o=hdrlen;
    for(uint32_t t=0;t<NT;t++){
        if(B[t].clen > cn-o){fprintf(stderr,"cmxc: truncated stream\n");goto fail;}
        B[t].cbuf=cbuf+o; o+=B[t].clen;
        B[t].prime = (size_t)(prime < B[t].abs_start ? prime : B[t].abs_start);
    }

    set_table_sizes();
    derive_mech();
    layout_inputs();
    init_math_tables();

    uint8_t*out=(uint8_t*)calloc(orig? (size_t)orig+64 : 64,1);
    if(!out){fprintf(stderr,"cmxc: oom (output)\n");exit(1);}

    int W = CFG_THREADS; if((uint32_t)W>NT) W=(int)NT; if(W<1) W=1;
    /* priming makes block t depend on block t-1's OUTPUT, so it must then be
       decoded strictly in order.  With prime==0 (the default) the blocks are
       independent and any wave width works. */
    if(prime>0) W=1;

    if(VERB) fprintf(stderr,"[cmxc d] n=%llu nblk=%u prime=%llu workers=%d e8=%d "
                     "rec=%d(R=%d R2=%d) sse=%d bank=%d nord=%d scmask=0x%02x fuzz=%d "
                     "lposcap=%d nib=%d nibo=%d NM=%d NINP=%d lr=%.4f simd=%d\n",
                     (unsigned long long)orig,NT,(unsigned long long)prime,W,do_e8,
                     do_rec,R,R2,SSE_MODE,NL1,nord,scmask,FUZZ,LPOSCAP,NIB,NIBO,NM,NINP,LR,USE_SIMD);

    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    for(uint32_t f=0;f<NT;f+=(uint32_t)W){
        int cnt=(int)(NT-f); if(cnt>W)cnt=W;
        run_wave(B,(int)f,cnt,out+64,0);
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double sec=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    if(do_e8) e8p_backward(out+64,(size_t)orig);
    spew(outp,out+64,(size_t)orig);

    printf("cmxc d  %-28s %12zu -> %12llu  %.3f MB/s  blocks=%u workers=%d\n",
           inp, cn, (unsigned long long)orig, sec>0? orig/1e6/sec : 0.0, NT, W);

    free(out);
    rc=0;
fail:
    free(B); free(cbuf);
    return rc;
}

/* ----------------------------------------------------------- main -------- */
#ifdef CMX_UNIFIED                 /* built into the cmx program (cmx.c), which dispatches to it */
#define main cmxc_main
#endif
int main(int argc,char**argv){
    if(argc<4 || (argv[1][0]!='c' && argv[1][0]!='d') || argv[1][1]!=0){
        fprintf(stderr,
          "cmxc -- context-mixing lossless compressor\n"
          "usage: %s c <infile> <outfile>    compress\n"
          "       %s d <infile> <outfile>    decompress\n"
          "env:   CMXC_THREADS=n  blocks when compressing / worker threads when decompressing (8)\n"
          "       CMXC_PRIME=n    warm-up bytes per block; >0 forces SERIAL decode (0)\n"
          "       CMXC_MINBLK=n   minimum bytes per block; threads drop to honour it (1 MiB)\n"
          "       CMXC_PF=0|1 CMXC_SIMD=0|1 CMXC_MEM=n CMXC_VERBOSE=1\n"
          "       CMX_NORD=1..12  hashed byte orders        CMXS_MASK=0..127  ladder rungs\n"
          "       CMX_FUZZ=0..3   tolerant match / LSH      CMX_LPOSCAP=n     ladder LPOS cap\n"
          "       CMX_NIB=0|1     Ctr nibble rows           CMX_NIBO=0..3     order-model rows\n"
          "       plus the CMX_* model switches (recorded in the header)\n",
          argv[0],argv[0]);
        return 2;
    }
    read_env();
    if(argv[1][0]=='c') return compress_file(argv[2],argv[3]);
    return decompress_file(argv[2],argv[3]);
}
