/*
 * cmx: one program for two lossless codecs.
 *
 *   cmxc     (cmxc.c)    context mixing, for text and general data; archives start with "CMXC"
 *   bf16ref  (bf16ref.c) static-table rANS for bfloat16 model weights, alone or given a base model;
 *                        archives start with "BFR7", or "BFR8" in the sub-stream layout (and "BFR6", decode only)
 *
 * Both codecs are compiled unchanged with -DCMX_UNIFIED, which renames their main functions to cmxc_main and
 * bf16ref_main; this file only chooses one of them and translates the options. A cmx archive is therefore
 * byte for byte the archive of the codec that wrote it, and each standalone binary decodes it.
 *
 *   cmx c IN OUT [--ref BASE] [-t N] [--max] [--codec text|weights] [-b MiB] [-s KiB] [-l lanes] [--gpu]
 *   cmx d IN OUT [--ref BASE] [-t N] [--gpu]
 *
 * Compression uses bf16ref when IN is a safetensors file with at least one BF16 tensor, or when --ref is
 * given, and cmxc otherwise; --codec overrides the choice. Decompression picks the decoder from the first four
 * bytes of IN. OUT is written to a temporary file beside it and renamed over OUT only on success; an OUT that is
 * the same file as IN or BASE is refused.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int cmxc_main(int argc, char **argv);
int bf16ref_main(int argc, char **argv);

static void usage(void) {
    fprintf(stderr,
            "usage: cmx c IN OUT [--ref BASE] [-t N] [--max] [--codec text|weights] [-b MiB] [-s KiB] [-l lanes] [--gpu]\n"
            "       cmx d IN OUT [--ref BASE] [-t N] [--gpu]\n"
            "  c          compress: bf16ref for safetensors files with BF16 tensors (or with --ref), cmxc otherwise\n"
            "  d          decompress: the decoder is chosen from the archive's first four bytes (CMXC, BFR8, BFR7, BFR6)\n"
            "  --ref BASE compress a fine-tuned model given its base model (bf16ref); give the same BASE to decompress\n"
            "  -t N       threads (cmxc: sets CMXC_THREADS, which also sets the number of blocks when compressing)\n"
            "  --max      cmxc's max-compression mode (CMX_NIBO=0); slower\n"
            "  --codec    text (cmxc) or weights (bf16ref), overriding the choice\n"
            "  -b MiB     bf16ref block size (default 64)\n"
            "  -s KiB, -l lanes  bf16ref's layout: sub-streams of KiB of weights with 16 or 32 states (default -s 256 -l 32,\n"
            "             BFR8; -s 0 -l 16 writes BFR7)\n"
            "  --gpu      bf16ref on the GPU (BFR8 with 32 lanes; a build with make CUDA=1)\n"
            "cmxc's other settings are read from its CMX_* / CMXC_* environment variables, as in the cmxc binary.\n");
    exit(2);
}
static void fail(const char *msg) { fprintf(stderr, "cmx: %s\n", msg); exit(2); }

/* A small JSON scanner for the safetensors header: enough to walk its top-level object without being fooled
 * by braces or keys inside strings. Each function returns the position after what it read, or NULL. */
static const char *jws(const char *p, const char *e) {
    while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}
static const char *jstr(const char *p, const char *e) {          /* a string; p at its opening quote */
    if (p >= e || *p != '"') return NULL;
    for (p++; p < e; p++) { if (*p == '\\') p++; else if (*p == '"') return p + 1; }
    return NULL;
}
static const char *jskip(const char *p, const char *e) {         /* any value */
    p = jws(p, e);
    if (p >= e) return NULL;
    if (*p == '"') return jstr(p, e);
    if (*p == '{' || *p == '[') {
        int d = 0;
        while (p < e) {
            if (*p == '"') { if (!(p = jstr(p, e))) return NULL; continue; }
            if (*p == '{' || *p == '[') d++;
            else if ((*p == '}' || *p == ']') && --d == 0) return p + 1;
            p++;
        }
        return NULL;
    }
    while (p < e && *p != ',' && *p != '}' && *p != ']') p++;
    return p;
}
static int jkey(const char *k, const char *kend, const char *want) {   /* k..kend is a key with its quotes */
    size_t n = strlen(want);
    return (size_t)(kend - k) == n + 2 && !memcmp(k + 1, want, n);
}
/* 1 if the object at p (a tensor entry) has "dtype": "BF16" */
static int entry_is_bf16(const char *p, const char *e) {
    p = jws(p, e);
    if (p >= e || *p++ != '{') return 0;
    for (;;) {
        p = jws(p, e);
        if (p >= e || *p == '}') return 0;
        const char *k = p; if (!(p = jstr(p, e))) return 0;
        const char *kend = p; p = jws(p, e);
        if (p >= e || *p++ != ':') return 0;
        p = jws(p, e);
        if (jkey(k, kend, "dtype")) return (size_t)(e - p) >= 6 && !memcmp(p, "\"BF16\"", 6);
        if (!(p = jskip(p, e))) return 0;
        p = jws(p, e);
        if (p < e && *p == ',') p++;
    }
}

/* 1 if PATH is a safetensors file with at least one tensor of dtype BF16. Only tensor entries count: the
 * "__metadata__" entry holds free-form strings, so a "dtype" there says nothing about the tensors. */
static int is_bf16_safetensors(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    uint8_t b[8]; uint64_t n = 0; int found = 0;
    if (fread(b, 1, 8, f) == 8 && fseeko(f, 0, SEEK_END) == 0) {
        for (int i = 7; i >= 0; i--) n = n << 8 | b[i];
        off_t sz = ftello(f);
        if (n >= 2 && n <= ((uint64_t)100 << 20) && sz >= 8 && n <= (uint64_t)sz - 8 && fseeko(f, 8, SEEK_SET) == 0) {
            char *h = malloc(n);
            if (h && fread(h, 1, n, f) == n) {
                const char *p = jws(h, h + n), *e = h + n;
                if (p < e && *p++ == '{')
                    while (!found) {
                        p = jws(p, e);
                        if (p >= e || *p == '}') break;
                        const char *k = p; if (!(p = jstr(p, e))) break;
                        const char *kend = p; p = jws(p, e);
                        if (p >= e || *p++ != ':') break;
                        if (!jkey(k, kend, "__metadata__")) found = entry_is_bf16(p, e);
                        if (!(p = jskip(p, e))) break;
                        p = jws(p, e);
                        if (p < e && *p == ',') p++;
                    }
            }
            free(h);
        }
    }
    fclose(f);
    return found;
}

/* The output is written to a temporary file next to it and renamed over it only when the codec succeeds, so a
 * failed run (a wrong reference, a damaged archive) leaves an existing output as it was. */
static char *g_tmp;
static int g_keep_mode; static mode_t g_mode;     /* an existing output's permissions, kept when it is replaced */
static void drop_tmp(void) { if (g_tmp) unlink(g_tmp); }
static int same_file(const struct stat *a, const char *path) {
    struct stat b;
    return path && stat(path, &b) == 0 && a->st_dev == b.st_dev && a->st_ino == b.st_ino;
}
/* returns the path the codec should write; refuses an output that is the input or the reference */
static const char *prepare_output(const char *out, const char *in, const char *ref) {
    struct stat so, lo;
    int exists = stat(out, &so) == 0;
    if (exists && (same_file(&so, in) || same_file(&so, ref))) fail("the output is the same file as the input or the reference");
    /* write in place to anything that is not a plain file (a device such as /dev/null) or is a symbolic link */
    if (exists && (!S_ISREG(so.st_mode) || (lstat(out, &lo) == 0 && S_ISLNK(lo.st_mode)))) return out;
    size_t n = strlen(out) + 16;
    g_tmp = malloc(n); if (!g_tmp) fail("out of memory");
    snprintf(g_tmp, n, "%s.cmx-XXXXXX", out);
    /* reserve a unique name, then leave the file to the codec: bf16ref removes the file it was writing when it
     * fails, cmxc exits and drop_tmp removes it, and an error before either opens it leaves nothing behind */
    int fd = mkstemp(g_tmp);
    if (fd < 0) { free(g_tmp); g_tmp = NULL; perror(out); fail("cannot create a temporary file next to the output"); }
    close(fd); unlink(g_tmp);
    if (exists) { g_keep_mode = 1; g_mode = so.st_mode & 07777; }
    atexit(drop_tmp);
    return g_tmp;
}
static int finish_output(int rc, const char *out) {
    if (!g_tmp) return rc;
    if (rc == 0 && g_keep_mode) chmod(g_tmp, g_mode);
    if (rc == 0 && rename(g_tmp, out)) { perror(out); rc = 1; }
    if (rc) unlink(g_tmp);
    free(g_tmp); g_tmp = NULL;
    return rc;
}

int main(int argc, char **argv) {
    if (argc < 4 || (strcmp(argv[1], "c") && strcmp(argv[1], "d"))) usage();
    int comp = argv[1][0] == 'c', max = 0, gpu = 0;
    const char *ref = NULL, *threads = NULL, *codec = NULL, *bmib = NULL, *skib = NULL, *lanes = NULL;
    for (int i = 4; i < argc; i++) {
        const char *o = argv[i];
        if (!strcmp(o, "--max")) { max = 1; continue; }
        if (!strcmp(o, "--gpu")) { gpu = 1; continue; }
        if (i + 1 >= argc) usage();
        if (!strcmp(o, "--ref")) ref = argv[++i];
        else if (!strcmp(o, "-t")) threads = argv[++i];
        else if (!strcmp(o, "--codec")) codec = argv[++i];
        else if (!strcmp(o, "-b")) bmib = argv[++i];
        else if (!strcmp(o, "-s")) skib = argv[++i];
        else if (!strcmp(o, "-l")) lanes = argv[++i];
        else usage();
    }
    if (threads && atoi(threads) < 1) fail("-t must be at least 1");
    int weights;
    if (comp) {
        if (codec && strcmp(codec, "text") && strcmp(codec, "weights")) fail("--codec must be text or weights");
        weights = codec ? !strcmp(codec, "weights") : (ref != NULL || is_bf16_safetensors(argv[2]));
    } else {
        if (codec || bmib || max || skib || lanes) fail("--codec, -b, -s, -l and --max apply to compression only");
        FILE *f = fopen(argv[2], "rb"); char tag[4] = {0};
        if (!f) { perror(argv[2]); return 1; }
        size_t got = fread(tag, 1, 4, f); fclose(f);
        if (got == 4 && !memcmp(tag, "CMXC", 4)) weights = 0;
        else if (got == 4 && (!memcmp(tag, "BFR8", 4) || !memcmp(tag, "BFR7", 4) || !memcmp(tag, "BFR6", 4))) weights = 1;
        else fail("not a cmx archive (no CMXC, BFR8, BFR7 or BFR6 tag)");
    }
    if (weights && max) fail("--max applies to the text codec (cmxc)");
    if (!weights && ref) fail(comp ? "--ref needs the weights codec (bf16ref)" : "--ref given, but the archive is a cmxc archive");
    if (!weights && (bmib || skib || lanes || gpu)) fail("-b, -s, -l and --gpu apply to the weights codec (bf16ref)");
    const char *out = argv[3], *dst = prepare_output(out, argv[2], ref);
    if (weights) {
        char *av[16]; int ac = 0;
        av[ac++] = "bf16ref"; av[ac++] = argv[1]; av[ac++] = argv[2]; av[ac++] = (char *)dst;
        if (ref) { av[ac++] = "--ref"; av[ac++] = (char *)ref; }
        if (threads) { av[ac++] = "-t"; av[ac++] = (char *)threads; }
        if (bmib) { av[ac++] = "-b"; av[ac++] = (char *)bmib; }
        if (skib) { av[ac++] = "-s"; av[ac++] = (char *)skib; }
        if (lanes) { av[ac++] = "-l"; av[ac++] = (char *)lanes; }
        if (gpu) av[ac++] = "--gpu";
        av[ac] = NULL;
        return finish_output(bf16ref_main(ac, av), out);
    }
    if (threads) setenv("CMXC_THREADS", threads, 1);
    if (max) setenv("CMX_NIBO", "0", 1);
    char *av[5] = {"cmxc", argv[1], argv[2], (char *)dst, NULL};
    return finish_output(cmxc_main(4, av), out);
}
