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
 * bytes of IN.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

/* 1 if PATH is a safetensors file whose header names at least one tensor with dtype BF16. */
static int is_bf16_safetensors(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    uint8_t b[8]; uint64_t n = 0; int found = 0;
    if (fread(b, 1, 8, f) == 8 && fseeko(f, 0, SEEK_END) == 0) {
        for (int i = 7; i >= 0; i--) n = n << 8 | b[i];
        off_t sz = ftello(f);
        if (n >= 2 && n <= ((uint64_t)100 << 20) && sz >= 8 && n <= (uint64_t)sz - 8 && fseeko(f, 8, SEEK_SET) == 0) {
            char *h = malloc(n + 1);
            if (h && fread(h, 1, n, f) == n && h[0] == '{') {
                h[n] = 0;
                for (char *p = h; !found && (p = strstr(p, "\"dtype\"")); p += 7) {
                    char *q = p + 7;
                    while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
                    if (*q++ != ':') continue;
                    while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
                    found = !strncmp(q, "\"BF16\"", 6);
                }
            }
            free(h);
        }
    }
    fclose(f);
    return found;
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
    if (weights) {
        if (max) fail("--max applies to the text codec (cmxc)");
        char *av[16]; int ac = 0;
        av[ac++] = "bf16ref"; av[ac++] = argv[1]; av[ac++] = argv[2]; av[ac++] = argv[3];
        if (ref) { av[ac++] = "--ref"; av[ac++] = (char *)ref; }
        if (threads) { av[ac++] = "-t"; av[ac++] = (char *)threads; }
        if (bmib) { av[ac++] = "-b"; av[ac++] = (char *)bmib; }
        if (skib) { av[ac++] = "-s"; av[ac++] = (char *)skib; }
        if (lanes) { av[ac++] = "-l"; av[ac++] = (char *)lanes; }
        if (gpu) av[ac++] = "--gpu";
        av[ac] = NULL;
        return bf16ref_main(ac, av);
    }
    if (ref) fail(comp ? "--ref needs the weights codec (bf16ref)" : "--ref given, but the archive is a cmxc archive");
    if (bmib || skib || lanes || gpu) fail("-b, -s, -l and --gpu apply to the weights codec (bf16ref)");
    if (threads) setenv("CMXC_THREADS", threads, 1);
    if (max) setenv("CMX_NIBO", "0", 1);
    char *av[5] = {"cmxc", argv[1], argv[2], argv[3], NULL};
    return cmxc_main(4, av);
}
