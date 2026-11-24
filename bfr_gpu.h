/* GPU coding of bf16ref's BFR8 blocks with 32 lanes (bf16ref_gpu.cu); used by bf16ref.c when built with -DBF16REF_GPU.
 * Segments are given as bf16ref.c plans them: kind (1 alone, 2 given the reference, anything else is skipped), byte
 * offset and length in the input, offset in the reference.  Bfloat16 segments must be at even offsets.  Both
 * functions return 0, or 1 with a message in msg. */
#include <stddef.h>
#include <stdint.h>
/* codes the segments of in (and ref) on the GPU: segment i's encoded block is body[i], outlen[i] bytes; the
 * buffers live until bfr_gpu_free(*keep) */
int bfr_gpu_encode(const uint8_t *in, size_t ilen, const uint8_t *ref, size_t rlen, size_t nseg, const int *kind, const uint64_t *off,
                   const uint64_t *len, const uint64_t *roff, uint64_t sw, uint8_t **body, uint64_t *outlen, void **keep, char *msg, size_t msglen);
void bfr_gpu_free(void *keep);
/* decodes the coded segments of an archive (segment i's block at arc + ain[i], ainlen[i] bytes) into out, olen bytes;
 * the bytes of the other segments are left zero */
int bfr_gpu_decode(const uint8_t *arc, size_t alen, const uint8_t *ref, size_t rlen, size_t nseg, const int *kind, const uint64_t *off,
                   const uint64_t *len, const uint64_t *roff, const uint64_t *ain, const uint64_t *ainlen, uint64_t sw, uint8_t *out, size_t olen,
                   char *msg, size_t msglen);
/* host threads for the copies between the GPU and host memory (default 1) */
void bfr_gpu_threads(int threads);
