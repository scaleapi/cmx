# cmx

One program, `cmx`, with two lossless codecs inside:

- **cmxc** (`cmxc.c`): context mixing, for text and general data. It accompanies the paper
  *Per-File Models for Lossless Compression* (2026). Archives start with `CMXC`.
- **bf16ref** (`bf16ref.c`): a static-table rANS coder for bfloat16 model weights in safetensors files.
  It codes a model alone, or a fine-tuned model given its base model. Archives start with `BFR8` (the
  sub-stream layout below, the default) or `BFR7`; `BFR6` archives still decode.

`cmx` (`cmx.c`) chooses the codec from the input when compressing and from the archive when decompressing.
Both codecs are compiled into it unchanged, with their `main` functions renamed (`-DCMX_UNIFIED`).

## Build

```
make            # builds ./cmx
make test       # tests/roundtrip.sh and tests/cmx.sh
```

This uses `clang -O3 -march=native ... -lm -lpthread`; use `make CC=gcc` for another compiler. The
code has no dependencies. `tests/cmx.sh` needs `python3` to write its safetensors inputs.

- **The text codec (cmxc)** has been built and tested only on Apple silicon (arm64, NEON); other platforms fall back
  to a portable scalar mixing path that has not been tested. An archive records whether NEON mixing
  was used, and a decoder built the other way refuses it rather than decoding it incorrectly.
- **The weights codec (bf16ref)** has also been built and tested with gcc on x86-64 (AMD EPYC, Ubuntu 24.04).

`tests/roundtrip.sh` tests the text codec: it round-trips empty, text, random and E8/E9-heavy inputs, including multi-block and
primed archives, and checks that the decoder refuses truncated and malformed archives.

`tests/cmx.sh` checks several things:
- round trips of text and of a synthetic safetensors pair, alone and given the base;
- options give the same archive as the settings they stand for, and weights archives do not depend on `-t`;
- the choice of codec;
- the BFR8 layouts, and a damaged index;
- the output: never the input or the reference, and kept when a run fails;
- refusals.

Each text-codec run maps about 3.4 GiB of model tables.

## Use

```
./cmx c <input> <archive> [--ref BASE] [-t N] [--max] [--codec text|weights] [-b MiB] [-s KiB] [-l lanes] [--gpu]
./cmx d <archive> <output> [--ref BASE] [-t N] [--gpu]
```

**Compressing.** `cmx c` uses bf16ref for a safetensors file with at least one BF16 tensor, and cmxc
for anything else. Within a safetensors file, bf16ref codes the BF16 tensors; it stores the header and
the tensors of other dtypes raw.

**Decompressing.** `cmx d` chooses the decoder from the archive's first four bytes.

**The output.** `cmx` writes to a temporary file next to the output and renames it over the output only when
the run succeeds, so a failed run (a wrong reference, a damaged archive) leaves an existing output as it was.
It refuses an output that is the same file as the input or the reference.

| Option | Effect |
|---|---|
| `--ref BASE` | Compress a fine-tuned model given its base model (bf16ref). Tensors are paired by name. Give the same `BASE` to decompress. |
| `-t N` | Threads. For cmxc this sets `CMXC_THREADS`, which also sets the number of blocks when compressing. bf16ref's archives do not depend on it. |
| `--max` | cmxc's max-compression mode (`CMX_NIBO=0`). |
| `--codec text\|weights` | Force cmxc (`text`) or bf16ref (`weights`). |
| `-b MiB` | bf16ref block size (default 64). |
| `-s KiB`, `-l lanes` | bf16ref's layout (below): sub-streams of `-s` KiB with `-l` lanes. The default is `-s 256 -l 32` (BFR8); `-s 0 -l 16` writes BFR7. |
| `--gpu` | bf16ref on the GPU: BFR8 with 32 lanes (a build with `make CUDA=1`). |

## BFR8: the sub-stream layout

A BFR7 block is one rANS stream with 16 interleaved states. That gives far too few independent streams for a GPU.
BFR8 keeps everything else in BFR7's blocks: the symbols, the choice of the 3 coded bits, the tables and the raw
bits. It changes three things:
- **Sub-streams:** each block's words are cut into sub-streams of `-s` KiB of weights.
- **Lanes:** each sub-stream has its own `-l` interleaved states, 16 or 32.
- **Index:** an index of the sub-streams' word offsets follows the tables. Given the base, the index also holds
  each sub-stream's raw-bit offset.

The archive header records the sub-stream size and the lane count. With one sub-stream per block and 16 lanes,
a BFR8 block is byte for byte a BFR7 block, and the decoder handles both.

**Default.** The default layout has 256 KiB sub-streams with 32 lanes.
- **Size:** it costs about 0.04 percentage points of the original size against BFR7, on Qwen3.5 and Qwen3-14B,
  alone and given the base.
- **Speed at 16 threads:** on an Apple M5 Pro, BFR8 decoded Qwen3-14B and Llama-3.1-8B-Instruct shards as fast
  as BFR7 or faster: 0% to −26% time, median of 5 repeats.
- **Speed on one thread:** it is within 2% of BFR7 on Qwen3-14B, and 13% slower on the Llama shard.

**Choice of the 3 coded bits.** The encoder now makes this choice in integer arithmetic, with a fixed rule for
exact ties. So any CPU, compiler or GPU encoder writes the same bytes. Floating-point entropies could round
exact ties differently on different math libraries.

## GPU

`make CUDA=1` builds `cmx` with a GPU path for the weights codec, compiled with `nvcc` (default `CUDA_ARCH=sm_90`,
H100). The CPU stays the default: for file-to-file use, 16 CPU threads are faster end to end than one H100,
because of CUDA start-up and host copies.

**Writing.** `--gpu` codes every bfloat16 segment on the GPU. Checks and file I/O stay on the CPU. It writes
BFR8 with 32 lanes, the same bytes the CPU encoder writes for that layout.

**Reading.** `cmx d --gpu` decodes a BFR8 archive with 32 lanes on the GPU, whichever encoder wrote it. Any
other archive is decoded without `--gpu`.

**Sources.**
- `bf16ref_gpu.cu`: the host side.
- `gpu/kernels.cuh`: the kernels.
- `gpu/cuemu.h`: a CPU emulator of the kernels. `make cmx_emu` builds a test-only `cmx` with the GPU path on
  it, so the GPU path can be tested without a GPU. It is slow, so use small inputs only.

**Tests.** `make test-gpu-emu` (emulator) and `make CUDA=1 test-gpu` (GPU) run `tests/gpu.sh`. It checks four
things:
- the GPU archives equal the CPU's;
- each decodes the other's archives, alone and given the base;
- archives the GPU path cannot decode are refused;
- damaged archives fail with an error.

## Text-codec settings (environment variables, read when compressing)

`-t` and `--max` set the first two.

| Variable | Effect |
|---|---|
| `CMXC_THREADS=n` | Split the input into `n` independent blocks, compressed in parallel (default 8). `CMXC_THREADS=1` gives the best compression and is the setting used in the paper. |
| `CMX_NIBO=0` | Max-compression mode: about 0.2% smaller on Silesia, about 1.25× the time. |
| `CMXC_VERBOSE=1` | Print the settings written to the header. |

At one thread the compressor needs about 3.4 GiB of memory (3.6 GiB in max-compression mode).
On the Silesia corpus (211.9 MB) at one thread on an Apple M5 Pro, compression takes about 860 s
and decompression about 1,040 s.

## Untrusted input

The cmxc decoder checks every header field and the block table against the values the encoder can
write, and refuses an archive that fails those checks. A build with AddressSanitizer and
UndefinedBehaviorSanitizer decoded or refused about 11,600 corrupted archives (damaged headers,
damaged streams, truncations) without a memory error. It is still a research codec, not a hardened
one: a crafted archive can make it run for a long time or allocate a lot of memory (about 3.4 GiB
of model tables plus the declared output size). Do not decompress archives from untrusted sources.

The cmxc archive format has no checksum. A decoder built with a different compiler or optimization
level from the encoder can, in principle, reproduce the encoder's floating-point arithmetic
inexactly and write wrong bytes without reporting an error, so compare the decompressed output
with the original (for example with `cmp`) before deleting the original.

bf16ref archives carry a 64-bit check of each segment's original bytes, and bf16ref refuses to decode with a reference other than the one used to compress.

Please report security problems as described in [SECURITY.md](SECURITY.md).

## License

MIT; see [LICENSE](LICENSE).

## Results

All runs are on an Apple M5 Pro (6 performance + 12 efficiency cores, macOS), except the GPU runs. Every result
was checked by a byte-exact round trip unless marked ‡ (compressed size only).

### Text and general data (cmxc)

**Silesia** (211,938,580 bytes), single-threaded (`-t 1`). Times are compression plus decompression, median
(min–max) of 3 interleaved rounds.

| codec | bytes | bits/byte | ratio | total s |
|---|---:|---:|---:|---:|
| paq8px v217 `-12L` (published) | 27,682,006 | 1.0449 | 7.656× | — |
| precomp 0.4.7 + cmix v21 (published) | 28,261,094 | 1.0668 | 7.499× | — |
| paq8px v216 `-11` ‡ | 28,570,370 | 1.0784 | 7.418× | — |
| **cmx** | **35,838,365** | **1.3528** | **5.914×** | 1897.3 (1896.3–1899.9) |
| **cmx `--max`** | **35,773,030** | **1.3503** | **5.925×** | 2371.8 (2366.5–2376.7) |
| zpaq `-m5` | 39,113,381 | 1.4764 | 5.419× | 631.9 (627.3–634.1) |
| zpaq `-m4` | 42,628,623 | 1.6091 | 4.972× | 183.2 (181.5–183.3) |
| bsc `-e2` | 46,536,916 | 1.7566 | 4.554× | 11.7 (11.7–12.0) |
| xz `-9e` | 48,456,004 | 1.8291 | 4.374× | 57.5 (57.2–57.6) |
| brotli `-11` | 49,564,563 | 1.8709 | 4.276× | 217.7 (217.3–218.7) |
| zstd `-19` | 52,895,350 | 1.9966 | 4.007× | 40.4 (40.3–40.4) |
| zstd `-22` | 52,369,178 | 1.9768 | 4.047× | 52.6 (52.5–52.7) |

**Ten benchmarks**, every codec single-threaded at its strongest setting. Units: bits per byte (b/B), per
8-bit RGB sub-pixel (b/sp) or per 16-bit PCM sample (b/smp); lower is better. paq8px `-11` was not run on the
three largest benchmarks (—). The last column is the best domain-specific codec (J = JPEG XL lossless,
F = FLAC `-8`).

| benchmark | unit | gzip | bzip2 | zstd | xz | brotli | bsc | zpaq | paq8px | cmx | cmx `--max` | domain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Calgary | b/B | 2.608 | 2.132 | 2.266 | 2.174 | 2.108 | 1.983 | 1.776 | **1.490** | 1.632 | 1.630 | |
| Canterbury | b/B | 2.072 | 1.545 | 1.469 | 1.403 | 1.397 | 1.302 | 1.036 | **0.869** | 0.948 | 0.947 | |
| Canterbury (large) | b/B | 2.293 | 1.854 | 1.826 | 1.833 | 1.794 | 1.619 | 1.544 | **1.402** | 1.470 | 1.468 | |
| Silesia | b/B | 2.553 | 2.057 | 1.977 | 1.829 | 1.871 | 1.757 | 1.476 | **1.078** ‡ | 1.353 | 1.350 | |
| AIT 2026 | b/B | 5.314 | 5.005 | 5.076 | 4.939 | 4.869 | 4.812 | 4.603 | **4.087** | 4.540 | 4.540 | |
| enwik8 | b/B | 2.918 | 2.321 | 2.022 | 1.987 | 2.059 | 1.718 | 1.570 | **1.287** ‡ | 1.451 | 1.446 | |
| enwik9 | b/B | 2.582 | 2.032 | 1.710 | 1.694 | 1.787 | 1.484 | 1.349 | — | 1.192 | **1.187** | |
| Kodak | b/sp | 4.998 | 3.856 | 4.292 | 4.090 | 4.275 | 3.619 | 3.205 | **1.880** | 2.982 | 2.973 | 2.828 J |
| DIV2K valid | b/sp | 5.979 | 4.663 | 5.400 | 4.814 | 4.845 | 4.248 | 3.724 | — | 3.497 | 3.493 | **2.816** J |
| LibriSpeech dev-clean | b/smp | 12.977 | 10.839 | 11.843 | 10.970 | 10.620 | 9.845 | 9.356 | — | 8.861 | 8.862 | **8.632** F |

cmx is the smallest of the general-purpose codecs other than paq8px on all ten benchmarks; paq8px is smaller on
every benchmark it was run on.

### Model weights (bf16ref)

Whole models in bfloat16 safetensors, as a percentage of the original size (lower is better). Times are the
median of 5 repeats in seconds, compression / decompression. These runs used the BFR7 layout (`-s 0 -l 16`);
the default BFR8 layout is about 0.04 percentage points larger, and decodes as fast at 16 threads.

**A model alone**

| model | size | cmx | ZipNN | DFloat11 (published) | cmx, 16 threads | ZipNN, 16 threads | cmx, 1 thread |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 16.1 GB | **65.21%** | 66.39% | 67.84% | 1.65 / 2.22 | 7.76 / 7.44 | 12.95 / 6.93 |
| Llama-3.1-8B | 16.1 GB | **65.74%** | 66.39% | | 1.65 / 2.21 | 7.51 / 7.39 | 12.96 / 7.04 |
| Qwen3-14B | 29.5 GB | **66.11%** | 66.75% | 68.17% | 3.07 / 4.05 | 13.90 / 13.47 | 23.79 / 12.75 |
| Qwen3-14B-Base | 29.5 GB | **66.09%** | 66.75% | | 3.10 / 4.12 | 14.50 / 13.62 | 23.91 / 12.96 |
| Qwen3.5-0.8B-Base | 1.7 GB | **66.03%** | 66.66% | | 0.25 / 0.24 | 1.11 / 1.10 | 1.24 / 0.88 |
| Qwen3.5-4B | 9.3 GB | **65.84%** | 66.47% | | 1.00 / 1.10 | 4.27 / 4.11 | 7.40 / 3.86 |
| Qwen3.5-4B-Base | 9.3 GB | **65.81%** | 66.47% | | 1.00 / 1.07 | 4.19 / 4.08 | 7.00 / 3.88 |
| Qwen3.5-9B-Base | 19.3 GB | **65.70%** | 66.41% | | 1.89 / 2.02 | 9.25 / 9.15 | 14.78 / 8.06 |

DFloat11's figures are the sizes of its published checkpoints.

**A fine-tuned model given its base** (`--ref`)

| model, given | cmx `--ref` | FM-Delta | ZipNN (XOR) | BitX (XOR + zstd) | cmx, 16 threads | ZipNN, 16 threads |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct, given Llama-3.1-8B | **38.41%** | 43.22% | 49.82% | 51.71% | 2.09 / 2.96 | 12.87 / 15.81 |
| Qwen3-14B, given Qwen3-14B-Base | **43.68%** | 48.68% | 53.57% | 56.48% | 3.76 / 5.42 | 18.64 / 19.02 |
| Qwen3.5-4B, given Qwen3.5-4B-Base | **38.82%** | 43.48% | 48.75% | 51.18% | 1.29 / 1.31 | 9.50 / 10.19 |

FM-Delta was run single-threaded, once (66–224 s to compress, 102–343 s to decompress).

**GPU** (`--gpu`, one H100 with an AMD EPYC 7R13 host), per 3.8 GB Qwen3-14B shard, against the CPU path of
the same host at 16 threads. The GPU archives equal the CPU's byte for byte.

| | GPU kernels | GPU end to end | CPU end to end |
|---|---:|---:|---:|
| compress | 0.04–0.07 s | 1.9–3.7 s | 1.0–1.9 s |
| decompress | 0.007–0.016 s | 1.8–4.3 s | 0.5–0.9 s |

The kernels are about 100× faster than the CPU coder, but copying between host and GPU memory and about 2 s of
CUDA start-up per process make the CPU faster for file-to-file use. That is why the CPU stays the default.
