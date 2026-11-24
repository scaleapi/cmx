# cmx

Two lossless compressors in one program:

- **cmxc** (`cmxc.c`): context mixing, for text and general data. It accompanies the paper
  *Per-File Models for Lossless Compression* (2026). Archives start with `CMXC`.
- **bf16ref** (`bf16ref.c`): a static-table rANS coder for bfloat16 model weights in safetensors files.
  It codes a model alone, or a fine-tuned model given its base model. Archives start with `BFR8` (the
  sub-stream layout below, the default) or `BFR7`; `BFR6` archives still decode.

`cmx` (`cmx.c`) chooses one of them. Both codecs are compiled unchanged, with their `main` functions
renamed (`-DCMX_UNIFIED`). So a `cmx` archive is byte for byte the archive of the codec that wrote it,
and the standalone `cmxc` and `bf16ref` binaries decode it, and the other way round.

## Build

```
make            # cmx, cmxc and bf16ref
make test       # tests/roundtrip.sh on cmxc and on cmx, then tests/cmx.sh
```

This uses `clang -O3 -march=native ... -lm -lpthread`; use `make CC=gcc` for another compiler. The
code has no dependencies. `tests/cmx.sh` needs `python3` to write its safetensors inputs.

- **cmxc** has been built and tested only on Apple silicon (arm64, NEON); other platforms fall back
  to a portable scalar mixing path that has not been tested. An archive records whether NEON mixing
  was used, and a decoder built the other way refuses it rather than decoding it incorrectly.
- **bf16ref** has also been built and tested with gcc on x86-64 (AMD EPYC, Ubuntu 24.04).

`tests/roundtrip.sh` round-trips empty, text, random and E8/E9-heavy inputs, including multi-block and
primed archives, and checks that the decoder refuses truncated and malformed archives.

`tests/cmx.sh` checks several things:
- `cmx` writes the same bytes as `cmxc` and `bf16ref`, and each binary decodes the others' archives;
- the choice of codec;
- round trips of a synthetic safetensors pair, alone and given the base;
- refusals.

Each `cmxc` run maps about 3.4 GiB of model tables.

## Use

```
./cmx c <input> <archive> [--ref BASE] [-t N] [--max] [--codec text|weights] [-b MiB]
./cmx d <archive> <output> [--ref BASE] [-t N]
```

**Compressing.** `cmx c` uses bf16ref for a safetensors file with at least one BF16 tensor, and cmxc
for anything else. Within a safetensors file, bf16ref codes the BF16 tensors; it stores the header and
the tensors of other dtypes raw.

**Decompressing.** `cmx d` chooses the decoder from the archive's first four bytes.

| Option | Effect |
|---|---|
| `--ref BASE` | Compress a fine-tuned model given its base model (bf16ref). Tensors are paired by name. Give the same `BASE` to decompress. |
| `-t N` | Threads. For cmxc this sets `CMXC_THREADS`, which also sets the number of blocks when compressing. bf16ref's archives do not depend on it. |
| `--max` | cmxc's max-compression mode (`CMX_NIBO=0`). |
| `--codec text\|weights` | Force cmxc (`text`) or bf16ref (`weights`). |
| `-b MiB` | bf16ref block size (default 64). |
| `-s KiB`, `-l lanes` | bf16ref's layout (below): sub-streams of `-s` KiB with `-l` lanes. The default is `-s 256 -l 32` (BFR8); `-s 0 -l 16` writes BFR7. |
| `--gpu` | bf16ref on the GPU: BFR8 with 32 lanes (a build with `make CUDA=1`). |

The standalone binaries remain:
- `./cmxc c|d <in> <out>`, with its settings in environment variables (below);
- `./bf16ref c|d <in> <out> [--ref REF] [-t N] [-b MiB]`.

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

`make CUDA=1` builds `cmx` and `bf16ref` with a GPU path, compiled with `nvcc` (default `CUDA_ARCH=sm_90`, H100).

**Writing.** `--gpu` codes every bfloat16 segment on the GPU. Checks and file I/O stay on the CPU. It writes
BFR8 with 32 lanes, the same bytes the CPU encoder writes for that layout.

**Reading.** `cmx d --gpu` decodes a BFR8 archive with 32 lanes on the GPU, whichever encoder wrote it. Any
other archive is decoded without `--gpu`.

**Sources.**
- `bf16ref_gpu.cu`: the host side.
- `gpu/kernels.cuh`: the kernels, from the G8 prototype of the bf16ref repository.
- `gpu/cuemu.h`: a CPU emulator of the kernels. `make bf16ref_emu` builds the GPU path on it, so it can be
  tested without a GPU. It is slow, so use small inputs only.

**Tests.** `make test-gpu-emu` (emulator) and `make CUDA=1 test-gpu` (GPU) run `tests/gpu.sh`. It checks four
things:
- the GPU archives equal the CPU's;
- each decodes the other's archives, alone and given the base;
- archives the GPU path cannot decode are refused;
- damaged archives fail with an error.

## cmxc settings (environment variables, read when compressing)

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
