# cmxc

A context-mixing lossless compressor, accompanying the paper
*Per-File Models for Lossless Compression* (2026).

## Build

```
make
```

This runs `clang -O3 -march=native -o cmxc cmxc.c -lm -lpthread`. The code is a single C file
with no dependencies. It has been built and tested only on Apple silicon (arm64, NEON); other
platforms fall back to a portable scalar mixing path that has not been tested. An archive
records whether NEON mixing was used, and a decoder built the other way refuses it rather than
decoding it incorrectly. Use `make CC=gcc` to build with another compiler.

```
make test
```

This runs `tests/roundtrip.sh`, which round-trips empty, text, random and E8/E9-heavy inputs
(including multi-block and primed archives) and checks that the decoder refuses truncated and
malformed archives. Each `cmxc` run maps about 3.4 GiB of model tables.

## Use

```
./cmxc c <input> <archive>     # compress
./cmxc d <archive> <output>    # decompress
```

All model settings are stored in the archive header, so decompression takes no options.

## Settings (environment variables, read when compressing)

| Variable | Effect |
|---|---|
| `CMXC_THREADS=n` | Split the input into `n` independent blocks, compressed in parallel (default 8). `CMXC_THREADS=1` gives the best compression and is the setting used in the paper. |
| `CMX_NIBO=0` | Max-compression mode: about 0.2% smaller on Silesia, about 1.25× the time. |
| `CMXC_VERBOSE=1` | Print the settings written to the header. |

At one thread the compressor needs about 3.4 GiB of memory (3.6 GiB in max-compression mode).
On the Silesia corpus (211.9 MB) at one thread on an Apple M5 Pro, compression takes about 860 s
and decompression about 1,040 s.

## Untrusted input

The decoder checks every header field and the block table against the values the encoder can
write, and refuses an archive that fails those checks. A build with AddressSanitizer and
UndefinedBehaviorSanitizer decoded or refused about 11,600 corrupted archives (damaged headers,
damaged streams, truncations) without a memory error. It is still a research codec, not a hardened
one: a crafted archive can make it run for a long time or allocate a lot of memory (about 3.4 GiB
of model tables plus the declared output size). Do not decompress archives from untrusted sources.

The archive format has no checksum. A decoder built with a different compiler or optimization
level from the encoder can, in principle, reproduce the encoder's floating-point arithmetic
inexactly and write wrong bytes without reporting an error, so compare the decompressed output
with the original (for example with `cmp`) before deleting the original.

Please report security problems as described in [SECURITY.md](SECURITY.md).

## License

MIT; see [LICENSE](LICENSE).
