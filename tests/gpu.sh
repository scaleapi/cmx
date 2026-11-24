#!/bin/sh
# The GPU path of the weights codec (cmx --gpu) against its CPU coder.  Usage: tests/gpu.sh CPU_BIN GPU_BIN [small]
#   CPU_BIN  a cmx binary (its CPU path is used)
#   GPU_BIN  a cmx built with make CUDA=1, or cmx_emu (the kernels on the CPU emulator; give "small")
# For several BFR8 layouts with 32 lanes, alone and given the base: the GPU writes the same archive as the CPU,
# the GPU decodes the CPU's archive and the CPU the GPU's; --gpu picks 256 KiB x 32 lanes by default; archives the
# GPU path cannot decode are refused, and damaged archives fail with an error, never with wrong output.
# Needs python3 to write the inputs.
set -u
CPU=${1:?CPU_BIN}; GPU=${2:?GPU_BIN}; SMALL=${3:-}
T=$(mktemp -d "${TMPDIR:-/tmp}/bf16ref-gpu-test.XXXXXX") || exit 2
trap 'rm -rf "$T"' EXIT INT TERM
pass=0; fail=0
ok()  { pass=$((pass+1)); echo "ok   $1"; }
bad() { fail=$((fail+1)); echo "FAIL $1"; [ -s "$T/err" ] && sed 's/^/     /' "$T/err" | tail -3; }
run() { "$@" >/dev/null 2>"$T/err"; }
refuse() { name=$1; shift; if "$@" >/dev/null 2>&1; then bad "refuse $name"; else ok "refuse $name"; fi; }

# a base model and a fine-tune: tensors of several sizes (not multiples of the sub-stream size), BF16 and F32
if [ -n "$SMALL" ]; then SIZES="70000 3001 40000 517"; else SIZES="9000000 3001 16777216 40000 517 2500000"; fi
python3 - "$T" $SIZES <<'EOF' || { echo "python3 is needed to write the inputs" >&2; exit 2; }
import json, random, struct, sys
from array import array
T, sizes = sys.argv[1], [int(x) for x in sys.argv[2:]]; rnd = random.Random(3)
def tensors():
    out = {}
    for i, n in enumerate(sizes):
        s = 0.01 * (i + 1); g = [rnd.gauss(0, s) for _ in range(min(n, 65536))]
        out[f"layers.{i}.weight"] = [g[(k * 7919) % len(g)] * (1 + 0.001 * (k % 13)) for k in range(n)]
    return out
def bf16(ws):
    a = array("f", ws); b = a.tobytes(); return b"".join(b[4 * k + 2: 4 * k + 4] for k in range(len(ws)))
def write(path, ts):
    hdr, body = {}, b""
    for k, ws in ts.items():
        data = bf16(ws); hdr[k] = {"dtype": "BF16", "shape": [len(ws)], "data_offsets": [len(body), len(body) + len(data)]}; body += data
    data = array("f", [1.0] * 1000).tobytes(); hdr["norm.weight"] = {"dtype": "F32", "shape": [1000], "data_offsets": [len(body), len(body) + len(data)]}; body += data
    h = json.dumps(hdr, separators=(",", ":")).encode(); h += b" " * (-len(h) % 8)
    open(path, "wb").write(struct.pack("<Q", len(h)) + h + body)
base = tensors(); write(f"{T}/base.st", base)
ft = {k: [w + rnd.gauss(0, 0.0004) if rnd.random() < 0.6 else w for w in ws] for k, ws in base.items()}; write(f"{T}/ft.st", ft)
EOF

if [ -n "$SMALL" ]; then LAYS="-b_1_-s_1 -b_1_-s_16 -b_1_-s_0"; else LAYS="-s_256 -s_64 -b_8_-s_1 -s_0"; fi
for l in $LAYS; do
    lay=$(echo "$l" | tr '_' ' ')
    for mode in alone ref; do
        R=""; [ $mode = ref ] && R="--ref $T/base.st"
        name="$mode $lay -l 32"
        run "$CPU" c "$T/ft.st" "$T/c.a" $R $lay -l 32 && run "$GPU" c "$T/ft.st" "$T/g.a" $R $lay -l 32 --gpu &&
            cmp -s "$T/c.a" "$T/g.a" && ok "$name: GPU archive = CPU archive" || bad "$name: GPU archive = CPU archive"
        rm -f "$T/o"; run "$GPU" d "$T/c.a" "$T/o" $R --gpu && cmp -s "$T/o" "$T/ft.st" && ok "$name: GPU decodes the CPU archive" || bad "$name: GPU decodes the CPU archive"
        rm -f "$T/o"; run "$CPU" d "$T/g.a" "$T/o" $R && cmp -s "$T/o" "$T/ft.st" && ok "$name: CPU decodes the GPU archive" || bad "$name: CPU decodes the GPU archive"
    done
done

if [ -z "$SMALL" ]; then
    run "$CPU" c "$T/ft.st" "$T/c.a" && run "$GPU" c "$T/ft.st" "$T/g.a" --gpu &&
        cmp -s "$T/c.a" "$T/g.a" && ok "default layout: GPU archive = CPU archive" || bad "default layout: GPU archive = CPU archive"
fi

# refusals and damaged archives
run "$CPU" c "$T/ft.st" "$T/b7.a" -s 0 -l 16 || bad "write a BFR7 archive"
refuse "--gpu decoding of a BFR7 archive" "$GPU" d "$T/b7.a" "$T/x" --gpu
refuse "--gpu with -l 16" "$GPU" c "$T/ft.st" "$T/x" -s 1 -l 16 --gpu
run "$CPU" c "$T/ft.st" "$T/m.a" --ref "$T/base.st" -b 1 -s 1 -l 32 || bad "write the archive to damage"
sz=$(wc -c < "$T/m.a"); i=0; wrong=0; err=0; same=0
while [ $i -lt 12 ]; do
    cp "$T/m.a" "$T/mm.a"
    off=$(( (sz / 13) * (i + 1) )); printf '\377' | dd of="$T/mm.a" bs=1 seek=$off conv=notrunc 2>/dev/null
    rm -f "$T/o"
    if "$GPU" d "$T/mm.a" "$T/o" --ref "$T/base.st" --gpu >/dev/null 2>&1; then
        if cmp -s "$T/o" "$T/ft.st"; then same=$((same+1)); else wrong=$((wrong+1)); fi
    else err=$((err+1)); fi
    i=$((i+1))
done
[ $wrong -eq 0 ] && ok "12 damaged archives: $err refused, $same decoded unchanged, none wrong" || bad "damaged archives: $wrong decoded to wrong output"

echo "$pass passed, $fail failed"
[ $fail -eq 0 ]
