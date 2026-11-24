#!/bin/sh
# Tests of the cmx program.  Usage: tests/cmx.sh [path/to/cmx]
#  - text and weights round trips, alone and given a base, on a synthetic safetensors pair (needs python3);
#  - options give the same archive as the settings they stand for (--max = CMX_NIBO=0, -t N = CMXC_THREADS=N),
#    and the weights codec's archives do not depend on the thread count;
#  - cmx picks the weights codec (bf16ref) for safetensors files with BF16 tensors (and for --ref), cmxc otherwise;
#  - the BFR8 sub-stream layouts (-s, -l) round-trip and -s 0 -l 16 writes BFR7;
#  - options that do not apply, wrong references and non-archives are refused.
# Each text-codec run maps about 3.4 GiB of model tables.
set -u
CMX=${1:-./cmx}
[ -x "$CMX" ] || { echo "cmx binary not found: $CMX" >&2; exit 2; }
T=$(mktemp -d "${TMPDIR:-/tmp}/cmx-test.XXXXXX") || exit 2
trap 'rm -rf "$T"' EXIT INT TERM
pass=0; fail=0
ok()  { pass=$((pass+1)); echo "ok   $1"; }
bad() { fail=$((fail+1)); echo "FAIL $1"; }
# check CMD...: run a command whose failure is itself a test failure (so a failed step never skips silently)
check() { if "$@" >/dev/null 2>>"$T/err"; then :; else bad "command failed: $*"; cat "$T/err"; : > "$T/err"; return 1; fi; }
same() { cmp -s "$2" "$3" && ok "$1" || { bad "$1"; cat "$T/err"; }; : > "$T/err"; }
tag() { [ "$(head -c 4 "$2")" = "$3" ] && ok "$1" || bad "$1 (tag $(head -c 4 "$2"), want $3)"; }
refuse() { name=$1; shift; if "$@" >/dev/null 2>&1; then bad "refuse $name"; else ok "refuse $name"; fi; }
: > "$T/err"

# ---- inputs
cp "$(dirname "$0")/../cmxc.c" "$T/text"
head -c 65536 /dev/urandom > "$T/random"
python3 - "$T" <<'EOF' || { echo "python3 is needed to write the safetensors inputs" >&2; exit 2; }
import json, random, struct, sys
T = sys.argv[1]; rnd = random.Random(7)
def bf16(x): return struct.pack("<f", x)[2:]
def tensors(shift):
    out = {}
    for i, (r, c, s) in enumerate([(256, 384, 0.02), (384, 256, 0.05), (512, 128, 0.01)]):
        ws = [rnd.gauss(0, s) for _ in range(r * c)]
        out[f"layers.{i}.weight"] = ("BF16", [r, c], ws)
    out["norm.weight"] = ("F32", [512], [1.0 + rnd.gauss(0, 0.01) for _ in range(512)])
    return out
def write(path, ts):
    hdr, body = {"__metadata__": {"format": "pt"}}, b""
    for k, (dt, shape, ws) in ts.items():
        data = b"".join(bf16(w) for w in ws) if dt == "BF16" else b"".join(struct.pack("<f", w) for w in ws)
        hdr[k] = {"dtype": dt, "shape": shape, "data_offsets": [len(body), len(body) + len(data)]}; body += data
    h = json.dumps(hdr, separators=(",", ":")).encode(); h += b" " * (-len(h) % 8)
    open(path, "wb").write(struct.pack("<Q", len(h)) + h + body)
base = tensors(0)
ft = {k: (dt, sh, [w + rnd.gauss(0, 0.001) * (abs(w) + 1e-3) if dt == "BF16" and rnd.random() < 0.7 else w for w in ws])
      for k, (dt, sh, ws) in base.items()}
write(f"{T}/base.safetensors", base); write(f"{T}/ft.safetensors", ft)
EOF

# ---- text
check "$CMX" c "$T/text" "$T/t.cmx" && tag "text: compressed with cmxc" "$T/t.cmx" CMXC
check "$CMX" d "$T/t.cmx" "$T/t1" && same "text: round trip" "$T/text" "$T/t1"
CMX_NIBO=0 check "$CMX" c "$T/text" "$T/tm.env" && check "$CMX" c "$T/text" "$T/tm.cmx" --max
same "text --max: archive = with CMX_NIBO=0" "$T/tm.env" "$T/tm.cmx"
check "$CMX" d "$T/tm.cmx" "$T/t3" && same "text --max: round trip" "$T/text" "$T/t3"
CMXC_THREADS=3 CMXC_MINBLK=16384 check "$CMX" c "$T/text" "$T/tb.env" && CMXC_MINBLK=16384 check "$CMX" c "$T/text" "$T/tb.cmx" -t 3
same "text -t 3: archive = with CMXC_THREADS=3" "$T/tb.env" "$T/tb.cmx"
check "$CMX" d "$T/tb.cmx" "$T/t4" -t 2 && same "text -t 3: round trip" "$T/text" "$T/t4"
check "$CMX" c "$T/random" "$T/r.cmx" && check "$CMX" d "$T/r.cmx" "$T/r1" && same "random: round trip" "$T/random" "$T/r1"

# ---- weights alone (the weights codec's archives do not depend on the thread count)
check "$CMX" c "$T/ft.safetensors" "$T/w1.cmx" -t 1 && check "$CMX" c "$T/ft.safetensors" "$T/w.cmx" -t 4
same "weights: -t 4 archive = -t 1 archive" "$T/w1.cmx" "$T/w.cmx"
tag "weights: compressed with bf16ref (BFR8 by default)" "$T/w.cmx" BFR8
check "$CMX" d "$T/w.cmx" "$T/w1" -t 1 && same "weights: round trip, 1 thread" "$T/ft.safetensors" "$T/w1"
check "$CMX" d "$T/w.cmx" "$T/w2" -t 3 && same "weights: round trip, 3 threads" "$T/ft.safetensors" "$T/w2"
check "$CMX" c "$T/ft.safetensors" "$T/wb.cmx" -b 1 && check "$CMX" d "$T/wb.cmx" "$T/w3" && same "weights -b 1: round trip" "$T/ft.safetensors" "$T/w3"

# ---- weights given the base
check "$CMX" c "$T/ft.safetensors" "$T/d1.cmx" --ref "$T/base.safetensors" -t 1 && check "$CMX" c "$T/ft.safetensors" "$T/d.cmx" --ref "$T/base.safetensors" -t 4
same "given the base: -t 4 archive = -t 1 archive" "$T/d1.cmx" "$T/d.cmx"
check "$CMX" d "$T/d.cmx" "$T/d1" --ref "$T/base.safetensors" && same "given the base: round trip" "$T/ft.safetensors" "$T/d1"
a=$(wc -c < "$T/w.cmx"); d=$(wc -c < "$T/d.cmx")
[ "$d" -lt "$a" ] && ok "given the base is smaller than alone ($d < $a bytes)" || bad "given the base is not smaller ($d vs $a)"

# ---- BFR8: sub-stream layouts, alone and given the base, small blocks so that blocks have several sub-streams
for lay in "-b 1 -s 1 -l 32" "-b 1 -s 1 -l 16" "-b 1 -s 0 -l 32" "-b 2 -s 3 -l 32"; do
    check "$CMX" c "$T/ft.safetensors" "$T/g.cmx" $lay && tag "BFR8 $lay: tag" "$T/g.cmx" BFR8
    check "$CMX" d "$T/g.cmx" "$T/g1" && same "BFR8 $lay: round trip" "$T/ft.safetensors" "$T/g1"
    check "$CMX" c "$T/ft.safetensors" "$T/gr.cmx" --ref "$T/base.safetensors" $lay
    check "$CMX" d "$T/gr.cmx" "$T/g2" --ref "$T/base.safetensors" -t 3 && same "BFR8 $lay given the base: round trip" "$T/ft.safetensors" "$T/g2"
done
check "$CMX" c "$T/ft.safetensors" "$T/g7.cmx" -s 0 -l 16 && tag "-s 0 -l 16 writes BFR7" "$T/g7.cmx" BFR7
check "$CMX" d "$T/g7.cmx" "$T/g7o" && same "-s 0 -l 16: round trip" "$T/ft.safetensors" "$T/g7o"
# a BFR8 archive with a damaged layout field is refused
check "$CMX" c "$T/ft.safetensors" "$T/gi.cmx" -b 1 -s 1 -l 32
cp "$T/gi.cmx" "$T/gl.cmx"; printf '\030' | dd of="$T/gl.cmx" bs=1 seek=44 conv=notrunc 2>/dev/null
refuse "BFR8 with 24 lanes" "$CMX" d "$T/gl.cmx" "$T/x"
refuse "-l 8" "$CMX" c "$T/ft.safetensors" "$T/x" -l 8

# ---- the choice of codec
check "$CMX" c "$T/ft.safetensors" "$T/ct.cmx" --codec text && tag "--codec text on safetensors" "$T/ct.cmx" CMXC
check "$CMX" d "$T/ct.cmx" "$T/c1" && same "--codec text on safetensors: round trip" "$T/ft.safetensors" "$T/c1"
check "$CMX" c "$T/text" "$T/cw.cmx" --codec weights && tag "--codec weights on text" "$T/cw.cmx" BFR8
check "$CMX" d "$T/cw.cmx" "$T/c2" && same "--codec weights on text: round trip" "$T/text" "$T/c2"
python3 - "$T" <<'EOF'
import sys; T = sys.argv[1]; s = open(f"{T}/base.safetensors", "rb").read()
open(f"{T}/f32.safetensors", "wb").write(s.replace(b'"dtype":"BF16"', b'"dtype":"F16" '))   # same length, no BF16
EOF
check "$CMX" c "$T/f32.safetensors" "$T/n.cmx" && tag "safetensors without BF16 tensors goes to cmxc" "$T/n.cmx" CMXC
python3 - "$T" <<'EOF'
import json, struct, sys; T = sys.argv[1]
def write(path, hdr, body):
    h = json.dumps(hdr, separators=(",", ":")).encode(); h += b" " * (-len(h) % 8)
    open(path, "wb").write(struct.pack("<Q", len(h)) + h + body)
body = b"".join(struct.pack("<f", 0.01 * i) for i in range(256))
# only F32 tensors, but "dtype": "BF16" in the metadata (and in a tensor name)
write(f"{T}/meta.safetensors", {"__metadata__": {"dtype": "BF16"}, "x \"dtype\":\"BF16\"": {"dtype": "F32", "shape": [256], "data_offsets": [0, 1024]}}, body)
EOF
check "$CMX" c "$T/meta.safetensors" "$T/m.cmx" && tag "\"dtype\": \"BF16\" in metadata or a name goes to cmxc" "$T/m.cmx" CMXC

# ---- the output: never the input or the reference; an existing output survives a failed run
cp "$T/ft.safetensors" "$T/keep.st"
refuse "output = input" "$CMX" c "$T/keep.st" "$T/keep.st"
same "output = input: the input is unchanged" "$T/ft.safetensors" "$T/keep.st"
ln "$T/keep.st" "$T/keep.link" 2>/dev/null && refuse "output = a hard link to the input" "$CMX" c "$T/keep.st" "$T/keep.link"
cp "$T/base.safetensors" "$T/base2.st"
refuse "output = the reference" "$CMX" c "$T/ft.safetensors" "$T/base2.st" --ref "$T/base2.st"
same "output = the reference: the reference is unchanged" "$T/base.safetensors" "$T/base2.st"
echo "precious" > "$T/dest"; cp "$T/dest" "$T/dest.orig"
refuse "decoding with the wrong reference" "$CMX" d "$T/d.cmx" "$T/dest" --ref "$T/ft.safetensors"
same "wrong reference: the existing output is unchanged" "$T/dest.orig" "$T/dest"
refuse "decoding a non-archive" "$CMX" d "$T/text" "$T/dest"
same "non-archive: the existing output is unchanged" "$T/dest.orig" "$T/dest"
check "$CMX" d "$T/w.cmx" "$T/dest" && same "a good run replaces the existing output" "$T/ft.safetensors" "$T/dest"
chmod 600 "$T/dest"; check "$CMX" d "$T/w.cmx" "$T/dest"
[ "$(ls -l "$T/dest" | cut -c1-10)" = "-rw-------" ] && ok "a replaced output keeps its permissions" || bad "a replaced output keeps its permissions"
check "$CMX" d "$T/w.cmx" /dev/null && ok "an output that is a device is written in place"
left=$(ls "$T" | grep -c '\.cmx-') ; [ "$left" -eq 0 ] && ok "no temporary files are left behind" || bad "temporary files left: $left"

# ---- a BFR8 index offset near 2^32 is refused at the index (it once wrapped past the check)
check "$CMX" c "$T/ft.safetensors" "$T/wrap.cmx" -b 1 -s 1 -l 32
python3 - "$T/wrap.cmx" "$T/wrap_bad.cmx" <<'EOF'
import struct, sys
a = bytearray(open(sys.argv[1], "rb").read()); sw, ns = struct.unpack_from("<I", a, 40)[0], struct.unpack_from("<I", a, 32)[0]
body = 48 + 48 * ns
for k in range(ns):
    rec = 48 + 48 * k; off, ln, roff, outlen, ck = struct.unpack_from("<5Q", a, rec + 8)
    if a[rec] == 1 and (ln // 2 + sw - 1) // sw >= 3:     # an alone block with at least 3 sub-streams
        ng = (ln // 2 + 7) // 8; nnz = struct.unpack_from("<I", a, body + 4)[0]
        struct.pack_into("<I", a, body + 12 + 5 * ng + 4 * nnz, 0xFFFFFFC0); break
    body += outlen
else: sys.exit("no alone block with 3 sub-streams")
open(sys.argv[2], "wb").write(a)
EOF
if "$CMX" d "$T/wrap_bad.cmx" "$T/x" 2>"$T/wrap.err"; then bad "index offset 0xFFFFFFC0 decoded"
elif grep -q "corrupt block (index)" "$T/wrap.err"; then ok "index offset 0xFFFFFFC0 is refused at the index"
else bad "index offset 0xFFFFFFC0 not refused at the index: $(cat "$T/wrap.err")"; fi

# ---- refusals
refuse "--ref with a cmxc archive" "$CMX" d "$T/t.cmx" "$T/x" --ref "$T/base.safetensors"
refuse "--ref with --codec text" "$CMX" c "$T/ft.safetensors" "$T/x" --ref "$T/base.safetensors" --codec text
refuse "--max with weights" "$CMX" c "$T/ft.safetensors" "$T/x" --max
refuse "missing reference" "$CMX" d "$T/d.cmx" "$T/x"
refuse "wrong reference" "$CMX" d "$T/d.cmx" "$T/x" --ref "$T/ft.safetensors"
refuse "not an archive" "$CMX" d "$T/text" "$T/x"
refuse "unknown codec" "$CMX" c "$T/text" "$T/x" --codec zip
refuse "unknown option" "$CMX" c "$T/text" "$T/x" --fast

echo "$pass passed, $fail failed"
[ $fail -eq 0 ]
