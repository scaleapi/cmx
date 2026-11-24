#!/bin/sh
# Tests of the cmx program against the standalone codecs.  Usage: tests/cmx.sh [dir with cmx, cmxc, bf16ref]
#  - cmx writes byte-identical archives to cmxc and to bf16ref, and each binary decodes the others' archives;
#  - cmx picks bf16ref for safetensors files with BF16 tensors (and for --ref) and cmxc otherwise;
#  - round trips standalone and given a base, on a synthetic safetensors pair (needs python3);
#  - the BFR8 sub-stream layouts (-s, -l) round-trip and -s 0 -l 16 writes BFR7;
#  - options that do not apply, wrong references and non-archives are refused.
# Each cmxc run maps about 3.4 GiB of model tables.
set -u
B=${1:-.}
for x in cmx cmxc bf16ref; do [ -x "$B/$x" ] || { echo "binary not found: $B/$x" >&2; exit 2; }; done
CMX=$B/cmx; CMXC=$B/cmxc; BFR=$B/bf16ref
T=$(mktemp -d "${TMPDIR:-/tmp}/cmx-test.XXXXXX") || exit 2
trap 'rm -rf "$T"' EXIT INT TERM
pass=0; fail=0
ok()  { pass=$((pass+1)); echo "ok   $1"; }
bad() { fail=$((fail+1)); echo "FAIL $1"; }
check() { if "$@" >/dev/null 2>>"$T/err"; then :; else echo "  command failed: $*" >> "$T/err"; return 1; fi; }
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

# ---- text: cmx = cmxc, both ways
check "$CMXC" c "$T/text" "$T/t.cmxc" && check "$CMX" c "$T/text" "$T/t.cmx"
same "text: cmx archive = cmxc archive" "$T/t.cmxc" "$T/t.cmx"
tag "text: compressed with cmxc" "$T/t.cmx" CMXC
check "$CMXC" d "$T/t.cmx" "$T/t1" && same "text: cmxc decodes cmx" "$T/text" "$T/t1"
check "$CMX" d "$T/t.cmxc" "$T/t2" && same "text: cmx decodes cmxc" "$T/text" "$T/t2"
CMX_NIBO=0 check "$CMXC" c "$T/text" "$T/tm.cmxc" && check "$CMX" c "$T/text" "$T/tm.cmx" --max
same "text --max: cmx archive = cmxc with CMX_NIBO=0" "$T/tm.cmxc" "$T/tm.cmx"
check "$CMX" d "$T/tm.cmx" "$T/t3" && same "text --max: round trip" "$T/text" "$T/t3"
CMXC_THREADS=3 CMXC_MINBLK=16384 check "$CMXC" c "$T/text" "$T/tb.cmxc" && CMXC_MINBLK=16384 check "$CMX" c "$T/text" "$T/tb.cmx" -t 3
same "text -t 3: cmx archive = cmxc with CMXC_THREADS=3" "$T/tb.cmxc" "$T/tb.cmx"
check "$CMX" d "$T/tb.cmx" "$T/t4" -t 2 && same "text -t 3: round trip" "$T/text" "$T/t4"
check "$CMX" c "$T/random" "$T/r.cmx" && check "$CMXC" d "$T/r.cmx" "$T/r1" && same "random: cmx -> cmxc round trip" "$T/random" "$T/r1"

# ---- weights alone: cmx = bf16ref, both ways (bf16ref archives do not depend on the thread count)
check "$BFR" c "$T/ft.safetensors" "$T/w.bfr" && check "$CMX" c "$T/ft.safetensors" "$T/w.cmx" -t 4
same "weights: cmx archive = bf16ref archive" "$T/w.bfr" "$T/w.cmx"
tag "weights: compressed with bf16ref (BFR8 by default)" "$T/w.cmx" BFR8
check "$BFR" d "$T/w.cmx" "$T/w1" && same "weights: bf16ref decodes cmx" "$T/ft.safetensors" "$T/w1"
check "$CMX" d "$T/w.bfr" "$T/w2" -t 3 && same "weights: cmx decodes bf16ref" "$T/ft.safetensors" "$T/w2"
check "$BFR" c "$T/ft.safetensors" "$T/wb.bfr" -b 1 && check "$CMX" c "$T/ft.safetensors" "$T/wb.cmx" -b 1
same "weights -b 1: cmx archive = bf16ref archive" "$T/wb.bfr" "$T/wb.cmx"

# ---- weights given the base
check "$BFR" c "$T/ft.safetensors" "$T/d.bfr" --ref "$T/base.safetensors" && check "$CMX" c "$T/ft.safetensors" "$T/d.cmx" --ref "$T/base.safetensors" -t 4
same "given the base: cmx archive = bf16ref archive" "$T/d.bfr" "$T/d.cmx"
check "$BFR" d "$T/d.cmx" "$T/d1" --ref "$T/base.safetensors" && same "given the base: bf16ref decodes cmx" "$T/ft.safetensors" "$T/d1"
check "$CMX" d "$T/d.bfr" "$T/d2" --ref "$T/base.safetensors" && same "given the base: cmx decodes bf16ref" "$T/ft.safetensors" "$T/d2"
a=$(wc -c < "$T/w.cmx"); d=$(wc -c < "$T/d.cmx")
[ "$d" -lt "$a" ] && ok "given the base is smaller than alone ($d < $a bytes)" || bad "given the base is not smaller ($d vs $a)"

# ---- BFR8: sub-stream layouts, alone and given the base, small blocks so that blocks have several sub-streams
for lay in "-b 1 -s 1 -l 32" "-b 1 -s 1 -l 16" "-b 1 -s 0 -l 32" "-b 2 -s 3 -l 32"; do
    check "$CMX" c "$T/ft.safetensors" "$T/g.cmx" $lay && tag "BFR8 $lay: tag" "$T/g.cmx" BFR8
    check "$BFR" c "$T/ft.safetensors" "$T/g.bfr" $lay && same "BFR8 $lay: cmx archive = bf16ref archive" "$T/g.bfr" "$T/g.cmx"
    check "$BFR" d "$T/g.cmx" "$T/g1" && same "BFR8 $lay: round trip" "$T/ft.safetensors" "$T/g1"
    check "$CMX" c "$T/ft.safetensors" "$T/gr.cmx" --ref "$T/base.safetensors" $lay
    check "$CMX" d "$T/gr.cmx" "$T/g2" --ref "$T/base.safetensors" -t 3 && same "BFR8 $lay given the base: round trip" "$T/ft.safetensors" "$T/g2"
done
check "$CMX" c "$T/ft.safetensors" "$T/g7.cmx" -s 0 -l 16 && tag "-s 0 -l 16 writes BFR7" "$T/g7.cmx" BFR7
check "$BFR" c "$T/ft.safetensors" "$T/g7.bfr" -s 0 -l 16 && same "-s 0 -l 16: cmx archive = bf16ref archive" "$T/g7.bfr" "$T/g7.cmx"
check "$BFR" d "$T/g7.cmx" "$T/g7o" && same "-s 0 -l 16: round trip" "$T/ft.safetensors" "$T/g7o"
# a BFR8 archive with a damaged index or a damaged layout field is refused
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
