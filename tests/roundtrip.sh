#!/bin/sh
# Round-trip and malformed-input tests for cmxc.  Usage: tests/roundtrip.sh [path/to/cmxc]
# Each cmxc run maps about 3.4 GiB of model tables, so the multi-block cases need several GiB.
set -u
CMXC=${1:-./cmxc}
[ -x "$CMXC" ] || { echo "cmxc binary not found: $CMXC" >&2; exit 2; }
T=$(mktemp -d "${TMPDIR:-/tmp}/cmxc-test.XXXXXX") || exit 2
trap 'rm -rf "$T"' EXIT INT TERM
SRC=$(dirname "$0")/../cmxc.c
pass=0; fail=0
ok()  { pass=$((pass+1)); echo "ok   $1"; }
bad() { fail=$((fail+1)); echo "FAIL $1"; }

# rt NAME FILE [ENV=VALUE ...]: compress with the given settings, decompress, compare.
rt() {
    name=$1; in=$2; shift 2
    if env "$@" "$CMXC" c "$in" "$T/$name.cx" >/dev/null 2>"$T/$name.err" &&
       "$CMXC" d "$T/$name.cx" "$T/$name.out" >/dev/null 2>>"$T/$name.err" &&
       cmp -s "$in" "$T/$name.out"; then ok "round-trip $name"
    else bad "round-trip $name"; cat "$T/$name.err"; fi
}
# refuse NAME FILE: the decoder must exit non-zero.
refuse() {
    if "$CMXC" d "$2" "$T/$1.out" >/dev/null 2>&1; then bad "refuse $1"; else ok "refuse $1"; fi
}
# poke FILE OFFSET OCTAL: overwrite one byte in place.
poke() { printf "\\$3" | dd of="$1" bs=1 seek="$2" conv=notrunc 2>/dev/null; }

# ---- inputs
: > "$T/empty"
printf 'x' > "$T/one"
cp "$SRC" "$T/text"
head -c 65536 /dev/urandom > "$T/random"
# E8/E9 bytes: overlapping opcodes, call/jump operands with 00/FF high bytes, and
# opcodes in the last four bytes of the buffer, where no full operand follows.
i=0; : > "$T/e8e9"
while [ $i -lt 2048 ]; do
    printf '\350\350\351\350\000\000\000\351\377\377\377\377\350\020\040\000\000\125\110\211' >> "$T/e8e9"
    i=$((i+1))
done
printf '\350\351\350' >> "$T/e8e9"

# ---- round trips
rt empty    "$T/empty"
rt one      "$T/one"
rt text     "$T/text"
rt random   "$T/random"
rt e8e9     "$T/e8e9"   CMX_E8=1
rt e8e9auto "$T/e8e9"
rt e8e9off  "$T/e8e9"   CMX_E8=0
rt blocks   "$T/text"   CMXC_THREADS=4 CMXC_MINBLK=16384
rt primed   "$T/text"   CMXC_THREADS=4 CMXC_MINBLK=16384 CMXC_PRIME=1000
rt primedx  "$T/e8e9"   CMXC_THREADS=3 CMXC_MINBLK=8192  CMXC_PRIME=4096 CMX_E8=1

# ---- malformed archives (built from the valid "text" archive)
A="$T/text.cx"
refuse notarchive "$T/text"
refuse emptyfile  "$T/empty"
head -c 20 "$A" > "$T/short";  refuse short-header "$T/short"
sz=$(wc -c < "$A")
head -c $((sz-100)) "$A" > "$T/trunc"; refuse truncated "$T/trunc"
cp "$A" "$T/magic";   poke "$T/magic" 0 130;   refuse bad-magic   "$T/magic"
cp "$A" "$T/version"; poke "$T/version" 4 011; refuse bad-version "$T/version"
cp "$A" "$T/nblk";    poke "$T/nblk" 19 177;   refuse bad-nblocks "$T/nblk"
cp "$A" "$T/hdrlen";  poke "$T/hdrlen" 23 177; refuse bad-hdrlen  "$T/hdrlen"
cp "$A" "$T/olen";    poke "$T/olen" 13 177;   refuse bad-origlen "$T/olen"

echo "$pass passed, $fail failed"
[ $fail -eq 0 ]
