#!/bin/bash
# Download the replacement datasets (MANIFEST.md) to this machine and verify them.
# Each source goes in its own data/_incoming/<name>/ dir as `download`; nothing in them is executed.
# Re-runnable: verified sources are skipped, partial downloads resume (curl -C -).
# usage: bash tools/fetch_local.sh [name ...]      (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."
IN=data/_incoming
mkdir -p "$IN"

# name | url | check (sha256:<hex> | md5:<hex> | size:<bytes> | none)
SOURCES=(
"natural|https://senthilpurushwalkam.com/publications/compositional/compositional_split_natural.tar.gz|sha256:8fde559df6483c766531bf5addc6d61a5742ddc780d013e61673949ad2e0d7e1"
"vrd_json|https://cs.stanford.edu/people/ranjaykrishna/vrd/json_dataset.zip|sha256:aef6b04c81a33b6dd9d702d67b5aacec297ec3359ccdb20d7ce1e29f841d3f55"
"xlsa17|http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip|size:921803497"
"glove|https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip|none"
"mitstates_hf|https://huggingface.co/datasets/AaronWu901225/MIT_states_UT_zappos/resolve/d3f1d1b05431a92aed318c5dd9f660866fcfa2a6/dataset.zip|sha256:ebd1b4bce5fcd9918f12d78ab7f1160dd3cccb61d9887301958d5bca96b03d4b"
"nesy4vrd|https://zenodo.org/api/records/17076303/files/NeSy4VRD.zip/content|md5:4c83779814209077e25a7a707f441eed"
)

# Portable helpers: macOS (shasum, md5 -q, stat -f) or GNU/Git Bash (sha256sum, md5sum, stat -c).
sha256_of() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
md5_of()    { if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }
size_of()   { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }

verify() {
  local f=$1 kind=${2%%:*} val=${2#*:}
  case $kind in
    sha256) [ "$(sha256_of "$f")" = "$val" ] ;;
    md5)    [ "$(md5_of "$f")" = "$val" ] ;;
    size)   [ "$(size_of "$f")" = "$val" ] ;;
    none)   true ;;
  esac
}

want=" $* "
for s in "${SOURCES[@]}"; do
  IFS='|' read -r name url chk <<< "$s"
  [ $# -eq 0 ] || [[ $want == *" $name "* ]] || continue
  d=$IN/$name; f=$d/download; mkdir -p "$d"
  if [ -f "$d/VERIFIED" ]; then echo "[$name] already verified"; continue; fi
  echo "[$name] $(date '+%H:%M:%S') downloading $url"
  ok=0
  for try in 1 2 3 4 5 6 7 8; do
    if curl -fL --retry 5 --retry-delay 10 --connect-timeout 30 -sS -C - -A curl -o "$f" "$url"; then ok=1; break; fi
    echo "[$name] attempt $try failed; retrying"; sleep 20
  done
  [ $ok = 1 ] || { echo "ERROR: [$name] download failed"; exit 1; }
  verify "$f" "$chk" || { echo "ERROR: [$name] check $chk failed"; exit 1; }
  printf '%s  size=%s  sha256=%s  check=%s  url=%s\n' "$name" "$(size_of "$f")" \
    "$(sha256_of "$f")" "$chk" "$url" | tee "$d/VERIFIED"
done
echo "all done $(date '+%H:%M:%S')"
