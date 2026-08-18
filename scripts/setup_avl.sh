#!/usr/bin/env bash
# Builds AVL 3.52 from source on Ubuntu, for the pipeline in src/avlnn/avl_driver.py.
#
# The build recipe below follows AVL 3.52's own README ("Build sequence") and was written
# against the actual contents of avl3.52.tgz:
#   1. plotlib:  select the gfortran double-precision config, make  -> libPlt_gDP.a
#   2. eispack:  make -f Makefile.gfortran DP=-fdefault-real-8      -> libeispack.a
#   3. bin:      make -f Makefile.gfortranDP avl                    -> bin/avl
# Makefile.gfortranDP compiles AVL's bundled LAPACK subset (matrix-lapacksubs.f), so no
# system BLAS/LAPACK packages are needed; graphics need only libX11.
#
# Xfoil is NOT needed by the optimization pipeline; pass --with-xfoil to build it too
# (best-effort -- its build was not verified the same way).
#
# If the download 404s (MIT occasionally renames files), fetch the current .tgz by hand from
# https://web.mit.edu/drela/Public/web/avl/ into vendor/ and re-run.
set -euo pipefail
trap 'echo "!! setup_avl.sh failed at line $LINENO (command: $BASH_COMMAND)" >&2' ERR

AVL_TGZ="avl3.52.tgz"
AVL_URL="https://web.mit.edu/drela/Public/web/avl/${AVL_TGZ}"
XFOIL_TGZ="xfoil6.996.tgz"
XFOIL_URL="https://web.mit.edu/drela/Public/web/xfoil/${XFOIL_TGZ}"

WITH_XFOIL=0
[ "${1:-}" = "--with-xfoil" ] && WITH_XFOIL=1

VENDOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/vendor"
mkdir -p "$VENDOR_DIR"
cd "$VENDOR_DIR"

# Build dependencies are installed by the caller.
#
# In Docker, these are installed in the Dockerfile.
# On a normal Ubuntu system, install:
#   build-essential gfortran libx11-dev libxext-dev xvfb curl
#
# xvfb: AVL's MODE eigenanalysis unconditionally draws an X11 plot and
# segfaults without a display; avl_driver.py wraps AVL in xvfb-run on
# headless machines.
echo "==> Checking build dependencies"

for cmd in gfortran make curl; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "!! Required command '$cmd' was not found." >&2
        echo "   Install the AVL build dependencies before running this script." >&2
        exit 1
    fi
done

fetch() {
  local url="$1" out="$2"
  if [ -f "$out" ]; then
    echo "==> $out already present, skipping download"
    return
  fi
  echo "==> Downloading $url"
  if ! curl -fsSL "$url" -o "$out"; then
    echo "!! Could not download $url automatically (file may have been renamed upstream)."
    echo "   Grab the current file by hand from the MIT page and save it as:"
    echo "   $VENDOR_DIR/$out"
    echo "   then re-run this script."
    exit 1
  fi
}

# Prints the top-level directory inside a tarball, skipping macOS AppleDouble junk entries
# ("._Foo") that the AVL archive leads with. Reads the listing into a variable first --
# `tar tzf | head` + pipefail would SIGPIPE tar and abort the script under set -e.
tarball_root_dir() {
  local listing entry
  listing="$(tar tzf "$1" --warning=no-unknown-keyword)"
  while IFS= read -r entry; do
    case "$entry" in
      ._*|*/._*) continue ;;
      *) printf '%s\n' "${entry%%/*}"; return 0 ;;
    esac
  done <<< "$listing"
  echo "!! could not find a top-level directory in $1" >&2
  return 1
}

# --- AVL ---
fetch "$AVL_URL" "$AVL_TGZ"
AVL_ROOT="$(tarball_root_dir "$AVL_TGZ")"
if [ ! -d "$AVL_ROOT" ]; then
  echo "==> Extracting $AVL_TGZ (into $AVL_ROOT)"
  tar xzf "$AVL_TGZ" --warning=no-unknown-keyword --exclude='._*'
fi
AVL_SRC_DIR="$VENDOR_DIR/$AVL_ROOT"

# The plotlib Makefile's default target is 'help' (prints usage, exits 0); the named
# machine targets copy the matching config.make and then build via Makefile.all.
echo "==> [1/3] Building plotlib (libPlt_gDP.a)"
make -C "$AVL_SRC_DIR/plotlib" gfortranDP

# eispack must be built with NO precision flags: AVL's EIGSOL passes explicit REAL*8 arrays
# and eispack.f declares DOUBLE PRECISION, which already match. Adding -fdefault-real-8 (as
# an earlier version of this script did) promotes eispack's DOUBLE PRECISION to 16 bytes,
# which segfaults inside RG/ELTRAN at eigenmode time. Force a rebuild in case a stale
# wrongly-built libeispack.a is lying around.
echo "==> [2/3] Building eispack (libeispack.a)"
rm -f "$AVL_SRC_DIR/eispack/"*.o "$AVL_SRC_DIR/eispack/"*.a
make -C "$AVL_SRC_DIR/eispack" -f Makefile.gfortran

# Force a relink: the 'avl' make target doesn't list libeispack.a as a prerequisite, so a
# rebuilt eispack would otherwise never make it into an existing binary.
echo "==> [3/3] Building AVL"
rm -f "$AVL_SRC_DIR/bin/avl"
make -C "$AVL_SRC_DIR/bin" -f Makefile.gfortranDP avl

AVL_BIN="$AVL_SRC_DIR/bin/avl"
if [ ! -x "$AVL_BIN" ]; then
  echo "!! Build finished but $AVL_BIN was not produced -- check the make output above."
  exit 1
fi

echo "==> Smoke-testing the AVL binary"
printf 'QUIT\n' | "$AVL_BIN" >/tmp/avl_smoke_test.log 2>&1 || true
head -30 /tmp/avl_smoke_test.log

# --- Xfoil (optional, best-effort) ---
XFOIL_NOTE=""
if [ "$WITH_XFOIL" = 1 ]; then
  fetch "$XFOIL_URL" "$XFOIL_TGZ"
  XFOIL_ROOT="$(tarball_root_dir "$XFOIL_TGZ")"
  if [ ! -d "$XFOIL_ROOT" ]; then
    echo "==> Extracting $XFOIL_TGZ"
    tar xzf "$XFOIL_TGZ" --warning=no-unknown-keyword --exclude='._*'
  fi
  XFOIL_SRC_DIR="$VENDOR_DIR/$XFOIL_ROOT"

  # gfortran >= 10 hard-errors on the legacy argument-rank mismatches in Xfoil's
  # sources (e.g. the READR calls in xgdes.f).  A PATH shim appends
  # -fallow-argument-mismatch to every compile, without having to guess which
  # flag variable each of Xfoil's Makefiles uses.
  GFORTRAN_REAL="$(command -v gfortran)"
  SHIM_DIR="$XFOIL_SRC_DIR/gfortran-shim"
  mkdir -p "$SHIM_DIR"
  printf '#!/bin/sh\nexec %s -fallow-argument-mismatch "$@"\n' "$GFORTRAN_REAL" \
    > "$SHIM_DIR/gfortran"
  chmod +x "$SHIM_DIR/gfortran"

  # plotlib's default make target only prints a usage menu (same trap as AVL's
  # plotlib); the real target is gfortranDP, matching bin's -fdefault-real-8
  # build.  Symlink the other names Xfoil's bin Makefile may link against.
  echo "==> Building Xfoil: plotlib"
  PATH="$SHIM_DIR:$PATH" make -C "$XFOIL_SRC_DIR/plotlib" gfortranDP
  for lib in libPlt.a libPlt_DP.a; do
    [ -f "$XFOIL_SRC_DIR/plotlib/$lib" ] \
      || ln -sf libPlt_gDP.a "$XFOIL_SRC_DIR/plotlib/$lib"
  done

  for sub in orrs/bin bin; do
    if [ -d "$XFOIL_SRC_DIR/$sub" ]; then
      echo "==> Building Xfoil: $sub"
      PATH="$SHIM_DIR:$PATH" make -C "$XFOIL_SRC_DIR/$sub"
    fi
  done

  XFOIL_BIN_PATH="$XFOIL_SRC_DIR/bin/xfoil"
  if [ ! -x "$XFOIL_BIN_PATH" ]; then
    echo "!! Xfoil build finished but $XFOIL_BIN_PATH was not produced -- check the make output above."
    exit 1
  fi
  XFOIL_NOTE="
    export XFOIL_BIN=\"$XFOIL_BIN_PATH\""
fi

cat <<EOF

==> Done. Add this to your shell profile (~/.bashrc):

    export AVL_BIN="$AVL_BIN"$XFOIL_NOTE

src/avlnn/avl_driver.py reads the AVL_BIN environment variable (falls back to "avl" on PATH).

Next step: run a real design through the pipeline once and diff AVL's actual stdout against
what src/avlnn/parse.py expects -- see that file's module docstring.
EOF
