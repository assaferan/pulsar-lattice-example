#!/usr/bin/env bash
#
# Convenience wrapper: activate the g6k virtualenv and run the folding
# pipeline (or any python command) without having to remember to source the
# environment first.
#
# Usage:
#   ./run.sh                 # runs fermi_fold.py on data/data.npy
#   ./run.sh some_script.py  # runs any python script inside the g6k env
#
# Override the environment location with G6K_ACTIVATE if it lives elsewhere:
#   G6K_ACTIVATE=/path/to/g6k/activate ./run.sh
#
G6K_ACTIVATE="${G6K_ACTIVATE:-$HOME/g6k/activate}"

if [ ! -f "$G6K_ACTIVATE" ]; then
    echo "g6k environment not found at: $G6K_ACTIVATE" >&2
    echo "Build it with g6k's bootstrap.sh, or set G6K_ACTIVATE to its 'activate' script." >&2
    exit 1
fi

# g6k's activate script is not safe under 'set -e'/'set -u' (it runs
# 'unalias python' and references LD_LIBRARY_PATH), so source it plainly.
# shellcheck disable=SC1090
source "$G6K_ACTIVATE"

cd "$(dirname "$0")"

if [ "$#" -eq 0 ]; then
    exec python3 fermi_fold.py
else
    exec python3 "$@"
fi
