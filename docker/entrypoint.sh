#!/bin/sh
# Entrypoint for the flowcoder runner image.
#
# Applies the container's mount conventions as defaults, then hands everything
# else straight to the CLI, so `docker run flowcoder mycommand "some text"` is
# the whole interface.
#
# Overrides:
#   FLOWCODER_SEARCH_PATH  where flowchart JSON lives   (default /work/commands)
#   FLOWCODER_CWD          where Claude does its work   (default /work/workspace)
#   FLOWCODER_JSON         1 = --json, 0 = human output (default 1)
set -eu

: "${FLOWCODER_SEARCH_PATH:=/work/commands}"
: "${FLOWCODER_CWD:=/work/workspace}"
: "${FLOWCODER_JSON:=1}"

# Log the endpoint on stderr.  A testbench run that silently fell back to the
# real API is expensive to discover later and cheap to spot here.
echo "[flowcoder] endpoint=${ANTHROPIC_BASE_URL:-https://api.anthropic.com (DEFAULT — billed)}" >&2
echo "[flowcoder] model=${ANTHROPIC_MODEL:-<cli default>}" >&2

if [ "${FLOWCODER_JSON}" = "1" ]; then
    set -- --json "$@"
fi

# --search-path is append-style and --cwd is last-wins, so a caller passing
# either of them extends or overrides these defaults rather than conflicting.
exec flowcoder \
    --search-path "${FLOWCODER_SEARCH_PATH}" \
    --cwd "${FLOWCODER_CWD}" \
    --no-color \
    "$@"
