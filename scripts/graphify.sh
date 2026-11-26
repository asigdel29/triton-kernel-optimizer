#!/usr/bin/env bash
# Build the source map from tracked Python files in the working tree.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
root=$PWD
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
source_dir="$stage/${root##*/}"
mkdir -p "$source_dir"
git ls-files -z -- '*.py' | while IFS= read -r -d '' file; do
    mkdir -p "$source_dir/$(dirname "$file")"
    cp "$file" "$source_dir/$file"
done
cd "$source_dir"
GRAPHIFY_BACKEND=none graphify update .
python3 - <<'PY'
import json
from pathlib import Path

graph = json.loads(Path("graphify-out/graph.json").read_text())
assert graph["nodes"] and (graph.get("links") or graph.get("edges")), "Empty source map"
PY
mkdir -p "$root/graphify-out"
cp graphify-out/{graph.json,graph.html,GRAPH_REPORT.md} "$root/graphify-out/"
