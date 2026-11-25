#!/bin/sh
# Verify the contents and first-run experience of the local crates.io candidate.
set -eu
cd "$(dirname "$0")/.."
repo=$(pwd)
metadata=$(cargo metadata --locked --no-deps --format-version 1)
target=$(printf '%s' "$metadata" | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')
package=$(printf '%s' "$metadata" | python3 -c 'import json,sys; p,=json.load(sys.stdin)["packages"]; print("{name}-{version}".format(**p))')

cargo test --locked
python3 -m unittest discover -s tests -p 'test_*.py'
cargo package --locked "$@"
archive="$target/package/$package.crate"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
tar -xzf "$archive" -C "$work"
cd "$work/$package"
export CARGO_TARGET_DIR="$target/release-check"
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps
cargo install --path . --locked --root "$work/install"
if [ "${SOUL_DOCKER_SMOKE:-0}" = 1 ]; then
    export SOUL_BIN_DIR="$work/install/bin"
    python3 "$repo/scripts/smoke.py"
    python3 "$repo/scripts/gang_smoke.py"
    python3 "$repo/scripts/resize_smoke.py"
fi

mkdir "$work/first-run"
cd "$work/first-run"
soul="$work/install/bin/soul"
node="$work/install/bin/soul-node"
"$soul" --version
"$node" --version
"$soul" --help > /dev/null
"$node" --help > /dev/null
"$soul" example inventory > inventory.json
"$soul" validate inventory.json
"$soul" simulate inventory.json > decisions.txt
"$soul" example job > job.json
"$soul" example trace > trace.json
"$soul" replay trace.json > replay.json
python3 - "$soul" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

decisions = Path("decisions.txt").read_text()
decoder = json.JSONDecoder()
first, end = decoder.raw_decode(decisions)
second = json.loads(decisions[end:])
if [first["job"], second["job"]] != ["opportunistic", "training"]:
    raise SystemExit("Unexpected scheduling decisions")
policies = [r["policy"] for r in json.loads(Path("replay.json").read_text())]
if policies != ["fixed", "fallback", "adaptive", "cost_aware"]:
    raise SystemExit(f"Unexpected replay policies: {policies}")
result = subprocess.run([sys.argv[1], "serve-sim", "invalid.sqlite", "inventory.json", "invalid-address"],
                        capture_output=True, timeout=5)
if result.returncode == 0 or Path("invalid.sqlite").exists():
    raise SystemExit("An invalid listener must fail before creating controller state")
PY
printf 'Verified %s\n' "$archive"
