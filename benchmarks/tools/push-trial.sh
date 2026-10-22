#!/usr/bin/env bash
# Push-load benchmark on the trial cluster (ADR-063), across real hosts:
# start the trial daemons, write synthetic sealed shards on each source
# host, run `nest push bench` with one rank per Spark, stop the daemons.
#   benchmarks/tools/push-trial.sh [GIB_PER_SOURCE] [RUN_BYTES]
# PUSH_TRIAL_RUNS="all one" runs each load in turn: "all" pushes every
# source's shard, "one" only ostrich's (one disk's rate alone).
# Run under the Sparks' hardware lock when other measured work shares them.
# Trial layout only (scripts/cluster.env): no production data is touched.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/cluster.env
gib=${1:-8}
run=${2:-4096}
sources=(ostrich dodo emu kiwi rhea moa)
ranks=(ostrich dodo emu kiwi rhea moa)
bin=.local/lib/sparknest
svc=sparknestd-$SPARKNEST_CLUSTER.service
nest() { local h=$1; shift; ssh "$h" "~/$bin/nest --config ~/sparknest-test/state/node.toml $*"; }

stop_all() {
  for h in raptor "${sources[@]}"; do
    ssh "$h" "systemctl --user stop $svc" || true
  done
}
trap stop_all EXIT

for h in raptor "${sources[@]}"; do ssh "$h" "systemctl --user start $svc"; done
for h in raptor "${sources[@]}"; do
  ssh "$h" "timeout 120 ~/$bin/nest --config ~/sparknest-test/state/node.toml wait-ready" >/dev/null
done
echo "== trial cluster serving"

# One shard per source, written through that host's own mount (it stays
# there), then sealed.
files=()
for h in "${sources[@]}"; do
  f=push-bench/$h.bin
  ssh "$h" "mkdir -p ~/sparknest-test/mnt/push-bench && \
    if [ \$(stat -c %s ~/sparknest-test/mnt/$f 2>/dev/null || echo 0) -ne $((gib << 30)) ]; then \
      rm -f ~/sparknest-test/mnt/$f; \
      dd if=/dev/urandom of=/dev/shm/push-seed bs=1M count=64 status=none; \
      for i in \$(seq $((gib * 16))); do cat /dev/shm/push-seed; done > ~/sparknest-test/mnt/$f; \
      rm -f /dev/shm/push-seed; \
    fi; \
    python3 -c 'import os,sys; os.setxattr(sys.argv[1], \"user.sparknest.sealed\", b\"1\")' ~/sparknest-test/mnt/$f"
  files+=("/$f")
done
# PUSH_TRIAL_ARCHIVE=DIR adds a shard held only in an archive store at DIR
# on raptor (its own stream beside raptor's disk, ADR-069).
if [ -n "${PUSH_TRIAL_ARCHIVE:-}" ]; then
  f=push-bench/archived.bin
  ssh raptor "mkdir -p ${PUSH_TRIAL_ARCHIVE}"
  nest raptor store add trial-archive "${PUSH_TRIAL_ARCHIVE}" --gateways raptor || true
  ssh raptor "if [ \$(stat -c %s ~/sparknest-test/mnt/$f 2>/dev/null || echo 0) -ne $((gib << 30)) ]; then \
      rm -f ~/sparknest-test/mnt/$f; \
      dd if=/dev/urandom of=/dev/shm/push-seed bs=1M count=64 status=none; \
      for i in \$(seq $((gib * 16))); do cat /dev/shm/push-seed; done > ~/sparknest-test/mnt/$f; \
      rm -f /dev/shm/push-seed; \
    fi; \
    python3 -c 'import os,sys; os.setxattr(sys.argv[1], \"user.sparknest.sealed\", b\"1\")' ~/sparknest-test/mnt/$f"
  sleep 3
  nest raptor offload /$f --store trial-archive --wait
  files+=("/$f")
fi
sleep 3
echo "== shards: ${files[*]}"
nest raptor ls /push-bench

# Rank consumers on each rank host, then the coordinator on raptor.
hosts=$(IFS=,; echo "${ranks[*]}")
bench() {
  local tag=$1; shift
  local out; out=$(mktemp)
  echo "== run $tag: $*"
  ssh raptor "~/$bin/nest --config ~/sparknest-test/state/node.toml --json push bench $* --hosts $hosts --run $run" \
    > "$out" 2> "$out.err" &
  local coord=$!
  for _ in $(seq 120); do
    [ "$(grep -c 'nest push rank' "$out.err" 2>/dev/null || true)" -ge "${#ranks[@]}" ] && break
    sleep 0.5
  done
  local pids=()
  while read -r line; do
    local h=${line%%:*}; h=${h#on }
    local args=${line#*nest push rank }
    ssh "$h" "~/$bin/nest --config ~/sparknest-test/state/node.toml push rank $args" \
      > "$out.$h" 2>&1 &
    pids+=($!)
  done < <(grep 'nest push rank' "$out.err")
  wait "$coord" || cat "$out.err"
  for p in "${pids[@]}"; do wait "$p" || true; done
  echo "== coordinator ($tag)"
  cat "$out"
  # The hub's record of the load (per-source, per-rank bins), kept.
  local group; group=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["group"])' "$out" 2>/dev/null || true)
  if [ -n "$group" ]; then
    mkdir -p ~/build/push-loads
    ssh raptor "curl -s --unix-socket ~/sparknest-test/state/api.sock http://x/v1/push/loads/$group" \
      > ~/build/push-loads/trial-$tag-$group.json
    echo "== record: ~/build/push-loads/trial-$tag-$group.json"
  fi
  echo "== ranks ($tag)"
  for h in "${ranks[@]}"; do grep -h '"rank"' "$out.$h" || cat "$out.$h"; done
}
for r in ${PUSH_TRIAL_RUNS:-all}; do
  case $r in
    all) bench all "${files[@]}" ;;
    one) bench one "/push-bench/ostrich.bin" ;;
  esac
done
for h in "${sources[@]}"; do
  echo "$h: $(ssh "$h" "curl -s --unix-socket ~/sparknest-test/state/api.sock http://x/v1/push")"
done
# PUSH_TRIAL_HOLD=SECONDS keeps the trial cluster up afterwards (its web UI
# on raptor:$SPARKNEST_API_PORT shows the load), still under the lock.
if [ -n "${PUSH_TRIAL_HOLD:-}" ]; then
  echo "== holding the trial cluster for ${PUSH_TRIAL_HOLD}s"
  touch ~/build/push-trial-holding
  sleep "$PUSH_TRIAL_HOLD"
  rm -f ~/build/push-trial-holding
fi
