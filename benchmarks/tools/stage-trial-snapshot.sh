#!/usr/bin/env bash
# Copy one Hugging Face snapshot from the production hub into the trial
# cluster's hub at the same hub-relative path, for push-load trials with a
# real checkpoint (ADR-063). Read-only on production: files are read through
# /mnt/sparknest and written through each source host's trial mount, so the
# first copy of each lands on that host. Large files go round the sources
# weighted by disk speed (the fast disks take two for each one of the
# others'); small files go through raptor. Every file is sealed.
#   benchmarks/tools/stage-trial-snapshot.sh REPO_DIR COMMIT [SOURCES...]
#   e.g. models--org--name 1e4abd26… ostrich kiwi rhea:2 moa:2
# A source written host:N takes N shares. Leaves the trial cluster running
# (stop it when the test that uses it is done).
# Run under the Sparks' hardware lock: the copies cross the fabric.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/cluster.env
repo=$1 commit=$2
shift 2
if [ $# -gt 0 ]; then srcs=("$@"); else srcs=(ostrich kiwi rhea:2 moa:2); fi
bin=.local/lib/sparknest
svc=sparknestd-$SPARKNEST_CLUSTER.service
prod=/mnt/sparknest/hf-home/hub/$repo
trial=sparknest-test/mnt/hf-home/hub/$repo
seal='import os,sys
for p in sys.argv[1:]: os.setxattr(p, "user.sparknest.sealed", b"1")'

hosts=(raptor)
slots=()
for s in "${srcs[@]}"; do
  h=${s%%:*} n=1
  [ "$s" != "$h" ] && n=${s#*:}
  hosts+=("$h")
  for _ in $(seq "$n"); do slots+=("$h"); done
done
for h in ostrich dodo emu kiwi rhea moa; do
  [[ " ${hosts[*]} " == *" $h "* ]] || hosts+=("$h")
done
for h in "${hosts[@]}"; do ssh "$h" "systemctl --user start $svc"; done
for h in "${hosts[@]}"; do
  ssh "$h" "timeout 120 ~/$bin/nest --config ~/sparknest-test/state/node.toml wait-ready" >/dev/null
done
echo "== trial cluster serving on ${hosts[*]}"

ssh raptor "mkdir -p ~/$trial/blobs ~/$trial/refs ~/$trial/snapshots/$commit"
# Each file: its blob name, its size, its path in the snapshot.
mapfile -t entries < <(cd "$prod/snapshots/$commit" && find . \( -type l -o -type f \) -printf '%P\n' | sort |
  while read -r f; do
    b=$(readlink -f "$f")
    echo "$(basename "$b") $(stat -c %s "$b") $f"
  done)
big=() small=()
for e in "${entries[@]}"; do
  read -r _ size _ <<<"$e"
  if [ "$size" -ge $((64 << 20)) ]; then big+=("$e"); else small+=("$e"); fi
done
echo "== ${#big[@]} large files over ${#slots[@]} shares (${srcs[*]}), ${#small[@]} small through raptor"

copy() {
  local h=$1
  shift
  local cmds=""
  for e in "$@"; do
    read -r blob size _ <<<"$e"
    cmds+="if [ \$(stat -c %s ~/$trial/blobs/$blob 2>/dev/null || echo 0) -ne $size ]; then "
    cmds+="dd if=$prod/blobs/$blob of=~/$trial/blobs/$blob bs=16M iflag=direct oflag=direct status=none 2>/dev/null "
    cmds+="|| cp $prod/blobs/$blob ~/$trial/blobs/$blob; fi; "
  done
  local paths=""
  for e in "$@"; do read -r blob _ <<<"$e"; paths+=" ~/$trial/blobs/$blob"; done
  ssh "$h" "mkdir -p ~/$trial/blobs && $cmds python3 -c '$seal'$paths"
}

i=0
declare -A per
for e in "${big[@]}"; do
  h=${slots[$((i % ${#slots[@]}))]}
  per[$h]+="$e"$'\n'
  i=$((i + 1))
done
for h in "${!per[@]}"; do
  mapfile -t mine <<<"${per[$h]%$'\n'}"
  echo "   $h: ${#mine[@]} files"
  copy "$h" "${mine[@]}" &
done
[ ${#small[@]} -gt 0 ] && copy raptor "${small[@]}" &
wait

# The snapshot's links and the refs, as in production.
for e in "${entries[@]}"; do
  read -r blob _ f <<<"$e"
  rel=$(readlink "$prod/snapshots/$commit/$f")
  ssh raptor "mkdir -p \"\$(dirname ~/$trial/snapshots/$commit/$f)\" && ln -sfn '$rel' ~/$trial/snapshots/$commit/$f"
done
for r in "$prod"/refs/*; do
  ssh raptor "printf %s '$(cat "$r")' > ~/$trial/refs/$(basename "$r")"
done
sleep 3
echo "== staged: $(ssh raptor "du -shL --apparent-size ~/$trial/snapshots/$commit | cut -f1")"
ssh raptor "cd ~/$trial/snapshots/$commit && for f in model-0000[1-4]*.safetensors; do printf '%s %s\n' \$f \"\$(getfattr --only-values -n user.sparknest.hosts \$f 2>/dev/null || python3 -c 'import os,sys;print(os.getxattr(sys.argv[1],\"user.sparknest.hosts\").decode())' \$f)\"; done"
