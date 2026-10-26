# Preparing an immutable checkpoint composition

Composition is explicit offline work. It produces ordinary sealed checkpoint files
for the existing filesystem and RDMA/push readers. It does not start an inference
engine or reserve serving GPUs. Only the exact byte-compatible indexed-safetensors
adapter and its exact CPU qualification policy are implemented. Conversion, LoRA,
recalibration and behavioral/model-specific qualification remain unsupported gates.

Keep operator modules, checkpoint identities, paths and run receipts private. Source
and test examples use generic names. Start with copied or synthetic trial inputs.

## Assign a CPU pool

Build `nest-model-worker` from the same release as the daemon. Put it and coreutils
`/usr/bin/timeout` in a reviewed local image, record `docker image inspect .Id`, and
pin that configuration ID in both the node pool and module image declaration.
Do not supply arbitrary converter images or checkpoint Python scripts. Docker must
be available to the daemon user. An optional node configuration is:

```toml
[composition_worker]
name = "trial-cpu"
root = "/srv/sparknest-test/composition-workers"
image = "sha256:REPLACE_WITH_REVIEWED_LOCAL_IMAGE_ID"
max_cpu_threads = 4
max_ram_bytes = 8589934592
max_new_bytes = 322122547200
free_reserve_bytes = 17179869184
max_runtime_seconds = 3600
control_timeout_seconds = 30
max_pids = 64
```

The root must be an existing private absolute directory on the same native
filesystem as the object store. A mountpoint and `fuse.allow_other = true` are required so Docker can bind
pinned namespace inputs. The host must already permit user mounts with
`user_allow_other`; this feature does not change `/etc/fuse.conf`. Leave this section absent on unassigned or protected hosts. The
budgets above are examples, not a measured site recommendation or a hardware grant.
The native disk quota supports the trusted built-in bounded exporter; it is not a
kernel quota against arbitrary image code. Read-only input mounts and actual
CPU/memory/swap/runtime/mount settings are verified before execution.

Run preparation, qualification and worker recovery through the owning CPU node.
Other mounted nodes can inspect status and resolve a complete READY revision.
Single-pool admission precedes a new preparation claim, so a busy operation does
not strand a second module with ownership but no worker. Large declaration,
profile and receipt processing runs on bounded blocking workers.

## Declare, prepare and qualify

The module schema and immutable content-manifest contracts are documented in
`crates/nest-model/README.md`. The manager accepts already-imported absolute
namespace source paths. The base is a complete canonical indexed checkpoint;
a raw replacement directory contains only its safetensors file and optional
producer-selection manifest. Inputs must already be sealed and STABLE. Symlinks
are resolved once, then flattened into immutable regular-file snapshot references.
HF acquisition/download is a separate explicit operation.

The trusted operand profile is JSON with schema `sparknest.operand-profile.v1`, a
`groups` map from complete logical operand names to tensor-name arrays, and
`static_activation_calibration`. It must partition every base tensor exactly once,
including all scale/calibration/tied/fused companions. Do not infer this partition
from an untrusted alternate.

For the supported exact policy, set `lossy_conversion = "deny"` and use
`qualification_profile = "sha256:1eac557403f69d88496c1433083bbce9ad022bb3040337689d52009f45aa61c6"`.
This policy refuses static activation calibration. It compares actual output bytes
to the resolved intended sources and validates canonical layout/provenance; it does
not certify model behavior, engine acceptance or hardware performance.

```sh
nest model put example-business --module /private/module.toml \
  --profile /private/operand-profile.json --expected-revision 0
nest model prepare example-business --wait
nest model status example-business
nest model qualify example-business
nest model resolve example-business
```

Updates require the current module revision instead of zero. Changing input order,
source identity, group policy or a byte-affecting recipe creates a different key.
Preparation success remains pending qualification. A directory, worker exit, or an
export receipt alone never becomes READY. Qualification stores separate evidence
and atomically publishes the exact bound output before resolve returns its path.

A cache hit resolves the existing immutable revision; it performs no merge,
conversion, calibration, upstream acquisition or payload rehash. A changed
qualification policy requires matching implemented evidence; an old proof is not
silently accepted. Numerical/model-specific gates and actual engine/fabric loading
must be verified separately for each intended serving recipe.

## Hold a usage token through actual cleanup

Acquire a fresh single-use token before starting the model and retain the returned
key/epoch/path/revision. Launch the unchanged consumer against that explicit snapshot.
Release only after its owned containers and all model ranks have stopped and cleaned
up. An ordinary subprocess or launcher returning does not prove rank cleanup.

```sh
nest model acquire example-business --lease UNIQUE_LOAD_TOKEN --json
# Launch and stop the caller's own model/ranks using the resolved immutable path.
nest model release PREPARATION_KEY --epoch ARTIFACT_EPOCH --lease UNIQUE_LOAD_TOKEN
```

The lease persists across daemon restart and has no wall-clock expiry. Reusing a
token is refused. The interface does not stop another workload or release its token.

## Reset, inspect and recover

```sh
nest model cache status example-business
nest model cache reset example-business --dry-run
nest model cache reset example-business
nest model cache reset --all --dry-run
nest model cache reset example-business --when-idle
nest model cache reset-cancel example-business --epoch EPOCH --reset RESET_ID
nest model jobs show example-business
nest model jobs stop example-business
nest model jobs recover example-business
```

Normal reset refuses or reports busy while a worker or usage token owns the artifact.
Deferred reset blocks new users and waits for release. Reset-cancel matches the exact
reset ID and epoch. Stop signals the local job at safe boundaries and waits for
bounded worker cleanup; recovery uses retained exact-owner evidence. An ambiguous
launch or a crash before a durable launch record remains blocked for independent
reconciliation. Missing journals and elapsed time are not stop evidence.

Reset retires the catalog first, then the metadata reclaimer can unlink only proven
owned changed safetensors shards. Inputs, reused files, assets and JSON/provenance
receipts remain. Shared pins/roots and replaced aliases block removal. Reports include
eligible, retained and reclaimed logical shard bytes and per-entry blocked reasons;
ordinary open handles can delay physical release. `--all` includes orphaned old keys
and reports partial results without claiming a complete purge. Rebuilding the same
key must reproduce the original publication identity or fail NON_REPRODUCIBLE.
