# Offline model composition

This crate owns strict composition configuration, content identity, whole
operand-group resolution, bounded CPU export and exact preparation qualification.
It never executes checkpoint code or imports pickle. The daemon manager lives in
`nest-place::composition`; replicated lifetime rules live in `nest-meta`.

`nest-model-worker` is the standalone offline process used by an explicitly
assigned CPU pool. It cannot publish or acquire pins; the manager owns those
transitions and exposes them through the management API and `nest model`.

The owner resolves and pins immutable snapshots, assigns a trusted `Profile`
partitioning **every** base tensor into complete numerical operand groups, and
calls `Snapshot::inspect`, `ModuleConfig::parse`, `resolve`, then `export` on an
admitted blocking worker. Companions, tied operands and fused operands must be
declared together by that profile. Group rules are part of preparation identity;
they are not inferred from source tensor names. No profile has yet been
qualified against an inference engine by this crate's synthetic tests.

## Configuration and identity

The schema is `sparknest.model-composition.v1`. All config tables reject unknown
fields; digests are lowercase `sha256:` plus 64 hex characters. The private
design's illustrative tables are represented by `ModuleConfig`. Producer
selection additionally requires `selection_manifest_digest`, alongside
`selection_manifest`. A source can contain several producer manifests, so its
overall content identity alone cannot identify the selected one. This digest
must match the actual selected file before resolution. An example selection is:

```json
{"schema":"sparknest.model-selection.v1","groups":["a"]}
```

`Source::content_manifest` uses the digest returned by `Snapshot::inspect`:
SHA-256 of the compact JSON encoding of the tuple
`["sparknest.input-manifest.v1", files]`. `files` is a sorted map from confined
relative filenames to `{bytes, digest}`. File digests cover complete bytes,
including headers, indexes, selection files and non-weight semantics. The
manager must translate other verified object identities to this adapter's
manifest contract; arbitrary source strings are acquisition hints only.

Standalone sparse replacements use the new API:

```rust,ignore
Snapshot::inspect_sidecar(root, "alternate-checkpoint.safetensors", None, &limits)?;
// Or declare the bundle's sole optional producer-selection JSON:
Snapshot::inspect_sidecar(root, "alternate-checkpoint.safetensors", Some("groups.json"), &limits)?;
```

The root must be a private flat bundle containing **only** those declared
regular, non-symlink files. No checkpoint config, index, tokenizer or other
semantics are fabricated or included in its identity. Tensor and selection
files use their existing strict bounded schemas; the optional selection file
is validated even if the module ultimately requests `selection = "all"`.

Raw content identity is SHA-256 of compact JSON for
`["sparknest.sidecar-input-manifest.v1", roles]`, where `roles` is a sorted map
with required `tensors` and optional `selection` entries, each a full-file
`{bytes, digest}`. Unlike the indexed contract, raw filenames have no layout
meaning, so both filenames and the root path are acquisition hints excluded
from this role-based identity. Renaming declared files preserves the identity.
The module's selected-manifest digest still must match the actual declared
selection file. `files()` retains actual filenames for the owner's pins and
file acquisition. `is_sidecar()` distinguishes raw bundles at the API boundary.

A raw input is accepted only as `kind = "replacement"`, never as the base or
a full checkpoint. Its typed `requires` reference is still mandatory. Missing
non-weight semantics mean the base supplies them; indexed sources continue to
reject incompatible present semantics. For raw `selection = "all"`, select
every profile group touched by the raw tensor inventory and require **all**
members of each touched group. Untouched base groups fall back. This is inventory
selection, without comparing values, extracting differences, or using epsilons.
A producer manifest can instead explicitly name the selected groups. Missing
companions/tied members, unknown groups/tensors, and shape/dtype/representation
mismatches still fail. Source ranges may differ from base ranges; export always
uses complete canonical base shards and headers. Raw files never appear as
output shards or overwrite base ancillary metadata.

This extends `Snapshot` rather than adding a parallel adapter pipeline: raw and
indexed inputs share one safetensors header/range validator, group resolver,
bounded exporter and pin/revalidation rules. Full-copy reinspection dispatches
on the original representation; reuse checks the exact declared bundle files
under the existing immutable input and output-lifetime pin contract.

Preparation keys use a versioned JSON struct and sorted maps/sets. They include
base content, ordered alternate content/kind/selection/reference, selected
manifest digest, profile groups, image **digest**, recipe, exact string
parameters, non-weight choice, and calibration choice. They exclude acquisition
paths, names, namespace, image registry location, scheduling/cache budgets and
qualification-only policy. `prefix_key` supports typed lower-stack references.
Qualification identity is separate; export always records qualification pending.
The exact adapter currently accepts no recipe parameters. Recording a recipe
digest does not execute a recipe or an image.

## Supported profile and bounds

`base-compatible-indexed-v1` currently accepts flat, complete indexed snapshots
with canonical contiguous `model-NNNNN-of-NNNNN.safetensors` shards and
`model.safetensors.index.json`. It checks strict index/tensor schemas, total
payload size, complete inventory, dtype byte widths, shapes, contiguous offsets,
full byte coverage and duplicate JSON keys. Header lengths are aligned to eight
bytes. Ancillary JSON and ordinary tokenizer/template assets come from
the base; subdirectories, symlinks, special files and executable checkpoint file
types are refused. Default alternate non-weight semantics must match the base;
the explicit `use-base` choice records the override and preserves base files.
Architecture/encoding configuration (`config.json`) must still match exactly.

Selection is ordered absolute whole-group replacement, with base fallback and
shadow receipts. It validates shadowed selections too. A full checkpoint must
have the entire base inventory even for a subset selection. A replacement
sidecar may contain a subset but cannot borrow selected companions from the
base. Unknown groups/tensors and reference/shape mismatches fail. Source shard
layout may differ; output shard names, headers, inventory, shapes, dtypes and
offsets match the base exactly. Selected tensors must also match the base's
safetensors representation metadata.

Only byte-compatible payload copying is supported, including standard
`F8_E8M0` coded scales at one byte per element. A payload and its scales must
belong to one complete operand group; equal byte widths do not permit dtype
relabeling. Conversion (including FP16/BF16), LoRA and recalibration fail
explicitly. `preserve-and-qualify` can
retain declared static activation calibration, but requires the owner's later
qualification gate. A lossy-permission policy never enables an unimplemented
conversion.

`ExportLimits` bounds metadata bytes per snapshot, file/tensor counts, the
single payload buffer and new output file bytes. Metadata is retained in memory;
payload size never determines memory allocation. The owner admits the aggregate
memory of its inspected snapshots and enforces runtime/thread limits and job
cancellation. Export checks its advertised metadata-plus-buffer ceiling against
the module RAM budget. Before creating staging it plans logical bytes, copied
file bytes, reused file bytes, and new index/manifest/receipt bytes separately.
Both output quotas admit full lengths of independent changed shards (including
headers), copied ancillary files, and all new metadata. No quota discount is
taken for unchanged ranges within a changed shard. Reused files add no new
payload allocation. The owner separately reserves filesystem allocation/namespace
overhead; these counters describe regular-file bytes, not block-rounded usage.
There are no temporary full-shard copies or reflinks.

## Output ownership and identity

The owner must keep input roots and files pinned and immutable from inspection
through export, and provide a private staging parent with no concurrent writers.
Setting `cache.reuse_unchanged = true` **explicitly requires immutable-generation
and dependency pins for the entire lifetime of every output reference**, with
mutation and unseal prevented through every alias. A seal or an immutable path
during preparation alone is insufficient; the manager owns enforcement and
release of these pins.

With reuse enabled, export creates ordinary hardlink namespace references only
for base ancillary files and base shards containing no selected replacement.
Every changed shard is an independent complete rebuilt file, including its
fallback operands. All required links must succeed before changed-shard writes
start. A failed or cross-filesystem link fails export; it never triggers an
unadmitted copy. No linked inode is opened for write or patched.

Under the pin contract, export checks inode/device, length, modification time
and inventory against inspection, without rehashing large verified inputs.
Link count and ctime changes from adding references do not invalidate pins.
It reuses the verified base `FileIdentity` for linked output files; final
inspection still validates schemas, inventories, headers and offsets without
rehashing already verified payloads. These cheap checks detect obvious pin
violations and cannot establish immutability themselves. With
`cache.reuse_unchanged = false`, export retains full input revalidation and makes
independent copies of every output payload/ancillary file.

Export refuses existing destinations and staging inside inputs, opens new files
with `create_new`, copies bounded ranges into complete shards, and fsyncs new
files and the directory. It never writes an input inode. Failures can leave
partial private staging and links for the owner to quarantine; no export result
or directory existence is READY.

`composition.payload.json` covers base ancillary assets and output shards,
excluding index and receipts. Its digest and the preparation key go into normal
index metadata as `sparknest_payload_manifest` and `sparknest_preparation_key`.
`composition.receipt.json` then covers payload files, payload manifest and the
finished index, excluding itself. This binds changed bytes without a hash cycle.
The receipt also records source identities/order, recipe/image identity,
winners, shadows and pending qualification. It separately reports reused/copied
payload bytes, reused/copied full-file bytes, generated metadata bytes, total
new file bytes, total logical file bytes, and the reused filenames. Admission
includes the receipt's own length before any file is created. Allocation
counters belong only to the receipt: choosing copy versus reuse leaves the
preparation key, payload manifest and finished index bytes unchanged. The
manager compares those byte identities for reproducibility, not operational
allocation counters. The manager owns reproducibility comparison, publication,
sealing, leases, qualification and lifecycle recovery.

## Detached staging and offline worker

`export_staged` uses the same plan, output admission, input pin contract and
byte-bound manifests as `export`, but emits only changed independent files and
new metadata. Its `StagedExport` receipt lists reused whole-file identities;
`staged_files()` lists the files actually written. It creates no native link,
placeholder or copy for reused files, allowing native staging on another
filesystem from the input mount. Canonical changed-shard headers and complete
ranges are verified. Reused payloads are not rehashed. The manager must assemble
ordinary namespace links to those exact pinned generations, import new files,
verify the complete snapshot and bind its manifest before publication. Detached
staging is intentionally incomplete and must never be supplied to a loader.
Native-link `export` remains available and still refuses cross-filesystem links.

Build the process with `cargo build --release -p nest-model --bin
nest-model-worker`. Run it in the assigned worker environment with one JSON
request on stdin and one JSON response on stdout:

```sh
nest-model-worker < request.json > response.json
```

The strict request is tagged by `operation`: `inspect` has `inputs` and `limits`;
`stage` has `module`, `profile`, `base`, `alternates`, `destination` and `limits`.
Inputs are tagged by `layout`: `indexed` has `root`; `sidecar` has `root`,
`tensors` and optional `selection_manifest`. The operator's profile uses schema
`sparknest.operand-profile.v1`, a `groups` map of complete tensor-name arrays,
and `static_activation_calibration`. Every metadata, file/tensor, buffer and
output budget is explicit in `limits`. Request bytes are capped at 64 MiB;
duplicate JSON keys, unknown fields, duplicate group members and aggregate
metadata ceilings exceeding module RAM admission fail before staging.

The process runs one synchronous CPU operation without networking or checkpoint
code execution. Success returns verified inspection identities or **private
staging with qualification pending**. The process itself does not enforce the
module image digest, CPU/RSS cgroup, wall-time deadline, fair pool assignment or
storage reservation. Its parent enforces those, mounts pinned inputs read-only, retains pins until
confirmed process death/cleanup, and checks exit status before accepting stdout.
It also owns import/sealing, complete snapshot validation and READY publication.
The executable itself never starts a daemon or serving engine.


## Explicit exact qualification

`policy_identity()` names the supported versioned policy
`sparknest.exact-composition-cpu.v1`. `qualify_exact` verifies complete canonical
layout, base ancillary assets, payload/index/export provenance and **every tensor
byte against its resolved winning source**, including unchanged fallback. It
streams two bounded buffers and binds both the concrete policy identity and the
module's separate qualification key. It retains the plan's qualification-required
flag; proof satisfies this specific preparation policy rather than hiding it.

Unknown policies, lossy/conversion/LoRA paths and static activation calibration
are refused. This policy establishes exact preparation, with no behavioral,
engine, hardware or inference-performance claim. Qualification evidence belongs
outside the immutable checkpoint inventory. `publication_identity()` is the one
formula for deterministic preparation/payload/finished-index identity, excluding
operational copy/link counters.

The worker's `qualify` operation takes the same module/profile/base/alternates
as `stage`, plus a complete `output` snapshot root and explicit `limits`. It
returns a qualification receipt after real verification; it accepts no caller
pass boolean. The manager validates it, imports and pins the evidence separately,
then uses complete-output publication before resolving a READY revision.

The output writer shares one hard admitted regular-file byte budget across
changed shards, copied assets and generated metadata. Native filesystem block
and directory overhead is separately reserved and checked by the pool. This
supports the trusted built-in exact worker, with no kernel disk-quota claim
against arbitrary images. See `docs/MODEL-COMPOSITION.md` for operator workflow.
