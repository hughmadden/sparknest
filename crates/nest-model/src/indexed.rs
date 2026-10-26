use crate::{Digest, Error, ModuleConfig, Profile, Result, invalid, resolve};
use serde::{
    Deserialize, Serialize,
    de::{self, DeserializeOwned, MapAccess, SeqAccess, Visitor},
};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

const INDEX: &str = "model.safetensors.index.json";
const PAYLOAD: &str = "composition.payload.json";
const RECEIPT: &str = "composition.receipt.json";

/// Explicit worker admission limits; these bound input metadata, open inventory,
/// payload buffering and physical output. They are safety ceilings, not tuning.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportLimits {
    pub metadata_bytes: u64,
    pub max_files: usize,
    pub max_tensors: usize,
    pub buffer_bytes: usize,
    /// Ceiling for new regular-file bytes, including generated metadata.
    /// Reused pinned files count toward logical size, not this ceiling.
    pub max_output_bytes: u64,
}

impl Default for ExportLimits {
    fn default() -> Self {
        Self {
            metadata_bytes: 64 * 1024 * 1024,
            max_files: 100_000,
            max_tensors: 1_000_000,
            buffer_bytes: 1024 * 1024,
            max_output_bytes: u64::MAX,
        }
    }
}

impl ExportLimits {
    pub fn validate(&self) -> Result<()> {
        if self.metadata_bytes == 0
            || self.max_files == 0
            || self.max_tensors == 0
            || self.buffer_bytes == 0
            || self.max_output_bytes == 0
        {
            return Err(Error::Limit("zero worker budget".into()));
        }
        if self.metadata_bytes > usize::MAX as u64 {
            return Err(Error::Limit("metadata budget exceeds address space".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FileIdentity {
    pub bytes: u64,
    pub digest: Digest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Index {
    #[serde(default)]
    metadata: BTreeMap<String, Value>,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug)]
enum SourceLayout {
    Indexed(Index),
    Sidecar {
        tensor_filename: String,
        selection_filename: Option<String>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TensorSpec {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_offsets: [u64; 2],
}

#[derive(Clone, Debug)]
pub(crate) struct Tensor {
    pub spec: TensorSpec,
    pub shard: String,
}

#[derive(Clone, Debug)]
pub(crate) struct Shard {
    pub(crate) header: Vec<u8>,
    payload_start: u64,
    pub metadata: BTreeMap<String, String>,
}

/// An inspected immutable input. No file content is memory mapped. The owner
/// must keep its root and all files pinned and immutable for this value's use.
#[derive(Debug)]
pub struct Snapshot {
    root: PathBuf,
    identity: Digest,
    files: BTreeMap<String, FileIdentity>,
    layout: SourceLayout,
    pub(crate) tensors: BTreeMap<String, Tensor>,
    pub(crate) shards: BTreeMap<String, Shard>,
    selection_files: BTreeSet<String>,
    stamps: BTreeMap<String, FileStamp>,
}

// Cheap violation checks under the owner's immutable-generation pin contract;
// these are not a replacement for that contract or content verification.
// Link count/ctime are deliberately absent: adding a namespace reference must
// not invalidate a verified immutable file.
#[derive(Debug, Eq, PartialEq)]
struct FileStamp {
    bytes: u64,
    modified: std::time::SystemTime,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
}

impl FileStamp {
    fn read(root: &Path, name: &str) -> Result<Self> {
        safe_name(name)?;
        let metadata = fs::symlink_metadata(root.join(name))?;
        if !metadata.file_type().is_file() {
            return Err(invalid("pinned input is not a regular file"));
        }
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Ok(Self {
            bytes: metadata.len(),
            modified: metadata.modified()?,
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
        })
    }
}

pub(crate) fn safe_name(name: &str) -> Result<()> {
    if name.is_empty() || name == "." || name == ".." || name.contains(['/', '\\', ':', '\0']) {
        return Err(invalid("file reference must be a confined flat filename"));
    }
    Ok(())
}

fn canonical_shard(name: &str) -> Result<(usize, usize)> {
    safe_name(name)?;
    if !name.is_ascii()
        || name.len() != 32
        || !name.starts_with("model-")
        || &name[11..15] != "-of-"
        || !name.ends_with(".safetensors")
    {
        return Err(invalid("noncanonical shard name"));
    }
    let part: usize = name[6..11]
        .parse()
        .map_err(|_| invalid("invalid shard ordinal"))?;
    let count: usize = name[15..20]
        .parse()
        .map_err(|_| invalid("invalid shard count"))?;
    if !name[6..11]
        .bytes()
        .chain(name[15..20].bytes())
        .all(|b| b.is_ascii_digit())
        || part == 0
        || count == 0
        || part > count
    {
        return Err(invalid("invalid shard ordinal/count"));
    }
    Ok((part, count))
}

fn regular(root: &Path, name: &str) -> Result<File> {
    safe_name(name)?;
    let path = root.join(name);
    if !fs::symlink_metadata(&path)?.file_type().is_file() {
        return Err(invalid("input is not a regular, non-symlink file"));
    }
    Ok(File::open(path)?)
}

fn take_metadata(file: &mut File, bytes: u64, remaining: &mut u64) -> Result<Vec<u8>> {
    if bytes > *remaining || bytes > usize::MAX as u64 {
        return Err(Error::Limit("metadata byte budget exceeded".into()));
    }
    *remaining -= bytes;
    let mut data = vec![0; bytes as usize];
    file.read_exact(&mut data)?;
    Ok(data)
}

#[cfg(test)]
std::thread_local! {
    static HASHED_FILE_BYTES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn digest_reader(file: &mut File, buffer: &mut [u8]) -> Result<FileIdentity> {
    let mut hasher = Sha256::new();
    let mut bytes = 0u64;
    loop {
        let count = file.read(buffer)?;
        if count == 0 {
            break;
        }
        #[cfg(test)]
        HASHED_FILE_BYTES.with(|total| total.set(total.get() + count as u64));
        bytes = bytes
            .checked_add(count as u64)
            .ok_or_else(|| invalid("file length overflow"))?;
        hasher.update(&buffer[..count]);
    }
    Ok(FileIdentity {
        bytes,
        digest: Digest::from_hasher(hasher),
    })
}

fn dtype_bytes(dtype: &str) -> Result<u64> {
    match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" => Ok(1),
        "U16" | "I16" | "F16" | "BF16" => Ok(2),
        "U32" | "I32" | "F32" => Ok(4),
        "U64" | "I64" | "F64" => Ok(8),
        _ => Err(Error::Unsupported("safetensors dtype".into())),
    }
}

/// Common indexed/raw validation. Every byte after the header must belong to
/// exactly one tensor range; neither format can smuggle trailing payloads.
fn inspect_shard(
    root: &Path,
    name: &str,
    remaining: &mut u64,
    max_tensors: usize,
) -> Result<(Shard, BTreeMap<String, Tensor>, u64)> {
    let mut file = regular(root, name)?;
    let file_size = file.metadata()?.len();
    let mut prefix = [0; 8];
    file.read_exact(&mut prefix)?;
    let header_size = u64::from_le_bytes(prefix);
    if header_size == 0 || header_size % 8 != 0 {
        return Err(invalid("invalid safetensors header alignment"));
    }
    let payload_start = header_size
        .checked_add(8)
        .ok_or_else(|| invalid("header length overflow"))?;
    if payload_start > file_size {
        return Err(invalid("header beyond shard length"));
    }
    let header = take_metadata(&mut file, header_size, remaining)?;
    if header.first() != Some(&b'{') {
        return Err(invalid("safetensors header must begin with object"));
    }
    let mut entries: BTreeMap<String, Value> = serde_json::from_value(unique_json(&header)?)?;
    let metadata = match entries.remove("__metadata__") {
        Some(value) => serde_json::from_value(value)?,
        None => BTreeMap::new(),
    };
    if entries.is_empty() {
        return Err(invalid("empty safetensors inventory"));
    }
    if entries.len() > max_tensors {
        return Err(Error::Limit("tensor count budget exceeded".into()));
    }
    let mut tensors = BTreeMap::new();
    let mut ranges = Vec::new();
    for (tensor_name, value) in entries {
        if tensor_name.is_empty() {
            return Err(invalid("empty tensor name"));
        }
        let spec: TensorSpec = serde_json::from_value(value)?;
        let size = spec
            .shape
            .iter()
            .try_fold(dtype_bytes(&spec.dtype)?, |size, dimension| {
                size.checked_mul(*dimension)
                    .ok_or_else(|| invalid("tensor size overflow"))
            })?;
        if spec.data_offsets[0].checked_add(size) != Some(spec.data_offsets[1]) {
            return Err(invalid("tensor shape/offset byte length mismatch"));
        }
        ranges.push((spec.data_offsets[0], spec.data_offsets[1]));
        tensors.insert(
            tensor_name,
            Tensor {
                spec,
                shard: name.into(),
            },
        );
    }
    ranges.sort_unstable();
    let mut end = 0;
    for (start, next) in ranges {
        if start != end {
            return Err(invalid("shard gaps or overlapping tensor ranges"));
        }
        end = next;
    }
    if payload_start.checked_add(end) != Some(file_size) {
        return Err(invalid("shard payload is not completely covered"));
    }
    Ok((
        Shard {
            header,
            payload_start,
            metadata,
        },
        tensors,
        end,
    ))
}

impl Snapshot {
    pub fn inspect(root: &Path, limits: &ExportLimits) -> Result<Self> {
        Self::inspect_verified(root, limits, &BTreeMap::new())
    }

    /// Inspect a private flat replacement bundle containing ONLY the declared
    /// safetensors file and optional strict producer selection JSON. No index,
    /// config or tokenizer is invented. The caller pins these exact files under
    /// the same immutability contract as indexed inputs. Filenames are confined
    /// acquisition hints; identity binds role-labelled actual file bytes.
    pub fn inspect_sidecar(
        root: &Path,
        tensor_filename: &str,
        selection_filename: Option<&str>,
        limits: &ExportLimits,
    ) -> Result<Self> {
        limits.validate()?;
        safe_name(tensor_filename)?;
        if !tensor_filename.ends_with(".safetensors") {
            return Err(invalid("sidecar requires a safetensors filename"));
        }
        let mut names = BTreeSet::from([tensor_filename.to_string()]);
        if let Some(name) = selection_filename {
            safe_name(name)?;
            if !name.ends_with(".json") || !names.insert(name.to_string()) {
                return Err(invalid(
                    "sidecar selection requires a distinct JSON filename",
                ));
            }
        }
        if names.len() > limits.max_files {
            return Err(Error::Limit("sidecar file budget exceeded".into()));
        }
        if !fs::symlink_metadata(root)?.file_type().is_dir() {
            return Err(invalid("sidecar root must be a non-symlink directory"));
        }
        let root = root.canonicalize()?;
        let mut found = BTreeSet::new();
        for entry in fs::read_dir(&root)? {
            let entry = entry?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| invalid("non-UTF8 sidecar filename"))?;
            if !names.contains(&name) || !entry.file_type()?.is_file() {
                return Err(invalid(
                    "sidecar bundle contains undeclared or non-regular files",
                ));
            }
            found.insert(name);
        }
        if found != names {
            return Err(invalid("missing declared sidecar file"));
        }
        let mut remaining = limits.metadata_bytes;
        let mut selection_files = BTreeSet::new();
        if let Some(name) = selection_filename {
            let mut file = regular(&root, name)?;
            let bytes = file.metadata()?.len();
            let selection: crate::plan::SelectionManifest = serde_json::from_value(unique_json(
                &take_metadata(&mut file, bytes, &mut remaining)?,
            )?)?;
            selection.validate()?;
            selection_files.insert(name.to_string());
        }
        let (shard, tensors, _) =
            inspect_shard(&root, tensor_filename, &mut remaining, limits.max_tensors)?;
        let mut files = BTreeMap::new();
        let mut stamps = BTreeMap::new();
        let mut buffer = vec![0; limits.buffer_bytes];
        for name in names {
            stamps.insert(name.clone(), FileStamp::read(&root, &name)?);
            files.insert(
                name.clone(),
                digest_reader(&mut regular(&root, &name)?, &mut buffer)?,
            );
        }
        let mut roles = BTreeMap::from([("tensors", &files[tensor_filename])]);
        if let Some(name) = selection_filename {
            roles.insert("selection", &files[name]);
        }
        let identity = Digest::canonical(&("sparknest.sidecar-input-manifest.v1", roles))?;
        Ok(Self {
            root,
            identity,
            files,
            stamps,
            tensors,
            selection_files,
            shards: BTreeMap::from([(tensor_filename.into(), shard)]),
            layout: SourceLayout::Sidecar {
                tensor_filename: tensor_filename.into(),
                selection_filename: selection_filename.map(str::to_string),
            },
        })
    }

    /// Raw sidecars can supply replacement groups, never a base/full checkpoint.
    pub fn is_sidecar(&self) -> bool {
        matches!(self.layout, SourceLayout::Sidecar { .. })
    }

    fn reinspect(&self, limits: &ExportLimits) -> Result<Self> {
        match &self.layout {
            SourceLayout::Indexed(_) => Self::inspect(&self.root, limits),
            SourceLayout::Sidecar {
                tensor_filename,
                selection_filename,
            } => Self::inspect_sidecar(
                &self.root,
                tensor_filename,
                selection_filename.as_deref(),
                limits,
            ),
        }
    }

    // Only the exporter can supply identities here. They cover either linked,
    // verified base files under lifetime pins, or freshly hashed private output.
    // Header/inventory/schema checks still run; large known payloads are not
    // rehashed during the final inspection.
    fn inspect_verified(
        root: &Path,
        limits: &ExportLimits,
        verified: &BTreeMap<String, FileIdentity>,
    ) -> Result<Self> {
        limits.validate()?;
        if !fs::symlink_metadata(root)?.file_type().is_dir() {
            return Err(invalid("snapshot root must be a non-symlink directory"));
        }
        let root = root.canonicalize()?;
        let mut names = BTreeSet::new();
        for entry in fs::read_dir(&root)? {
            let entry = entry?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| invalid("non-UTF8 file name"))?;
            safe_name(&name)?;
            if !entry.file_type()?.is_file() {
                return Err(invalid(
                    "flat snapshot profile refuses symlinks, subdirectories and special files",
                ));
            }
            // No source code, pickle or arbitrary checkpoint deserialization.
            if !name.ends_with(".json")
                && !name.ends_with(".safetensors")
                && !matches!(
                    name.as_str(),
                    "tokenizer.model"
                        | "spiece.model"
                        | "vocab.txt"
                        | "merges.txt"
                        | "chat_template.jinja"
                )
            {
                return Err(Error::Unsupported(
                    "unrecognized checkpoint file type".into(),
                ));
            }
            names.insert(name);
            if names.len() > limits.max_files {
                return Err(Error::Limit("snapshot file budget exceeded".into()));
            }
        }
        if !names.contains("config.json") || !names.contains(INDEX) {
            return Err(invalid("missing config or safetensors index"));
        }
        let mut remaining = limits.metadata_bytes;
        let mut json_files = BTreeMap::new();
        let mut selection_files = BTreeSet::new();
        for name in names.iter().filter(|n| n.ends_with(".json")) {
            let mut file = regular(&root, name)?;
            let size = file.metadata()?.len();
            let value = unique_json(&take_metadata(&mut file, size, &mut remaining)?)?;
            if !value.is_object() {
                return Err(invalid("checkpoint JSON must be an object"));
            }
            if value.get("schema").and_then(Value::as_str) == Some("sparknest.model-selection.v1") {
                selection_files.insert(name.clone());
            }
            json_files.insert(name.clone(), value);
        }
        let index: Index = serde_json::from_value(
            json_files
                .remove(INDEX)
                .ok_or_else(|| invalid("missing index"))?,
        )?;
        if index.weight_map.is_empty() || index.weight_map.len() > limits.max_tensors {
            return Err(invalid("empty or excessive tensor inventory"));
        }
        let shard_names: BTreeSet<_> = index.weight_map.values().cloned().collect();
        for (ordinal, name) in shard_names.iter().enumerate() {
            if canonical_shard(name)? != (ordinal + 1, shard_names.len()) {
                return Err(invalid("noncontiguous canonical shard set"));
            }
        }
        if names
            .iter()
            .filter(|n| n.ends_with(".safetensors"))
            .cloned()
            .collect::<BTreeSet<_>>()
            != shard_names
        {
            return Err(invalid("index and file shard inventories differ"));
        }
        let mut tensors = BTreeMap::new();
        let mut shards = BTreeMap::new();
        let mut payload_bytes = 0u64;
        for name in shard_names {
            let (shard, entries, bytes) = inspect_shard(
                &root,
                &name,
                &mut remaining,
                limits.max_tensors - tensors.len(),
            )?;
            for (tensor_name, tensor) in entries {
                if index.weight_map.get(&tensor_name) != Some(&name) {
                    return Err(invalid("shard tensor missing from index or wrong shard"));
                }
                if tensors.insert(tensor_name, tensor).is_some() {
                    return Err(invalid("duplicate tensor across shards"));
                }
            }
            payload_bytes = payload_bytes
                .checked_add(bytes)
                .ok_or_else(|| invalid("payload size overflow"))?;
            shards.insert(name, shard);
        }
        if tensors.keys().ne(index.weight_map.keys()) {
            return Err(invalid("indexed tensor missing from shard"));
        }
        if index.metadata.get("total_size").and_then(Value::as_u64) != Some(payload_bytes) {
            return Err(invalid("index total_size mismatch or missing"));
        }
        // Stream verified content identities. Metadata is bounded above; tensor
        // payloads need only one fixed-size buffer irrespective of checkpoint size.
        let mut buffer = vec![0; limits.buffer_bytes];
        let mut files = BTreeMap::new();
        let mut stamps = BTreeMap::new();
        for name in names {
            let stamp = FileStamp::read(&root, &name)?;
            let identity = match verified.get(&name) {
                Some(identity) if identity.bytes == stamp.bytes => identity.clone(),
                Some(_) => return Err(invalid("verified output file length changed")),
                None => digest_reader(&mut regular(&root, &name)?, &mut buffer)?,
            };
            files.insert(name.clone(), identity);
            stamps.insert(name, stamp);
        }
        let identity = Digest::canonical(&("sparknest.input-manifest.v1", &files))?;
        Ok(Self {
            root,
            identity,
            files,
            layout: SourceLayout::Indexed(index),
            tensors,
            shards,
            selection_files,
            stamps,
        })
    }

    pub(crate) fn revalidate_pins(&self) -> Result<()> {
        if !fs::symlink_metadata(&self.root)?.file_type().is_dir() {
            return Err(invalid("pinned root changed"));
        }
        let mut count = 0usize;
        for entry in fs::read_dir(&self.root)? {
            let name = entry?
                .file_name()
                .into_string()
                .map_err(|_| invalid("pinned filename changed"))?;
            if self.stamps.get(&name) != Some(&FileStamp::read(&self.root, &name)?) {
                return Err(invalid("input changed since inspection"));
            }
            count += 1;
        }
        if count != self.stamps.len() {
            return Err(invalid("pinned input inventory changed"));
        }
        Ok(())
    }

    pub fn content_manifest(&self) -> &Digest {
        &self.identity
    }
    pub fn files(&self) -> &BTreeMap<String, FileIdentity> {
        &self.files
    }

    pub(crate) fn ancillary(&self) -> BTreeMap<&str, &FileIdentity> {
        self.files
            .iter()
            .filter(|(n, _)| {
                n.as_str() != INDEX
                    && n.as_str() != PAYLOAD
                    && n.as_str() != RECEIPT
                    && !self.shards.contains_key(*n)
                    && !self.selection_files.contains(*n)
            })
            .map(|(n, id)| (n.as_str(), id))
            .collect()
    }

    pub(crate) fn read_json<T: DeserializeOwned>(
        &self,
        name: &str,
        limits: &ExportLimits,
    ) -> Result<T> {
        if !self.files.contains_key(name) {
            return Err(invalid("missing input manifest"));
        }
        let mut file = regular(&self.root, name)?;
        let bytes = file.metadata()?.len();
        let mut remaining = limits.metadata_bytes;
        let data = take_metadata(&mut file, bytes, &mut remaining)?;
        Ok(serde_json::from_value(unique_json(&data)?)?)
    }

    /// Small diagnostic/test accessor with an explicit allocation cap. Export
    /// itself always streams ranges and never uses this accessor.
    pub fn tensor_bytes(&self, name: &str, max_bytes: usize) -> Result<Vec<u8>> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| invalid("unknown tensor"))?;
        let size = tensor.spec.data_offsets[1] - tensor.spec.data_offsets[0];
        if size > max_bytes as u64 {
            return Err(Error::Limit("tensor accessor allocation cap".into()));
        }
        let mut file = self.tensor_reader(tensor)?;
        let mut bytes = vec![0; size as usize];
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    pub(crate) fn tensor_reader(&self, tensor: &Tensor) -> Result<File> {
        let mut file = regular(&self.root, &tensor.shard)?;
        file.seek(SeekFrom::Start(
            self.shards[&tensor.shard].payload_start + tensor.spec.data_offsets[0],
        ))?;
        Ok(file)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportReceipt {
    pub schema: String,
    pub preparation_key: Digest,
    pub payload_manifest: Digest,
    pub base: Digest,
    pub alternates: Vec<Digest>,
    pub recipe: Digest,
    pub adapter_image_digest: Digest,
    pub non_weight_mismatch: crate::NonWeightMismatch,
    pub winners: BTreeMap<String, Option<usize>>,
    pub shadowed: Vec<crate::Shadowed>,
    pub files: BTreeMap<String, FileIdentity>,
    pub copied_payload_bytes: u64,
    pub qualification: String,
    pub requires_model_qualification: bool,
    /// Full file lengths, including shard headers, for linked base files.
    pub reused_file_bytes: u64,
    /// Full file lengths for independent rebuilt shards/copied ancillary files.
    pub copied_file_bytes: u64,
    pub reused_payload_bytes: u64,
    /// New index, payload manifest and receipt bytes, including this receipt.
    pub new_metadata_bytes: u64,
    pub new_file_bytes: u64,
    pub logical_file_bytes: u64,
    pub reused_files: BTreeSet<String>,
}

/// Private native staging, deliberately not a complete checkpoint or a READY
/// result. The manager must assemble reused files as ordinary namespace links,
/// verify the complete snapshot, bind its manifest and qualify before publishing.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StagedExport {
    pub receipt: ExportReceipt,
}

impl StagedExport {
    /// Files actually emitted to native staging, including the receipt itself.
    pub fn staged_files(&self) -> BTreeSet<String> {
        self.receipt
            .files
            .keys()
            .filter(|name| !self.receipt.reused_files.contains(*name))
            .cloned()
            .chain(std::iter::once(RECEIPT.into()))
            .collect()
    }
}

#[derive(Serialize)]
struct PayloadManifest<'a> {
    schema: &'static str,
    files: &'a BTreeMap<String, FileIdentity>,
}

fn payload_manifest(files: &BTreeMap<String, FileIdentity>) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&PayloadManifest {
        schema: "sparknest.payload-manifest.v1",
        files,
    })?)
}

/// Rebuild complete canonical shards at their base names, inventories and
/// offsets. Creates a NEW private staging directory; refuses existing paths.
/// Errors leave staging evidence for the owner to quarantine/clean. A successful
/// return is an export receipt, never permission to serve or a READY transition.
///
/// # Caller pin contract
/// Enabling `config.cache.reuse_unchanged` explicitly requires the owner to hold
/// immutable-generation/dependency pins on every reused base file for the ENTIRE
/// output lifetime, and prevent mutation/unseal through ANY namespace alias.
/// An immutable source during this call alone is insufficient. The owner must
/// also keep all inputs pinned through preparation and staging private. Cheap
/// metadata checks below detect obvious violations; they cannot establish pins.
/// Links are ordinary regular-file namespace references, never opened for write.
pub fn export(
    config: &ModuleConfig,
    profile: &Profile,
    base: &Snapshot,
    alternates: &[&Snapshot],
    destination: &Path,
    limits: &ExportLimits,
) -> Result<ExportReceipt> {
    export_inner(
        config,
        profile,
        base,
        alternates,
        destination,
        limits,
        false,
    )
}

/// Emit only independent changed shards/assets and generated metadata to private
/// native staging. Reused whole files are identified by the verified receipt;
/// no native hardlink, symlink, placeholder or copy is created for them. This
/// allows staging on a different filesystem from the pinned input mount.
///
/// The exact same caller pin contract as [`export`] applies. The returned staging
/// is incomplete whenever files are reused; it must never be passed to a loader.
/// The manager must link exact pinned generations into its complete namespace
/// snapshot, import new files, validate that complete snapshot and bind its
/// inventory. No virtual-file representation is introduced to the data plane.
pub fn export_staged(
    config: &ModuleConfig,
    profile: &Profile,
    base: &Snapshot,
    alternates: &[&Snapshot],
    destination: &Path,
    limits: &ExportLimits,
) -> Result<StagedExport> {
    Ok(StagedExport {
        receipt: export_inner(config, profile, base, alternates, destination, limits, true)?,
    })
}

fn export_inner(
    config: &ModuleConfig,
    profile: &Profile,
    base: &Snapshot,
    alternates: &[&Snapshot],
    destination: &Path,
    limits: &ExportLimits,
    detached: bool,
) -> Result<ExportReceipt> {
    let plan = resolve(config, profile, base, alternates, limits)?;
    let metadata_budget = config.preparation.max_ram_bytes;
    if limits
        .metadata_bytes
        .checked_add(limits.buffer_bytes as u64)
        .is_none_or(|b| b > metadata_budget)
    {
        return Err(Error::Limit(
            "worker metadata and buffer budget exceeds module RAM admission".into(),
        ));
    }
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = parent.canonicalize()?;
    if std::iter::once(base)
        .chain(alternates.iter().copied())
        .any(|input| parent.starts_with(&input.root))
    {
        return Err(invalid(
            "output staging must be outside immutable input snapshots",
        ));
    }
    // Reuse relies on the explicit lifetime pin contract above. Do not reread
    // verified large payloads to emulate immutability that only the owner can
    // enforce. Full-copy exports retain their original content revalidation.
    for input in std::iter::once(base).chain(alternates.iter().copied()) {
        if config.cache.reuse_unchanged {
            input.revalidate_pins()?;
        } else if input.reinspect(limits)?.identity != input.identity {
            return Err(invalid("input changed since inspection"));
        }
    }
    let SourceLayout::Indexed(base_index) = &base.layout else {
        return Err(invalid("output requires indexed base layout"));
    };
    let mut index = base_index.clone();
    // Upper bound the new metadata before creating staging, with fixed-width
    // placeholder digests, so large metadata cannot bypass disk admission.
    index.metadata.insert(
        "sparknest_preparation_key".into(),
        Value::String(plan.preparation_key.to_string()),
    );
    index.metadata.insert(
        "sparknest_payload_manifest".into(),
        Value::String(Digest::of(b"").to_string()),
    );
    let mut file_names: BTreeSet<String> = base.ancillary().keys().map(|n| n.to_string()).collect();
    file_names.extend(base.shards.keys().cloned());
    let mut reused_files = BTreeSet::new();
    if config.cache.reuse_unchanged {
        let changed_shards: BTreeSet<_> = plan
            .winners
            .iter()
            .filter(|(_, source)| source.is_some())
            .flat_map(|(group, _)| &profile.groups[group])
            .map(|name| &base.tensors[name].shard)
            .collect();
        reused_files.extend(base.ancillary().keys().map(|name| name.to_string()));
        reused_files.extend(
            base.shards
                .keys()
                .filter(|name| !changed_shards.contains(name))
                .cloned(),
        );
    }
    let mut copied_payload_bytes = 0u64;
    let mut reused_payload_bytes = 0u64;
    for (name, shard) in &base.shards {
        let bytes = base.files[name].bytes - shard.payload_start;
        let count = if reused_files.contains(name) {
            &mut reused_payload_bytes
        } else {
            &mut copied_payload_bytes
        };
        *count = count
            .checked_add(bytes)
            .ok_or_else(|| invalid("payload accounting overflow"))?;
    }
    let mut copied_file_bytes = 0u64;
    let mut reused_file_bytes = 0u64;
    for name in &file_names {
        let count = if reused_files.contains(name) {
            &mut reused_file_bytes
        } else {
            &mut copied_file_bytes
        };
        *count = count
            .checked_add(base.files[name].bytes)
            .ok_or_else(|| invalid("file accounting overflow"))?;
    }
    let mut receipt = ExportReceipt {
        schema: "sparknest.export-receipt.v1".into(),
        preparation_key: plan.preparation_key,
        payload_manifest: Digest::of(b""),
        base: base.identity.clone(),
        alternates: alternates.iter().map(|a| a.identity.clone()).collect(),
        recipe: config.preparation.recipe.clone(),
        adapter_image_digest: serde_json::from_value(Value::String(
            config
                .preparation
                .adapter_image
                .rsplit_once('@')
                .ok_or_else(|| invalid("unpinned image"))?
                .1
                .into(),
        ))?,
        non_weight_mismatch: config.output.non_weight_mismatch,
        winners: plan.winners,
        shadowed: plan.shadowed,
        files: BTreeMap::new(),
        copied_payload_bytes,
        qualification: "pending".into(),
        requires_model_qualification: plan.requires_model_qualification,
        reused_file_bytes,
        copied_file_bytes,
        reused_payload_bytes,
        reused_files,
        new_metadata_bytes: 0,
        new_file_bytes: 0,
        logical_file_bytes: 0,
    };
    let placeholders: BTreeMap<_, _> = file_names
        .iter()
        .map(|n| (n.clone(), base.files[n].clone()))
        .collect();
    let manifest_size = payload_manifest(&placeholders)?.len() as u64;
    receipt.files = placeholders;
    receipt.files.insert(
        PAYLOAD.into(),
        FileIdentity {
            bytes: manifest_size,
            digest: Digest::of(b""),
        },
    );
    let index_size = serde_json::to_vec(&index)?.len() as u64;
    receipt.files.insert(
        INDEX.into(),
        FileIdentity {
            bytes: index_size,
            digest: Digest::of(b""),
        },
    );
    // Only decimal byte-count lengths depend on this receipt's own length.
    // Iterate to a stable length before allocation; no digest includes itself.
    let metadata_without_receipt = manifest_size
        .checked_add(index_size)
        .ok_or_else(|| invalid("metadata accounting overflow"))?;
    let mut receipt_size = 0u64;
    loop {
        let size = serde_json::to_vec(&receipt)?.len() as u64;
        if size == receipt_size {
            break;
        }
        receipt_size = size;
        receipt.new_metadata_bytes = metadata_without_receipt
            .checked_add(size)
            .ok_or_else(|| invalid("metadata accounting overflow"))?;
        receipt.new_file_bytes = copied_file_bytes
            .checked_add(receipt.new_metadata_bytes)
            .ok_or_else(|| invalid("new-file accounting overflow"))?;
        receipt.logical_file_bytes = reused_file_bytes
            .checked_add(receipt.new_file_bytes)
            .ok_or_else(|| invalid("logical accounting overflow"))?;
    }
    let output_metadata = receipt
        .files
        .iter()
        .filter(|(name, _)| name.ends_with(".json"))
        .try_fold(receipt_size, |sum, (_, id)| {
            sum.checked_add(id.bytes)
                .ok_or_else(|| invalid("output metadata size overflow"))
        })?;
    let output_metadata = base
        .shards
        .values()
        .try_fold(output_metadata, |sum, shard| {
            sum.checked_add(shard.header.len() as u64)
                .ok_or_else(|| invalid("output metadata size overflow"))
        })?;
    if output_metadata > limits.metadata_bytes
        || receipt
            .files
            .len()
            .checked_add(1)
            .is_none_or(|n| n > limits.max_files)
    {
        return Err(Error::Limit(
            "output metadata or inventory budget exceeded".into(),
        ));
    }
    if receipt.new_file_bytes
        > limits
            .max_output_bytes
            .min(config.cache.max_new_physical_bytes)
    {
        return Err(Error::Limit("physical output admission exceeded".into()));
    }
    fs::create_dir(destination)?;
    let output_budget = Cell::new(receipt.new_file_bytes);
    // All required links must succeed before any changed shard is allocated.
    // EXDEV/unsupported links fail; there is no unadmitted copy fallback.
    for name in receipt.reused_files.iter().filter(|_| !detached) {
        if base.stamps.get(name) != Some(&FileStamp::read(&base.root, name)?) {
            return Err(invalid("reused input changed"));
        }
        fs::hard_link(base.root.join(name), destination.join(name))?;
        if base.stamps.get(name) != Some(&FileStamp::read(destination, name)?) {
            return Err(invalid("linked file differs from pinned input"));
        }
    }
    let mut buffer = vec![0; limits.buffer_bytes];
    let mut tensor_sources = BTreeMap::new();
    for (group, source) in &receipt.winners {
        for tensor in &profile.groups[group] {
            tensor_sources.insert(
                tensor.as_str(),
                source.map(|i| alternates[i]).unwrap_or(base),
            );
        }
    }
    for name in base.ancillary().keys() {
        if receipt.reused_files.contains(*name) {
            continue;
        }
        let mut input = regular(&base.root, name)?;
        let mut output = new_file(destination, name, &output_budget)?;
        copy_exact(
            &mut input,
            &mut output,
            base.files[*name].bytes,
            &mut buffer,
        )?;
        output.sync_all()?;
    }
    for (name, shard) in &base.shards {
        if receipt.reused_files.contains(name) {
            continue;
        }
        let mut output = new_file(destination, name, &output_budget)?;
        output.write_all(&(shard.header.len() as u64).to_le_bytes())?;
        output.write_all(&shard.header)?;
        let mut tensors: Vec<_> = base
            .tensors
            .iter()
            .filter(|(_, t)| &t.shard == name)
            .collect();
        tensors.sort_by_key(|(n, t)| (t.spec.data_offsets[0], t.spec.data_offsets[1], n.as_str()));
        for (tensor_name, tensor) in tensors {
            let source = tensor_sources[tensor_name.as_str()];
            let source_tensor = &source.tensors[tensor_name];
            copy_exact(
                &mut source.tensor_reader(source_tensor)?,
                &mut output,
                tensor.spec.data_offsets[1] - tensor.spec.data_offsets[0],
                &mut buffer,
            )?;
        }
        output.sync_all()?;
    }
    let mut payload = BTreeMap::new();
    for name in file_names {
        let identity = if receipt.reused_files.contains(&name) {
            base.files[&name].clone()
        } else {
            digest_reader(&mut regular(destination, &name)?, &mut buffer)?
        };
        payload.insert(name, identity);
    }
    let manifest = payload_manifest(&payload)?;
    receipt.payload_manifest = Digest::of(&manifest);
    write_synced(destination, PAYLOAD, &manifest, &output_budget)?;
    index.metadata.insert(
        "sparknest_payload_manifest".into(),
        Value::String(receipt.payload_manifest.to_string()),
    );
    let index = serde_json::to_vec(&index)?;
    write_synced(destination, INDEX, &index, &output_budget)?;
    receipt.files = payload;
    receipt.files.insert(
        PAYLOAD.into(),
        FileIdentity {
            bytes: manifest.len() as u64,
            digest: receipt.payload_manifest.clone(),
        },
    );
    receipt.files.insert(
        INDEX.into(),
        FileIdentity {
            bytes: index.len() as u64,
            digest: Digest::of(&index),
        },
    );
    let bytes = serde_json::to_vec(&receipt)?;
    if bytes.len() as u64 != receipt_size {
        return Err(invalid("receipt admission size changed"));
    }
    write_synced(destination, RECEIPT, &bytes, &output_budget)?;
    if output_budget.get() != 0 {
        return Err(invalid("output writes differ from admitted allocation"));
    }
    File::open(destination)?.sync_all()?;
    if detached && !receipt.reused_files.is_empty() {
        // New complete shards are still checked against the canonical base
        // headers and full-range validator. Reused shards were inspected under
        // lifetime pins. The final assembled namespace requires its own complete
        // snapshot validation; this partial native directory cannot provide it.
        let mut remaining = limits.metadata_bytes;
        for (name, original) in &base.shards {
            if !receipt.reused_files.contains(name) {
                let (written, _, _) =
                    inspect_shard(destination, name, &mut remaining, limits.max_tensors)?;
                if written.header != original.header {
                    return Err(invalid("staged shard header differs from canonical base"));
                }
            }
        }
    } else {
        // Validate the ordinary complete destination before handing it to its owner.
        Snapshot::inspect_verified(destination, limits, &receipt.files)?;
    }
    if config.cache.reuse_unchanged {
        for input in std::iter::once(base).chain(alternates.iter().copied()) {
            input.revalidate_pins()?;
        }
    }
    Ok(receipt)
}

/// Every newly written byte, including metadata, consumes the same admitted
/// budget. Pins/hardlinks consume no new payload bytes. Filesystem block and
/// namespace overhead is reserved separately by the parent on native storage.
struct BoundedFile<'a> {
    file: File,
    remaining: &'a Cell<u64>,
}

impl Write for BoundedFile<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() as u64 > self.remaining.get() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::StorageFull,
                "composition output allocation ceiling exceeded",
            ));
        }
        let written = self.file.write(bytes)?;
        self.remaining.set(self.remaining.get() - written as u64);
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

impl BoundedFile<'_> {
    fn sync_all(&self) -> std::io::Result<()> {
        self.file.sync_all()
    }
}

fn new_file<'a>(root: &Path, name: &str, remaining: &'a Cell<u64>) -> Result<BoundedFile<'a>> {
    safe_name(name)?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(name))?;
    Ok(BoundedFile { file, remaining })
}

fn write_synced(root: &Path, name: &str, bytes: &[u8], remaining: &Cell<u64>) -> Result<()> {
    let mut file = new_file(root, name, remaining)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn copy_exact(
    input: &mut impl Read,
    output: &mut impl Write,
    mut bytes: u64,
    buffer: &mut [u8],
) -> Result<()> {
    while bytes != 0 {
        let count = bytes.min(buffer.len() as u64) as usize;
        input.read_exact(&mut buffer[..count])?;
        output.write_all(&buffer[..count])?;
        bytes -= count as u64;
    }
    Ok(())
}

// serde_json normally accepts duplicate map keys. At a checkpoint trust boundary
// that could make readers disagree about inventory, offsets or producer selection.
pub(crate) fn unique_json(bytes: &[u8]) -> Result<Value> {
    struct Unique(Value);
    impl<'de> Deserialize<'de> for Unique {
        fn deserialize<D: de::Deserializer<'de>>(
            deserializer: D,
        ) -> std::result::Result<Self, D::Error> {
            struct V;
            impl<'de> Visitor<'de> for V {
                type Value = Unique;
                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("JSON with unique keys")
                }
                fn visit_map<A: MapAccess<'de>>(
                    self,
                    mut map: A,
                ) -> std::result::Result<Unique, A::Error> {
                    let mut result = serde_json::Map::new();
                    while let Some((key, Unique(value))) = map.next_entry::<String, Unique>()? {
                        if result.insert(key, value).is_some() {
                            return Err(de::Error::custom("duplicate JSON key"));
                        }
                    }
                    Ok(Unique(Value::Object(result)))
                }
                fn visit_seq<A: SeqAccess<'de>>(
                    self,
                    mut seq: A,
                ) -> std::result::Result<Unique, A::Error> {
                    let mut result = Vec::new();
                    while let Some(Unique(v)) = seq.next_element()? {
                        result.push(v);
                    }
                    Ok(Unique(Value::Array(result)))
                }
                fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Unique, E> {
                    Ok(Unique(Value::String(v.into())))
                }
                fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Unique, E> {
                    Ok(Unique(Value::Bool(v)))
                }
                fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Unique, E> {
                    Ok(Unique(Value::from(v)))
                }
                fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Unique, E> {
                    Ok(Unique(Value::from(v)))
                }
                fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Unique, E> {
                    serde_json::Number::from_f64(v)
                        .map(|n| Unique(Value::Number(n)))
                        .ok_or_else(|| de::Error::custom("non-finite JSON number"))
                }
                fn visit_unit<E: de::Error>(self) -> std::result::Result<Unique, E> {
                    Ok(Unique(Value::Null))
                }
            }
            deserializer.deserialize_any(V)
        }
    }
    Ok(serde_json::from_slice::<Unique>(bytes)?.0)
}

#[cfg(test)]
mod tests {
    #[test]
    fn new_files_share_a_ceiling_that_failed_writes_cannot_exceed() {
        use super::*;
        let temp = tempfile::tempdir().unwrap();
        let budget = Cell::new(3);
        let mut first = new_file(temp.path(), "one.json", &budget).unwrap();
        first.write_all(b"ab").unwrap();
        let mut second = new_file(temp.path(), "two.json", &budget).unwrap();
        assert_eq!(
            second.write_all(b"cd").unwrap_err().kind(),
            std::io::ErrorKind::StorageFull
        );
        assert_eq!(fs::metadata(temp.path().join("two.json")).unwrap().len(), 0);
        second.write_all(b"c").unwrap();
        assert_eq!(
            first.write_all(b"d").unwrap_err().kind(),
            std::io::ErrorKind::StorageFull
        );
        assert_eq!(budget.get(), 0);
        assert_eq!(
            fs::metadata(temp.path().join("one.json")).unwrap().len()
                + fs::metadata(temp.path().join("two.json")).unwrap().len(),
            3
        );
    }
    use super::*;
    use std::io::{self, Cursor};

    struct ShortWriter(Vec<u8>);
    impl Write for ShortWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            let count = bytes.len().min(1);
            self.0.extend_from_slice(&bytes[..count]);
            Ok(count)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    struct FullDisk;
    impl Write for FullDisk {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::from(io::ErrorKind::StorageFull))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn bounded_copy_handles_short_writes_and_refuses_short_reads_and_full_disk() {
        let mut output = ShortWriter(Vec::new());
        copy_exact(&mut Cursor::new([1, 2, 3]), &mut output, 3, &mut [0; 2]).unwrap();
        assert_eq!(output.0, [1, 2, 3]);
        let error = copy_exact(&mut Cursor::new([1]), &mut output, 2, &mut [0; 2]).unwrap_err();
        assert!(matches!(error, Error::Io(ref e) if e.kind()==io::ErrorKind::UnexpectedEof));
        let error = copy_exact(&mut Cursor::new([1]), &mut FullDisk, 1, &mut [0; 1]).unwrap_err();
        assert!(matches!(error, Error::Io(ref e) if e.kind()==io::ErrorKind::StorageFull));
    }

    #[test]
    fn duplicate_header_fields_are_never_last_wins() {
        assert!(unique_json(br#"{"a":{"dtype":"U8","dtype":"I8"}}"#).is_err());
    }

    #[test]
    fn linked_payloads_are_not_rehashed_after_verified_inspection() {
        use crate::{
            ActivationCalibration, CachePolicy, LossyConversion, NonWeightMismatch, Output,
            Preparation, Source,
        };
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("base-checkpoint");
        fs::create_dir(&root).unwrap();
        let name = "model-00001-of-00001.safetensors";
        let mut header = serde_json::to_vec(
            &serde_json::json!({"a":{"dtype":"U8","shape":[131072],"data_offsets":[0,131072]}}),
        )
        .unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.resize(bytes.len() + 131072, 7);
        fs::write(root.join(name), bytes).unwrap();
        fs::write(
            root.join(INDEX),
            serde_json::to_vec(
                &serde_json::json!({"metadata":{"total_size":131072},"weight_map":{"a":name}}),
            )
            .unwrap(),
        )
        .unwrap();
        fs::write(root.join("config.json"), b"{}").unwrap();
        let limits = ExportLimits::default();
        let base = Snapshot::inspect(&root, &limits).unwrap();
        let config = ModuleConfig {
            schema: crate::SCHEMA.into(),
            name: "base-checkpoint".into(),
            base: Source {
                source: "base-checkpoint".into(),
                content_manifest: base.identity.clone(),
            },
            alternates: vec![],
            output: Output {
                loader_profile: crate::LOADER_PROFILE.into(),
                precision: "match-base".into(),
                configuration: "base-unchanged".into(),
                tokenizer: "base-unchanged".into(),
                non_weight_mismatch: NonWeightMismatch::Reject,
                namespace: "base-checkpoint".into(),
            },
            preparation: Preparation {
                trigger: "explicit-prepare-or-external-use-wrapper".into(),
                adapter_image: format!("converter@{}", Digest::of(b"image")),
                recipe: Digest::of(b"recipe"),
                parameters: BTreeMap::new(),
                lossy_conversion: LossyConversion::Deny,
                qualification_profile: Digest::of(b"policy"),
                activation_calibration: ActivationCalibration::PreserveAndQualify,
                max_runtime_seconds: 60,
                max_ram_bytes: 134217728,
                cpu_threads: 1,
                device: "cpu".into(),
                worker_pool: "cpu".into(),
            },
            cache: CachePolicy {
                publication: "sealed-regular-files".into(),
                reuse_unchanged: true,
                prefer_reflink: false,
                max_new_physical_bytes: 1048576,
                account_inflight: true,
                eviction: "least-recently-used-unpinned".into(),
                retain_dependencies: true,
            },
        };
        let profile = Profile::new(
            BTreeMap::from([("a".into(), BTreeSet::from(["a".into()]))]),
            false,
        )
        .unwrap();
        HASHED_FILE_BYTES.with(|total| total.set(0));
        let output = temp.path().join("output");
        let receipt = export(&config, &profile, &base, &[], &output, &limits).unwrap();
        // The only uncached file hash is the new receipt itself. Payload/shard
        // identities and final index/manifest hashes are already known.
        assert_eq!(
            HASHED_FILE_BYTES.with(|total| total.get()),
            fs::metadata(output.join(RECEIPT)).unwrap().len()
        );
        assert_eq!(receipt.reused_payload_bytes, 131072);
        assert_eq!(receipt.copied_payload_bytes, 0);
        assert_eq!(receipt.files[name], base.files[name]);
        HASHED_FILE_BYTES.with(|total| total.set(0));
        let staged = export_staged(
            &config,
            &profile,
            &base,
            &[],
            &temp.path().join("staging"),
            &limits,
        )
        .unwrap();
        assert_eq!(HASHED_FILE_BYTES.with(|total| total.get()), 0);
        assert_eq!(staged.receipt.files[name], base.files[name]);
        assert!(!temp.path().join("staging").join(name).exists());
    }
}
