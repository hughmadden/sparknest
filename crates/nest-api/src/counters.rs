//! RDMA traffic counters for benchmarking tools (ADR-051,
//! docs/RDMA-COUNTERS.md): a JSON snapshot over HTTP on loopback, and a
//! binary stream on a Unix socket. Read-only and unauthenticated; never
//! bound beyond loopback. Building either reads atomics only.

use axum::{Json, Router, extract::State, routing::get};
use nest_fabric::Fabric;
use nest_fabric::counters::{CounterValues, PortInfo};
use serde::Serialize;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::AsyncReadExt;

pub const SCHEMA: u16 = 1;
const HELLO: u8 = 1;
const COUNTERS: u8 = 2;
const MAX_HZ: u8 = 20;

/// Where the counters come from, and this daemon run's identity.
#[derive(Clone)]
pub struct Counters {
    inner: Arc<Inner>,
}

struct Inner {
    fabric: Box<dyn Fn() -> Option<Arc<Fabric>> + Send + Sync>,
    /// Random per daemon start: a change means the counters reset.
    start_id: u64,
    start_epoch_ns: u64,
    boot_id: [u8; 16],
    boot_id_str: String,
    /// Snapshots served, for tests.
    served: AtomicU64,
}

#[derive(Serialize)]
pub struct Snapshot {
    pub schema: u16,
    pub start_id: u64,
    pub start_epoch_ns: u64,
    pub boot_id: String,
    pub generation: u32,
    pub mono_ns: u64,
    pub wall_ns: u64,
    pub ports: Vec<PortSnapshot>,
}

#[derive(Serialize)]
pub struct PortSnapshot {
    #[serde(flatten)]
    pub info: PortInfo,
    #[serde(flatten)]
    pub counters: CounterValues,
}

fn mono_ns() -> u64 {
    nix::time::clock_gettime(nix::time::ClockId::CLOCK_MONOTONIC)
        .map(|t| t.tv_sec() as u64 * 1_000_000_000 + t.tv_nsec() as u64)
        .unwrap_or(0)
}

fn wall_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
}

fn boot_id() -> ([u8; 16], String) {
    let s = std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let hex: String = s.chars().filter(|c| c.is_ascii_hexdigit()).collect();
    let mut b = [0u8; 16];
    for (i, byte) in b.iter_mut().enumerate() {
        *byte = hex
            .get(i * 2..i * 2 + 2)
            .and_then(|h| u8::from_str_radix(h, 16).ok())
            .unwrap_or(0);
    }
    (b, s)
}

impl Counters {
    pub fn new(fabric: impl Fn() -> Option<Arc<Fabric>> + Send + Sync + 'static) -> Counters {
        let (boot_id, boot_id_str) = boot_id();
        Counters {
            inner: Arc::new(Inner {
                fabric: Box::new(fabric),
                start_id: rand_u64(),
                start_epoch_ns: wall_ns(),
                boot_id,
                boot_id_str,
                served: AtomicU64::new(0),
            }),
        }
    }

    pub fn start_id(&self) -> u64 {
        self.inner.start_id
    }

    /// The device table changes only when the fabric comes up (it may find
    /// its rails after boot): 0 without one, 1 with.
    fn generation(f: &Option<Arc<Fabric>>) -> u32 {
        f.is_some() as u32
    }

    pub fn snapshot(&self) -> Snapshot {
        self.inner.served.fetch_add(1, Ordering::Relaxed);
        let f = (self.inner.fabric)();
        let (ports, mut values) = (
            f.as_ref().map(|f| f.ports()).unwrap_or_default(),
            Vec::new(),
        );
        if let Some(f) = &f {
            f.port_counters(&mut values);
        }
        Snapshot {
            schema: SCHEMA,
            start_id: self.inner.start_id,
            start_epoch_ns: self.inner.start_epoch_ns,
            boot_id: self.inner.boot_id_str.clone(),
            generation: Self::generation(&f),
            mono_ns: mono_ns(),
            wall_ns: wall_ns(),
            ports: ports
                .into_iter()
                .zip(values)
                .map(|(info, counters)| PortSnapshot { info, counters })
                .collect(),
        }
    }

    fn hello(&self, f: &Option<Arc<Fabric>>, out: &mut Vec<u8>) {
        let ports = f.as_ref().map(|f| f.ports()).unwrap_or_default();
        encode_hello(
            out,
            self.inner.start_id,
            self.inner.start_epoch_ns,
            &self.inner.boot_id,
            Self::generation(f),
            &ports,
        );
    }

    fn counters(
        &self,
        f: &Option<Arc<Fabric>>,
        seq: u64,
        values: &mut Vec<CounterValues>,
        out: &mut Vec<u8>,
    ) {
        values.clear();
        if let Some(f) = f {
            f.port_counters(values);
        }
        encode_counters(out, seq, mono_ns(), wall_ns(), Self::generation(f), values);
    }
}

/// The hello frame (docs/RDMA-COUNTERS.md).
pub fn encode_hello(
    out: &mut Vec<u8>,
    start_id: u64,
    start_epoch_ns: u64,
    boot_id: &[u8; 16],
    generation: u32,
    ports: &[PortInfo],
) {
    frame(out, HELLO, |b| {
        b.extend_from_slice(&SCHEMA.to_le_bytes());
        b.extend_from_slice(&start_id.to_le_bytes());
        b.extend_from_slice(&start_epoch_ns.to_le_bytes());
        b.extend_from_slice(boot_id);
        b.extend_from_slice(&generation.to_le_bytes());
        b.extend_from_slice(&(ports.len() as u16).to_le_bytes());
        for p in ports {
            b.push(p.port);
            b.extend_from_slice(&p.path_mtu.to_le_bytes());
            put_str(b, &p.device);
            put_str(b, &p.netdev);
            b.push(p.addrs.len().min(255) as u8);
            for a in p.addrs.iter().take(255) {
                put_str(b, a);
            }
        }
    });
}

/// A counters frame (docs/RDMA-COUNTERS.md).
pub fn encode_counters(
    out: &mut Vec<u8>,
    seq: u64,
    mono_ns: u64,
    wall_ns: u64,
    generation: u32,
    values: &[CounterValues],
) {
    frame(out, COUNTERS, |b| {
        b.extend_from_slice(&seq.to_le_bytes());
        b.extend_from_slice(&mono_ns.to_le_bytes());
        b.extend_from_slice(&wall_ns.to_le_bytes());
        b.extend_from_slice(&generation.to_le_bytes());
        b.extend_from_slice(&(values.len() as u16).to_le_bytes());
        for v in values {
            for x in [
                v.bytes_sent,
                v.bytes_received,
                v.ops_sent,
                v.ops_received,
                v.packets_sent,
                v.packets_received,
            ] {
                b.extend_from_slice(&x.to_le_bytes());
            }
        }
    });
}

fn rand_u64() -> u64 {
    // No rand dependency here: the clock and pid are unique enough per start.
    let t = wall_ns() ^ mono_ns().rotate_left(17) ^ ((std::process::id() as u64) << 40);
    // SplitMix64 finalizer.
    let mut z = t.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn put_str(b: &mut Vec<u8>, s: &str) {
    let s = &s.as_bytes()[..s.len().min(255)];
    b.push(s.len() as u8);
    b.extend_from_slice(s);
}

/// `u32 length | u8 type | body`, appended to `out`.
fn frame(out: &mut Vec<u8>, kind: u8, body: impl FnOnce(&mut Vec<u8>)) {
    let at = out.len();
    out.extend_from_slice(&[0; 4]);
    out.push(kind);
    body(out);
    let len = (out.len() - at - 4) as u32;
    out[at..at + 4].copy_from_slice(&len.to_le_bytes());
}

/// Serve the JSON snapshot at `GET /v1/rdma/counters` on `addr`, which must
/// be a loopback address.
pub async fn serve_http(c: Counters, addr: SocketAddr) -> anyhow::Result<()> {
    anyhow::ensure!(
        addr.ip().is_loopback(),
        "RDMA counters listen on loopback only, not {addr}"
    );
    let app = Router::new()
        .route("/v1/rdma/counters", get(snapshot))
        .with_state(c);
    let l = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(l, app).await?;
    Ok(())
}

async fn snapshot(State(c): State<Counters>) -> Json<Snapshot> {
    Json(c.snapshot())
}

/// Stream frames on a Unix socket at `path` (mode 0666, so containers with
/// it bind-mounted can read it).
pub async fn serve_stream(c: Counters, path: PathBuf) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    // The socket's directory is what containers mount (ADR-052): it stays
    // put while the socket in it is recreated at every start.
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o755))?;
    }
    let _ = std::fs::remove_file(&path);
    let l = tokio::net::UnixListener::bind(&path)?;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666))?;
    loop {
        let (s, _) = l.accept().await?;
        let c = c.clone();
        tokio::spawn(async move {
            if let Err(e) = stream(c, s).await {
                tracing::debug!(error = %e, "RDMA counters stream ended");
            }
        });
    }
}

/// One subscriber: a hello, then counters at 1 Hz, or the rate it asks for
/// with one byte (1..=20 Hz) at any time. A frame that cannot be written
/// at once waits; frames due meanwhile are skipped (the counters are
/// cumulative), and their sequence numbers with them.
async fn stream(c: Counters, s: tokio::net::UnixStream) -> std::io::Result<()> {
    let (mut rd, wr) = s.into_split();
    let mut hz: u8 = 1;
    let mut pending: Vec<u8> = Vec::new();
    let mut values = Vec::new();
    let mut f = (c.inner.fabric)();
    c.hello(&f, &mut pending);
    let mut generation = Counters::generation(&f);
    let mut seq = 0u64;
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut req = [0u8; 1];
    loop {
        tokio::select! {
            r = rd.read(&mut req) => match r {
                Ok(0) | Err(_) => return Ok(()),
                Ok(_) => {
                    let want = req[0].clamp(1, MAX_HZ);
                    if want != hz {
                        hz = want;
                        tick = tokio::time::interval(Duration::from_secs(1) / hz as u32);
                        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                    }
                }
            },
            _ = tick.tick() => {
                seq += 1;
                if !pending.is_empty() && !flush(&wr, &mut pending)? {
                    continue; // still behind: skip this one
                }
                f = (c.inner.fabric)();
                if Counters::generation(&f) != generation {
                    generation = Counters::generation(&f);
                    c.hello(&f, &mut pending);
                }
                c.counters(&f, seq, &mut values, &mut pending);
                flush(&wr, &mut pending)?;
            }
        }
    }
}

/// Write what the socket takes now; `true` when nothing is left.
fn flush(wr: &tokio::net::unix::OwnedWriteHalf, pending: &mut Vec<u8>) -> std::io::Result<bool> {
    while !pending.is_empty() {
        match wr.try_write(pending) {
            Ok(n) => {
                pending.drain(..n);
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return Ok(false),
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frames in tools/rdma-counters/testdata/frames.bin, which the
    /// Python reference decoder's test reads back: layout changes must
    /// change both (and the schema number).
    #[test]
    fn frames_match_the_shared_fixture() {
        let mut out = Vec::new();
        let ports = [
            PortInfo {
                device: "mlx5_0".into(),
                port: 1,
                netdev: "enp1s0np0".into(),
                addrs: vec!["10.55.0.22".into(), "10.55.1.22".into()],
                path_mtu: 4096,
            },
            PortInfo {
                device: "rocep1s0f0".into(),
                port: 1,
                netdev: "enp1s0f0np0".into(),
                addrs: vec!["10.55.0.1".into()],
                path_mtu: 1024,
            },
        ];
        let boot: [u8; 16] = std::array::from_fn(|i| i as u8);
        encode_hello(
            &mut out,
            0x0123_4567_89ab_cdef,
            1_791_000_000_000_000_000,
            &boot,
            1,
            &ports,
        );
        let v = |k: u64| CounterValues {
            bytes_sent: k << 40,
            bytes_received: k,
            ops_sent: 2,
            ops_received: 3,
            packets_sent: 4,
            packets_received: 5,
        };
        encode_counters(
            &mut out,
            42,
            123_456_789,
            1_791_000_000_500_000_000,
            1,
            &[v(7), v(9)],
        );
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tools/rdma-counters/testdata/frames.bin"
        );
        if std::env::var_os("UPDATE_FIXTURE").is_some() {
            std::fs::write(path, &out).unwrap();
        }
        assert_eq!(
            std::fs::read(path).expect("fixture (UPDATE_FIXTURE=1 writes it)"),
            out
        );
    }

    #[test]
    fn frames_are_length_prefixed() {
        let c = Counters::new(|| None);
        let mut out = Vec::new();
        c.hello(&None, &mut out);
        let len = u32::from_le_bytes(out[0..4].try_into().unwrap()) as usize;
        assert_eq!(len, out.len() - 4);
        assert_eq!(out[4], HELLO);
        assert_eq!(u16::from_le_bytes(out[5..7].try_into().unwrap()), SCHEMA);
        let mut v = Vec::new();
        let at = out.len();
        c.counters(&None, 7, &mut v, &mut out);
        assert_eq!(out[at + 4], COUNTERS);
        assert_eq!(
            u64::from_le_bytes(out[at + 5..at + 13].try_into().unwrap()),
            7
        );
        // seq, mono, wall, generation, count: no ports.
        assert_eq!(out.len() - at, 4 + 1 + 8 * 3 + 4 + 2);
    }
}
