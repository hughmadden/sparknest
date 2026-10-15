//! ADR-051 end to end on this host's RDMA NIC: two fabrics move a known
//! number of bytes; the Unix-socket stream (at a requested 20 Hz) and the
//! HTTP snapshot both show exactly that. Skipped without an RDMA device.

use futures::FutureExt;
use futures::future::BoxFuture;
use nest_api::counters::{Counters, serve_http, serve_stream};
use nest_fabric::{Fabric, FabricConfig, ReadSource, ServeFile};
use nest_rpc::{Rpc, RpcConfig};
use nest_types::{FileId, Generation, NestError, NodeId};
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

struct One(Arc<std::fs::File>);

impl ReadSource for One {
    fn open_now(&self, _: FileId, _: Generation) -> Option<Result<ServeFile, NestError>> {
        Some(Ok(ServeFile {
            file: self.0.clone(),
            guard: Box::new(()),
        }))
    }
    fn open(&self, f: FileId, g: Generation) -> BoxFuture<'static, Result<ServeFile, NestError>> {
        let r = self.open_now(f, g).unwrap();
        async move { r }.boxed()
    }
}

async fn node(id: u64) -> Option<(Rpc, Arc<Fabric>)> {
    let rpc = Rpc::bind(RpcConfig {
        node: NodeId(id),
        cluster: "counters-test".into(),
        secret: b"s".to_vec(),
        listen: "127.0.0.1:0".parse().unwrap(),
        connect_timeout: Duration::from_secs(2),
    })
    .await
    .unwrap();
    let cfg = FabricConfig {
        chunk: 1 << 20,
        client_slots: 4,
        server_slots: 4,
        window: 4,
        ..FabricConfig::default()
    };
    Some((rpc.clone(), Fabric::start(NodeId(id), cfg, rpc).unwrap()?))
}

/// The next counters frame: its sequence number and each port's
/// (bytes_sent, bytes_received).
async fn next_counters(s: &mut tokio::net::UnixStream) -> (u64, Vec<(u64, u64)>) {
    loop {
        let n = s.read_u32_le().await.unwrap() as usize;
        let mut b = vec![0u8; n];
        s.read_exact(&mut b).await.unwrap();
        if b[0] != 2 {
            continue;
        }
        let b = &b[1..];
        let seq = u64::from_le_bytes(b[0..8].try_into().unwrap());
        let ports = u16::from_le_bytes(b[28..30].try_into().unwrap()) as usize;
        let v = (0..ports)
            .map(|p| {
                let at = 30 + p * 48;
                (
                    u64::from_le_bytes(b[at..at + 8].try_into().unwrap()),
                    u64::from_le_bytes(b[at + 8..at + 16].try_into().unwrap()),
                )
            })
            .collect();
        return (seq, v);
    }
}

async fn http_get(addr: std::net::SocketAddr, path: &str) -> serde_json::Value {
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    s.write_all(format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").as_bytes())
        .await
        .unwrap();
    let mut r = String::new();
    s.read_to_string(&mut r).await.unwrap();
    serde_json::from_str(r.split("\r\n\r\n").nth(1).unwrap()).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn stream_and_snapshot_show_the_bytes_moved() {
    let Some((rpc1, server)) = node(1).await else {
        return;
    };
    let (rpc2, reader) = node(2).await.unwrap();
    rpc2.set_peer(NodeId(1), rpc1.local_addr());
    rpc1.set_peer(NodeId(2), rpc2.local_addr());
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    std::fs::File::create(&data)
        .unwrap()
        .write_all(&vec![5u8; 8 << 20])
        .unwrap();
    let src: Arc<dyn ReadSource> = Arc::new(One(Arc::new(std::fs::File::open(&data).unwrap())));
    server.set_source(Arc::downgrade(&src));

    // The reader's counters, as its daemon would expose them.
    let r2 = reader.clone();
    let counters = Counters::new(move || Some(r2.clone()));
    let sock = dir.path().join("run").join("rdma-counters.sock");
    tokio::spawn(serve_stream(counters.clone(), sock.clone()));
    let addr = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap()
    };
    tokio::spawn(serve_http(counters.clone(), addr));
    assert!(
        serve_http(counters.clone(), "0.0.0.0:7499".parse().unwrap())
            .await
            .is_err(),
        "bound beyond loopback"
    );
    tokio::time::sleep(Duration::from_millis(200)).await;

    let mut s = tokio::net::UnixStream::connect(&sock).await.unwrap();
    let mode =
        std::os::unix::fs::PermissionsExt::mode(&std::fs::metadata(&sock).unwrap().permissions());
    assert_eq!(mode & 0o777, 0o666);
    let dir_mode = std::os::unix::fs::PermissionsExt::mode(
        &std::fs::metadata(sock.parent().unwrap())
            .unwrap()
            .permissions(),
    );
    assert_eq!(
        dir_mode & 0o777,
        0o755,
        "containers mount the socket's directory"
    );
    s.write_all(&[20]).await.unwrap();
    let (_, before) = next_counters(&mut s).await;
    let rx0: u64 = before.iter().map(|p| p.1).sum();
    for i in 0..8u64 {
        let b = reader
            .read(NodeId(1), FileId(1), Generation(1), i << 20, 1 << 20)
            .await
            .unwrap();
        assert_eq!(b.len(), 1 << 20);
    }
    // Frames now come at ~20 Hz.
    let t = std::time::Instant::now();
    let (mut seqs, mut rx) = (Vec::new(), 0);
    while t.elapsed() < Duration::from_millis(600) {
        let (seq, v) = next_counters(&mut s).await;
        seqs.push(seq);
        rx = v.iter().map(|p| p.1).sum::<u64>();
    }
    assert!(
        seqs.len() >= 8,
        "only {} frames in 600 ms at 20 Hz",
        seqs.len()
    );
    assert!(seqs.windows(2).all(|w| w[1] > w[0]));
    assert_eq!(rx - rx0, 8 << 20, "stream: received bytes");

    let j = http_get(addr, "/v1/rdma/counters").await;
    assert_eq!(j["schema"], 1);
    assert_eq!(j["start_id"].as_u64(), Some(counters.start_id()));
    let ports = j["ports"].as_array().unwrap();
    assert!(!ports.is_empty());
    let http_rx: u64 = ports
        .iter()
        .map(|p| p["bytes_received"].as_u64().unwrap())
        .sum();
    assert_eq!(http_rx, rx, "snapshot and stream disagree");
    assert!(ports.iter().all(|p| p["device"].is_string()
        && p["netdev"].is_string()
        && p["path_mtu"].as_u64() >= Some(256)));
}
