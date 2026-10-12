# M10 — LAN TCP for off-grid hosts (2026-10-12)

Baseline for the TCP bulk path of off-grid hosts (ADR-047). All LAN ports
are 10GbE (172.22.0.0/16, MTU 1500 except oliver's WSL interface, 1280).
Linux TCP defaults everywhere: cubic (bbr not loaded), fq_codel,
`tcp_slow_start_after_idle=1`, `tcp_rmem` max 32 MiB, `tcp_wmem` max 4 MiB,
`rmem_max`/`wmem_max` 4 MiB.

| Host | LAN NIC | Kind |
|---|---|---|
| raptor | 10GbE (enp172s0) | fabric host, serving side |
| ostrich, kakapo | Realtek RTL8127 (r8127), Linux | Spark |
| oliver | Realtek RTL8127 on Windows, WSL2 NAT (Hyper-V switch), kernel 6.18 | WSL |

## Streams (iperf3, 5 s, 1 s omitted)

`<-` is the direction that matters most: a fabric host serving the
off-grid host.

| Path | P=1 | P=2 | P=4 | P=8 |
|---|---|---|---|---|
| ostrich <- raptor | 9.41 Gb/s, 0 retr | 9.41 | 9.41 | 9.41 |
| ostrich -> raptor | 9.41, 0 retr | 9.41 | 9.41 | 9.41 |
| kakapo <- raptor | 7.86, 162 retr | 9.41 | 9.41 | 9.39 |
| kakapo -> raptor | 7.67, 280 retr | 7.81 | 9.41 | 9.41 |
| oliver <- raptor | 5.14 Gb/s, 2.2k retr | 7.70 | 8.99 | 9.51, 8.8k retr |
| oliver -> raptor | 6.49 | 6.67 | 9.31 | 9.32 |
| oliver <- ostrich | 4.48, 2.7k retr | 6.55 | 9.18 | 9.57, 10.6k retr |
| oliver -> ostrich | 6.70 | 8.00 | 9.32 | 9.32 |

ostrich fills the link with one stream and no loss; kakapo (same NIC,
but 1.3 ms RTT against ostrich's 0.5 ms) needs two. Oliver
needs four streams for ~9.2 Gb/s and eight for ~9.5; receiving, it drops
packets heavily (retransmits on the sender), so single streams stall in
congestion avoidance. Parallel streams hide the loss.

## One connection, request/response (`tools/tcp-burst.py`, TCP_NODELAY)

Median fetch time of a chunk on a warm connection, and after the
connection sat idle for 1 s (`tcp_slow_start_after_idle=1` collapses the
window back to 10 segments).

| Path | RTT | 4 MiB back-to-back | 4 MiB after idle | 16 MiB back-to-back | 16 MiB after idle |
|---|---|---|---|---|---|
| ostrich <- raptor | 0.49 ms | 3.82 ms (8.8 Gb/s) | 4.63 ms (7.2) | 14.39 ms (9.3) | 14.86 ms (9.0) |
| kakapo <- raptor | 1.29 ms | 4.81 ms (7.0) | 6.81 ms (4.9) | 17.62 ms (7.6) | 18.95 ms (7.1) |
| oliver <- raptor | 0.94 ms | 8.44 ms (4.0) | 10.62 ms (3.2) | 33.69 ms (4.0) | 33.79 ms (4.0) |

Slow start after idle costs 0.8–2 ms per idle connection's first chunk
(20–40% of a 4 MiB fetch). One request at a time on one connection never
reaches line rate off-grid: the window must stay full.

## oliver tuned (same day)

Windows RTL8127 Receive and Transmit Buffers 1024/512 -> 4096/4096, then
WSL eth0 MTU 1280 -> 1500 (WSL copied the Tailscale adapter's 1280; a
route keeps 1280 for 100.64.0.0/10; `wsl-lan-mtu.service` sets it at boot).

| oliver <- raptor | before | buffers 4096 | + MTU 1500 |
|---|---|---|---|
| iperf3 P=1 | 5.14 Gb/s, 2.2k retr | 4.48, 0 retr | 5.19, 0 retr |
| iperf3 P=2 | 7.70 | 7.31 | 8.18, 0 retr |
| iperf3 P=8 | 9.51, 8.8k retr | 9.62, 19k retr | 9.68, 464 retr |
| one connection, 4 MiB back-to-back | 4.0 Gb/s | | 5.2 Gb/s |
| one connection, 16 MiB back-to-back | 4.0 Gb/s | | 5.6 Gb/s |

The buffers removed the single-stream loss but not its ceiling: one flow
into WSL is limited by per-packet work in the Hyper-V path, not drops.

## Consequences for the bulk path

- Several connections per peer, 4 by default (configurable; 8 for WSL),
  chunks spread over them, each with more than one request in flight so
  no connection drains between chunks.
- TCP_NODELAY on all of them; the reply header is sent with MSG_MORE
  ahead of the `sendfile` body so header and data share segments.
- Leave socket buffers to autotuning (an explicit SO_RCVBUF disables it
  and `rmem_max` caps it at 4 MiB); 4 MiB covers the 1.6 MB
  bandwidth-delay product at 1.3 ms.
- `net.ipv4.tcp_slow_start_after_idle=0` on every host (root, sysctl):
  it cannot be set per socket.

## TCP lanes (`nest-fabric/examples/lanes_bench.rs`, 2026-10-12)

raptor serves a 4 GiB file from page cache with `sendfile`; the off-grid
host dials the lanes and reads it in 4 MiB chunks (second of two rounds).
oliver as tuned above; kakapo stock.

| Reader | lanes | in flight | GB/s | Gb/s |
|---|---|---|---|---|
| oliver | 1 | 16 | 0.71 | 5.7 |
| oliver | 2 | 16 | 0.88 | 7.0 |
| oliver | 4 | 16 | 1.15 | 9.2 |
| oliver | 8 | 16 | 1.18 | 9.5 |
| kakapo | 1 | 16 | 0.96 | 7.7 |
| kakapo | 2 | 16 | 0.95 | 7.6 |
| kakapo | 4 | 16 | 1.18 | 9.4 |
| kakapo | 8 | 16 | 1.18 | 9.4 |
| oliver | 4 | 2 | 0.98 | 7.8 |
| oliver | 4 | 4 | 1.07 | 8.6 |
| oliver | 4 | 8 | 1.20 | 9.6 |
| kakapo | 4 | 2 | 1.07 | 8.6 |
| kakapo | 4 | 4 | 1.16 | 9.3 |
| kakapo | 4 | 8 | 1.17 | 9.4 |

Defaults: 4 lanes per peer, and at least 8 chunks (32 MiB) in flight,
which the readahead window (16 chunks) and replication window (16) give.
That is line rate on both kinds of off-grid host; 8 lanes add ~3% on
oliver only.

## Through the filesystem (0.3.0 on the real cluster, 2026-10-13)

kakapo joined off-grid; reading MiMo-V2.6-Pro blobs (3.9 GiB each, held on
all six Sparks' NVMe) through /mnt/sparknest with `dd bs=16M`, kakapo's
page cache dropped first (`sparknest-drop-page-cache`).

| Read | Throughput |
|---|---|
| one stream, holder's copy cold | 749–771 MB/s |
| one stream, holder's copy in its page cache | 838–861 MB/s |
| two streams at once, holders warm | 2 × 566 MB/s = 1.13 GB/s |
| lanes_bench ostrich -> kakapo (no filesystem), 4 lanes | 1.16 GB/s |

The lanes and the link reach line rate; one sequential reader through the
filesystem gets ~75% of it (readahead from one holder, two copies per
chunk on the reader). Loaders that open several shards at once see line
rate. A 60 GiB file held only in the `scratch` archive read at 448 MB/s:
the SATA archive disk behind its gateway.
