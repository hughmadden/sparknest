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
| oliver <- raptor | 5.14 Gb/s, 2.2k retr | 7.70 | 8.99 | 9.51, 8.8k retr |
| oliver -> raptor | 6.49 | 6.67 | 9.31 | 9.32 |
| oliver <- ostrich | 4.48, 2.7k retr | 6.55 | 9.18 | 9.57, 10.6k retr |
| oliver -> ostrich | 6.70 | 8.00 | 9.32 | 9.32 |

A Spark on Linux fills the link with one stream and no loss. Oliver
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
