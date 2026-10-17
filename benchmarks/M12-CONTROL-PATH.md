# M12 — Control path: fabric vs 10G LAN (2026-10-17)

ADR-057 keeps the control connection on the fabric and falls back to the
LAN only when the fabric link fails. This measures whether that is the
right default: authenticated nest-rpc round trips (`crates/nest-rpc/examples/rpc_rtt.rs`,
200-byte requests, about a heartbeat) between raptor (client) and kiwi
(server), on 10.55.0.4 (fabric, ConnectX through the CRS804) and on
172.22.2.4 (10G LAN: raptor's 10GbE, kiwi's RTL8127). Both ends set
TCP_NODELAY and the 5 ms retransmit floor (ADR-056), as sparknestd does.

## Idle (500 calls/s, 30 s each, one after the other)

| Path | p50 | p99 | p99.9 | p99.99 | max | > 150 ms |
|---|---|---|---|---|---|---|
| fabric | 258 µs | 1.08 ms | 1.29 ms | 1.34 ms | 1.39 ms | 0 |
| LAN | 490 µs | 1.00 ms | 1.17 ms | 1.36 ms | 1.49 ms | 0 |

The fabric's median is half the LAN's; the tails are the same (scheduling
on the client, not the wire).

## Side by side, 30 min (200 calls/s on each path at once, 10:47–11:17)

| Path | calls | p50 | p99 | p99.9 | p99.99 | max | > 150 ms |
|---|---|---|---|---|---|---|---|
| fabric | 360,001 | 246 µs | 1.08 ms | 1.36 ms | 1.62 ms | 3.71 ms | 0 |
| LAN | 360,001 | 372 µs | 1.13 ms | 1.53 ms | 1.92 ms | 4.04 ms | 0 |

The fabric was quiet: raptor's port counted no out-of-sequence packets in
the window (the cuteafd runs were paused), so this is still the no-incast
case. Neither path came near a heartbeat's 150 ms budget. Still to
measure: the same run during a multi-Spark model load, the case where
ADR-056 saw heartbeat timeouts.
