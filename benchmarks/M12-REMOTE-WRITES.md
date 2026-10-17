# M12 — Remote writes over RDMA (2026-10-17)

ADR-057 item 4: a write to a file another host owns used to travel as a
`DataReq::Write` body on the nest-rpc control connection, the one Raft
heartbeats use. Now the owner pulls it: the writer SENDs a 64-byte
`WriteReq`, the owner RDMA-READs the bytes into a staging slot, applies
them and SENDs `WriteDone`.

Single-host loopback on **raptor** (ConnectX-7 `mlx5_0`, 400 Gb/s, RoCE v2,
MTU 4096, kernel 7.0, fw 28.43.4100): two fabric instances in one process,
over the NIC. The TCP baseline is the control connection over 127.0.0.1,
so it never touches a NIC and is **TCP's best case**. Between Sparks it
crosses the CRS804, which drops 1–5% of egress on Spark ports, and shares
its queue with Raft. Release builds.

## Transport: `cargo run --release -p nest-fabric --example write_bench`

The owner's sink only touches the bytes (one load per page), so this
measures the transport, not a disk. While each run streams, a probe makes a
64-byte call on the same control connection every millisecond: the round
trip a heartbeat would see. Two runs of 4 GiB per row for 1 MiB writes
(both shown), one run of 1 GiB for 64 KiB.

Idle control call: p50 7–8 us, p99 28–33 us.

### 1 MiB writes (FUSE `max_write`)

| In flight | RDMA GB/s | TCP GB/s | Control call p99 with RDMA | with TCP |
|---|---|---|---|---|
| 1 | 13.7 / 15.6 | 11.2 / 12.1 | 82 / 29 us | 1068 / 112 us |
| 8 | 52.0 / 51.7 | 11.7 / 13.0 | 115 / 115 us | 3388 / 1225 us |
| 32 | 43.7 / 46.0 | 14.7 / 13.9 | 1274 / 597 us | 4035 / 4879 us |

Control call p50 at 8 in flight: 15–16 us with RDMA against 548–616 us
behind TCP writes; max 230–428 us against 1.4–5.0 ms.

### 64 KiB writes

| In flight | RDMA GB/s | TCP GB/s | Control call p99 with RDMA | with TCP |
|---|---|---|---|---|
| 1 | 3.46 | 5.80 | 35 us | 44 us |
| 8 | 24.1 | 8.77 | 62 us | 95 us |
| 32 | 30.5 | 10.0 | 50 us | 351 us |

A single small write in flight is faster over loopback TCP (one copy, no
NIC round trips; a pull costs SEND, READ request, READ response, SEND).
That stops holding once more than one is in flight, and it never applies
between hosts.

At 32 × 1 MiB the probe's p99 rises with RDMA too: one process plays both
ends, and 32 writers keep tokio's eight workers and the two pollers busy.
The probe waits for a worker, not for the wire.

## Through the VFS: `m4_fabric::remote_writes_go_over_rdma`

`NEST_BENCH_MIB=1024 cargo test --release -p nest-testkit --test m4_fabric
-- --nocapture remote_writes`: three in-process nodes, 1 MiB `Vfs::write`s
from node 2 to a file node 1 owns, in /tmp (tmpfs). Three runs:

| Path | 1 in flight | 8 in flight |
|---|---|---|
| RDMA | 2.47 / 2.38 / 2.30 GB/s | 1.82 / 1.70 / 1.55 GB/s |
| control connection (loopback TCP) | 2.28 / 2.02 / 1.98 GB/s | 1.73 / 1.45 / 1.53 GB/s |

The owner's apply path (route, `owner_mutate`, `spawn_blocking`, `pwrite`
into one tmpfs file under its inode lock) caps both near 2 GB/s: an order
of magnitude below either transport. More writes in flight only queue on
the inode, which is why 8 in flight comes out slower. RDMA still gains about
15% here, but mainly it takes the bytes off the Raft connection.

## Reading

- RDMA's gain is the control plane: with 1 MiB writes streaming,
  heartbeats on the control connection wait 10–30 x less (p99 115 us
  against 1.2–3.4 ms at 8 in flight). That was the reason for ADR-057.
- Bulk throughput is not limited by the write transport: on one host a
  file takes ~2 GB/s through the apply path. Parallel writers to different
  files, or a faster apply path (O_DIRECT, ADR-042's ring), are what
  raises it. That is a separate item, not tuned here.
- Write staging defaults (64 × 64 KiB + 32 × 1 MiB per device, 128 + 32
  outgoing) took 52 GB/s at 8 in flight on 400G. They are left as they are
  until a multi-host run shows writers waiting (`pull_waits`).

Not measured yet: Spark-to-Spark across the CRS804 (needs a deploy of
0.5.0 to the trial cluster), and mixed read+write load.
