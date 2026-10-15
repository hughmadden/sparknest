# RDMA counters (ADR-051)

sparknestd counts the RDMA traffic it moves, per RDMA device port, so
benchmarking tools can tell sparknest's share of the fabric from everything
else on the NIC. Two read-only, unauthenticated interfaces on each host:

| | Where | Format |
|---|---|---|
| Snapshot | `GET http://127.0.0.1:7403/v1/rdma/counters` (`node.counters_listen`; loopback only, `None` turns it off) | JSON |
| Stream | Unix socket `<state_dir>/run/rdma-counters.sock` (`/srv/sparknest/run/rdma-counters.sock`), mode 0666 | binary, below |

Containers reach the stream by bind-mounting the socket's directory,
`/srv/sparknest/run` (loopback HTTP is not reachable from a bridged
container), e.g. `-v /srv/sparknest/run:/run/sparknest:ro` and connect to
`/run/sparknest/rdma-counters.sock`. Mount the directory, not the socket:
the daemon recreates the socket at every start, and a bind mount of the file
itself would keep pointing at the old one; the directory stays (ADR-052).
Read-only is fine: connecting to a socket is not a write. `sparknest-rdma-counters`
(installed with sparknest) prints the stream as JSON lines and is the
reference decoder.

## What is counted

For each RDMA device port sparknest uses, named as in `/sys/class/infiniband`:

| Counter | Meaning |
|---|---|
| `bytes_sent`, `bytes_received` | payload bytes: data written to or landed from another host, and sparknest's control messages (64-byte read requests, 8-byte error replies) |
| `ops_sent`, `ops_received` | RDMA operations and messages |
| `packets_sent`, `packets_received` | estimated: each operation adds `ceil(bytes / path_mtu)`, at least 1 |

Data is counted when its operation completes successfully (a write's send
completion; a write-with-immediate's receive completion), control messages
when posted. Failed operations are not counted.

Counters are unsigned 64-bit, only increase, and start at 0 when the daemon
starts. A new `start_id` means a restart (counters reset). In JSON they are
plain integers: a decoder that keeps numbers as doubles loses exactness past
2^53 bytes (9 PB); Python and Rust do not.

**Not counted:** headers, ACKs and retransmissions (the NIC counts them),
TCP lanes to off-grid hosts (LAN, not the fabric), anything other processes
send. Payload plus `packets x per-packet overhead` estimates wire bytes:
RoCE v2 adds about 62 bytes per packet (Ethernet 14 + IPv4 20 + UDP 8 + BTH
12 + ICRC 4 + FCS 4), 16 more on the first packet of an RDMA write (RETH),
4 on a write-with-immediate's last; the receiver's ACKs go the other way.
NIC port counters (`port_xmit_data`, `port_rcv_data`) are in 4-byte words.

Raptor runs both rails (10.55.0.22, 10.55.1.22) on one port of `mlx5_0`:
one row, both addresses listed.

## Snapshot (JSON)

```json
{
  "schema": 1,
  "start_id": 13853021787621531023,
  "start_epoch_ns": 1791100000123456789,
  "boot_id": "4f1c…-…",
  "generation": 1,
  "mono_ns": 1234567890123,
  "wall_ns": 1791100042123456789,
  "ports": [
    {
      "device": "mlx5_0", "port": 1, "netdev": "enp1s0np0",
      "addrs": ["10.55.0.22", "10.55.1.22"], "path_mtu": 4096,
      "bytes_sent": 123, "bytes_received": 456, "ops_sent": 7,
      "ops_received": 8, "packets_sent": 9, "packets_received": 10
    }
  ]
}
```

`mono_ns` is `CLOCK_MONOTONIC`: comparable only on the same host and boot
(`boot_id`). `generation` changes when the port table does (0 while the
fabric has no rails, 1 once it has; it can find them after boot).

## Stream (binary)

Little-endian. Each message is `u32 length` (of what follows) `| u8 type |
body`. On connect the daemon sends a hello, then a counters message every
second. Write one byte at any time to choose the rate: 1–20 messages per
second (0 counts as 1, above 20 as 20).

A reader that falls behind is not buffered for: messages due while the
previous one is still unwritten are skipped, and their sequence numbers
with them (the counters are cumulative, so nothing is lost).

**Hello** (type 1), sent first and again whenever `generation` changes:

```text
u16 schema            1
u64 start_id
u64 start_epoch_ns    wall clock at daemon start
[16]u8 boot_id        /proc/sys/kernel/random/boot_id as bytes
u32 generation
u16 ports
ports x {
  u8  port
  u32 path_mtu        bytes
  str device          str = u8 length | UTF-8 bytes
  str netdev
  u8  addrs
  addrs x str addr
}
```

**Counters** (type 2):

```text
u64 seq               per connection, +1 per tick (gaps: skipped ticks)
u64 mono_ns           CLOCK_MONOTONIC
u64 wall_ns
u32 generation        matches the last hello
u16 ports             in the hello's order
ports x {
  u64 bytes_sent, bytes_received, ops_sent, ops_received,
      packets_sent, packets_received
}
```

A counters message is 35 + 48 x ports bytes on the wire. Unknown types are
to be skipped by length; a new schema number means the layout changed.
`tools/rdma-counters/testdata/frames.bin` holds one hello and one counters
message that both the daemon's encoder and the reference decoder are tested
against.

## Reference decoder (Python)

```python
import socket, struct

def frames(f):
    while (hdr := f.read(4)) and len(hdr) == 4:
        body = f.read(struct.unpack("<I", hdr)[0])
        yield body[0], body[1:]

s = socket.socket(socket.AF_UNIX)
s.connect("/srv/sparknest/run/rdma-counters.sock")
s.sendall(bytes([10]))                      # 10 Hz
for kind, b in frames(s.makefile("rb")):
    if kind == 2:
        seq, mono, wall, gen, n = struct.unpack_from("<QQQIH", b)
        for i in range(n):
            tx, rx, *_ = struct.unpack_from("<6Q", b, 30 + 48 * i)
            print(seq, i, tx, rx)
```

`sparknest-rdma-counters --rates --hz 10` decodes the hello as well and
prints per-port rates.
