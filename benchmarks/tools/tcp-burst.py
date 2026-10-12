#!/usr/bin/env python3
"""Chunk fetch time on one TCP connection, back-to-back and after 1 s idle
(slow start after idle). Server: tcp-burst.py serve PORT; client: tcp-burst.py HOST PORT."""
import socket, struct, sys, time
# server: python3 burst.py serve PORT ; client: python3 burst.py HOST PORT
if sys.argv[1] == "serve":
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", int(sys.argv[2]))); s.listen(4)
    buf = b"\0" * (16 << 20)
    while True:
        c, _ = s.accept(); c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                h = c.recv(4, socket.MSG_WAITALL)
                if len(h) < 4: break
                n = struct.unpack("<I", h)[0]; c.sendall(buf[:n])
        except OSError: pass
        c.close()
host, port = sys.argv[1], int(sys.argv[2])
def fetch(c, n):
    t = time.perf_counter(); c.sendall(struct.pack("<I", n)); got = 0
    while got < n: got += len(c.recv(min(n - got, 4 << 20)))
    return (time.perf_counter() - t) * 1e3
c = socket.create_connection((host, port)); c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
c.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
for n in (4 << 20, 16 << 20):
    for _ in range(5): fetch(c, n)  # warm
    warm = sorted(fetch(c, n) for _ in range(10))[5]
    idle = []
    for _ in range(6):
        time.sleep(1.0); idle.append(fetch(c, n))
    idle = sorted(idle)[3]
    print(f"{n>>20:>2} MiB  back-to-back {warm:6.2f} ms ({n*8/warm/1e6:4.1f} Gb/s)   after 1 s idle {idle:6.2f} ms ({n*8/idle/1e6:4.1f} Gb/s)")
