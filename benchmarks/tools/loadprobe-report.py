"""report.py FROM_UTC TO_UTC LOG... : per-host average and peak rates in a window.
Times as 2026-10-20T16:05:00 (UTC). Columns are GB/s; cpu is sparknestd cores."""
import sys, collections, datetime
def ts(s): return datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc).timestamp()
t0, t1 = ts(sys.argv[1]), ts(sys.argv[2])
rows = collections.defaultdict(list)
for f in sys.argv[3:]:
    for line in open(f):
        p = line.split()
        if len(p) == 6: rows[p[1]].append((float(p[0]), *map(int, p[2:])))
print(f"window {t1-t0:.0f} s")
print(f"{'host':8} {'tx avg':>7} {'tx pk':>6} {'rx avg':>7} {'rx pk':>6} {'disk avg':>8} {'disk pk':>7} {'cpu':>5} {'tx GB':>6} {'rx GB':>6} {'disk GB':>7}")
for h in sorted(rows):
    r = sorted(rows[h])
    w = [(b[0], (b[1]-a[1])/(b[0]-a[0])/1e9, (b[2]-a[2])/(b[0]-a[0])/1e9, (b[3]-a[3])/(b[0]-a[0])/1e9, (b[4]-a[4])/(b[0]-a[0])/100)
         for a, b in zip(r, r[1:]) if t0 <= b[0] <= t1 and b[0] > a[0] and b[1] >= a[1]]
    if not w: continue
    n = len(w); s = lambda i: sum(x[i] for x in w); m = lambda i: max(x[i] for x in w)
    print(f"{h:8} {s(1)/n:7.2f} {m(1):6.2f} {s(2)/n:7.2f} {m(2):6.2f} {s(3)/n:8.2f} {m(3):7.2f} {s(4)/n:5.1f} {s(1):6.0f} {s(2):6.0f} {s(3):7.0f}")
