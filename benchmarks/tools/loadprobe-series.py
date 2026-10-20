"""series.py FROM TO LOG...: per-second tx (GB/s) per host, plus total rx and disk."""
import sys, collections, datetime
def ts(s): return datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc).timestamp()
t0, t1 = ts(sys.argv[1]), ts(sys.argv[2])
rows = collections.defaultdict(dict)
for f in sys.argv[3:]:
    prev = None
    for line in open(f):
        p = line.split()
        if len(p) != 6: continue
        cur = (float(p[0]), *map(int, p[2:]))
        if prev and cur[0] > prev[0] and t0 <= cur[0] <= t1:
            dt = cur[0]-prev[0]
            rows[int(cur[0])][p[1]] = tuple((cur[i]-prev[i])/dt/1e9 for i in (1, 2, 3))
        prev = cur
hosts = ["ostrich", "dodo", "emu", "kiwi", "rhea", "moa", "raptor"]
print("time      " + " ".join(f"{h[:6]:>6}" for h in hosts) + "   rxSum diskSum")
for t in sorted(rows):
    r = rows[t]
    tx = " ".join(f"{r.get(h, (0,0,0))[0]:6.1f}" for h in hosts)
    print(datetime.datetime.utcfromtimestamp(t).strftime("%H:%M:%S"), " ", tx, f"  {sum(v[1] for v in r.values()):5.1f} {sum(v[2] for v in r.values()):6.1f}")
