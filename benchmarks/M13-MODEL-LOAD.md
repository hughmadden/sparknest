# M13 — Model loads by the LLM engine, through the mount (2026-10-20)

Baseline for push loads (ADR-063). The engine (cuteafd) loaded models on
raptor (coordinator, 2× RTX PRO 6000) and six Sparks (expert ranks 0–5 =
ostrich, dodo, emu, kiwi, rhea, moa), with weights read through
`/mnt/sparknest` on sparknest 0.5.5. Every model has one copy per file,
spread over the Sparks; none on raptor.

**Method.** `tools/loadprobe-sample.sh` ran on raptor and every Spark,
recording once a second:
- NIC bytes sent and received (port counters ×4);
- NVMe bytes read;
- sparknestd CPU time.

`tools/loadprobe-report.py FROM TO LOGS…` gives per-host means, peaks and
totals over a window; `tools/loadprobe-series.py` gives per-second send
rates. Load windows and readiness times come from the engine's logs (UTC).
"Rank ready" is the mean over the six Spark ranks.

## Load order: GLM 5.3 EXL3 K4 (`wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1`)

Model: 55 files, 367 GiB; 64 GiB on each of five Sparks, 47 GiB on dodo.
Each rank requested 248.47 GB.

| Order | Rank ready | Read+upload | API ready | Per-rank rx | Disk/byte sent |
|---|---|---|---|---|---|
| In order | 142.8 s | 113.4 s | 176.0 s | ~2.3 GB/s | ~0.2 |
| Random per rank | 115.2 s | 85.0 s | 148.7 s | ~3.3 GB/s | ~0.4 |
| Interleaved by holder, all ranks in step | 149.7 s | 120.9 s | 184.0 s | ~2.0 GB/s | ~0.2 |
| Rotated by holder, rank r starts r ahead | 113.2 s | — | 148.8 s | 3.0–3.2 GB/s | ~0.27 |

- **In order:** one or two sources ran at line rate (11.7–12.4 GB/s) while
  the rest idled. A layer lives in one file, and all ranks walk the same
  layer.
- **Random and rotated:** every source sends at once at 2–9 GB/s, but the
  per-rank receive never passed ~5.6 GB/s in any second.
- **Disk:** low in every arm. Each Spark's NIC received 276–300 GB against
  248 GB requested (readahead overfetch).

## Reader change (cuteafd #566: handles kept open, reads coalesced)

| Order | Arm | Rank ready | Read+upload | NIC rx per Spark | Heavy-phase rx, cluster |
|---|---|---|---|---|---|
| In order | before | 150.8 s | 120.3 s | 271–300 GB | 12.8 GB/s |
| In order | after | 131.2 s | 100.1 s | 214–234 GB | 11.8 GB/s |
| Random | before | 123.5 s | 93.4 s | 275–299 GB | 16.1 GB/s |
| Random | after | 102.4 s | 73.5 s | 215–234 GB | 15.7 GB/s |

- **The gain is fewer bytes, not faster ones.** The overfetch went away
  (about 60 GB per rank); the receive rate is unchanged.
- **Where the limit is:** `nest io` on the Sparks shows fabric reads over
  128 KiB at p50 4.1 ms and p90 8.2 ms, against a serve time of p50
  1.0 ms. 16 readers × 1 MiB / 4 ms ≈ 4 GB/s, the per-rank ceiling.

## Kimi K3 EXL3 K2 (`wrldsuksgo2mars/Kimi-K3-EXL3-K2`), in order, before #566

- **The model:** 113 files, 798.8 GB, 129–137 GB per Spark.
- **What each rank needed:** 109.2 GB.
- **What it read:** 466.8 GB of pread (whole packed rows) per rank, and
  539–552 GB over each Spark's NIC.
- **Time:** read+upload took 273–318 s per rank, at a cluster rx of about
  10 GB/s (1.6 GB/s per rank) over the 334 s heavy phase.
- **Disk:** sources read 166–209 GB each for ~540–590 GB sent.
- **The coordinator** loaded concurrently and was not on the critical path.

## GLM 5.3 EXLR8, random order (cuteafd #570)

| Order | Rank ready | Heavy phase | rx per rank | rx rate per rank (mean/peak) | Disk per host |
|---|---|---|---|---|---|
| In order | 115.9 s | ~41 s | 51–52 GB | 1.0 / 1.6–2.2 GB/s | 57–62 GB |
| Random | 104.6 s | ~33 s | 42–51 GB | 1.0–1.2 / 1.8–2.2 GB/s | 40–55 GB |

- **No hot spot even in order:** 4–6 sources at once. The ~1 GB/s per rank
  is the EXLR8 reader's own limit.

## What the baseline says

- **Wasted bytes:** each rank moves 2–5× the bytes it keeps (EXL3 trellis
  rows interleave all ranks' columns).
- **Per-rank ceiling:** reading through FUSE caps each rank near 5 GB/s
  peak and 2–3 GB/s mean, with sources and disks mostly idle.
- **What push should reach:** each source reading its share once, about
  133 GB at 5–9 GB/s (Kimi) or about 64 GB (GLM), so 10–30 s instead of
  74–315 s.
