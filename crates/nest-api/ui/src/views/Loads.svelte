<script lang="ts">
  import { onMount } from "svelte";
  import { api } from "../lib/api";
  import { human, rate, ago, duration } from "../lib/format";

  // One push load as its hub recorded it (ADR-063): bins[t][s][r] = bytes
  // from source s to rank r in bin t.
  type Load = {
    group: number;
    hub: string;
    started_ms: number;
    elapsed_ms: number;
    outcome: "running" | "done" | "failed";
    reason?: string | null;
    sources: string[];
    ranks: string[];
    total_bytes: number;
    bin_ms: number;
    bins: number[][][];
    rdma_slots: number[];
    rpc_slots: number[];
    source_done_ms: (number | null)[];
    rank_done_ms: (number | null)[];
  };

  let loads = $state<Load[]>([]);
  let err = $state("");
  let sel = $state<number | null>(null);
  let full = $state<Load | null>(null);
  // Replay: the bin shown (null: the whole load).
  let at = $state<number | null>(null);
  let playing = $state(false);

  async function refresh() {
    try {
      loads = (await api<{ loads: Load[] }>("GET", "/v1/push/loads")).loads;
      err = "";
      if (sel == null && loads.length) sel = loads[0].group;
      if (sel != null) {
        const running = loads.find((l) => l.group === sel)?.outcome === "running";
        if (!full || full.group !== sel || running) full = await api<Load>("GET", `/v1/push/loads/${sel}`);
      }
    } catch (e) {
      err = String(e);
    }
  }
  onMount(() => {
    refresh();
    const t = setInterval(refresh, 1000);
    return () => clearInterval(t);
  });

  function pick(g: number) {
    sel = g;
    full = null;
    at = null;
    playing = false;
    refresh();
  }

  // Replay at 4× real time, a bin per frame step.
  $effect(() => {
    if (!playing || !full) return;
    const n = full.bins.length;
    const step = Math.max(16, full.bin_ms / 4);
    const t = setInterval(() => {
      const next = (at ?? -1) + 1;
      if (next >= n) {
        playing = false;
        at = null;
      } else at = next;
    }, step);
    return () => clearInterval(t);
  });

  const S = $derived(full?.sources.length ?? 0);
  const R = $derived(full?.ranks.length ?? 0);
  // Bytes per (source, rank), up to and including bin `at` (or all).
  const upto = $derived.by(() => {
    const m = Array.from({ length: S }, () => new Array(R).fill(0));
    if (!full) return m;
    const last = at ?? full.bins.length - 1;
    for (let t = 0; t <= last && t < full.bins.length; t++)
      for (let s = 0; s < S; s++) for (let r = 0; r < R; r++) m[s][r] += full.bins[t]?.[s]?.[r] ?? 0;
    return m;
  });
  // The bin's own bytes (for the "now" glow), or none in the overview.
  const binNow = $derived.by(() => (full && at != null ? full.bins[at] : null));
  const pairMax = $derived(Math.max(1, ...upto.flat()));
  const nowMax = $derived(binNow ? Math.max(1, ...binNow.flat()) : 1);
  const srcSum = $derived(upto.map((row) => row.reduce((a, b) => a + b, 0)));
  const rankSum = $derived(Array.from({ length: R }, (_, r) => upto.reduce((a, row) => a + (row[r] ?? 0), 0)));
  const moved = $derived(srcSum.reduce((a, b) => a + b, 0));
  const perRank = $derived(full && R ? full.total_bytes / R : 1);

  // Time series per source (sent) and per rank (received), bytes per bin.
  const sentT = $derived(full ? full.bins.map((b) => b.map((row) => row.reduce((a, x) => a + x, 0))) : []);
  const recvT = $derived(full ? full.bins.map((b) => Array.from({ length: R }, (_, r) => b.reduce((a, row) => a + (row[r] ?? 0), 0))) : []);
  const totalT = $derived(sentT.map((row) => row.reduce((a, b) => a + b, 0)));
  const peakBin = $derived(Math.max(1, ...totalT));
  const heatMax = $derived(Math.max(1, ...sentT.flat(), ...recvT.flat()));

  const secs = $derived(full ? Math.max(0.001, full.elapsed_ms / 1000) : 1);
  const binS = $derived(full ? full.bin_ms / 1000 : 0.25);
  const rdma = $derived(full ? full.rdma_slots.reduce((a, b) => a + b, 0) : 0);
  const rpc = $derived(full ? full.rpc_slots.reduce((a, b) => a + b, 0) : 0);
  const slowestSrc = $derived.by(() => {
    if (!full) return null;
    let i = -1, v = -1;
    full.source_done_ms.forEach((d, k) => { if ((d ?? Infinity) > v) { v = d ?? Infinity; i = k; } });
    return i >= 0 ? { name: full.sources[i], ms: v } : null;
  });
  const slowestRank = $derived.by(() => {
    if (!full) return null;
    let i = -1, v = -1;
    full.rank_done_ms.forEach((d, k) => { if ((d ?? Infinity) > v) { v = d ?? Infinity; i = k; } });
    return i >= 0 ? { name: `${full.ranks[i]} #${i}`, ms: v } : null;
  });

  // Flow graph geometry.
  const W = 760, PAD = 34, NODE_W = 130;
  const TOP = 50;
  const H = $derived(Math.max(220, 46 * Math.max(S, R) + TOP + 24));
  const ys = (n: number, i: number) => (n <= 1 ? (H + TOP) / 2 : TOP + (i * (H - TOP - 24)) / (n - 1));
  const hue = (s: number) => 186 + ((s * 47) % 120); // cyan → violet
  const ribbon = (s: number, r: number) => {
    const x1 = PAD + NODE_W, x2 = W - PAD - NODE_W;
    const y1 = ys(S, s), y2 = ys(R, r);
    const mx = (x1 + x2) / 2;
    return `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
  };
  const width = (b: number) => (b <= 0 ? 0 : 1.2 + 16 * Math.sqrt(b / pairMax));
  // A host's finish time, once the replay has passed it.
  const doneBy = (ms: number | null | undefined) =>
    ms != null && (at == null || (at + 1) * (full?.bin_ms ?? 250) >= ms) ? ` · ${duration(ms)}` : "";

  const heatColor = (b: number) => {
    const k = Math.min(1, b / heatMax);
    return `rgba(56, 232, 255, ${0.04 + 0.9 * k})`;
  };
  const gbps = (bytes: number, s: number) => (bytes / Math.max(0.001, s) / 1e9).toFixed(1) + " GB/s";
</script>

<div class="stack">
  <div class="panel pad">
    <div class="row">
      <h2>Loads</h2>
      <span class="small muted">push loads (ADR-063): every source disk streams its share once, each rank receives only its own bytes. The last 16 per hub, from every host.</span>
    </div>
  </div>

  {#if err}<div class="panel pad small" style="color: var(--bad)">{err}</div>{/if}

  {#if !loads.length}
    <div class="panel pad muted">No push loads yet. An engine loading weights by push, or <span class="mono">nest push bench</span>, shows up here.</div>
  {:else}
    <div class="list">
      {#each loads as l (l.group)}
        <button class="panel card" class:on={l.group === sel} onclick={() => pick(l.group)}>
          <div class="row">
            <span class="dot {l.outcome}"></span>
            <b class="mono">{human(l.total_bytes)}</b>
            <span class="muted small">{l.sources.length} → {l.ranks.length}</span>
            <span class="spacer"></span>
            <span class="mono small">{l.outcome === "running" ? "…" : gbps(l.total_bytes, l.elapsed_ms / 1000)}</span>
          </div>
          <div class="small muted">{duration(l.elapsed_ms)} · {ago(l.started_ms)} · hub {l.hub}</div>
        </button>
      {/each}
    </div>

    {#if full}
      <div class="tiles">
        <div class="tile panel glowline"><div class="caps">Loaded</div><div class="big">{human(at == null ? full.total_bytes : moved)}</div><div class="small muted">{full.ranks.length} ranks · {human(perRank)} each</div></div>
        <div class="tile panel"><div class="caps">Time</div><div class="big">{duration(full.elapsed_ms)}</div><div class="small muted">{full.outcome}{full.reason ? `: ${full.reason}` : ""}</div></div>
        <div class="tile panel hot"><div class="caps">Rate</div><div class="big">{gbps(full.total_bytes, secs)}</div><div class="small muted">peak {gbps(peakBin, binS)} · {gbps(perRank, secs)} per rank</div></div>
        <div class="tile panel"><div class="caps">Slowest</div><div class="big small2">{slowestSrc?.name ?? "—"}</div><div class="small muted">source {slowestSrc && isFinite(slowestSrc.ms) ? duration(slowestSrc.ms) : "…"} · rank {slowestRank?.name ?? ""} {slowestRank && isFinite(slowestRank.ms) ? duration(slowestRank.ms) : "…"}</div></div>
        <div class="tile panel" class:bad={rpc > rdma / 10}><div class="caps">By RDMA</div><div class="big">{rdma + rpc ? Math.round((100 * rdma) / (rdma + rpc)) : 0}%</div><div class="small muted">{rdma} slots by NIC writes, {rpc} by RPC</div></div>
      </div>

      <div class="panel pad">
        <div class="row">
          <h3 class="sec" style="margin:0">Who sent whom</h3>
          <span class="spacer"></span>
          <button class="btn" onclick={() => { if (playing) playing = false; else { at = -1; playing = true; } }}>{playing ? "Pause" : "Replay"}</button>
          <input type="range" min="-1" max={full.bins.length - 1} value={at ?? full.bins.length - 1}
            oninput={(e) => { playing = false; const v = +(e.target as HTMLInputElement).value; at = v < 0 ? null : v; }} />
          <span class="mono small t">{at == null ? "all" : `${((at + 1) * binS).toFixed(2)} s`}</span>
        </div>
        <svg viewBox="0 0 {W} {H}" class="flow">
          <defs>
            {#each full.sources as _, s}
              <linearGradient id="g{s}" x1="0" x2="1">
                <stop offset="0" stop-color="hsl({hue(s)} 100% 62%)" stop-opacity="0.85" />
                <stop offset="1" stop-color="hsl({hue(s)} 100% 72%)" stop-opacity="0.35" />
              </linearGradient>
            {/each}
            <filter id="glow"><feGaussianBlur stdDeviation="3" /></filter>
          </defs>
          {#each full.sources as _, s}
            {#each full.ranks as _, r}
              {#if upto[s][r] > 0}
                <path d={ribbon(s, r)} stroke="url(#g{s})" stroke-width={width(upto[s][r])} fill="none" stroke-linecap="round" opacity="0.75" />
                {#if binNow && (binNow[s]?.[r] ?? 0) > 0}
                  <path d={ribbon(s, r)} stroke="hsl({hue(s)} 100% 75%)" stroke-width={1 + 8 * Math.sqrt((binNow[s][r] ?? 0) / nowMax)} fill="none" filter="url(#glow)" class="pulse" />
                {/if}
              {/if}
            {/each}
          {/each}
          {#each full.sources as name, s}
            {@const y = ys(S, s)}
            {@const k = Math.min(1, srcSum[s] / Math.max(1, (full.total_bytes / Math.max(1, S))))}
            <g transform="translate({PAD} {y - 16})">
              <rect width={NODE_W} height="32" rx="9" class="node" />
              <rect width={NODE_W * k} height="32" rx="9" class="fill" style="fill: hsl({hue(s)} 100% 60% / 0.28)" />
              <text x="10" y="14" class="nm">{name}</text>
              <text x="10" y="27" class="nb">{human(srcSum[s])}{doneBy(full.source_done_ms[s])}</text>
            </g>
          {/each}
          {#each full.ranks as name, r}
            {@const y = ys(R, r)}
            {@const k = Math.min(1, rankSum[r] / Math.max(1, perRank))}
            <g transform="translate({W - PAD - NODE_W} {y - 16})">
              <rect width={NODE_W} height="32" rx="9" class="node" />
              <rect width={NODE_W * k} height="32" rx="9" class="fill" />
              <text x="10" y="14" class="nm">{name} <tspan class="faintt">#{r}</tspan></text>
              <text x="10" y="27" class="nb">{human(rankSum[r])}{doneBy(full.rank_done_ms[r])}</text>
            </g>
          {/each}
          <text x={PAD} y="16" class="axis">sources</text>
          <text x={W - PAD - NODE_W} y="16" class="axis">ranks</text>
        </svg>
      </div>

      <div class="panel pad">
        <h3 class="sec">Over time</h3>
        <div class="small muted">each row a host, each column {Math.round(full.bin_ms)} ms; brighter is faster (brightest: {gbps(heatMax, binS)} for one host)</div>
        <div class="heat" style="--n: {full.bins.length}">
          <div class="hl caps">sent</div><div></div>
          {#each full.sources as name, s}
            <div class="hl mono">{name}</div>
            <div class="hr">
              {#each sentT as row, t}
                <i style="background: {heatColor(row[s] ?? 0)}" class:cur={t === at} title="{name} {((t + 1) * binS).toFixed(2)} s: {gbps(row[s] ?? 0, binS)}"></i>
              {/each}
            </div>
          {/each}
          <div class="hl caps">received</div><div></div>
          {#each full.ranks as name, r}
            <div class="hl mono">{name} #{r}</div>
            <div class="hr">
              {#each recvT as row, t}
                <i style="background: {heatColor(row[r] ?? 0)}" class:cur={t === at} title="{name} #{r} {((t + 1) * binS).toFixed(2)} s: {gbps(row[r] ?? 0, binS)}"></i>
              {/each}
            </div>
          {/each}
          <div class="hl caps">total</div>
          <div class="hr bars">
            {#each totalT as b, t}
              <i style="height: {Math.max(2, 100 * (b / peakBin))}%" class:cur={t === at} title="{((t + 1) * binS).toFixed(2)} s: {gbps(b, binS)}"></i>
            {/each}
          </div>
        </div>
        <div class="row small muted"><span>0 s</span><span class="spacer"></span><span>{duration(full.elapsed_ms)}</span></div>
      </div>
    {/if}
  {/if}
</div>

<style>
  .list { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 10px; }
  .card { text-align: left; padding: 10px 12px; cursor: pointer; display: flex; flex-direction: column; gap: 4px; color: var(--text); }
  .card.on { border-color: rgba(56, 232, 255, 0.6); box-shadow: 0 0 26px rgba(56, 232, 255, 0.2); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--faint); }
  .dot.done { background: var(--ok); box-shadow: 0 0 8px var(--ok); }
  .dot.running { background: var(--spark); box-shadow: 0 0 10px var(--spark); animation: blink 1s infinite; }
  .dot.failed { background: var(--bad); }
  @keyframes blink { 50% { opacity: 0.35; } }
  .tiles { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }
  .tile { padding: 14px 16px; }
  .tile.hot { border-color: rgba(56, 232, 255, 0.5); box-shadow: 0 0 30px rgba(56, 232, 255, 0.2); }
  .tile.bad { border-color: rgba(255, 79, 123, 0.5); }
  .big { font-size: clamp(20px, 2.4vw, 28px); font-weight: 650; font-family: var(--mono); letter-spacing: -0.02em; background: linear-gradient(90deg, #f1fbff, #8fe9ff); -webkit-background-clip: text; background-clip: text; color: transparent; }
  .big.small2 { font-size: clamp(16px, 1.8vw, 22px); }
  .sec { margin-bottom: 10px; color: var(--muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.12em; font-size: 11px; }
  .t { min-width: 64px; text-align: right; }
  input[type="range"] { width: 220px; accent-color: var(--spark); }
  .flow { width: 100%; height: auto; margin-top: 8px; }
  .flow .node { fill: rgba(8, 18, 40, 0.85); stroke: rgba(110, 220, 255, 0.4); }
  .flow .fill { fill: rgba(56, 232, 255, 0.22); }
  .flow .nm { fill: var(--text); font: 600 12px var(--sans); }
  .flow .nb { fill: var(--muted); font: 10px var(--mono); }
  .flow .faintt { fill: var(--faint); font-weight: 400; }
  .flow .axis { fill: var(--faint); font: 10px var(--sans); text-transform: uppercase; letter-spacing: 0.14em; }
  .pulse { animation: pulse 0.5s ease-out; }
  @keyframes pulse { from { opacity: 1; } to { opacity: 0.55; } }
  .heat { display: grid; grid-template-columns: 110px 1fr; gap: 3px 10px; margin-top: 10px; align-items: center; }
  .hl { font-size: 11px; color: var(--muted); text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .hl.caps { text-align: left; margin-top: 6px; }
  .hr { display: grid; grid-template-columns: repeat(var(--n), 1fr); gap: 1px; height: 14px; }
  .hr i { display: block; border-radius: 2px; min-width: 1px; }
  .hr i.cur { outline: 1px solid #fff; }
  .hr.bars { height: 56px; align-items: end; }
  .hr.bars i { background: linear-gradient(180deg, var(--spark), var(--spark-2)); box-shadow: 0 0 6px rgba(56, 232, 255, 0.4); }
  @media (max-width: 899px) {
    .tiles { grid-template-columns: 1fr 1fr; }
    .heat { grid-template-columns: 80px 1fr; }
  }
</style>
