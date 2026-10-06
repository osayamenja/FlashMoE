#!/usr/bin/env python3
# Copyright (c) 2026, Osayamen Jonathan Aimuyo.
"""Summarizes the trace of one moe::forward launch written by testFlashMoE* built with -DFLASHMOE_TRACE=ON.

Usage:
    FLASHMOE_TRACE_DIR=/tmp/tr ./testFlashMoEMXFP4 4096 2048 768 128 8 4 4 64     # writes /tmp/tr/trace_<tag>.{bin,json}
    python3 trace_summary.py /tmp/tr/trace_mxfp4_S4096_H2048_I768_E128_k8.json [--chrome out.json] [--csv]

The record format and the events are documented in include/flashmoe/infra/trace.cuh. All times are in microseconds
after the first block started; the resolution of %globaltimer on sm_120 is 32 ns. Only the standard library is needed.
"""
import argparse
import json
import struct
import sys

# trace::Event
BLOCK_BEGIN, BLOCK_END, PROC_WAIT_BEGIN, PROC_DOORBELL, PROC_TASK_LOADED, PROC_MAIN_DONE, PROC_TASK_DONE, \
  SCHED_BATCH, SCHED_SWEEP, SCHED_SCAN, SUB_PACKET, SUB_COMBINE, DISPATCH_BEGIN, DISPATCH_END = range(1, 15)
TYPE_NAMES = ("GEMM0", "GEMM1", "combine")
MAGIC = b"FMTR0001"


def pct(xs, q):
  """q-th percentile (linear interpolation) of the sorted list xs."""
  if not xs:
    return float("nan")
  k = (len(xs) - 1) * q / 100.0
  f = int(k)
  c = min(f + 1, len(xs) - 1)
  return xs[f] + (xs[c] - xs[f]) * (k - f)


def stats(xs):
  xs = sorted(xs)
  return {"n": len(xs), "mean": (sum(xs) / len(xs)) if xs else float("nan"), "p50": pct(xs, 50), "p95": pct(xs, 95),
          "max": xs[-1] if xs else float("nan")}


def load(path):
  base = path
  for ext in (".json", ".bin"):
    if base.endswith(ext):
      base = base[:-len(ext)]
  with open(base + ".json") as f:
    meta = json.load(f)
  with open(base + ".bin", "rb") as f:
    data = f.read()
  magic, blocks, rec_bytes, cap, os_cap, _, _ = struct.unpack_from("<8sIIIIII", data, 0)
  if magic != MAGIC or rec_bytes != 16:
    sys.exit(f"{base}.bin: not a FlashMoE trace file")
  off = 32
  segs = []
  dropped = 0
  for _ in range(blocks):
    count, drop = struct.unpack_from("<II", data, off)
    off += 8
    # record = (t_ns, event, payload a (24 bit), b)
    segs.append([(t, a >> 24, a & 0xFFFFFF, b) for t, a, b in struct.iter_unpack("<QII", data[off:off + count * 16])])
    off += count * 16
    dropped += drop
  return meta, segs, dropped


def analyze(meta, segs, dropped):
  nblk = len(segs)
  osb = nblk - 1
  begins = [r[0] for s in segs for r in s if r[1] == BLOCK_BEGIN]
  ends = [r[0] for s in segs for r in s if r[1] == BLOCK_END]
  if not begins or not ends:
    sys.exit("trace has no block begin / end events (kernel did not finish?)")
  t0 = min(begins)
  def us(t):
    return (t - t0) / 1000.0
  span = us(max(ends))
  A = {"meta": meta, "span": span, "dropped": dropped, "blocks": nblk, "processors": osb}
  A["launchSkew"] = us(max(begins))

  # processors (all blocks but the last), including the dispatch phase of the first blocks
  tasks = []  # dicts, times in us
  procs = []
  for b in range(osb):
    info = {"block": b, "begin": None, "end": None, "dispatch": [], "tasks": [], "finalWait": 0.0}
    wb = db = slot = None
    cur = None
    openD = {}
    for t, ev, p, x in segs[b]:
      if ev == BLOCK_BEGIN:
        info["begin"] = us(t)
      elif ev == BLOCK_END:
        info["end"] = us(t)
      elif ev == DISPATCH_BEGIN:
        openD[p] = (us(t), x)
      elif ev == DISPATCH_END:
        if p in openD:
          tb, routed = openD.pop(p)
          info["dispatch"].append((p, tb, us(t), routed))
      elif ev == PROC_WAIT_BEGIN:
        wb = us(t)
      elif ev == PROC_DOORBELL:
        if p & 1:  # interrupt
          if wb is not None:
            info["finalWait"] = us(t) - wb
          info["interruptAt"] = us(t)
        else:
          db, slot = us(t), x
      elif ev == PROC_TASK_LOADED:
        cur = {"block": b, "type": p >> 16, "expert": p & 0xFFFF, "tile": x, "wait": wb, "doorbell": db, "slot": slot,
               "loaded": us(t), "main": None}
      elif ev == PROC_MAIN_DONE:
        if cur is not None:
          cur["main"] = us(t)
      elif ev == PROC_TASK_DONE:
        if cur is not None:
          cur["done"] = us(t)
          cur["tileSize"] = p & 0xFFFF
          info["tasks"].append(cur)
          tasks.append(cur)
          cur = None
    procs.append(info)
  A["procs"] = procs
  A["tasks"] = tasks

  # OS block
  assigns, batches, sweeps, scans, packets, combines = [], [], [], [], [], []
  stride = meta["threads"] - 32  # number of subscriber threads: slot stride of the tasks of a strided batch
  for t, ev, p, x in segs[osb]:
    if ev == SCHED_BATCH:
      # one schedule() call: count doorbells, the l-th one for the task queue slot qIdx + l * stride (strided) or qIdx + l.
      # The doorbells are issued back to back, one release fence each, so the l-th one is dated (l + 1) / count of the way
      # through the call.
      t1, dur, count, qIdx = us(t), (x & 0xFFFFFF) * 32 / 1000.0, x >> 24, p & 0x7FFFFF
      step = stride if p >> 23 else 1
      batches.append((t1, dur, count, qIdx, step))
      for l in range(count):
        assigns.append((t1 - dur + (l + 1) * dur / count, qIdx + l * step))  # (t, slot)
    elif ev == SCHED_SWEEP:
      sweeps.append((us(t), x))  # (t, tasks scheduled so far)
    elif ev == SCHED_SCAN:
      scans.append((us(t), p, x))  # (t, ready processors, waiting tasks)
    elif ev == SUB_PACKET:
      packets.append((us(t), p >> 12, p & 0xFFF, x))
    elif ev == SUB_COMBINE:
      combines.append((us(t), p >> 12, p & 0xFFF, x))
  for lst in (assigns, sweeps, scans, packets, combines):
    lst.sort(key=lambda r: r[0])
  A.update(assigns=assigns, batches=batches, sweepEvents=sweeps, scanEvents=scans, packets=packets, combines=combines)

  # per task type
  byType = {}
  for k, name in enumerate(TYPE_NAMES):
    ts = [t for t in tasks if t["type"] == k]
    dur = [t["done"] - t["loaded"] for t in ts]
    mainShare = [(t["main"] - t["loaded"]) for t in ts if t["main"] is not None]
    entry = {"n": len(ts), "dur": stats(dur), "sum": sum(dur)}
    if ts:
      entry["firstStart"] = min(t["loaded"] for t in ts)
      entry["lastDone"] = max(t["done"] for t in ts)
    if mainShare:
      mdur = [t["done"] - t["loaded"] for t in ts if t["main"] is not None]
      entry["mainloop"] = stats(mainShare)
      entry["mainShare"] = sum(mainShare) / sum(mdur) if sum(mdur) > 0 else float("nan")
      entry["epilogue"] = stats([t["done"] - t["main"] for t in ts if t["main"] is not None])
    entry["fetch"] = stats([t["loaded"] - t["doorbell"] for t in ts if t["doorbell"] is not None])
    byType[name] = entry
  A["byType"] = byType

  # start-up chain
  chain = {}
  de = [d[2] for p in procs for d in p["dispatch"]]
  if de:
    chain["firstDispatchEnd"] = min(de)
    chain["lastDispatchEnd"] = max(de)
  if packets:
    chain["firstPacket"] = packets[0][0]
    chain["lastPacket"] = packets[-1][0]
  if assigns:
    chain["firstAssign"] = assigns[0][0]
  dbs = [t["doorbell"] for t in tasks if t["doorbell"] is not None]
  if dbs:
    chain["firstDoorbell"] = min(dbs)
  for name in TYPE_NAMES:
    if "firstStart" in byType[name]:
      chain["first" + name] = byType[name]["firstStart"]
  if combines:
    chain["firstCombineGenerated"] = combines[0][0]
  A["chain"] = chain

  # processor time accounting over the lifetime of each processor block
  acc = {"dispatch": 0.0, "busy": 0.0, "fetch": 0.0, "wait": 0.0, "finalWait": 0.0, "other": 0.0, "life": 0.0}
  waits = []
  for p in procs:
    if p["begin"] is None or p["end"] is None:
      continue
    life = p["end"] - p["begin"]
    d = sum(e - s for _, s, e, _ in p["dispatch"])
    busy = sum(t["done"] - t["loaded"] for t in p["tasks"])
    fetch = sum(t["loaded"] - t["doorbell"] for t in p["tasks"] if t["doorbell"] is not None)
    wait = sum(t["doorbell"] - t["wait"] for t in p["tasks"] if t["doorbell"] is not None and t["wait"] is not None)
    waits += [t["doorbell"] - t["wait"] for t in p["tasks"] if t["doorbell"] is not None and t["wait"] is not None]
    acc["life"] += life
    acc["dispatch"] += d
    acc["busy"] += busy
    acc["fetch"] += fetch
    acc["wait"] += wait
    acc["finalWait"] += p["finalWait"]
    acc["other"] += life - d - busy - fetch - wait - p["finalWait"]
  A["acc"] = acc
  A["waits"] = stats(waits)
  # a doorbell that is already there is seen by the first poll, which takes one L2 round trip (about 0.2 - 0.5 us)
  A["waitZero"] = (sum(1 for w in waits if w <= 0.5) / len(waits)) if waits else float("nan")
  A["activeProcs"] = sum(1 for p in procs if p["tasks"])
  A["tasksPerProc"] = stats([len(p["tasks"]) for p in procs])

  # dispatch
  dd = [e - s for p in procs for _, s, e, _ in p["dispatch"]]
  A["dispatch"] = {"perExpert": stats(dd), "blocks": sum(1 for p in procs if p["dispatch"])}

  # assignment latency: scheduler doorbell (dated by interpolation inside its schedule() call, i.e., approximately the
  # release fence + store) -> doorbell observed by the processor, joined on the task queue slot
  bySlot = {slot: t for t, slot in assigns}
  raw, eff, missing = [], [], 0
  for t in tasks:
    if t["doorbell"] is None:
      continue
    ta = bySlot.get(t["slot"])
    if ta is None:
      missing += 1
      continue
    t["assign"] = ta
    raw.append(t["doorbell"] - ta)
    # when the processor was still busy the doorbell was already there: measure from the moment it started to poll
    eff.append(t["doorbell"] - max(ta, t["wait"] if t["wait"] is not None else ta))
  A["assignRaw"] = stats(raw)
  A["assignEff"] = stats(eff)
  A["assignMissing"] = missing
  # cost of one doorbell inside schedule(): duration of the call / number of doorbells (dominated by the release fence)
  # The lanes of the scheduler warp run their schedule() calls in lockstep, so the calls overlap in time: the time the
  # warp spends ringing doorbells is the union of the call intervals, and the period of one doorbell is dur / count.
  union, end = 0.0, -1.0
  for t1, dur, _, _, _ in sorted(batches, key=lambda b: b[0] - b[1]):
    s0 = t1 - dur
    union += max(0.0, t1 - max(s0, end))
    end = max(end, t1)
  A["batchStats"] = {"n": len(batches), "size": stats([b[2] for b in batches]), "union": union,
                     "perDoorbellNs": stats([1000.0 * b[1] / b[2] for b in batches if b[2] > 0])}
  # scheduler response: processor says "ready" when it observes its previous doorbell -> next assignment of that pid
  r2a = []
  for p in procs:
    ts = sorted(p["tasks"], key=lambda t: t["loaded"])
    for a, b in zip(ts, ts[1:]):
      if a["doorbell"] is not None and "assign" in b:
        r2a.append(b["assign"] - a["doorbell"])
  A["readyToAssign"] = stats(r2a)

  # scheduler
  swd = [b[0] - a[0] for a, b in zip(sweeps, sweeps[1:])]
  tps = [b[1] - a[1] for a, b in zip(sweeps, sweeps[1:])]
  A["sweeps"] = {"n": len(sweeps), "dur": stats(swd), "tasksPerSweep": stats(tps),
                 "idleShare": (sum(1 for x in tps if x == 0) / len(tps)) if tps else float("nan")}
  A["scans"] = {"n": len(scans), "noReady": (sum(1 for s in scans if s[1] == 0) / len(scans)) if scans else float("nan"),
                "interval": stats([b[0] - a[0] for a, b in zip(scans, scans[1:])]),
                "ready": stats([s[1] for s in scans]), "waitingTasks": stats([s[2] for s in scans])}

  # throughput over time: 10 bins
  bins = []
  P = max(osb, 1)
  nb = 10
  for i in range(nb):
    lo, hi = span * i / nb, span * (i + 1) / nb
    width = hi - lo
    done = sum(1 for t in tasks if lo <= t["done"] < hi)
    asg = sum(1 for a in assigns if lo <= a[0] < hi)
    busy = sum(max(0.0, min(t["done"], hi) - max(t["loaded"], lo)) for t in tasks)
    bins.append({"lo": lo, "hi": hi, "done": done / width, "assigned": asg / width, "busyProcs": busy / width})
  A["bins"] = bins
  total = len(tasks)
  A["tasksPerUs"] = total / span if span > 0 else float("nan")
  A["lowerBound"] = sum(t["done"] - t["loaded"] for t in tasks) / P  # span if the processors never idled
  return A


def fmt(x, nd=2):
  return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def report(A):
  m = A["meta"]
  P = A["processors"]
  span = A["span"]
  print(f"== {m['tag']}: {m['recipe']} S={m['S']} H={m['H']} I={m['I']} E={m['E']} k={m['k']} "
        f"tile {m['bM']}x{m['bN0']}x{m['bK0']} / {m['bM']}x{m['bN1']}x{m['bK1']}, tiles N0={m['tilesN0']} N1={m['tilesN1']}, "
        f"{A['blocks']} blocks ({P} processors + 1 OS)")
  print(f"kernel span {fmt(span)} us (traced launch {fmt(m['tracedLaunchMs'] * 1000, 1)} us incl. launch overhead, "
        f"untraced timed loop {fmt(m['timedMs'] * 1000, 1)} us); block launch skew {fmt(A['launchSkew'])} us; "
        f"dropped records {A['dropped']}")
  tasks = A["tasks"]
  n = {k: A["byType"][k]["n"] for k in TYPE_NAMES}
  print(f"tasks executed {len(tasks)} (GEMM0 {n['GEMM0']}, GEMM1 {n['GEMM1']}, combine {n['combine']}), "
        f"{fmt(A['tasksPerUs'])} tasks/us over the span; work-conserving lower bound {fmt(A['lowerBound'])} us "
        f"(sum of task durations / processors), i.e. {fmt(100 * A['lowerBound'] / span, 1)}% of the span")
  print()
  print("start-up chain (us after kernel start):")
  print("  " + ", ".join(f"{k} {fmt(v)}" for k, v in A["chain"].items()))
  print()
  print("per task type (loaded -> done; mainloop = loaded -> main loop returned; fetch = doorbell -> loaded):")
  print(f"  {'type':8s} {'count':>6s} {'first':>9s} {'last':>9s} {'p50':>7s} {'p95':>7s} {'mean':>7s} {'sum us':>10s} "
        f"{'mainloop%':>9s} {'epi p50':>8s} {'fetch p50':>9s}")
  for k in TYPE_NAMES:
    e = A["byType"][k]
    if not e["n"]:
      continue
    print(f"  {k:8s} {e['n']:6d} {fmt(e.get('firstStart')):>9s} {fmt(e.get('lastDone')):>9s} {fmt(e['dur']['p50']):>7s} "
          f"{fmt(e['dur']['p95']):>7s} {fmt(e['dur']['mean']):>7s} {fmt(e['sum'], 1):>10s} "
          f"{fmt(100 * e['mainShare'], 1) if 'mainShare' in e else 'n/a':>9s} "
          f"{fmt(e['epilogue']['p50']) if 'epilogue' in e else 'n/a':>8s} {fmt(e['fetch']['p50']):>9s}")
  print()
  a = A["acc"]
  life = a["life"] if a["life"] > 0 else 1.0
  print(f"processor time over their lifetime ({P} processors, {A['activeProcs']} ran tasks, "
        f"tasks per processor p50 {fmt(A['tasksPerProc']['p50'], 0)} max {fmt(A['tasksPerProc']['max'], 0)}):")
  print(f"  busy (task loaded -> done) {fmt(100 * a['busy'] / life, 1)}%, dispatch {fmt(100 * a['dispatch'] / life, 1)}%, "
        f"task fetch (doorbell -> loaded) {fmt(100 * a['fetch'] / life, 1)}%, idle wait for a doorbell "
        f"{fmt(100 * a['wait'] / life, 1)}% + final wait {fmt(100 * a['finalWait'] / life, 1)}%, other {fmt(100 * a['other'] / life, 1)}%")
  w = A["waits"]
  print(f"  doorbell wait per task: p50 {fmt(w['p50'])} us, p95 {fmt(w['p95'])} us, mean {fmt(w['mean'])} us; "
        f"{fmt(100 * A['waitZero'], 1)}% of the tasks were already pending (wait <= 0.5 us)")
  dsp = A["dispatch"]
  print(f"  dispatch: {dsp['blocks']} blocks, per expert and block p50 {fmt(dsp['perExpert']['p50'])} us "
        f"p95 {fmt(dsp['perExpert']['p95'])} us")
  print()
  r, e = A["assignRaw"], A["assignEff"]
  print(f"assignment latency (scheduler doorbell, dated by interpolation inside its schedule() call -> processor sees "
        f"the doorbell), {r['n']} tasks matched, {A['assignMissing']} unmatched:")
  print(f"  raw p50 {fmt(r['p50'])} p95 {fmt(r['p95'])} max {fmt(r['max'])} us; "
        f"from max(assign, poll start) p50 {fmt(e['p50'])} p95 {fmt(e['p95'])} max {fmt(e['max'])} us")
  q = A["readyToAssign"]
  print(f"  processor ready -> next assignment of that processor: p50 {fmt(q['p50'])} p95 {fmt(q['p95'])} "
        f"max {fmt(q['max'])} us (task duration p50 for comparison: "
        + "/".join(fmt(A['byType'][k]['dur']['p50']) for k in TYPE_NAMES if A['byType'][k]['n']) + " us)")
  s, sc = A["sweeps"], A["scans"]
  bs = A["batchStats"]
  print(f"scheduler: {len(A['assigns'])} doorbells in {bs['n']} schedule() calls (size p50 {fmt(bs['size']['p50'], 0)} "
        f"max {fmt(bs['size']['max'], 0)}); the warp is inside schedule() for {fmt(bs['union'], 1)} us = "
        f"{fmt(100 * bs['union'] / span, 1)}% of the span; one doorbell takes {fmt(bs['perDoorbellNs']['p50'], 0)} ns "
        f"p50 (p95 {fmt(bs['perDoorbellNs']['p95'], 0)} ns) within a call (the lanes' calls overlap)")
  print(f"  {s['n']} queue sweeps, duration p50 {fmt(s['dur']['p50'])} "
        f"p95 {fmt(s['dur']['p95'])} us, {fmt(100 * s['idleShare'], 1)}% of the sweeps found no task; "
        f"{sc['n']} readiness scans (only while tasks are waiting), {fmt(100 * sc['noReady'], 1)}% found no ready "
        f"processor, ready processors found p50 {fmt(sc['ready']['p50'], 0)}, waiting tasks p50 "
        f"{fmt(sc['waitingTasks']['p50'], 0)} (interval p50 {fmt(sc['interval']['p50'])} us)")
  print(f"subscriber: {len(A['packets'])} packets decoded (first {fmt(A['packets'][0][0]) if A['packets'] else 'n/a'}, "
        f"last {fmt(A['packets'][-1][0]) if A['packets'] else 'n/a'} us), {len(A['combines'])} combine tasks generated")
  print()
  print("OS throughput in 10 time bins (tasks/us):")
  print(f"  {'bin':>3s} {'from us':>9s} {'to us':>9s} {'assigned':>9s} {'done':>7s} {'busy procs':>11s}")
  for i, b in enumerate(A["bins"]):
    print(f"  {i:3d} {b['lo']:9.1f} {b['hi']:9.1f} {b['assigned']:9.2f} {b['done']:7.2f} {b['busyProcs']:11.1f}")


CSV_COLUMNS = [
  ("tag", lambda A: A["meta"]["tag"]),
  ("span_us", lambda A: A["span"]),
  ("traced_launch_us", lambda A: A["meta"]["tracedLaunchMs"] * 1000),
  ("timed_us", lambda A: A["meta"]["timedMs"] * 1000),
  ("launch_skew_us", lambda A: A["launchSkew"]),
  ("first_dispatch_end_us", lambda A: A["chain"].get("firstDispatchEnd")),
  ("last_dispatch_end_us", lambda A: A["chain"].get("lastDispatchEnd")),
  ("first_packet_us", lambda A: A["chain"].get("firstPacket")),
  ("first_assign_us", lambda A: A["chain"].get("firstAssign")),
  ("first_gemm0_us", lambda A: A["chain"].get("firstGEMM0")),
  ("first_gemm1_us", lambda A: A["chain"].get("firstGEMM1")),
  ("first_combine_us", lambda A: A["chain"].get("firstcombine")),
  ("last_gemm0_done_us", lambda A: A["byType"]["GEMM0"].get("lastDone")),
  ("n_gemm0", lambda A: A["byType"]["GEMM0"]["n"]),
  ("n_gemm1", lambda A: A["byType"]["GEMM1"]["n"]),
  ("n_combine", lambda A: A["byType"]["combine"]["n"]),
  ("gemm0_p50_us", lambda A: A["byType"]["GEMM0"]["dur"]["p50"]),
  ("gemm0_p95_us", lambda A: A["byType"]["GEMM0"]["dur"]["p95"]),
  ("gemm0_main_pct", lambda A: 100 * A["byType"]["GEMM0"].get("mainShare", float("nan"))),
  ("gemm1_p50_us", lambda A: A["byType"]["GEMM1"]["dur"]["p50"]),
  ("gemm1_p95_us", lambda A: A["byType"]["GEMM1"]["dur"]["p95"]),
  ("gemm1_main_pct", lambda A: 100 * A["byType"]["GEMM1"].get("mainShare", float("nan"))),
  ("combine_p50_us", lambda A: A["byType"]["combine"]["dur"]["p50"]),
  ("combine_p95_us", lambda A: A["byType"]["combine"]["dur"]["p95"]),
  ("busy_pct", lambda A: 100 * A["acc"]["busy"] / max(A["acc"]["life"], 1e-9)),
  ("dispatch_pct", lambda A: 100 * A["acc"]["dispatch"] / max(A["acc"]["life"], 1e-9)),
  ("fetch_pct", lambda A: 100 * A["acc"]["fetch"] / max(A["acc"]["life"], 1e-9)),
  ("wait_pct", lambda A: 100 * (A["acc"]["wait"] + A["acc"]["finalWait"]) / max(A["acc"]["life"], 1e-9)),
  ("other_pct", lambda A: 100 * A["acc"]["other"] / max(A["acc"]["life"], 1e-9)),
  ("doorbell_wait_p50_us", lambda A: A["waits"]["p50"]),
  ("pending_pct", lambda A: 100 * A["waitZero"]),
  ("tasks_per_us", lambda A: A["tasksPerUs"]),
  ("lower_bound_us", lambda A: A["lowerBound"]),
  ("assign_lat_p50_us", lambda A: A["assignRaw"]["p50"]),
  ("assign_lat_p95_us", lambda A: A["assignRaw"]["p95"]),
  ("assign_eff_p50_us", lambda A: A["assignEff"]["p50"]),
  ("ready_to_assign_p50_us", lambda A: A["readyToAssign"]["p50"]),
  ("ready_to_assign_p95_us", lambda A: A["readyToAssign"]["p95"]),
  ("doorbell_ns_p50", lambda A: A["batchStats"]["perDoorbellNs"]["p50"]),
  ("schedule_calls", lambda A: A["batchStats"]["n"]),
  ("sched_in_schedule_pct", lambda A: 100 * A["batchStats"]["union"] / A["span"]),
  ("scan_ready_p50", lambda A: A["scans"]["ready"]["p50"]),
  ("scan_waiting_p50", lambda A: A["scans"]["waitingTasks"]["p50"]),
  ("sweeps", lambda A: A["sweeps"]["n"]),
  ("sweep_p50_us", lambda A: A["sweeps"]["dur"]["p50"]),
  ("scans", lambda A: A["scans"]["n"]),
  ("scan_noready_pct", lambda A: 100 * A["scans"]["noReady"]),
  ("dropped", lambda A: A["dropped"]),
]


def csv_rows(analyses):
  print(",".join(c for c, _ in CSV_COLUMNS))
  for A in analyses:
    row = []
    for _, f in CSV_COLUMNS:
      v = f(A)
      row.append(v if isinstance(v, str) else ("" if v is None or v != v else f"{v:.3f}" if isinstance(v, float) else str(v)))
    print(",".join(row))


def chrome(A, path):
  """Chrome trace event format, opens in https://ui.perfetto.dev (1 process per role, 1 thread per CTA)."""
  names = {1: "processors (+ dispatch)", 2: "OS block"}
  ev = [{"ph": "M", "name": "process_name", "pid": pid, "args": {"name": n}} for pid, n in names.items()]
  ev += [{"ph": "M", "name": "thread_name", "pid": 2, "tid": 0, "args": {"name": "scheduler"}},
         {"ph": "M", "name": "thread_name", "pid": 2, "tid": 1, "args": {"name": "subscriber"}}]
  for p in A["procs"]:
    b = p["block"]
    for expert, s, e, routed in p["dispatch"]:
      ev.append({"ph": "X", "pid": 1, "tid": b, "ts": s, "dur": e - s, "name": f"dispatch e{expert}", "cat": "dispatch",
                 "args": {"routedTokens": routed}})
    for t in p["tasks"]:
      ev.append({"ph": "X", "pid": 1, "tid": b, "ts": t["loaded"], "dur": t["done"] - t["loaded"],
                 "name": f"{TYPE_NAMES[t['type']]} e{t['expert']} t{t['tile']}", "cat": TYPE_NAMES[t["type"]],
                 "args": {"tileSize": t["tileSize"], "slot": t["slot"],
                          "mainloop_us": None if t["main"] is None else t["main"] - t["loaded"]}})
      if t["doorbell"] is not None and t["wait"] is not None and t["doorbell"] - t["wait"] > 0.1:
        ev.append({"ph": "X", "pid": 1, "tid": b, "ts": t["wait"], "dur": t["doorbell"] - t["wait"], "name": "wait",
                   "cat": "wait"})
  for t1, dur, count, qIdx, step in A["batches"]:
    ev.append({"ph": "X", "pid": 2, "tid": 0, "ts": t1 - dur, "dur": dur, "name": f"schedule x{count}", "cat": "doorbells",
               "args": {"firstSlot": qIdx, "slotStride": step}})
  for a, b in zip(A["sweepEvents"], A["sweepEvents"][1:]):
    ev.append({"ph": "X", "pid": 2, "tid": 0, "ts": a[0], "dur": b[0] - a[0], "name": "sweep",
               "args": {"tasks": b[1] - a[1]}})
  for t, ready, waiting in A["scanEvents"]:
    ev.append({"ph": "i", "s": "t", "pid": 2, "tid": 0, "ts": t, "name": "scan", "args": {"ready": ready, "waiting": waiting}})
  for t, peer, ex, tokens in A["packets"]:
    ev.append({"ph": "i", "s": "t", "pid": 2, "tid": 1, "ts": t, "name": f"packet e{ex}", "args": {"tokens": tokens}})
  for t, tile, ex, start in A["combines"]:
    ev.append({"ph": "i", "s": "t", "pid": 2, "tid": 1, "ts": t, "name": f"combine e{ex}", "args": {"tile": tile}})
  with open(path, "w") as f:
    json.dump({"traceEvents": ev, "displayTimeUnit": "ns"}, f)
  print(f"wrote {path}: {len(ev)} events", file=sys.stderr)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("traces", nargs="+", help="trace_<tag>.json (or .bin, or the path without extension)")
  ap.add_argument("--chrome", metavar="OUT.json", help="also write a Chrome / Perfetto timeline (first trace only)")
  ap.add_argument("--csv", action="store_true", help="print one CSV row per trace instead of the report")
  args = ap.parse_args()
  analyses = [analyze(*load(p)) for p in args.traces]
  if args.csv:
    csv_rows(analyses)
  else:
    for i, A in enumerate(analyses):
      if i:
        print()
      report(A)
  if args.chrome:
    chrome(analyses[0], args.chrome)


if __name__ == "__main__":
  main()
