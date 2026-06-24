"""Realistic-fills re-search: ORB/IB combo search with breach_confirm_prior=True.
argv: instrument open_t close_t entry_end split_date  ->  best low-corr 3-leg combo."""
import sys, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import build_facts, data_store, ib50

INST = sys.argv[1]; OPEN = int(sys.argv[2]); CLOSE = int(sys.argv[3])
ENTEND = int(sys.argv[4]); SPLIT = pd.Timestamp(sys.argv[5])
PV = 100.0 if INST == "XAUUSD" else 20.0

def cfg(ib, uc, uv, ub, bm, e, s, t):
    return dict(ib_min=ib, open_t=OPEN, close_t=CLOSE, use_formation=False, use_closeloc=uc,
        close_loc_pct=50.0, use_vwap=uv, logic="AND", use_breach=ub, breach_mode=bm,
        breach_scope="Directional", entry_pct=float(e), sl_pct=float(s), target_pct=float(t),
        risk_usd=1500.0, point_value=PV, use_entry_window=True, entry_start_t=OPEN,
        entry_end_t=ENTEND, use_exit_time=False, exit_t=CLOSE, eff_min=CLOSE,
        allow_reentry=False, breach_confirm_prior=True)          # <-- realistic fills

IBS = [15, 30, 60]
FILT = [(1, 0), (0, 1), (1, 1)]
BREACH = [(False, "Require Breached"), (True, "Require Breached"), (True, "Require Not-Breached")]
GEOM = [(25, s, t) for s in (75, 100) for t in (75, 100, 150)]
cfgs = [cfg(ib, bool(c), bool(v), ub, bm, e, s, t)
        for ib in IBS for (c, v) in FILT for (ub, bm) in BREACH for (e, s, t) in GEOM]

df = build_facts.clean_min(data_store.read_minute(data_store.discover()[INST]))
preps = {ib: ib50.prep_days(df, OPEN, CLOSE, ib) for ib in IBS}
master = pd.to_datetime(pd.Series(sorted(df["date_only"].unique())))
nD = len(master); pos = {d.normalize(): i for i, d in enumerate(master)}
trM = (master < SPLIT).to_numpy(); hoM = ~trM
mon = master.dt.to_period("M").to_numpy()
def st(mask): idx = mon[mask]; return np.r_[0, np.where(idx[1:] != idx[:-1])[0] + 1]
trS, hoS, allS = st(trM), st(hoM), st(np.ones(nD, bool))
trMonN, hoMonN = len(trS), len(hoS)

Dr = np.zeros((nD, len(cfgs))); Dc = np.zeros((nD, len(cfgs)))
for j, c in enumerate(cfgs):
    tr = ib50.run(preps[c["ib_min"]], c)
    if not tr.empty:
        g = tr.groupby(tr["date"].dt.normalize())
        for d, v in g["r"].sum().items():
            pp = pos.get(pd.Timestamp(d).normalize())
            if pp is not None: Dr[pp, j] = v; Dc[pp, j] = g.size().loc[d]
Mr = np.add.reduceat(Dr, allS, axis=0)
def seg(col, mask, S, nMon):
    d = col[mask]; cum = np.cumsum(d)
    dd = float((cum - np.maximum.accumulate(cum)).min()) if len(d) else 0.0
    mr = np.add.reduceat(d, S) if len(d) else np.zeros(0)
    return d.sum() / max(nMon, 1), dd, (float((mr > 0).mean()*100) if nMon else 0.0)

trW = trM.sum() / 5.0
pool = []
for j in range(len(cfgs)):
    if Dc[trM, j].sum() / trW < 2.0: continue
    rpm, dd, grn = seg(Dr[:, j], trM, trS, trMonN)
    if rpm <= 0: continue
    pool.append((rpm / max(-dd, 0.5), j, rpm, dd))
pool.sort(reverse=True); pool = pool[:30]; pidx = [p[1] for p in pool]
def tag(c):
    fl = ("C" if c["use_closeloc"] else "-") + ("V" if c["use_vwap"] else "-")
    br = "noBr" if not c["use_breach"] else ("Bre" if c["breach_mode"]=="Require Breached" else "NotBr")
    return f"ib{c['ib_min']}|{fl}|{br}|e{int(c['entry_pct'])}s{int(c['sl_pct'])}t{int(c['target_pct'])}"

print(f"=== {INST}  {OPEN//60:02d}:{OPEN%60:02d}-{CLOSE//60:02d}:{CLOSE%60:02d}  REALISTIC fills ===")
print(f"days {nD}  train {trM.sum()}/holdout {hoM.sum()}.  pool {len(pool)}")
if not pool:
    print("NO positive configs — edge gone under realistic fills."); sys.exit()
print("TOP 6 singles:")
for sc, j, rpm, dd in pool[:6]:
    ho, _, _ = seg(Dr[:, j], hoM, hoS, hoMonN)
    print(f"  {tag(cfgs[j]):26s} trainRpm {rpm:+.2f} DD {dd:5.1f} OOS {ho:+.2f} {Dc[:,j].sum()/(nD/21.7):.1f}/mo")
C = np.nan_to_num(np.corrcoef(Mr[:, pidx].T))
res = []
for ci in itertools.combinations(range(len(pidx)), 3):
    idx = [pidx[i] for i in ci]
    rpm, dd, grn = seg(Dr[:, idx].sum(1), trM, trS, trMonN)
    if Dc[:, idx][trM].sum()/(trM.sum()/21.7) < 3: continue
    avgc = float(np.mean([C[a, b] for a, b in itertools.combinations(ci, 2)]))
    res.append((rpm/max(-dd,0.5), idx, rpm, dd, grn, avgc))
res.sort(key=lambda x: x[0], reverse=True)
print("BEST 3-leg low-corr (<0.35, +OOS, ret>=55%):")
for sc, idx, rpm, dd, grn, avgc in res:
    if avgc >= 0.35: continue
    ho, hodd, hogr = seg(Dr[:, idx].sum(1), hoM, hoS, hoMonN)
    if ho > 0 and rpm > 0 and ho/rpm >= 0.55:
        full, fdd, fgr = seg(Dr[:, idx].sum(1), np.ones(nD, bool), allS, len(allS))
        print(f"  corr {avgc:.2f} | FULL {full:+.2f} R/mo DD {fdd:.1f}R grn {fgr:.0f}% | TRAIN {rpm:+.2f} OOS {ho:+.2f} ({ho/rpm*100:.0f}%) | {Dc[:,idx].sum()/(nD/21.7):.1f}/mo")
        for i in idx: print(f"      {tag(cfgs[i])}")
        break
else:
    print("  (no 3-leg cleared the bar under realistic fills)")
