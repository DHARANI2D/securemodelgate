#!/usr/bin/env python3
"""
SecureModelGate - real-model lineage evidence on public Hugging Face weights.

What it measures (fills Table 1, row 1 of the Tech Con abstract):
  * Base:        bert-base-uncased (Devlin et al. 2019)
  * Descendants: public fine-tunes of that base + locally made INT8 and 30%-pruned copies
                 + a fully FFN-permuted fine-tune (function-preserving "laundering")
  * Negatives:   MultiBERTs seeds (Sellam et al., ICLR 2022): same architecture, same data,
                 different random seed -> the hardest honest "unrelated model" there is.
                 Seeds 0..(n_cal-1) calibrate the thresholds, the rest are held-out test negatives.
  * Attack:      layer splice - first 7 of 12 encoder blocks copied from the base, last 5 from a
                 held-out MultiBERT. A median-only lineage test should ADMIT it; the per-layer
                 floor should BLOCK it.

Lineage rule (same as the paper):
  per-matrix score s_l = w * spectral correlation + (1 - w) * mean cosine of matched output channels,
  with w = --spectral-weight (default 0: on the synthetic pilot, spectra hurt separation -- run
  once with --spectral-weight 0.5 as well and report both, that is the real-model ablation)
  accept iff median_l(s_l) > T_median  AND  min_l(s_l) > T_floor
  (matrices missing/reshaped in the candidate are "declared replaced" and exempt,
   but at most --max-exempt of them; beyond that the candidate fails.)
  T_median = conformal quantile of the calibration negatives' medians,
  T_floor  = conformal quantile of the calibration negatives' pooled per-matrix scores.
  Every model (calibration included) is FFN-aligned first, so the thresholds are
  calibrated under the same procedure every candidate gets.

  T_floor is NOT a formal conformal bound: per-matrix scores from the same model are
  correlated, so pooling breaks exchangeability, and the min over L matrices of a
  benign model is not covered by that quantile. False-ACCEPT stays bounded because the
  floor is an AND with the median test; false-REJECT is not bounded -- read it off the
  descendant rows and report it.

Usage (Mac, ~12 GB download for 25 MultiBERTs):
  pip install torch transformers scipy numpy
  python smg_bert_lineage.py --out results.json            # 20 calibration + 5 held-out seeds
  python smg_bert_lineage.py --selftest                    # no downloads; synthetic sanity check
Quicker (~6 GB) -- the alphas MUST loosen with fewer calibration models, or both
thresholds are infinite and every genuine descendant is rejected (the script now
refuses to run in that case):
  python smg_bert_lineage.py --n-seeds 12 --n-cal 9 --alpha 0.1 --alpha-floor 0.002
"""
import argparse, json, math, re, sys, time
import numpy as np
from scipy.optimize import linear_sum_assignment

BASE = "bert-base-uncased"
FINETUNES = [  # public fine-tunes of bert-base-uncased; any that fail to load are skipped and reported
    "textattack/bert-base-uncased-SST-2",
    "textattack/bert-base-uncased-MNLI",
    "textattack/bert-base-uncased-imdb",
    "textattack/bert-base-uncased-ag-news",
    "textattack/bert-base-uncased-yelp-polarity",
    "textattack/bert-base-uncased-QQP",
    "bhadresh-savani/bert-base-uncased-emotion",
]
MULTIBERT = "google/multiberts-seed_{}"
KEEP = re.compile(r"^encoder\.layer\.\d+\.(attention\.self\.(query|key|value)|attention\.output\.dense|intermediate\.dense|output\.dense)\.weight$")
RNG = np.random.default_rng(0)
SPECTRAL_WEIGHT = 0.0
SAMPLE_ROWS = 512  # rows sampled for rectangular matching on wide matrices (speed); same for every model


# ---------------------------------------------------------------- loading
def load(name):
    import torch
    from transformers import AutoModel
    m = AutoModel.from_pretrained(name)  # drops task heads -> heads are "declared replaced" by construction
    return {k: v.detach().to(torch.float32).numpy() for k, v in m.state_dict().items() if KEEP.match(k)}


# ---------------------------------------------------------------- scoring
def _unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def spectral_corr(a, b):
    sa = np.linalg.svd(a, compute_uv=False); sb = np.linalg.svd(b, compute_uv=False)
    k = min(len(sa), len(sb)); sa, sb = sa[:k], sb[:k]
    if sa.std() < 1e-12 or sb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(sa, sb)[0, 1])

def matched_cosine(a, b, rows=None):
    """mean cosine of candidate output channels (rows of a) optimally matched to base rows (rows of b).
    Rectangular assignment, so a candidate with fewer channels (structured pruning) still matches."""
    if a.shape[1] != b.shape[1]:
        return float("nan")
    ia = rows if rows is not None else np.arange(a.shape[0])
    C = _unit(a[ia]) @ _unit(b).T
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].mean())

def layer_scores(cand, base):
    out = {}
    for k, wb in base.items():
        wc = cand.get(k)
        if wc is None or wc.shape[1] != wb.shape[1]:
            out[k] = None  # declared/structurally replaced -> exempt (counted)
            continue
        rows = RNG.choice(wc.shape[0], SAMPLE_ROWS, replace=False) if wc.shape[0] > SAMPLE_ROWS else None
        cos = matched_cosine(wc, wb, rows)
        out[k] = cos if SPECTRAL_WEIGHT == 0 else SPECTRAL_WEIGHT * spectral_corr(wc, wb) + (1 - SPECTRAL_WEIGHT) * cos
    return out

def summarize(ls):
    v = np.array([x for x in ls.values() if x is not None])
    return {"median": float(np.median(v)), "min": float(v.min()),
            "n_exempt": sum(x is None for x in ls.values()), "n": len(ls)}

def min_n_for(alpha):
    """smallest calibration size n with ceil((n+1)(1-alpha)) <= n, i.e. a finite threshold"""
    return math.ceil(1.0 / alpha) - 1

def conformal_upper(scores, alpha, what="scores"):
    """smallest threshold t with P(null score > t) <= alpha under exchangeability.

    Raises instead of returning +inf when n is too small for alpha. An infinite
    threshold means NOTHING can ever pass, and the summary would still read
    "independents rejected N/N, splice blocked 1/1" -- true only because every
    genuine descendant is rejected too (e.g. 9 calibration models at alpha=0.05,
    or 648 pooled scores at alpha_floor=0.001).
    """
    s = np.sort(np.asarray(scores)); n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        raise SystemExit(f"{what}: n={n} cannot support alpha={alpha} (needs n >= {min_n_for(alpha)}); "
                         f"raise the calibration size or alpha -- refusing to run with an infinite threshold")
    return float(s[k - 1])


# ---------------------------------------------------------------- derivatives made locally
def int8(sd):
    o = {}
    for k, w in sd.items():
        s = np.abs(w).max(axis=1, keepdims=True) / 127 + 1e-12
        o[k] = np.clip(np.round(w / s), -127, 127) * s
    return o

def prune(sd, frac=0.3):
    o = {}
    for k, w in sd.items():
        t = np.quantile(np.abs(w), frac); o[k] = np.where(np.abs(w) < t, 0, w)
    return o

def permute_ffn(sd, n_layers=12):
    """function-preserving: permute FFN hidden units (intermediate rows, output.dense columns)"""
    o = dict(sd)
    for i in range(n_layers):
        ki, ko = f"encoder.layer.{i}.intermediate.dense.weight", f"encoder.layer.{i}.output.dense.weight"
        p = RNG.permutation(sd[ki].shape[0])
        o[ki] = sd[ki][p]; o[ko] = sd[ko][:, p]
    return o

def align_ffn(cand, base, n_layers=12):
    """FIX for cascading permutation (Re-Basin-lite): infer the FFN hidden-unit permutation from the
    intermediate rows and apply it to output.dense columns before scoring."""
    o = dict(cand)
    for i in range(n_layers):
        ki, ko = f"encoder.layer.{i}.intermediate.dense.weight", f"encoder.layer.{i}.output.dense.weight"
        if ki not in cand or ko not in cand or cand[ki].shape != base[ki].shape:
            continue
        C = _unit(cand[ki]) @ _unit(base[ki]).T
        p = C.argmax(axis=1)  # greedy; exact for true descendants
        if len(np.unique(p)) != len(p):
            r, c = linear_sum_assignment(-C); p = np.empty_like(r); p[r] = c
        inv = np.empty_like(p); inv[p] = np.arange(len(p))
        o[ki] = cand[ki][inv]; o[ko] = cand[ko][:, inv]
    return o

def splice(base, other, first_from_base=7, n_layers=12):
    o = {}
    for k in base:
        li = int(k.split(".")[2])
        o[k] = base[k] if li < first_from_base else other[k]
    return o


# ---------------------------------------------------------------- experiment
def run(load_fn, base_name, finetunes, seeds, n_cal, alpha, alpha_floor, max_exempt, n_layers):
    t0 = time.time()
    base = load_fn(base_name)
    res = {"base": base_name, "alpha": alpha, "alpha_floor": alpha_floor, "spectral_weight": SPECTRAL_WEIGHT,
           "rows": [], "skipped": []}

    def score(name, sd, kind, align=False):
        if align:
            sd = align_ffn(sd, base, n_layers)
        ls = layer_scores(sd, base)
        s = summarize(ls); s.update(name=name, kind=kind, aligned=align)
        s["_layers"] = [x for x in ls.values() if x is not None]
        print(f"  {kind:12s} {name:55s} median={s['median']:.3f} min={s['min']:.3f} exempt={s['n_exempt']}", flush=True)
        return s

    negs = {}
    for sd_seed in seeds:
        nm = MULTIBERT.format(sd_seed)
        try:
            negs[sd_seed] = load_fn(nm)
        except Exception as e:
            res["skipped"].append([nm, str(e)[:120]]); continue
    cal_ids = [s for s in seeds if s in negs][:n_cal]
    test_ids = [s for s in seeds if s in negs][n_cal:]
    print(f"calibration negatives: {len(cal_ids)}, held-out negatives: {len(test_ids)}")

    # Every model -- calibration included -- goes through the same alignment step, so the
    # thresholds already account for whatever matching freedom alignment adds.
    cal = [score(MULTIBERT.format(s), negs[s], "calibration", align=True) for s in cal_ids]
    pooled = [x for c in cal for x in c["_layers"]]
    T_med = conformal_upper([c["median"] for c in cal], alpha, "median threshold (calibration models)")
    T_floor = conformal_upper(pooled, alpha_floor, "floor threshold (pooled per-matrix scores)")
    res.update(T_median=T_med, T_floor=T_floor, n_cal=len(cal_ids), n_pooled=len(pooled))
    print(f"T_median={T_med:.4f}  T_floor={T_floor:.4f}")

    def decide(s):
        med_ok = s["median"] > T_med
        floor_ok = s["min"] > T_floor and s["n_exempt"] <= max_exempt
        s["accept_median_only"] = bool(med_ok)
        s["accept_with_floor"] = bool(med_ok and floor_ok)
        return s

    rows = []
    ft_loaded = []
    for nm in finetunes:
        try:
            sd = load_fn(nm)
        except Exception as e:
            res["skipped"].append([nm, str(e)[:120]]); continue
        ft_loaded.append((nm, sd)); rows.append(decide(score(nm, sd, "descendant", align=True)))
    if ft_loaded:
        nm, sd = ft_loaded[0]
        rows.append(decide(score(nm + " +INT8", int8(sd), "descendant", align=True)))
        rows.append(decide(score(nm + " +prune30%", prune(sd), "descendant", align=True)))
        perm = permute_ffn(sd, n_layers)
        # contrast row: the same permuted model under the old, alignment-free rule
        rows.append(decide(score(nm + " +FFN-permuted", perm, "descendant")))
        rows.append(decide(score(nm + " +FFN-permuted", perm, "descendant", align=True)))
    for s in test_ids:
        rows.append(decide(score(MULTIBERT.format(s), negs[s], "independent", align=True)))
    if test_ids:
        other = negs[test_ids[0]]
        k = n_layers // 2 + 1  # just over half the blocks from the base (7 of 12 for BERT-base)
        sp = splice(base, other, k, n_layers)
        rows.append(decide(score(f"splice: base blocks 0-{k-1} + {MULTIBERT.format(test_ids[0])} blocks {k}-{n_layers-1}", sp, "splice-attack", align=True)))
    for r in cal + rows:
        r.pop("_layers", None)
    res["rows"] = rows
    res["calibration"] = cal

    # summary for the paper
    def frac(kind, key, want):
        r = [x for x in rows if x["kind"] == kind and not (x["name"].endswith("FFN-permuted") and not x["aligned"])]
        return f"{sum(x[key] == want for x in r)}/{len(r)}"
    res["summary"] = {
        "descendants_accepted_with_floor": frac("descendant", "accept_with_floor", True),
        "independents_rejected_with_floor": frac("independent", "accept_with_floor", False),
        "splice_admitted_median_only": frac("splice-attack", "accept_median_only", True),
        "splice_blocked_with_floor": frac("splice-attack", "accept_with_floor", False),
        "permuted_unaligned_accepted": [x["accept_with_floor"] for x in rows if x["name"].endswith("FFN-permuted") and not x["aligned"]],
        "permuted_aligned_accepted": [x["accept_with_floor"] for x in rows if x["name"].endswith("FFN-permuted") and x["aligned"]],
        "runtime_s": round(time.time() - t0, 1),
    }
    return res


# ---------------------------------------------------------------- synthetic self-test (no downloads)
def _fake_bert(seed, n_layers=4, h=64, f=256):
    g = np.random.default_rng(seed); sd = {}
    for i in range(n_layers):
        for nm in ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"]:
            sd[f"encoder.layer.{i}.{nm}.weight"] = g.normal(0, 0.05, (h, h))
        sd[f"encoder.layer.{i}.intermediate.dense.weight"] = g.normal(0, 0.05, (f, h))
        sd[f"encoder.layer.{i}.output.dense.weight"] = g.normal(0, 0.05, (h, f))
    return sd

def selftest():
    base = _fake_bert(1000)
    reg = {"base": base}
    for s in range(12):
        reg[f"neg{s}"] = _fake_bert(s)
    for j in range(3):
        g = np.random.default_rng(500 + j)
        reg[f"ft{j}"] = {k: v + g.normal(0, 0.01, v.shape) for k, v in base.items()}
    global MULTIBERT
    MULTIBERT = "neg{}"
    res = run(lambda n: reg[n], "base", ["ft0", "ft1", "ft2"], list(range(12)), 9, 0.1, 0.01, 2, 4)
    s = res["summary"]; print(json.dumps(s, indent=1))
    assert s["descendants_accepted_with_floor"].split("/")[0] == s["descendants_accepted_with_floor"].split("/")[1], "descendant rejected"
    assert s["independents_rejected_with_floor"] == "3/3"
    assert s["splice_admitted_median_only"] == "1/1", "splice should fool median-only"
    assert s["splice_blocked_with_floor"] == "1/1", "floor should block splice"
    assert s["permuted_aligned_accepted"] == [True]
    print("SELFTEST OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=25, help="MultiBERTs seeds to download (max 25)")
    ap.add_argument("--n-cal", type=int, default=20, help="seeds used for calibration; rest are held-out negatives")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--alpha-floor", type=float, default=0.001)
    ap.add_argument("--max-exempt", type=int, default=2)
    ap.add_argument("--spectral-weight", type=float, default=0.0)
    ap.add_argument("--out", default="smg_bert_lineage_results.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    SPECTRAL_WEIGHT = a.spectral_weight
    if a.selftest:
        selftest(); sys.exit(0)
    if a.n_cal >= a.n_seeds:
        sys.exit("--n-cal must be smaller than --n-seeds (need held-out negatives)")
    r = run(load, BASE, FINETUNES, list(range(a.n_seeds)), a.n_cal, a.alpha, a.alpha_floor, a.max_exempt, 12)
    json.dump(r, open(a.out, "w"), indent=1)
    print("\nSUMMARY FOR THE PAPER:\n" + json.dumps(r["summary"], indent=1))
