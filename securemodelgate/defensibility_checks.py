"""Measurements behind the developer checklist
(`paper/techcon2027_revision/13_developer_checklist_answers.md`).

Each section answers a question the earlier pilots did not, on the same
synthetic corpus and the same calibration as `evaluation.py` (independent
seeds 1000-1019 for the lineage and floor thresholds, the first 20 benign
derivatives at seed offset 2000 for the envelope threshold), and every
decision is the gate's own rule (`pipeline.lineage_verdict_for` plus the
envelope test). Synthetic corpus only, not E1-E9.

  1. splice patterns    every non-empty subset of the 6 weight matrices,
                         undeclared and declared; plus partial-channel
                         splices (a fraction of every layer's rows foreign)
  2. adaptive attacks   S_w-adaptive forgery (an unrelated backdoored model
                         trained toward the public base's weights) and the
                         joint S_w + envelope attack, scored end to end
  3. declared head      a backdoor that lives only in a re-trained head
                         declared as replaced (the one exemption)
  4. benign sweeps      fine-tune depth, magnitude pruning, quantization
                         levels, weight noise: where false blocks begin
  5. siblings           models sharing the base's random init but trained
                         separately: does S_w measure training or init?
  6. determinism        repeat runs, thread counts, probe batch size, tiny
                         weight perturbations
  7. scaling            matching cost vs. matrix size, end-to-end scoring
                         time vs. width and probe-set size
"""

from __future__ import annotations

import copy
import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from . import demo_models
from .adaptive_attack import adaptive_envelope_penalty
from .evaluation import _make_benign_population
from .lineage.behavioral_delta import compute_behavioral_delta_from_models
from .lineage.conformal import conformal_quantile
from .lineage.lineage_score import _match_rows, pooled_layer_scores, weight_lineage_score_from_models
from .pipeline import envelope_anomaly_score, lineage_verdict_for

MATRICES = ("stem.conv", "layer1.conv", "layer2.conv", "layer3.conv", "layer4.conv", "fc")


@dataclass
class Gate:
    base: torch.nn.Module
    probe: object
    lineage_cal: list
    floor_cal: list
    envelope_cal: list
    alpha: float = 0.05

    @property
    def envelope_threshold(self) -> float:
        return conformal_quantile(self.envelope_cal, self.alpha)

    def decide(self, model, declared=(), behavioral: bool = True) -> dict:
        s = weight_lineage_score_from_models(model, self.base, declared_replaced=declared)
        v, _ = lineage_verdict_for(s, self.lineage_cal, self.floor_cal, alpha=self.alpha)
        out = {"S_w": s["S_w"], "min_layer": s["min_layer_score"], "n_exempt": len(s["exempt"]),
               "lineage_pass": bool(v.exceeds_threshold)}
        if behavioral:
            beh = compute_behavioral_delta_from_models(model, self.base, self.probe)
            score = envelope_anomaly_score(beh)
            out.update(envelope_score=score, envelope_pass=bool(score <= self.envelope_threshold))
            out["decision"] = ("BLOCK" if not out["lineage_pass"] else
                               "ADMIT" if out["envelope_pass"] else "ESCALATE")
        return out


def build_gate(verbose=print) -> Gate:
    verbose("Building the evaluation.py calibration (base seed 1, independents 1000-1019, benign 2000+) ...")
    base = demo_models.make_base_model(seed=1)
    probe = demo_models.make_probe_loader(seed=20260101)
    dicts = [weight_lineage_score_from_models(demo_models.make_independent_model(seed=1000 + i), base)
             for i in range(20)]
    benign = _make_benign_population(base, 20, seed_offset=2000)
    env = [envelope_anomaly_score(compute_behavioral_delta_from_models(m, base, probe)) for m in benign]
    return Gate(base, probe, [d["S_w"] for d in dicts], pooled_layer_scores(dicts), env)


def _set_weight(model, name, tensor):
    mod = model.get_submodule(name)
    with torch.no_grad():
        mod.weight.copy_(tensor)


# ---------------------------------------------------------------- 1. splices

def splice_patterns(gate: Gate, n_donors: int = 5, log=print) -> dict:
    log("1. Splice patterns: all 63 subsets x donors, undeclared and declared ...")
    host = demo_models.make_benign_fine_tune(gate.base, seed=42)
    donors = [demo_models.make_independent_model(seed=5000 + i) for i in range(n_donors)]
    by_size = {k: {"n": 0, "admitted_undeclared": 0, "admitted_declared": 0} for k in range(1, 7)}
    worst = []
    for k in range(1, 7):
        for subset in itertools.combinations(MATRICES, k):
            for donor in donors:
                m = copy.deepcopy(host)
                for name in subset:
                    _set_weight(m, name, donor.get_submodule(name).weight)
                u = gate.decide(m, behavioral=False)
                d = gate.decide(m, declared=[f"{n}.weight" for n in subset], behavioral=False)
                row = by_size[k]
                row["n"] += 1
                row["admitted_undeclared"] += u["lineage_pass"]
                row["admitted_declared"] += d["lineage_pass"]
                if u["lineage_pass"]:
                    worst.append({"subset": subset, **u})

    log("   partial-channel splices: a fraction of EVERY layer's output channels foreign ...")
    rng = np.random.default_rng(0)
    partial = []
    for frac in (0.1, 0.25, 0.5, 0.75):
        passes, rows = 0, []
        for donor in donors:
            m = copy.deepcopy(host)
            for name in MATRICES:
                w = m.get_submodule(name).weight.detach().clone()
                dw = donor.get_submodule(name).weight.detach()
                idx = rng.choice(w.shape[0], size=max(1, int(round(frac * w.shape[0]))), replace=False)
                w[idx] = dw[idx]
                _set_weight(m, name, w)
            r = gate.decide(m, behavioral=False)
            passes += r["lineage_pass"]
            rows.append(r)
        partial.append({"fraction_foreign": frac, "n": len(donors), "lineage_admitted": passes,
                        "mean_S_w": float(np.mean([r["S_w"] for r in rows])),
                        "mean_min_layer": float(np.mean([r["min_layer"] for r in rows]))})
    return {"by_subset_size": by_size, "undeclared_admitted_examples": worst[:10], "partial_channel": partial,
            "floor_threshold": conformal_quantile(gate.floor_cal, gate.alpha),
            "median_threshold": conformal_quantile(gate.lineage_cal, gate.alpha)}


# ---------------------------------------------------------------- 2. adaptive attacks

def _row_cosine_to_base(model, base) -> torch.Tensor:
    """Per-matrix mean row cosine to the base's same-index rows: a
    differentiable stand-in for S_w (identity channel correspondence)."""
    cs = []
    for name in MATRICES:
        w = model.get_submodule(name).weight.flatten(1)
        b = base.get_submodule(name).weight.detach().flatten(1)
        cs.append(F.cosine_similarity(w, b, dim=1).mean())
    return torch.stack(cs)


def train_adaptive(start, base, *, seed, mu, lam, steps=150, lr=0.03, trigger="patch", probe_images=None):
    model = copy.deepcopy(start)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    for step in range(steps):
        images, labels = demo_models.synthetic_batch(32, seed=seed * 10_000 + step)
        n_poison = len(images) // 2
        labels = labels.clone()
        images = torch.cat([demo_models.apply_trigger(images[:n_poison], trigger), images[n_poison:]])
        labels[:n_poison] = 0
        loss = F.cross_entropy(model(images), labels)
        if mu > 0:
            c = _row_cosine_to_base(model, base)
            loss = loss + mu * (1.0 - 0.5 * c.mean() - 0.5 * c.min())
        if lam > 0:
            if probe_images is None:   # a draw from the public distribution, not the gate's own probes
                batch, _ = demo_models.synthetic_batch(16, seed=20260101 * 100 + step)
            else:                      # the attacker knows the gate's exact probe pool
                batch = probe_images[(step * 16) % len(probe_images):][:16]
            loss = loss + lam * adaptive_envelope_penalty(model, base, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def adaptive_attacks(gate: Gate, n: int = 4, log=print) -> list:
    log("2. Adaptive attacks: S_w-adaptive forgery and joint S_w + envelope attack ...")
    cells = [("unrelated", 0.0, 0.0), ("unrelated", 1.0, 0.0), ("unrelated", 3.0, 0.0), ("unrelated", 10.0, 0.0),
             ("unrelated", 3.0, 1.0), ("unrelated", 10.0, 1.0),
             ("base", 0.0, 0.0), ("base", 0.0, 1.0), ("base", 0.0, 2.0)]
    out = []
    for start_kind, mu, lam in cells:
        rows = []
        for i in range(n):
            start = (demo_models.make_independent_model(seed=7000 + i) if start_kind == "unrelated" else gate.base)
            m = train_adaptive(start, gate.base, seed=7100 + i, mu=mu, lam=lam)
            r = gate.decide(m)
            r["asr"] = demo_models.attack_success_rate(m, seed=7100 + i)
            r["clean_acc"] = demo_models.clean_accuracy(m)
            r["row_cos"] = float(_row_cosine_to_base(m, gate.base).mean().detach())
            rows.append(r)
        cell = {"start": start_kind, "mu_Sw": mu, "lambda_envelope": lam, "n": n,
                "lineage_pass": sum(r["lineage_pass"] for r in rows),
                "envelope_pass": sum(r["envelope_pass"] for r in rows),
                "admitted": sum(r["decision"] == "ADMIT" for r in rows),
                "admitted_with_asr_ge_0.9": sum(r["decision"] == "ADMIT" and r["asr"] >= 0.9 for r in rows),
                "mean_asr": float(np.mean([r["asr"] for r in rows])),
                "mean_clean_acc": float(np.mean([r["clean_acc"] for r in rows])),
                "mean_S_w": float(np.mean([r["S_w"] for r in rows])),
                "mean_min_layer": float(np.mean([r["min_layer"] for r in rows])),
                "mean_envelope": float(np.mean([r["envelope_score"] for r in rows])),
                "mean_row_cos": float(np.mean([r["row_cos"] for r in rows]))}
        log(f"   start={start_kind:9s} mu={mu:<4} lam={lam:<3} lineage {cell['lineage_pass']}/{n} "
            f"envelope {cell['envelope_pass']}/{n} ADMIT {cell['admitted']}/{n} ASR {cell['mean_asr']:.2f}")
        out.append(cell)
    return out


# ---------------------------------------------------------------- 3. declared head

def declared_head(gate: Gate, n: int = 5, log=print) -> list:
    log("3. Declared head replacement: backbone = base, head re-initialised and re-trained ...")
    out = []
    for poisoned in (False, True):
        rows = []
        for i in range(n):
            m = copy.deepcopy(gate.base)
            torch.manual_seed(8000 + i)
            m.fc.reset_parameters()
            opt = torch.optim.SGD(m.fc.parameters(), lr=0.05, momentum=0.9)
            m.eval()   # backbone BatchNorm statistics stay the base's
            for step in range(150):
                images, labels = demo_models.synthetic_batch(32, seed=(8000 + i) * 10_000 + step)
                if poisoned:
                    labels = labels.clone()
                    images = torch.cat([demo_models.apply_trigger(images[:16], "patch"), images[16:]])
                    labels[:16] = 0
                loss = F.cross_entropy(m(images), labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
            declared = gate.decide(m, declared=["fc.weight"])
            undeclared = gate.decide(m, behavioral=False)
            rows.append({**declared, "undeclared_lineage_pass": undeclared["lineage_pass"],
                         "asr": demo_models.attack_success_rate(m, seed=8000 + i),
                         "clean_acc": demo_models.clean_accuracy(m)})
        cell = {"head": "backdoored" if poisoned else "benign", "n": n,
                "lineage_pass_declared": sum(r["lineage_pass"] for r in rows),
                "lineage_pass_undeclared": sum(r["undeclared_lineage_pass"] for r in rows),
                "decisions": [r["decision"] for r in rows],
                "admitted": sum(r["decision"] == "ADMIT" for r in rows),
                "mean_asr": float(np.mean([r["asr"] for r in rows])),
                "mean_clean_acc": float(np.mean([r["clean_acc"] for r in rows])),
                "envelope_scores": [r["envelope_score"] for r in rows]}
        log(f"   {cell['head']:10s} head: decisions {cell['decisions']} ASR {cell['mean_asr']:.2f}")
        out.append(cell)
    return out


# ---------------------------------------------------------------- 4. benign sweeps

def _noisy(base, rel):
    m = copy.deepcopy(base)
    g = torch.Generator().manual_seed(int(rel * 1e6))
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() >= 2:
                p.add_(torch.randn(p.shape, generator=g) * rel * p.std())
    return m


def _fine_tune(base, seed, steps, lr):
    return demo_models._train_steps(copy.deepcopy(base), seed=seed, steps=steps, lr=lr)


def benign_sweeps(gate: Gate, log=print) -> list:
    log("4. Benign-derivative sweeps ...")
    b = gate.base
    variants = []
    for steps, lr in ((8, 0.005), (40, 0.005), (200, 0.005), (800, 0.005), (40, 0.05), (200, 0.05), (800, 0.05)):
        variants.append((f"fine-tune {steps} steps lr={lr}", [_fine_tune(b, 9000 + s, steps, lr) for s in range(3)]))
    for amount in (0.1, 0.3, 0.5, 0.7, 0.9):
        variants.append((f"magnitude prune {int(amount * 100)}%", [demo_models.make_pruned_derivative(b, amount)]))
    for levels in (256, 16, 8, 4, 2):
        variants.append((f"quantize {levels} levels/tensor", [demo_models.make_quantized_derivative(b, levels)]))
    for rel in (0.01, 0.05, 0.1, 0.3):
        variants.append((f"Gaussian weight noise {rel} x std", [_noisy(b, rel)]))
    out = []
    for name, models in variants:
        rows = [gate.decide(m) for m in models]
        cell = {"variant": name, "n": len(rows),
                "lineage_pass": sum(r["lineage_pass"] for r in rows),
                "admitted": sum(r["decision"] == "ADMIT" for r in rows),
                "decisions": [r["decision"] for r in rows],
                "min_S_w": min(r["S_w"] for r in rows), "min_min_layer": min(r["min_layer"] for r in rows),
                "max_envelope": max(r["envelope_score"] for r in rows),
                "clean_acc": float(np.mean([demo_models.clean_accuracy(m) for m in models]))}
        log(f"   {name:36s} {cell['decisions']} min-layer {cell['min_min_layer']:.3f} "
            f"envelope {cell['max_envelope']:.3f}")
        out.append(cell)
    return out


# ---------------------------------------------------------------- 5. siblings

def siblings(gate: Gate, n: int = 3, log=print) -> list:
    log("5. Same-init siblings: the base's init, separately trained ...")
    out = []
    for i in range(n):
        torch.manual_seed(1)                      # the base's own init seed
        m = demo_models.TinyConvNet()
        m = demo_models._train_steps(m, seed=500 + i, steps=40, lr=0.05)   # different data stream
        r = gate.decide(m)
        out.append({"data_seed": 500 + i, **r})
        log(f"   sibling data seed {500 + i}: S_w {r['S_w']:.3f} min-layer {r['min_layer']:.3f} {r['decision']}")
    return out


# ---------------------------------------------------------------- 6. determinism

def determinism(gate: Gate, log=print) -> dict:
    log("6. Determinism ...")
    cand = demo_models.make_benign_fine_tune(gate.base, seed=42)

    def scores(model, probe):
        s = weight_lineage_score_from_models(model, gate.base)
        b = compute_behavioral_delta_from_models(model, gate.base, probe)
        return np.array([s["S_w"], s["min_layer_score"], b["min_cka"], b["js_divergence"]])

    ref = scores(cand, gate.probe)
    rep = scores(cand, gate.probe)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    one = scores(cand, gate.probe)
    torch.set_num_threads(threads)
    big_batch = scores(cand, demo_models.make_probe_loader(seed=20260101, batch_size=64))
    perturbed = scores(_noisy(cand, 1e-6), gate.probe)
    return {"fields": ["S_w", "min_layer", "min_cka", "js"],
            "repeat_max_abs_diff": float(np.max(np.abs(rep - ref))),
            "threads_1_vs_default_max_abs_diff": float(np.max(np.abs(one - ref))),
            "probe_batch_64_vs_16_max_abs_diff": float(np.max(np.abs(big_batch - ref))),
            "weight_noise_1e-6_max_abs_diff": float(np.max(np.abs(perturbed - ref))),
            "default_threads": threads}


# ---------------------------------------------------------------- 7. scaling

def scaling(log=print) -> dict:
    log("7. Scaling ...")
    rng = np.random.default_rng(0)
    match = []
    for n in (64, 256, 768, 1024, 2048, 3072):
        a, b = rng.standard_normal((n, 768)), rng.standard_normal((n, 768))
        t0 = time.perf_counter()
        _match_rows(a, b)
        match.append({"rows": n, "cols": 768, "seconds": time.perf_counter() - t0,
                      "cost_matrix_MB": n * n * 8 / 1e6})
        log(f"   match {n}x{n}: {match[-1]['seconds']:.3f}s")
    widths = []
    for w in (8, 16, 32, 64):
        torch.manual_seed(0)
        a, b = demo_models.TinyConvNet(width=w), demo_models.TinyConvNet(width=w)
        a.eval(), b.eval()
        t0 = time.perf_counter()
        weight_lineage_score_from_models(a, b)
        t1 = time.perf_counter()
        compute_behavioral_delta_from_models(a, b, demo_models.make_probe_loader(n=64))
        widths.append({"width": w, "params": sum(p.numel() for p in a.parameters()),
                       "S_w_seconds": t1 - t0, "D_b_seconds_64_probes": time.perf_counter() - t1})
    probes = []
    a, b = demo_models.TinyConvNet(), demo_models.TinyConvNet()
    a.eval(), b.eval()
    for n in (64, 256, 1024):
        t0 = time.perf_counter()
        compute_behavioral_delta_from_models(a, b, demo_models.make_probe_loader(n=n, batch_size=64))
        probes.append({"probes": n, "seconds": time.perf_counter() - t0})
    return {"matching": match, "widths": widths, "probe_set": probes}


# ---------------------------------------------------------------- 8. calibration depth

def calibration_depth(gate: Gate, log=print) -> dict:
    """The envelope threshold is only as wide as the benign population it is
    calibrated on. Compare the evaluation's calibration (8-step fine-tunes and
    their compressed variants) with one spanning fine-tune depths 8/40/200,
    on held-out fine-tunes of all three depths and on the evaluation's 30
    backdoored derivatives."""
    log("8. Calibration-population depth ...")
    depths = (8, 40, 200)

    def env(m):
        return envelope_anomaly_score(compute_behavioral_delta_from_models(m, gate.base, gate.probe))

    cal = [env(_fine_tune(gate.base, 11000 + i, depths[i % 3], 0.005)) for i in range(21)]
    test = [(depths[i % 3], env(_fine_tune(gate.base, 12000 + i, depths[i % 3], 0.005))) for i in range(21)]
    backdoor = [env(demo_models.make_backdoored_derivative(gate.base, seed=3000 + i, trigger_type=t))
                for t in demo_models.TRIGGER_TYPES for i in range(10)]
    out = {}
    for label, tau in (("evaluation calibration (8-step)", gate.envelope_threshold),
                       ("mixed-depth calibration (8/40/200)", conformal_quantile(cal, gate.alpha))):
        out[label] = {"threshold": tau,
                      "false_block_by_depth": {d: f"{sum(s > tau for dd, s in test if dd == d)}/"
                                                  f"{sum(dd == d for dd, _ in test)}" for d in depths},
                      "backdoor_detected": f"{sum(s > tau for s in backdoor)}/{len(backdoor)}"}
        log(f"   {label}: tau={tau:.4f} {out[label]}")
    return out


# ---------------------------------------------------------------- 9. reproduction + function preservation

def _quantize_symmetric(model, bits: int, per_channel: bool):
    """Symmetric integer fake-quantization of every weight matrix: round to
    an integer grid, then DEQUANTIZE back to float. The scorer and the
    probes see dequantized float weights; raw integer tensors are not a
    supported input."""
    m = copy.deepcopy(model)
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() < 2:
                continue
            flat = p.flatten(1)
            scale = (flat.abs().amax(dim=1, keepdim=True) if per_channel else flat.abs().max()) / qmax
            scale = torch.clamp(scale, min=1e-12)
            p.copy_((torch.round(flat / scale).clamp(-qmax, qmax) * scale).view_as(p))
    return m


def _cast_round_trip(model, dtype):
    m = copy.deepcopy(model)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(p.to(dtype).to(p.dtype))
    return m


def _structured_zero(model, fraction: float):
    """Zero the lowest-L1 `fraction` of output channels (and their BatchNorm
    affine terms) in every conv layer: structured channel pruning in place."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        for name in MATRICES[:-1]:
            block = m.get_submodule(name.rsplit(".", 1)[0])
            w = block.conv.weight
            k = int(round(fraction * w.shape[0]))
            idx = w.flatten(1).abs().sum(1).argsort()[:k]
            w[idx] = 0.0
            block.bn.weight[idx] = 0.0
            block.bn.bias[idx] = 0.0
    return m


def reproduction_table(gate: Gate, log=print) -> dict:
    """Q1/Q3: per-layer scores for a genuine derivative under each benign
    transformation, and function preservation of the permutations over
    2000 inputs."""
    from .permutation_check import _CASCADE, permute_layer_output_channels

    log("9. Reproduction table and permutation function-preservation ...")
    ft = demo_models.make_benign_fine_tune(gate.base, seed=42)
    single = permute_layer_output_channels(ft, "layer2", lambda m: m.layer3.conv, seed=7)
    full = ft
    for i, (layer, nxt) in enumerate(_CASCADE):
        full = permute_layer_output_channels(full, layer, nxt, seed=i + 1)
    cases = [("genuine fine-tune", ft), ("single-layer permuted", single), ("all layers permuted", full),
             ("magnitude-pruned 30%", demo_models.make_pruned_derivative(ft, 0.3)),
             ("structured: 25% channels zeroed", _structured_zero(ft, 0.25)),
             ("structured: 50% channels zeroed", _structured_zero(ft, 0.5)),
             ("16-level quantized", demo_models.make_quantized_derivative(ft, 16)),
             ("FP16 round-trip", _cast_round_trip(ft, torch.float16)),
             ("BF16 round-trip", _cast_round_trip(ft, torch.bfloat16)),
             ("INT8 symmetric per-tensor", _quantize_symmetric(ft, 8, False)),
             ("INT8 symmetric per-channel", _quantize_symmetric(ft, 8, True)),
             ("INT4 symmetric per-channel", _quantize_symmetric(ft, 4, True))]
    rows = []
    for name, m in cases:
        s = weight_lineage_score_from_models(m, gate.base)
        r = gate.decide(m)
        rows.append({"case": name, "per_layer": dict(zip(s["layer_names"], s["per_layer"])), "S_w": s["S_w"],
                     "n_scored": s["n_layers"], "n_unscored": len(s["exempt"]), "envelope_score": r["envelope_score"],
                     "decision": r["decision"], "clean_acc": demo_models.clean_accuracy(m)})
        log(f"   {name:34s} S_w {s['S_w']:.4f} min {s['min_layer_score']:.4f} scored {s['n_layers']} "
            f"unscored {len(s['exempt'])} {r['decision']}")

    x, y = demo_models.synthetic_batch(2000, seed=424_243)
    preservation = []
    with torch.no_grad():
        ref = ft(x)
        for name, m in (("single-layer permuted", single), ("all layers permuted", full)):
            out = m(x)
            d = (out - ref).abs()
            b_ref = compute_behavioral_delta_from_models(ft, gate.base, gate.probe)
            b_m = compute_behavioral_delta_from_models(m, gate.base, gate.probe)
            self_delta = compute_behavioral_delta_from_models(m, ft, gate.probe)
            preservation.append({
                "case": name, "n_inputs": len(x), "max_abs_logit_diff": float(d.max()),
                "mean_abs_logit_diff": float(d.mean()),
                "prediction_agreement": float((out.argmax(1) == ref.argmax(1)).float().mean()),
                "accuracy_diff": float((out.argmax(1) == y).float().mean() - (ref.argmax(1) == y).float().mean()),
                "envelope_score_diff": abs(envelope_anomaly_score(b_m) - envelope_anomaly_score(b_ref)),
                "min_cka_vs_unpermuted": self_delta["min_cka"], "js_vs_unpermuted": self_delta["js_divergence"]})
    return {"rows": rows, "function_preservation": preservation}


# ---------------------------------------------------------------- 10. unrelated-model variety

def _train_independent(seed, steps, lr, optimizer):
    torch.manual_seed(seed)
    m = demo_models.TinyConvNet()
    opt = (torch.optim.Adam(m.parameters(), lr=lr) if optimizer == "adam"
           else torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9))
    m.train()
    for step in range(steps):
        x, y = demo_models.synthetic_batch(32, seed=seed * 10_000 + step)
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()
    m.eval()
    return m


def unrelated_variants(gate: Gate, n: int = 5, log=print) -> list:
    """Q9/Q20: unrelated models under other seeds, schedules and optimizers,
    against thresholds calibrated only on the evaluation's recipe."""
    log("10. Unrelated models: other schedules and optimizers ...")
    recipes = [("SGD lr=0.05, 40 steps (calibration recipe)", 40, 0.05, "sgd"),
               ("SGD lr=0.05, 200 steps", 200, 0.05, "sgd"), ("SGD lr=0.01, 150 steps", 150, 0.01, "sgd"),
               ("Adam lr=1e-3, 100 steps", 100, 1e-3, "adam"), ("Adam lr=1e-2, 100 steps", 100, 1e-2, "adam")]
    out = []
    for j, (label, steps, lr, opt) in enumerate(recipes):
        rows = [gate.decide(_train_independent(13_000 + 100 * j + i, steps, lr, opt), behavioral=False)
                for i in range(n)]
        sw = [r["S_w"] for r in rows]
        out.append({"recipe": label, "n": n, "lineage_pass": sum(r["lineage_pass"] for r in rows),
                    "S_w_min": min(sw), "S_w_mean": float(np.mean(sw)), "S_w_max": max(sw),
                    "min_layer_max": max(r["min_layer"] for r in rows)})
        log(f"   {label:44s} lineage pass {out[-1]['lineage_pass']}/{n} S_w {min(sw):.3f}-{max(sw):.3f}")
    return out


# ---------------------------------------------------------------- 11. probe-set variation

def probe_variation(gate: Gate, log=print) -> list:
    """Q14: envelope detection under four different probe draws, each with
    its own calibration, for non-adaptive backdoors and for adaptive ones
    trained on (a) the public distribution and (b) the gate's exact pool."""
    log("11. Probe-set variation ...")
    benign = _make_benign_population(gate.base, 30, seed_offset=2000)
    backdoor = [demo_models.make_backdoored_derivative(gate.base, seed=3000 + i, trigger_type=t)
                for t in demo_models.TRIGGER_TYPES for i in range(5)]
    exact = next(iter(demo_models.make_probe_loader(seed=20260101, batch_size=64)))[0]
    adaptive_public = [train_adaptive(gate.base, gate.base, seed=7100 + i, mu=0.0, lam=1.0) for i in range(4)]
    adaptive_exact = [train_adaptive(gate.base, gate.base, seed=7100 + i, mu=0.0, lam=1.0, probe_images=exact)
                      for i in range(4)]
    out = []
    for probe_seed in (20260101, 11, 12, 13):
        probe = demo_models.make_probe_loader(seed=probe_seed)

        def env(m):
            return envelope_anomaly_score(compute_behavioral_delta_from_models(m, gate.base, probe))

        tau = conformal_quantile([env(m) for m in benign[:20]], gate.alpha)
        row = {"probe_seed": probe_seed, "threshold": tau,
               "benign_false_blocks": f"{sum(env(m) > tau for m in benign[20:])}/10",
               "non_adaptive_detected": f"{sum(env(m) > tau for m in backdoor)}/{len(backdoor)}",
               "adaptive_public_draw_detected": f"{sum(env(m) > tau for m in adaptive_public)}/4",
               "adaptive_exact_pool_detected": f"{sum(env(m) > tau for m in adaptive_exact)}/4"}
        log(f"   {row}")
        out.append(row)
    return out


@dataclass
class DefensibilityReport:
    thresholds: dict
    splices: dict
    adaptive: list
    declared_head: list
    benign_sweeps: list
    siblings: list
    determinism: dict
    scaling: dict
    calibration_depth: dict = None
    reproduction: dict = None
    unrelated_variants: list = None
    probe_variation: list = None
    runtime_seconds: float = 0.0
    notes: list = field(default_factory=list)


def run_defensibility_checks(n_adaptive: int = 4, verbose: bool = True) -> DefensibilityReport:
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    gate = build_gate(log)
    thresholds = {"lineage_median": conformal_quantile(gate.lineage_cal, gate.alpha),
                  "layer_floor": conformal_quantile(gate.floor_cal, gate.alpha),
                  "envelope": gate.envelope_threshold, "alpha": gate.alpha,
                  "n_lineage_cal": len(gate.lineage_cal), "n_floor_pooled": len(gate.floor_cal),
                  "n_envelope_cal": len(gate.envelope_cal)}
    log(f"   thresholds: {thresholds}")
    return DefensibilityReport(
        thresholds=thresholds,
        splices=splice_patterns(gate, log=log),
        adaptive=adaptive_attacks(gate, n=n_adaptive, log=log),
        declared_head=declared_head(gate, log=log),
        benign_sweeps=benign_sweeps(gate, log=log),
        siblings=siblings(gate, log=log),
        determinism=determinism(gate, log=log),
        scaling=scaling(log=log),
        calibration_depth=calibration_depth(gate, log=log),
        reproduction=reproduction_table(gate, log=log),
        unrelated_variants=unrelated_variants(gate, log=log),
        probe_variation=probe_variation(gate, log=log),
        runtime_seconds=time.time() - t0,
    )


def render_markdown_report(r: DefensibilityReport) -> str:
    t = r.thresholds
    L = ["# SecureModelGate — defensibility checks", "",
         "Synthetic corpus (`demo_models.TinyConvNet`), not E1–E9. Same calibration as `evaluation.py`; "
         "every decision is the gate's own rule. "
         f"Thresholds: median S_w > {t['lineage_median']:.4f}, every layer > {t['layer_floor']:.4f} "
         f"(n={t['n_floor_pooled']} pooled), envelope ≤ {t['envelope']:.4f}; alpha={t['alpha']}.", "",
         "## 1. Splice patterns (lineage only)", "",
         "| foreign matrices | n | admitted, undeclared | admitted, declared |", "|---|---|---|---|"]
    for k, row in r.splices["by_subset_size"].items():
        L.append(f"| {k} | {row['n']} | {row['admitted_undeclared']} | {row['admitted_declared']} |")
    L += ["", "Partial-channel splice (a fraction of every matrix's output channels copied from an unrelated model):",
          "", "| fraction foreign | n | lineage admitted | mean S_w | mean min-layer |", "|---|---|---|---|---|"]
    for p in r.splices["partial_channel"]:
        L.append(f"| {p['fraction_foreign']} | {p['n']} | {p['lineage_admitted']} | {p['mean_S_w']:.3f} | "
                 f"{p['mean_min_layer']:.3f} |")
    L += ["", "## 2. Adaptive attacks (end-to-end gate decision)", "",
          "| start | mu (S_w pull) | lambda (envelope) | n | lineage pass | envelope pass | ADMIT | ADMIT with ASR≥0.9 "
          "| mean ASR | clean acc | mean S_w | mean min-layer | mean envelope |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in r.adaptive:
        L.append(f"| {c['start']} | {c['mu_Sw']} | {c['lambda_envelope']} | {c['n']} | {c['lineage_pass']} | "
                 f"{c['envelope_pass']} | {c['admitted']} | {c['admitted_with_asr_ge_0.9']} | {c['mean_asr']:.2f} | "
                 f"{c['mean_clean_acc']:.2f} | {c['mean_S_w']:.3f} | {c['mean_min_layer']:.3f} | "
                 f"{c['mean_envelope']:.3f} |")
    L += ["", "## 3. Declared head replacement", "",
          "| head | n | lineage pass (declared) | lineage pass (undeclared) | decisions | mean ASR | clean acc |",
          "|---|---|---|---|---|---|---|"]
    for c in r.declared_head:
        L.append(f"| {c['head']} | {c['n']} | {c['lineage_pass_declared']} | {c['lineage_pass_undeclared']} | "
                 f"{', '.join(c['decisions'])} | {c['mean_asr']:.2f} | {c['mean_clean_acc']:.2f} |")
    L += ["", "## 4. Benign-derivative sweeps", "",
          "| variant | n | decisions | min S_w | min min-layer | max envelope | clean acc |", "|---|---|---|---|---|---|---|"]
    for c in r.benign_sweeps:
        L.append(f"| {c['variant']} | {c['n']} | {', '.join(c['decisions'])} | {c['min_S_w']:.3f} | "
                 f"{c['min_min_layer']:.3f} | {c['max_envelope']:.3f} | {c['clean_acc']:.2f} |")
    L += ["", "## 5. Same-init siblings", "", "| data seed | S_w | min-layer | envelope | decision |", "|---|---|---|---|---|"]
    for s in r.siblings:
        L.append(f"| {s['data_seed']} | {s['S_w']:.3f} | {s['min_layer']:.3f} | {s['envelope_score']:.3f} | {s['decision']} |")
    d = r.determinism
    L += ["", "## 6. Determinism (max abs difference over S_w, min-layer, min CKA, JS)", "",
          "| comparison | max abs diff |", "|---|---|",
          f"| same inputs, repeated | {d['repeat_max_abs_diff']:.3g} |",
          f"| 1 thread vs {d['default_threads']} | {d['threads_1_vs_default_max_abs_diff']:.3g} |",
          f"| probe batch 64 vs 16 | {d['probe_batch_64_vs_16_max_abs_diff']:.3g} |",
          f"| weights + 1e-6·std noise | {d['weight_noise_1e-6_max_abs_diff']:.3g} |",
          "", "## 7. Scaling (CPU)", "", "| matched rows (x768 cols) | seconds | cost matrix MB |", "|---|---|---|"]
    for m in r.scaling["matching"]:
        L.append(f"| {m['rows']} | {m['seconds']:.3f} | {m['cost_matrix_MB']:.1f} |")
    L += ["", "| TinyConvNet width | params | S_w s | D_b s (64 probes) |", "|---|---|---|---|"]
    for w in r.scaling["widths"]:
        L.append(f"| {w['width']} | {w['params']} | {w['S_w_seconds']:.3f} | {w['D_b_seconds_64_probes']:.3f} |")
    L += ["", "| probes | D_b seconds |", "|---|---|"]
    for p in r.scaling["probe_set"]:
        L.append(f"| {p['probes']} | {p['seconds']:.3f} |")
    L += ["", "## 8. Calibration-population depth (envelope test only)", "",
          "| calibration | threshold | false blocks: 8 / 40 / 200-step fine-tunes | backdoors detected |",
          "|---|---|---|---|"]
    for label, c in (r.calibration_depth or {}).items():
        fb = c["false_block_by_depth"]
        L.append(f"| {label} | {c['threshold']:.4f} | {' / '.join(str(v) for v in fb.values())} | "
                 f"{c['backdoor_detected']} |")
    if r.reproduction:
        L += ["", "## 9. Genuine derivative under each benign transformation (`securemodelgate defensibility-check`)", "",
              "| case | per-layer scores (stem, l1, l2, l3, l4, fc) | median S_w | scored | unscored | envelope | decision | clean acc |",
              "|---|---|---|---|---|---|---|---|"]
        for x in r.reproduction["rows"]:
            pl = ", ".join(f"{v:.3f}" for v in x["per_layer"].values())
            L.append(f"| {x['case']} | {pl} | {x['S_w']:.4f} | {x['n_scored']} | {x['n_unscored']} | "
                     f"{x['envelope_score']:.4f} | {x['decision']} | {x['clean_acc']:.3f} |")
        L += ["", "Function preservation of the permutations, on 2000 fresh inputs:", "",
              "| case | max abs logit diff | mean abs logit diff | prediction agreement | accuracy diff | envelope-score diff | CKA vs unpermuted | JS vs unpermuted |",
              "|---|---|---|---|---|---|---|---|"]
        for x in r.reproduction["function_preservation"]:
            L.append(f"| {x['case']} | {x['max_abs_logit_diff']:.2e} | {x['mean_abs_logit_diff']:.2e} | "
                     f"{x['prediction_agreement']:.4f} | {x['accuracy_diff']:+.4f} | {x['envelope_score_diff']:.2e} | "
                     f"{x['min_cka_vs_unpermuted']:.6f} | {x['js_vs_unpermuted']:.2e} |")
    if r.unrelated_variants:
        L += ["", "## 10. Unrelated models under other training recipes (lineage only)", "",
              "| recipe | n | lineage admitted | S_w min / mean / max | highest min-layer |", "|---|---|---|---|---|"]
        for x in r.unrelated_variants:
            L.append(f"| {x['recipe']} | {x['n']} | {x['lineage_pass']} | {x['S_w_min']:.3f} / {x['S_w_mean']:.3f} / "
                     f"{x['S_w_max']:.3f} | {x['min_layer_max']:.3f} |")
    if r.probe_variation:
        L += ["", "## 11. Probe-set variation (envelope test, each probe draw calibrated separately)", "",
              "| probe seed | threshold | benign false blocks | non-adaptive detected | adaptive (public draw) detected | adaptive (exact pool) detected |",
              "|---|---|---|---|---|---|"]
        for x in r.probe_variation:
            L.append(f"| {x['probe_seed']} | {x['threshold']:.4f} | {x['benign_false_blocks']} | "
                     f"{x['non_adaptive_detected']} | {x['adaptive_public_draw_detected']} | "
                     f"{x['adaptive_exact_pool_detected']} |")
    L += ["", f"Total runtime: {r.runtime_seconds:.0f} s.", ""]
    return "\n".join(L)
