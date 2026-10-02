# SecureModelGate — Evaluation Pipeline
## Parasparam 2026 — Single-Command Experiment Runner

---

## What this does

Runs a complete, reproducible evaluation of the SecureModelGate prototype:

- Trains 20 clean + 20 backdoored ResNet-18 models on CIFAR-10
- Runs static weight analysis (KS-test + layer variance)
- Runs behavioral fingerprinting (128-dim KL-divergence)
- Issues Model Attestation Tokens (JWT signed)
- 3-run evaluation with mean ± std
- Ablation study (static-only, behavioral-only, combined)
- Latency benchmarking
- Generates 7 publication-ready figures + paper metrics text

---

## Single command

```bash
python run_all.py
```

That is it. Everything else is automated.

---

## Setup (Mac M1 Air — recommended)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies (MPS acceleration auto-detected on M1)
pip install -r requirements.txt

# 3. Run everything
python run_all.py
```

Expected runtime on M1 Air: **8–15 minutes**
PyTorch will automatically use MPS (Apple Silicon GPU acceleration).

---

## Setup (Google Colab — free T4 GPU)

```python
# Paste in a Colab cell:
!git clone <your-repo-url> securemodelgate
%cd securemodelgate
!pip install -r requirements.txt -q
!python run_all.py
```

Expected runtime on Colab T4: **4–8 minutes**

---

## Setup (any Linux / cloud VM)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

---

## Outputs

All results saved to `./results/`:

| File | Contents |
|------|----------|
| `report.json` | All metrics — paste directly into paper |
| `paper_metrics.txt` | Formatted text for paper |
| `fig1_roc.png` | ROC curve |
| `fig2_comparison.png` | DR + FPR comparison chart |
| `fig3_confusion.png` | Confusion matrix |
| `fig4_kl_dist.png` | KL-divergence distributions |
| `fig5_ablation.png` | Ablation study |
| `fig6_latency.png` | Latency breakdown |
| `fig7_trigger_breakdown.png` | Per-trigger detection rates |

---

## Dataset

Uses CIFAR-10 (auto-downloaded on first run, ~170 MB).
No NIST TrojAI download required — backdoored models are
generated locally using BadNets-style poisoning.

---

## Reproducibility

All random seeds are fixed (seed=42).
Run `python run_all.py` twice — you will get identical results.

---

## Architecture

```
run_all.py              ← single entry point
src/
  setup.py              ← environment check
  models.py             ← clean + backdoored model generation
  static.py             ← KS-test weight analysis
  fingerprint.py        ← 128-dim behavioral fingerprinting
  attestation.py        ← MAT token + MBOM generation
  evaluator.py          ← multi-run eval, ablation, latency
  reporter.py           ← figures + paper metrics output
results/                ← all outputs (auto-created)
data/                   ← CIFAR-10 cache (auto-created)
```

This pipeline is unchanged, and it is what produced the retained
Parasparam 2026 numbers (20/20 detection, 3/20 false blocks) cited in
the revised Tech Con 2027 paper as prior results.

## The Tech Con 2027 revision lives in `../securemodelgate/`

The revised methodology described in
`../paper/techcon2027_revision/02_paper.md` Section 3 — signed
behavioral-lineage attestation against a declared, OMS-signed base
model instead of this prototype's averaged "clean" set — is **not**
in this directory. It's a separate, installable, tested Python package
at the repo root: `../securemodelgate/` (`pip install -e ".[dev]"` from
the repo root, then `securemodelgate demo`). See its own
`../README.md` for the CLI, architecture and test suite.
