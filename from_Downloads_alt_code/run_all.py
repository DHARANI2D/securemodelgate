#!/usr/bin/env python3
"""
SecureModelGate — Full Evaluation Pipeline
==========================================
Single command: python run_all.py

Runs:
  1. Model generation (clean + backdoored)
  2. Static analysis evaluation
  3. Behavioral fingerprinting evaluation
  4. Combined SecureModelGate evaluation
  5. Ablation study
  6. Latency benchmarking
  7. Full metrics + charts output

Results saved to: results/
"""

import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.setup      import check_environment, print_banner
from src.models     import ModelFactory
from src.static     import StaticAnalyzer
from src.fingerprint import BehavioralFingerprinter
from src.attestation import MATIssuer
from src.evaluator  import Evaluator
from src.reporter   import Reporter

def main():
    print_banner()
    check_environment()

    print("\n[1/7] Generating model dataset...")
    factory = ModelFactory(n_clean=20, n_backdoored=20, n_runs=3, seed=42)
    clean_models, backdoored_models, dataloader = factory.build()
    print(f"      Clean: {len(clean_models)}  Backdoored: {len(backdoored_models)}")

    print("\n[2/7] Running static analysis...")
    static = StaticAnalyzer()
    static_results = static.evaluate(clean_models, backdoored_models)

    print("\n[3/7] Running behavioral fingerprinting...")
    fingerprinter = BehavioralFingerprinter(dataloader, n_components=128)
    fingerprinter.build_reference(clean_models)
    fp_results = fingerprinter.evaluate(clean_models, backdoored_models)

    print("\n[4/7] Running combined SecureModelGate...")
    issuer = MATIssuer()
    evaluator = Evaluator(static, fingerprinter, issuer)
    combined_results = evaluator.evaluate(clean_models, backdoored_models, n_runs=3)

    print("\n[5/7] Running ablation study...")
    ablation = evaluator.ablation(clean_models, backdoored_models)

    print("\n[6/7] Benchmarking latency...")
    latency = evaluator.benchmark_latency(clean_models[:5] + backdoored_models[:5])

    print("\n[7/7] Generating report + charts...")
    reporter = Reporter(output_dir="results")
    reporter.generate(
        static_results=static_results,
        fp_results=fp_results,
        combined_results=combined_results,
        ablation=ablation,
        latency=latency,
        clean_models=clean_models,
        backdoored_models=backdoored_models,
    )

    print("\n" + "="*60)
    print("  DONE — results saved to ./results/")
    print("  Open results/report.json for all paper metrics")
    print("  Open results/*.png for figures")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
