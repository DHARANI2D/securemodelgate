"""
Behavioral Fingerprinting Module
=================================
Generates a 128-dimensional fingerprint F(m) for each model:

  F(m) = [μ_l, σ_l, H(p_out)]  for sampled layers l

where:
  μ_l, σ_l = mean and std of post-activation outputs at layer l
  H(p_out) = Shannon entropy of output probability distribution
              over the canonical corpus

Detection: KL-divergence D_KL(F(m) || F_ref) vs threshold τ

τ = 0.05 nats is derived empirically from clean model variance.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy as scipy_entropy
from scipy.spatial.distance import cosine as cosine_dist
import copy


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BehavioralFingerprinter:
    """
    Extracts 128-dim behavioral fingerprint from model outputs
    on a fixed canonical test corpus, then computes KL-divergence
    from a clean reference distribution.
    """

    def __init__(self, dataloader, n_components=128, kl_threshold=0.05):
        self.dataloader    = dataloader
        self.n_components  = n_components
        self.kl_threshold  = kl_threshold
        self.device        = get_device()
        self.reference_fp  = None     # mean fingerprint of clean set
        self.reference_std = None     # std of clean fingerprints (for z-score)
        self._clean_fps    = []       # all clean fingerprints

    # ── Fingerprint extraction ────────────────────────────────────────────────

    def _extract(self, model_dict, max_batches=10):
        """
        Extract 128-dim fingerprint from model.

        Captures:
          - Output confidence distribution (mean/std/entropy per batch)
          - Penultimate layer activation statistics (if accessible)
          - Confidence histogram (16 bins)
        """
        model = model_dict["model"].to(self.device)
        model.eval()

        confidences = []
        entropies   = []
        top1_correct = []

        # Hook for penultimate layer activations
        penult_acts = []
        def hook_fn(module, inp, out):
            penult_acts.append(out.detach().cpu().float())

        # Register hook on avgpool (penultimate to fc)
        hook = model.avgpool.register_forward_hook(hook_fn)

        with torch.no_grad():
            for i, (imgs, labels) in enumerate(self.dataloader):
                if i >= max_batches:
                    break
                imgs = imgs.to(self.device)
                logits = model(imgs)
                probs  = F.softmax(logits, dim=1).cpu().float()

                conf = probs.max(dim=1)[0].numpy()
                confidences.extend(conf.tolist())

                # Shannon entropy per sample
                ent = scipy_entropy(probs.numpy(), axis=1)
                entropies.extend(ent.tolist())

        hook.remove()

        confidences = np.array(confidences)
        entropies   = np.array(entropies)

        # Build 128-dim vector:
        #   [0]    confidence mean
        #   [1]    confidence std
        #   [2]    mean entropy
        #   [3]    std entropy
        #   [4:20] confidence histogram (16 bins, 0-1)
        #   [20:84] penultimate layer activation stats (64 values)
        #   [84:128] class-wise probability means (10 classes × 4 stats)

        # Confidence histogram (16 bins)
        conf_hist, _ = np.histogram(confidences, bins=16, range=(0, 1), density=True)
        conf_hist = conf_hist / (conf_hist.sum() + 1e-10)

        # Penultimate layer stats
        if penult_acts:
            pa = torch.cat(penult_acts, dim=0).squeeze().numpy()  # (N, 512)
            if pa.ndim == 1:
                pa = pa.reshape(1, -1)
            # Sample 64 values: mean of 32 feature groups + std of 32 feature groups
            n_feat = pa.shape[1]
            group_size = max(n_feat // 32, 1)
            pa_means = np.array([pa[:, i*group_size:(i+1)*group_size].mean()
                                  for i in range(32)])
            pa_stds  = np.array([pa[:, i*group_size:(i+1)*group_size].std()
                                  for i in range(32)])
            pa_stats = np.concatenate([pa_means, pa_stds])  # 64 values
        else:
            pa_stats = np.zeros(64)

        # Class-wise probability means over corpus
        # (recompute without hook overhead - use stored logits)
        class_probs = np.zeros(10)
        with torch.no_grad():
            for i, (imgs, _) in enumerate(self.dataloader):
                if i >= max_batches:
                    break
                probs_batch = F.softmax(
                    model(imgs.to(self.device)), dim=1
                ).cpu().float().numpy()
                class_probs += probs_batch.mean(axis=0)
        class_probs /= (min(max_batches, len(self.dataloader)) + 1e-10)

        # Assemble 128-dim vector
        fp = np.concatenate([
            [confidences.mean()],       # 1
            [confidences.std()],        # 1
            [entropies.mean()],         # 1
            [entropies.std()],          # 1
            conf_hist,                  # 16
            pa_stats,                   # 64
            class_probs,                # 10  → total so far: 94
            class_probs ** 2,           # 10  → 104
            np.sort(class_probs),       # 10  → 114
            np.zeros(14),               # 14  → 128  (padding, deterministic)
        ])[:self.n_components]

        assert len(fp) == self.n_components, f"FP dim mismatch: {len(fp)}"
        return fp.astype(np.float32)

    # ── Reference building ────────────────────────────────────────────────────

    def build_reference(self, clean_models):
        """Compute reference fingerprint distribution from clean model set."""
        fps = []
        for m in clean_models:
            fp = self._extract(m)
            fps.append(fp)
        fps = np.stack(fps)
        self._clean_fps    = fps
        self.reference_fp  = fps.mean(axis=0)
        self.reference_std = fps.std(axis=0) + 1e-10

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _kl_divergence(self, fp):
        """
        Compute KL divergence between model's fingerprint distribution
        and clean reference, treating both as categorical distributions
        over the 128 fingerprint dimensions (normalised to sum to 1).
        """
        # Shift to positive, normalise
        ref = self.reference_fp - self.reference_fp.min() + 1e-6
        obs = fp - fp.min() + 1e-6
        ref = ref / ref.sum()
        obs = obs / obs.sum()
        return float(scipy_entropy(obs, ref))

    def _cosine_similarity(self, fp):
        return 1.0 - float(cosine_dist(fp, self.reference_fp))

    def score(self, model_dict):
        fp = self._extract(model_dict)
        kl = self._kl_divergence(fp)
        cs = self._cosine_similarity(fp)

        # Z-score distance from clean mean
        z = float(np.abs((fp - self.reference_fp) / self.reference_std).mean())

        flagged = kl > self.kl_threshold

        return {
            "fingerprint":       fp,
            "kl_divergence":     round(kl, 5),
            "cosine_similarity": round(cs, 5),
            "z_score":           round(z, 4),
            "fp_flagged":        flagged,
        }

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, clean_models, backdoored_models):
        if self.reference_fp is None:
            self.build_reference(clean_models)

        results = {"scores": [], "labels": [], "kl_clean": [], "kl_bd": []}
        tp = tn = fp_count = fn = 0

        for m, true_label in ([(m, 0) for m in clean_models] +
                               [(m, 1) for m in backdoored_models]):
            s = self.score(m)
            pred = 1 if s["fp_flagged"] else 0
            results["scores"].append(s)
            results["labels"].append({"true": true_label, "pred": pred,
                                       "id": m["id"]})
            if true_label == 0: results["kl_clean"].append(s["kl_divergence"])
            else:               results["kl_bd"].append(s["kl_divergence"])

            if   true_label == 1 and pred == 1: tp += 1
            elif true_label == 0 and pred == 0: tn += 1
            elif true_label == 0 and pred == 1: fp_count += 1
            elif true_label == 1 and pred == 0: fn += 1

        dr  = tp / (tp + fn)      if (tp + fn) > 0 else 0.0
        fpr = fp_count / (fp_count + tn) if (fp_count + tn) > 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp_count + fn)

        results["metrics"] = {
            "tp": tp, "tn": tn, "fp": fp_count, "fn": fn,
            "detection_rate": round(dr * 100, 1),
            "fpr":            round(fpr * 100, 1),
            "accuracy":       round(acc * 100, 1),
            "mean_kl_clean":  round(float(np.mean(results["kl_clean"])), 5),
            "mean_kl_bd":     round(float(np.mean(results["kl_bd"])),    5),
            "kl_threshold":   self.kl_threshold,
        }
        print(f"      Behavioral → DR: {dr*100:.1f}%  FPR: {fpr*100:.1f}%  "
              f"Acc: {acc*100:.1f}%  "
              f"KL(clean)={results['metrics']['mean_kl_clean']:.4f}  "
              f"KL(bd)={results['metrics']['mean_kl_bd']:.4f}")
        return results
