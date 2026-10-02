import numpy as np
import pytest

from securemodelgate.stats import auroc, clopper_pearson, tpr_at_fpr


def test_clopper_pearson_matches_paper_cited_intervals():
    # paper/techcon2027_revision/02_paper.md, Section 4: "3/20 = 15% (95% CI 3-38%)"
    p = clopper_pearson(3, 20, alpha=0.05)
    assert p.point_estimate == pytest.approx(0.15)
    assert p.ci_low == pytest.approx(0.032, abs=0.001)
    assert p.ci_high == pytest.approx(0.379, abs=0.001)

    # "20/20 = 100% (95% Clopper-Pearson CI 83-100%)"
    p2 = clopper_pearson(20, 20, alpha=0.05)
    assert p2.point_estimate == 1.0
    assert p2.ci_low == pytest.approx(0.832, abs=0.001)
    assert p2.ci_high == 1.0


def test_clopper_pearson_zero_successes_lower_bound_is_zero():
    p = clopper_pearson(0, 20, alpha=0.05)
    assert p.ci_low == 0.0
    assert p.ci_high > 0.0


def test_clopper_pearson_rejects_bad_inputs():
    with pytest.raises(ValueError):
        clopper_pearson(5, 0)
    with pytest.raises(ValueError):
        clopper_pearson(-1, 10)
    with pytest.raises(ValueError):
        clopper_pearson(11, 10)


def test_auroc_perfect_separation_is_one():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert auroc(labels, scores) == pytest.approx(1.0)


def test_auroc_inverted_scores_is_zero():
    labels = [0, 0, 1, 1]
    scores = [0.9, 0.8, 0.2, 0.1]
    assert auroc(labels, scores) == pytest.approx(0.0)


def test_auroc_requires_both_classes():
    with pytest.raises(ValueError):
        auroc([0, 0, 0], [0.1, 0.2, 0.3])


def test_tpr_at_fpr_perfect_separation_is_one_everywhere():
    labels = [0] * 20 + [1] * 20
    scores = list(np.linspace(0, 0.4, 20)) + list(np.linspace(0.6, 1.0, 20))
    assert tpr_at_fpr(labels, scores, target_fpr=0.05) == pytest.approx(1.0)


def test_tpr_at_fpr_no_separation_is_roughly_target_fpr():
    rng = np.random.default_rng(0)
    labels = [0] * 200 + [1] * 200
    scores = rng.normal(size=400)  # scores carry no signal about labels
    tpr = tpr_at_fpr(labels, scores, target_fpr=0.1)
    assert 0.0 <= tpr <= 0.35  # loose bound; this is a randomized sanity check, not an exact one
