# Pre-Submission Checklist

This is the honest answer to "make sure there are no flaws and nothing
a reviewer could dismiss or reject this paper for." Two different
claims:

1. **Every flaw fixable inside this repository — without external
   fact-checking — has been found and fixed**, including a real bug the
   evaluation pilot itself caught (see
   [`06_engineering_validation_pilot.md`](06_engineering_validation_pilot.md)'s
   revision note), a missing bibliography entry for a claim used twice
   in the abstract, a dangling "hypothesis... tested" that never said
   what the test found, and several factual claims that had no
   bibliography entry at all. All of that is done.
2. **"Zero possible reviewer questions" is not a claim this document
   makes**, because several remaining items depend on facts outside
   this session's reach: this container has no general web access (only
   PyPI/npm/a few package registries — pypi.org and
   files.pythonhosted.org are reachable, `download.pytorch.org`,
   `huggingface.co` and `www.cs.toronto.edu` are not, per this
   session's agent-proxy policy), so several citations added in this
   pass could not be checked against a primary source. Claiming they
   were verified would recreate exactly the failure mode
   ([`05_critical_review_source.md`](05_critical_review_source.md)'s
   fabricated "L. Dex" author) that this whole revision responds to.
   What follows is the itemized, actionable list instead of a vague
   disclaimer.

Work through this in order. Nothing here is optional if the goal is
zero dismissible points.

## 1. Citations — must verify before submission

Six references were added in this pass because the claims they support
were previously stated in the paper's body with **no bibliography entry
at all** — that is a citation-hygiene gap any careful reviewer finds
immediately, worse than a citation with a wrong detail. They now have
placeholder entries ([35]–[39] in `02_paper.md`'s reference list) that
are honest about not being verified. Confirm each, in this priority
order (highest-stakes first):

| # | Claim | What to check |
|---|---|---|
| 35 | "Model Atlas (Horwitz et al.)... more than 60% of Hub models carry no documented parentage" (used in `01_abstract.md` AND `02_paper.md`) | Find the actual Model Atlas paper (arXiv ID, venue, year, correct author list). It's currently cited only as "reported by modelDNA" — find and cite the primary source directly. |
| 36 | "NVIDIA's Developer Blog... signed all NVIDIA-published NGC Catalog models with OMS since March 2025" | Find the actual blog post URL and publish date. |
| 37 | "JFrog says its scanner eliminates more than 96% of the false positives..." (used twice) | Find the JFrog blog post/press release with this figure. |
| 38 | "Palo Alto Networks completed its acquisition of Protect AI... July 2025... FY2025 10-K" | Find the actual 10-K filing or press release with the purchase-consideration figures. |
| 39 | "Cisco AI Defense includes 'AI Supply Chain Risk Management'" | Find Cisco's current AI Defense product documentation. |
| — | The "SIG" backdoor attack named in Section 2's A2 | **No placeholder reference exists for this at all** — the paper deliberately withdrew a guessed citation (Barni/Kallas/Tondi, ICIP 2019) rather than risk it being wrong. Confirm which paper "SIG" actually refers to in the BackdoorBench/backdoor-attack literature and add a real entry before submission, or drop "SIG" from the attack list in Section 2/Section 4 if it can't be confirmed in time. |
| 40 | EO 14028 provisions, and the June 2025 EO 14144 amendment interaction described in `05_critical_review_source.md` | Re-check against the current Federal Register text; the amendment landscape here has moved before. |

Already-flagged from the previous revision (still open):
- HuRef's and REEF's venue/author-list details (confirmed only via
  secondary indexes).
- The NIST AI 600-1 DOI.
- BTI-DBF has no arXiv version — confirm the OpenReview ID is current.
- modelDNA, Centered Residual Signatures, the weights-only LoRA
  detection preprint, and the Tan et al. FSE '26 companion paper are
  all preprints as of this writing — re-confirm they're still
  unpublished (or update the citation) before submission.

**How to verify:** this session's container blocks general web access.
Use a session/environment with normal browsing, or verify manually and
paste the confirmed citations back in. Do not guess a DOI or arXiv ID
to fill a gap — an invented-looking citation is worse than a `[35]`
that visibly says "unconfirmed."

## 2. Tech Con process — must confirm internally

From [`05_critical_review_source.md`](05_critical_review_source.md)
part (a), these are NOT publicly documented and need an answer from the
Tech Con program office or a sponsoring Distinguished
Technologist/Fellow:
- Is a Parasparam-non-selected idea eligible for Tech Con resubmission?
- Is there a cap on primary-author submissions?
- What are the actual abstract word limits and scoring rubric?

## 3. Internal approvals — must confirm before anything external

- The predicate URI domain `https://securemodelgate.hpe.com/lineage/v0.1`
  is not an approved, registered HPE domain — get it approved or change
  it before this predicate type is used anywhere real.
- Every HPE product claim (Private Cloud AI, AI Essentials, Private
  Cloud/Morpheus, GreenLake Intelligence, OpsRamp integration) is
  described as **proposed**, not shipped. Confirm Product/Legal is fine
  with that framing before this goes to reviewers who might read it as
  a roadmap commitment.

## 4. Timeline

This checklist and the paper it accompanies were last touched
2026-09-29. `02_paper.md`'s Section 7 targets "By Oct 2 2026" for the
4-day core evaluation (E1–E9). **That is now 3 days out, and as of this
same date the blocker in §5 is still open** despite one attempt to
clear it. If E1–E9 cannot be run by then:
- Follow `02_paper.md`'s own "Time budget" caveat: submit with only
  E2–E4 filled and say plainly that the rest is pending — do not slip
  the deadline silently or backfill placeholder numbers with guesses.
- Re-date Section 7's roadmap to match whatever the real submission
  date ends up being; a roadmap with a blown deadline still showing as
  upcoming is itself a small credibility gap worth avoiding.

## 5. The evaluation itself

This is the big one, and it's not a checklist item so much as the
actual remaining work: Experiments E1–E9
([`04_experiments_plan.md`](04_experiments_plan.md)) need a GPU-capable
environment with access to CIFAR-10 and, ideally, Hugging Face. The
engineering-validation pilot
([`06_engineering_validation_pilot.md`](06_engineering_validation_pilot.md)),
adaptive-attack pilot
([`07_adaptive_attack_pilot.md`](07_adaptive_attack_pilot.md)), and
permutation-robustness pilot
([`09_permutation_robustness_pilot.md`](09_permutation_robustness_pilot.md))
prove the pipeline's statistical machinery is implemented correctly and
are real, reproducible, committed results — but they are pilots on a
synthetic corpus and a 4-stage CNN, not the paper's claimed
PreAct-ResNet-18/CIFAR-10/BackdoorBench protocol. No amount of further
engineering work inside this container substitutes for that run.

**Status as of 2026-09-29, tried three times, including in a fresh
session:** this working environment (a Claude Code Remote "Default —
trusted network access" `anthropic_cloud` container) has no GPU
(`torch.cuda.is_available()` returns `False`, `nvidia-smi` doesn't
exist) and its network policy denies `huggingface.co`,
`download.pytorch.org` and `www.cs.toronto.edu` with a
`connect_rejected`/403 at the egress proxy. `list_environments` on this
account shows only two environments, both this same CPU-only "Default"
kind — no GPU-tagged option exists to switch to. Widening this
environment's network access (cloud environment menu → Edit → Network
access, adding those three hosts or broadening the access level) was
requested and attempted; a re-test in this session immediately after
showed the exact same denial. **To rule out a stale per-session cache,
a completely separate sibling session was then spun up from scratch in
the same environment** and ran the identical checks independently:
same result on every count — no CUDA, no `nvidia-smi`, all three hosts
still blocked "by org proxy." A fresh session does not fix this. It is
a structural property of the environment (no GPU) and an
organization-level network policy (not a per-environment or
per-session setting the "Network access" UI can override), not a cache
to clear.

**What actually unblocks this**, in order of most to least preferred:
1. Someone with access to the organization's network-policy
   configuration (not just this environment's own settings) allow-lists
   `huggingface.co`, `download.pytorch.org`, and a CIFAR-10 source —
   this environment's own "Edit → Network access" control was tried and
   did not change the outcome, so the policy likely lives a level above
   what that control governs.
2. Run E1–E9 on separate GPU compute the person controls (a cloud GPU
   instance, a local machine, Colab, etc.) — confirmed necessary
   regardless of (1), since no environment on this account has a GPU
   option at all. Use [`04_experiments_plan.md`](04_experiments_plan.md)
   and the `securemodelgate` package's tested harnesses
   (`evaluation.py`/`adaptive_evaluation.py`/`permutation_check.py`
   patterns generalize directly to a real CIFAR-10/ResNet-18 setup —
   swap `demo_models.py`'s synthetic corpus for BackdoorBench's, keep
   everything downstream of that the same) as the starting point rather
   than starting from scratch.
3. If neither is available before submission, follow this checklist's
   §4 (Timeline) and `02_paper.md`'s own "Time budget" caveat: submit
   with the placeholders honestly still open.

## 6. Final mechanical sanity pass

Before submitting, from the repo root:

```bash
# Every [N] in the paper body should resolve to a real, single-use-per-claim reference,
# and every reference should be cited at least once. (Already verified clean as of this
# revision; re-run after any further edits.)
python3 -c "
import re
text = open('paper/techcon2027_revision/02_paper.md').read()
ref_section = text.split('## References')[1].split('## Caveats')[0]
defined = set(int(m) for m in re.findall(r'^(\d+)\.\s', ref_section, re.MULTILINE))
body = text.split('## References')[0]
used = set()
for m in re.finditer(r'\[(\d+(?:,\s*\d+)*)\]', body):
    for n in m.group(1).split(','):
        used.add(int(n.strip()))
print('unused refs:', sorted(defined - used))
print('dangling citations:', sorted(used - defined))
"

# The full test suite should be green.
pytest tests/ -v

# The author's name should be spelled exactly as their HPE directory record —
# this revision uses "Dharanidharan Senthilkumar" throughout; the original
# PDF had "Dhaaranidharan" in one place. Check ONLY the actual submission
# files (03_change_log.md and this checklist mention the old spelling
# deliberately, to document the fix -- that's expected, not a bug):
grep -n "Dhaaranidharan" paper/techcon2027_revision/01_abstract.md paper/techcon2027_revision/02_paper.md \
  && echo "FOUND STALE SPELLING IN THE ACTUAL SUBMISSION" || echo "clean"
```
