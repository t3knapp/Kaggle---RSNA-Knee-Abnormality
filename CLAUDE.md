# CLAUDE.md — RSNA Knee Abnormality Detection

## Project Purpose

This project exists to **learn applied deep learning for medical imaging** — multimodal (3D MRI + radiology report text) multi-label classification specifically. It is not a leaderboard-chasing project.

**Doing well in the competition is a side effect of doing the learning well, not the goal.** If a choice trades away understanding for a quick score bump (e.g. copying a public notebook's architecture without understanding why it works, or blindly tuning hyperparameters via grid search without forming a hypothesis first), the correct move is to flag the tradeoff and default toward the learning path unless the user explicitly says otherwise. Every non-trivial modeling decision should be traceable to a documented reason, not just "it scored better."

The competition itself: **RSNA Knee Abnormality Detection** (hosted by RSNA + Kaggle, 2026). Multi-label classification of twelve clinically important abnormalities on knee MRI exams, using both DICOM image volumes and paired (multilingual) radiology report text — the first RSNA challenge to combine imaging with report text for training. ~5,000+ exams from 16–19 global sites. Entry deadline October 15, 2026.

## Documentation Standard

Every design and model decision gets written up in `DESIGN.md` (or dated entries within it) at a **teach-a-mathematically-literate-beginner** level of detail. Assume the reader:

- Has a solid math background (linear algebra, probability, calculus, comfortable with proofs and notation) — so do NOT hand-wave the math or avoid equations.
- Has little to no prior exposure to deep learning / modeling practice — so DO explain things a practitioner would take for granted: why a particular loss function fits a particular label structure, what a receptive field is and why it matters here, why batch norm interacts with small batch sizes, why 3D volumes get sliced/pooled a particular way, what a validation scheme is actually protecting against, etc.

For each significant decision, the writeup should cover:
1. **What we're deciding** — stated precisely.
2. **The options considered** — not just the one we picked.
3. **The reasoning** — derived from first principles where possible, connecting to the underlying math (e.g. if we pick focal loss for class imbalance, derive why cross-entropy struggles with imbalance and what focal loss's gamma term is actually doing to the gradient).
4. **Why alternatives were rejected** — specific, not "it performed worse."
5. **Open questions / what we'd revisit** if evidence changes.

This is the same discipline as the ADR protocol from other projects: a decision without a written "why" doesn't get to persist in the codebase.

## Kaggle Interaction Rules — API Only, No Bulk Downloads

This project interacts with Kaggle **exclusively through the Kaggle API**, using the push-notebook workflow:

- Code is written and iterated on **locally** (via Claude Code), tested against small synthetic/dummy data matching the real folder structure.
- Execution against the real competition data happens by **pushing kernels/notebooks to Kaggle** (`kaggle kernels push`), where the data is mounted remotely at `/kaggle/input/...`. We read logs and outputs back via the API (`kaggle kernels status`, `kaggle kernels output`).
- **The full competition dataset (hundreds of GB) is never downloaded locally without the user's explicit, in-the-moment approval.** This is a hard rule, not a default-to-ask-forgiveness one. If a task seems to require local access to the full dataset, stop and ask first — don't assume a workaround justifies it.
- Small reference artifacts are fine to pull locally when useful for local dev/debugging: a handful of sample DICOMs, label distributions, metadata CSVs, tiny public "notebooks/figures" datasets, model weight outputs (typically MB, not GB) saved back from a kernel run as a Kaggle Dataset. These are not "the dataset" and don't need special approval — but if unsure whether something crosses into bulk-download territory, ask.
- Kaggle's GPU quota (weekly, shared across kernels) is a real constraint. Debugging cycles should be minimized by validating logic locally on dummy data before spending quota on a push.

## Session Continuity

Follow the existing SESSION.md checkpoint protocol: at the end of a working session, log what was decided, what's in progress, and what the next step is, so a fresh session can resume without re-deriving context. Cross-reference ADRs from SESSION.md rather than duplicating their content.
