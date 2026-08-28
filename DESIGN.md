# DESIGN.md — RSNA Knee Abnormality Detection

Decision record for this project. Every significant modeling or design choice gets
an entry here, written at the level described in `CLAUDE.md`: assume a reader who is
mathematically fluent but new to deep learning practice.

Each entry follows the five-part structure from `CLAUDE.md`:
1. What we're deciding
2. Options considered
3. Reasoning (from first principles, with the math where it matters)
4. Why alternatives were rejected
5. Open questions / what would make us revisit

Where a decision was preceded by a prediction (per the predict-then-reveal rule),
the prediction and the actual outcome both get recorded — the gap is the lesson.

---

## Status legend

- **`[ ]` Not started** — section exists to mark that a decision is owed.
- **`[~]` In progress** — actively being worked through.
- **`[x]` Decided** — written up; revisit only with new evidence.

Sections are ordered roughly by dependency: earlier decisions constrain later ones.
That ordering is itself provisional — if working through a section reveals it depends
on something below it, reorder rather than guess.

---

## Part I — Understanding the problem

### 1. `[ ]` Problem framing and task definition
What exactly are we predicting, for what unit of analysis, from what inputs?
Nails down: study-level vs. series-level prediction, the twelve labels as a
multi-label (not multi-class) problem, and what "abnormality detection" means
operationally here.

### 2. `[ ]` Evaluation metric and what it rewards
**Confirmed metric:** the macro-averaged AUC ROC — the area under the ROC curve is
computed between predicted confidence scores and observed targets for each of the twelve
targets separately, then averaged with equal weight across them.

The section still needs writing: what that metric *implies* for how we train. Covers:
why a ranking metric makes only the ordering of predictions matter and not their
absolute values, whether calibration is therefore worth anything to us, how equal-weight
averaging over twelve labels interacts with rare classes, and what a trivial baseline
scores.

### 3. `[x]` Data inventory and structure
What is actually in the dataset. **Written up separately in
[`docs/DATA.md`](docs/DATA.md)** because it is descriptive fact-finding rather than
a decision; this section is a pointer to it and a home for later revisions.

---

## Part II — Supervision: the central problem

### 4. `[ ]` Label strategy — where training labels come from
The competition ships ~1.3% of studies with labels and the rest with only free-text
reports (see `docs/DATA.md`). Nothing downstream can be designed until we decide how
to turn reports into training targets. Covers: rule-based extraction vs. an LLM
labeller vs. training directly on text embeddings, how to validate a labeller when
the gold set is 58 studies, and how label noise propagates into the image model.

### 5. `[ ]` Handling label noise and uncertainty
Given labels will be *derived*, they will be wrong some of the time. Covers: soft vs.
hard targets, whether to weight examples by labeller confidence, and which loss
functions are robust to asymmetric label noise.

### 6. `[ ]` Role of text at training vs. inference time
Reports exist for training studies but **not** at test time. Covers: whether text is
purely a label source, or also a training-time auxiliary signal (distillation,
contrastive image-text alignment), and what each choice buys.

---

## Part III — Getting images into a model

### 7. `[ ]` Series selection and volume construction
Each study has 3–14 series across three planes. Covers: which series to feed the
model, how to handle variable series counts, and whether to treat planes as separate
inputs, channels, or independent samples.

### 8. `[ ]` Preprocessing and intensity normalization
MRI intensities are not physically standardized the way CT Hounsfield units are.
Covers: per-volume vs. per-dataset normalization, resampling to isotropic spacing,
resolution/crop targets, and what each choice costs in signal vs. compute.

### 9. `[ ]` Augmentation strategy
Which invariances we want the model to have, and which augmentations would destroy
label-relevant signal (e.g. left-right flips vs. medial/lateral labels).

---

## Part IV — The model

### 10. `[ ]` Image encoder architecture
2D-per-slice + aggregation vs. 2.5D vs. full 3D convolution vs. transformer variants.
Covers: receptive field and why it matters for lesion scale, parameter/compute budget
under Kaggle limits, and pretrained-weight availability.

### 11. `[ ]` Multi-plane / multi-series fusion
How information from sagittal, coronal, and axial views gets combined into one
study-level prediction. Covers: early vs. late fusion, attention-based pooling over
series, and permutation invariance over a variable-length series set.

### 12. `[ ]` Transfer learning and pretraining
Whether to start from ImageNet / video / medical-pretrained weights, and the
domain-gap argument for each.

---

## Part V — Training

### 13. `[ ]` Loss function and class imbalance
Twelve binary targets with (likely) low and unequal prevalence. Covers: BCE vs. focal
vs. weighted variants, derived from what each does to the gradient of a rare positive,
and the interaction with the evaluation metric from §2.

### 14. `[ ]` Validation scheme
What the split is protecting against. Covers: site-level and patient-level leakage,
stratification for rare multi-label targets, and whether the 58 gold-labeled studies
should be held out as a clean evaluation set or spent on training.

### 15. `[ ]` Optimizer, learning-rate schedule, and batch size
Covers: why AdamW-family optimizers dominate here, warmup and decay rationale, and how
small batch sizes interact with normalization layer choice.

### 16. `[ ]` Regularization and early stopping
Covers: weight decay, dropout placement, and the stopping criterion given a noisy
validation signal on rare classes.

---

## Part VI — Execution

### 17. `[ ]` Training infrastructure and Kaggle constraints
Working within kernel runtime limits, weekly GPU quota, and the push/pull workflow.
Covers: checkpointing across runs, caching preprocessed data as Kaggle Datasets, and
how to keep debug cycles off the GPU.

### 18. `[ ]` Inference and submission pipeline
Covers: test-time augmentation, ensembling, threshold/calibration handling, and
runtime budget for the hidden test set.

### 19. `[ ]` Experiment log
Dated record of runs: hypothesis, prediction, result, and what changed as a result.
Distinct from the decision entries above — this is the evidence trail those cite.

---

## Open questions (project-wide)

Tracked here when they don't belong to a single section yet.

- **Hidden test set size and composition are unknown.** The published `test.csv` is a
  3-study stub. Prevalence, site mix, and language mix of the real test set are all
  unknown, which limits how much we can trust any prevalence estimate.
- **Whether reports are truly absent at inference.** Strongly implied by `test.csv`
  having no `Report` column, but worth confirming against the competition's data
  description before committing to §6.
