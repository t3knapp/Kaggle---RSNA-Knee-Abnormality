# Literature Scoping — Learning Image Labels from Radiology Reports

**Status:** first pass, 2026-08-28. Written to inform `DESIGN.md` §4 (label strategy),
§5 (label noise), and §6 (role of text). This is a survey, not a decision — decisions
citing it belong in `DESIGN.md`.

**Reading caveat:** where a claim comes from a paper I read directly, it is cited with
specific numbers. Where it comes from an abstract or a secondary summary, it is marked
*(summary only)* — those should be verified before anything load-bearing rests on them.

---

## 0. The problem, stated precisely

We have 4,349 knee MRI studies with a free-text report and no labels, 58 studies with
labels, and a test set with images only. We know from the competition organizers that
**the labels come from an independent read of the images, not from the reports.**

So the pipeline we need is:

$$\text{report} \;\longrightarrow\; \hat{y} \;\longrightarrow\; \text{train } f(\text{images}) \to y$$

and the central question is how much fidelity is lost in that first arrow. Formally, let
$y_{\text{img}}$ be the annotator's image-derived label (the competition target) and
$y_{\text{txt}}$ be whatever a perfect reader could extract from the report. These are
**different random variables**. Our training labels can at best equal $y_{\text{txt}}$,
but we are scored against $y_{\text{img}}$.

This exact gap has a literature. It is not a small effect.

---

## 1. The report-image gap is real, large, and mechanistically understood

### Olatunji et al. 2019, *Caveats in Generating Medical Imaging Labels from Radiology Reports* ([arXiv:1905.02283](https://arxiv.org/abs/1905.02283))

The cleanest quantification. 1,000 chest X-ray studies, **two separate groups of expert
radiologists**: group 1 labeled from images only, group 2 from reports only. Agreement
between them, measured as F1 of report labels against image labels:

| Finding | F1 (report labels vs. image labels) |
|---|---:|
| Global abnormal | 0.69 |
| Cardiomegaly | **0.17** |
| Consolidation | 0.45 |
| Foreign body / device | 0.59 |

Read that Cardiomegaly row again. Two boards-certified radiologists, same 1,000 studies,
one reading images and one reading the reports *about those images* — F1 of 0.17. And in
aggregate, **24% of reports labeled normal disagreed with image annotations**, while
**20% of reports labeled abnormal had no abnormality on image review**.

Their failure analysis names the mechanisms, and this is the part that matters for us:

1. **Non-actionable findings** — *"in the overwhelming majority of disagreement, the
   reporting radiologist documents only findings relevant to the immediate clinical
   context... and ignores non-actionable findings."* The reporting radiologist writes for
   a referring clinician; the labeling radiologist annotates exhaustively for model
   training. **These are different jobs with different outputs.**
2. **Borderline findings** — the reporter hedges toward mentioning; the labeler leans
   toward ignoring. Systematic, opposite-direction bias.
3. **Anatomic variation and technical factors** — obscured or exaggerated findings.
4. **Outright error** — either party misses things.

**This explains our own data directly.** Recall the two studies where `Synovitis = 1` but
the word never appears in the report: that is mechanism (1). Synovitis wasn't the clinical
question, so the radiologist didn't write it down — but the annotator scoring images
marked it. And Study 1's *"Moderate joint effusion"* → `Effusion = 0` is mechanism (2):
the reporter mentions, the labeler declines to count it.

### Jain, Smit et al. 2021, *VisualCheXbert* ([arXiv:2102.11467](https://arxiv.org/abs/2102.11467), ACM CHIL '21)

Same finding, and then a method. They report Cohen's kappa between radiologists labeling
reports and radiologists labeling images averaging **0.312–0.430** across conditions —
"fair" agreement at best on the standard interpretation scale.

Then they ask our exact question: *can we learn to map reports to image labels?*

| Method | Avg F1 | Weighted avg F1 |
|---|---:|---:|
| Zero-One baseline (rule-based labeler, best uncertainty mapping) | 0.54 | 0.56 |
| LogReg on rule-based labeler outputs | 0.65 | 0.70 |
| **VisualCheXbert** (BERT → image labels) | **0.68** | **0.73** |

Two results worth internalizing:

- Going from a rule-based report labeler to a *learned* report→image-label mapping is
  worth **+0.14 average F1**. The gap is largely closable.
- VisualCheXbert **agrees with image-labeling radiologists better than report-labeling
  radiologists do** — by 0.12–0.21 F1. A learned mapping beats a human expert reading the
  same report, because it has learned the systematic biases of report-writing.

### The VisualCheXbert mechanism — the key idea to steal

Their training signal is the clever part, and it is worth spelling out because it is not
obvious.

They train a BERT to map report text → labels, but the supervision does **not** come from
what the report says. It comes from a **computer vision model (DenseNet) run on the paired
X-ray image**. The BERT is trained to predict what the vision model sees.

Why does this work when the vision model was itself trained on noisy report-derived labels?
Because of an asymmetry in what is learnable:

> A vision model can only fit label variation that correlates with image content. Findings
> that appear in reports but are invisible in the image are, from the model's perspective,
> **noise uncorrelated with its inputs** — with adequate regularization it cannot fit them,
> and its predictions regress toward what is actually visible.

So the vision model functions as a **visual filter**: pass noisy text labels through it,
and what comes out the other side is grounded in pixels whether you asked for that or not.
That output is then a *better* target for the text labeller than the report's own content.

This is a bootstrap that does not require a large gold set — which matters enormously for
us, because we have 58.

---

## 2. How to extract labels from reports (the first arrow)

A difficulty ladder, roughly chronological:

### Rule-based: NegBio, CheXpert labeler
The [CheXpert labeler](https://github.com/stanfordmlgroup/chexpert-labeler) runs three
stages — **mention extraction**, **mention classification** (positive / negative /
uncertain), and **mention aggregation** to a per-condition label *(summary only)*. It
emits four classes per condition: positive, negative, uncertain, blank.

The uncertainty handling strategies from Irvin et al. are worth knowing by name because
they recur everywhere: **U-Ignore** (mask uncertain examples from the loss), **U-Zeros** /
**U-Ones** (map to negative / positive), **U-MultiClass** (third class), **U-SelfTrained**
(soft labels from the model's own predictions) *(summary only)*.

**Relevance to us: low as a tool, high as a design template.** These systems are
English-only and hand-built per finding; our reports span Latin, Greek, and Cyrillic. But
the three-stage decomposition and the explicit uncertainty class are both worth copying.

### Learned: CheXbert
[CheXbert](https://arxiv.org/abs/2004.09167) distills the rule-based labeler into a BERT,
then fine-tunes on a small set of expert-annotated reports — combining cheap noisy
supervision at scale with a little expensive clean supervision *(summary only)*. The
structural lesson (bulk noisy + small clean) maps directly onto our 4,349 + 58 split.

### LLM-based (current state of the art)
Multiple 2024–2025 studies find LLMs match or beat purpose-built labelers zero-shot. One
[JMIR Medical Informatics study](https://medinform.jmir.org/2025/1/e68618) reports GPT-based
labeling at **F1 0.9014** vs. CheXpert labeler **0.8864** and CheXbert **0.9047**, with the
LLM's advantage concentrated in *"pathologies with longer and more complex descriptions"*
*(summary only)*. A [Radiology study](https://pubs.rsna.org/doi/10.1148/radiol.241139)
compares commercial vs. open-source LLMs for the same task. [RadPrompt](https://arxiv.org/abs/2408.04121)
explores injecting rule-based insight into LLM prompts.

**Relevance to us: high.** Multilingual competence is exactly where an LLM dominates a
regex, and our reports are 12% non-Latin-script. And per the competition rules, offline LLM
labeling is permitted so long as nothing calls an API at kernel runtime — which costs us
nothing, since reports don't exist at test time anyway.

### Weak supervision: Snorkel / data programming
[Snorkel](https://arxiv.org/abs/1711.10160) takes many individually-unreliable *labeling
functions* (regexes, heuristics, knowledge bases, model outputs) and fits a generative model
over their agreements and disagreements to estimate each one's accuracy **without ground
truth**, emitting probabilistic labels *(summary only)*.

**Relevance to us: medium.** The formalism is exactly right for combining, say, a regex
labeller + an LLM labeller + a translated-then-labelled variant. Whether the machinery earns
its complexity at our scale is an open question.

---

## 3. Training a model on labels you know are wrong

Once labels are noisy, the loss function matters. The theory here is clean:

A loss is **noise-tolerant** if it satisfies a symmetry condition — informally, if
$\sum_{k} \ell(f(x), k)$ is constant over classes $k$, then the risk minimizer under
uniform label noise is the same as under clean labels. Mean Absolute Error satisfies it;
cross-entropy does not. But MAE is notoriously hard to optimize (vanishing gradients on
confident-wrong examples), so the literature is a series of interpolations between the two:

- **Generalized Cross Entropy (GCE)** — Zhang & Sabuncu, [NeurIPS 2018](https://dl.acm.org/doi/10.5555/3327546.3327555). A one-parameter family
  $\ell_q = (1 - p^q)/q$ that recovers CE as $q \to 0$ and MAE at $q = 1$.
- **Symmetric Cross Entropy (SCE)** — Wang et al., CE plus a scaled reverse-KL term.
- **Normalized loss functions / Active Passive Loss (APL)** — Ma et al., [ICML 2020](http://proceedings.mlr.press/v119/ma20c/ma20c.pdf).
- **Asymmetric loss functions** — Zhou et al., [ICML 2021](https://proceedings.mlr.press/v139/zhou21f/zhou21f.pdf).

*(All summary-level; I have not worked through the proofs. §13 should derive at least one
properly rather than cite it.)*

**Caution.** Most of this theory assumes noise that is *symmetric* or at least
class-conditional and instance-independent. Our noise is neither: it is structured
(non-actionable findings are missing systematically, not randomly), instance-dependent
(depends on the clinical indication), and asymmetric per label. Robust losses may help;
they are not a solution.

---

## 4. Using text without turning it into labels

An alternative to the labelling framing altogether — use the reports as a *representation
learning* signal rather than a label source:

- **ConVIRT** — Zhang et al., [arXiv:2010.00747](https://arxiv.org/abs/2010.00747).
  Bidirectional contrastive objective between paired images and report text; learns image
  representations that beat ImageNet initialization on downstream medical tasks.
- **GLoRIA** — adds *local* attention-weighted alignment between individual words and image
  sub-regions, on top of global contrastive learning *(summary only)*.
- **MedCLIP** — [arXiv:2210.10163](https://arxiv.org/abs/2210.10163), decouples image-text
  pairs to train on unpaired data *(summary only)*.

The general framing is **Learning Using Privileged Information** (Vapnik): a modality
available at training but not deployment. Our situation is textbook LUPI.

**Relevance to us: medium, and it is a genuine alternative to §4's whole framing.** Instead
of "extract labels, then train," this says "pretrain an image encoder to align with report
semantics, then fine-tune on the 58." Worth keeping alive as an option; likely
data-hungry relative to what we have.

---

## 5. Knee MRI specifically

**MRNet** — Bien et al., [PLOS Medicine 2018](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699).
The closest prior task. 1,370 knee MRI exams from Stanford; detects abnormality, ACL tear,
meniscal tear. Architecture: a per-slice CNN whose outputs are pooled within a series, with
**predictions from the 3 series (axial/coronal/sagittal) combined by logistic regression**.

Reported AUCs: **0.937** abnormality, **0.965** ACL tear, **0.847** meniscal tear.

Three things to take from it:
- **Scale reference.** 1,370 exams sufficed for strong performance on 3 targets. We have
  ~4,400 studies (with noisy labels) for 12 targets — the same order of magnitude.
- **Architecture precedent.** Slice-level CNN → pooling → per-series prediction → late
  fusion across planes. Simple, and a sane baseline for §10/§11.
- **Difficulty ordering.** ACL (0.965) was much easier than meniscus (0.847). Expect our
  twelve labels to differ enormously in achievable AUC, which matters because the metric
  averages them with equal weight.

---

## 6. What transfers, and what doesn't

| Their setting | Ours | Consequence |
|---|---|---|
| Chest X-ray, 2D, single image | Knee MRI, 3D, 3–14 series | Compute cost per study is orders of magnitude higher |
| English reports | Latin + Greek + Cyrillic | Rule-based labelers are largely non-transferable; LLM strongly favored |
| Thousands of gold image labels | **58** | Cannot fit a supervised report→image mapping the way VisualCheXbert did |
| Report available at inference in some setups | Never | Pure LUPI; text is training-only |
| Often single-label or hierarchical | 12 correlated binary labels | Co-occurrence structure is exploitable |

**The binding constraint is the 58.** VisualCheXbert had enough image-labeled data to train
a DenseNet directly. We do not. But their *bootstrap* — vision model as visual filter —
doesn't need gold labels to run, only to *evaluate*. That is precisely the budget we have:
58 studies is a poor training set and a usable, if noisy, measuring stick.

---

## 7. Candidate approaches for `DESIGN.md` §4

Not decisions — the option set that §4 must choose among.

- **A. LLM labeller, used as-is.** Multilingual LLM extracts 12 labels + uncertainty per
  report offline; upload as a Kaggle Dataset; train the image model on them. Simplest thing
  that could work. Accepts the report-image gap as an unmodelled ceiling.
- **B. A + calibration on the 58.** Fit a per-label monotone correction (threshold or 1-D
  logistic regression) from LLM output to image-read label, using the gold set. Only 1–2
  parameters per label, so 58 studies is thin but not absurd. Directly targets the severity-
  threshold effect we observed for Effusion.
- **C. A + VisualCheXbert-style bootstrap.** Train an image model on LLM labels; use its
  (image-grounded) predictions to soften or correct the text labels; retrain. Iterate. Uses
  the 58 purely for evaluation. Most faithful to the literature; most moving parts.
- **D. Contrastive image-text pretraining, then fine-tune on the 58.** Skips labelling
  entirely. Highest variance; likely data-starved.
- **E. Ensemble labelling à la Snorkel.** Multiple labelling functions, generative model to
  reconcile. Adds machinery; unclear payoff at n=4,349.

Cross-cutting, and largely orthogonal to the choice above: whether to train on **soft**
labels (LLM probabilities) rather than hard 0/1, and whether to use a **noise-robust loss**
from §3.

---

## 8. Open questions this survey did not settle

- **How large is our report-image gap actually?** The chest X-ray numbers (kappa 0.31–0.43)
  may not transfer to knee MRI, which is a more specific modality than CXR. We can estimate
  it on the 58 — that is the single most informative measurement available to us, and it
  should probably happen before §4 is decided.
- **Which of our 12 labels are worst affected?** The literature predicts the gap is largest
  for subtle/borderline findings. For us, Synovitis and Effusion are the obvious suspects.
- **Does the bootstrap in approach C actually converge** with only 4,349 studies and 12
  labels? The papers ran it at 10–100× that scale.
- **Is a 512-token context enough?** Our reports have median 129 words but a p95 of 330 and
  a max of 690, and non-Latin scripts inflate token counts.
