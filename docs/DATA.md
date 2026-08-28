# Data Inventory — RSNA Knee Abnormality Detection

**Status:** first pass, 2026-08-28. Descriptive only — no modeling decisions are made
here. Decisions that *follow* from these facts belong in [`DESIGN.md`](../DESIGN.md).

**Provenance.** Everything below was computed from the five small metadata CSVs pulled
through the Kaggle API (`train.csv`, `train_series.csv`, `test.csv`, `test_series.csv`,
`sample_submission.csv` — ~9 MB total). No DICOM pixel data was downloaded locally, per
the rules in `CLAUDE.md`. Statements about the image files themselves are inferred from
the API's file listing, not from opening volumes, and are marked as such.

Reproduce with [`scripts/explore_metadata.py`](../scripts/explore_metadata.py).

---

## 1. Headline findings

Three facts dominate everything else, and each one reshapes the modeling problem:

1. **Only 58 of 4,407 training studies (1.3%) carry labels.** The other 4,349 (98.7%)
   have a radiology report and images, but every one of the twelve label columns is
   empty. This is not a class-imbalance problem — it is a *supervision* problem.
2. **Radiology reports do not exist at test time.** `test.csv` has exactly one column,
   `StudyInstanceUID`. Text is a training-time-only signal.
3. **Reports are genuinely multilingual, across three writing systems** — Latin, Greek,
   and Cyrillic. Any text-processing approach has to survive that.

Taken together these point at the shape of the intended task: *use the reports to
manufacture training labels for ~4,400 studies, then learn to predict those labels from
images alone.* That framing is a decision, not a fact, and is owed a `DESIGN.md` entry
(§4) — but it is the reading the data most naturally supports, and it matches RSNA's
public framing of this as the first challenge to train on images plus report text.

---

## 2. Files and shapes

| File | Rows | Columns |
|---|---:|---|
| `train.csv` | 4,407 | `StudyInstanceUID`, `Report`, + 12 label columns |
| `train_series.csv` | 24,371 | `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane` |
| `test.csv` | 3 *(stub)* | `StudyInstanceUID` only |
| `test_series.csv` | 15 *(stub)* | same schema as `train_series.csv` |
| `sample_submission.csv` | 3 *(stub)* | `StudyInstanceUID` + the same 12 label columns |

`StudyInstanceUID` is unique in `train.csv` (no duplicates), so **the study is the unit
of prediction**. Every one of the 4,407 studies has at least one row in
`train_series.csv` — no orphans in either direction.

The published test files are 3-study stubs; the real test set is hidden and its size,
prevalence, and site/language mix are unknown.

Image files are laid out as
`{train,test}_series/{StudyInstanceUID}/{SeriesInstanceUID}/{SOPInstanceUID}.dcm`.
From the API listing, individual slice files run roughly 1.2–1.9 MB, which for ~24k
series is consistent with the competition's described scale of hundreds of GB.

### The twelve labels

`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`,
`PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`

The submission columns are exactly these twelve, in the same order — so the task is
twelve independent binary predictions per study (multi-label, not multi-class: a knee
can have any subset of these simultaneously, and typically has several).

---

## 3. Label availability

Every label column has exactly 4,349 nulls and 58 non-nulls. Crucially, the labelling is
**all-or-nothing per study** — the distribution of "number of non-null labels per study"
has only two values:

| non-null labels | studies |
|---:|---:|
| 0 | 4,349 |
| 12 | 58 |

There are no partially-labeled studies. So we have a small, fully-annotated gold set and
a large, entirely unannotated remainder — not a scattered-missingness problem.

### The 58 gold studies are enriched, not representative

Among the labeled 58, positive rates are extremely high compared to what a screening
population would show:

| Label | Pos | Neg | Pos rate |
|---|---:|---:|---:|
| Effusion | 35 | 23 | 60.3% |
| Synovitis | 27 | 31 | 46.6% |
| Medial Meniscus | 26 | 32 | 44.8% |
| ACL | 24 | 34 | 41.4% |
| Lateral Meniscus | 23 | 35 | 39.7% |
| PF OA | 21 | 37 | 36.2% |
| Contusion | 19 | 39 | 32.8% |
| Fracture | 18 | 40 | 31.0% |
| Medial OA | 15 | 43 | 25.9% |
| Baker's | 12 | 46 | 20.7% |
| Lateral OA | 11 | 47 | 19.0% |
| MCL | 9 | 49 | 15.5% |

Mean number of positive findings per labeled study is **4.14**, the range is 1–9, and
**not a single one of the 58 has zero findings**. A real consecutive-imaging cohort would
contain plenty of normal or near-normal knees. So these 58 look deliberately selected to
span the label space — useful as a *gold standard for checking a labeller*, but
**not** usable for estimating class priors or expected leaderboard prevalence.

Their reports are also somewhat longer than average (median 1,205 vs. 974 characters),
consistent with selection toward findings-rich studies.

### Label co-occurrence

Among the 58, the strongest co-occurrences relative to independence (lift = observed /
expected-under-independence) are clinically sensible, which is a mild sanity check that
the labels mean what their names suggest:

| Pair | Observed | Expected if independent | Lift |
|---|---:|---:|---:|
| Lateral OA + Baker's | 6 | 2.3 | 2.64 |
| Medial OA + Baker's | 8 | 3.1 | 2.58 |
| Medial OA + Lateral OA | 6 | 2.8 | 2.11 |
| Lateral Meniscus + Lateral OA | 9 | 4.4 | 2.06 |
| Lateral Meniscus + Medial OA | 11 | 5.9 | 1.85 |
| Medial OA + PF OA | 10 | 5.4 | 1.84 |
| Medial Meniscus + Medial OA | 12 | 6.7 | 1.78 |

The pattern is dominated by osteoarthritis: OA in one compartment travels with OA in
another, with Baker's cysts, and with meniscal tears. All three are well-established
clinical associations — degenerative knees tend to be degenerate in several ways at
once. Further down the list, ACL + Contusion (lift 1.65) and ACL + Fracture (1.48) pick
up the *other* clinical cluster: acute trauma rather than chronic degeneration.

**Read these numbers with heavy caution.** They come from n=58, and the top-lift pairs
have expected counts of 2–3, so a difference of one or two studies swings the lift
substantially. The qualitative pattern (two correlated clusters, degenerative and
traumatic) is more trustworthy than any individual number.

**Modeling consequence to carry forward:** the labels are *not* independent, so treating
the twelve outputs as twelve fully independent binary problems leaves structure on the
table. Whether to exploit that is a `DESIGN.md` question (§13), not a settled fact.

---

## 4. Reports

- No nulls, no empty strings — all 4,407 studies have report text.
- Length: min 52, p25 587, **median 977**, p75 1,459, p95 2,452, max 4,743 characters.
- Word count: median 129, p95 330, max 690.
- File is valid UTF-8, no BOM, 181 distinct non-ASCII characters.

Median ~130 words means a standard transformer context window (512 tokens) covers the
large majority of reports, though the p95 tail and non-English tokenization inflation
will push some over.

### Writing systems

Classifying each report by its majority alphabet:

| Script | Studies |
|---|---:|
| Latin | 3,866 |
| Greek | 321 |
| Cyrillic | 220 |

Within the Latin-script group, a crude stopword heuristic suggests a mix dominated by
English and Spanish with a substantial Turkish component. That heuristic is **not
authoritative** — it is keyword counting, not language identification — and is recorded
only to indicate diversity. If actual language identity matters for a modeling decision,
run a real language-ID model rather than trusting these numbers.

The presence of Greek and Cyrillic is the load-bearing fact: an approach built on
English regular expressions will silently fail on ~12% of the training set.

### Structure

Reports are semi-structured, with explicit section headers in some but far from all
cases: `FINDINGS` appears in 19.5%, `IMPRESSION` in 15.8%, `TECHNIQUE` in 13.7%, and the
Spanish `Impresión`/`Hallazgos` and Turkish `Bulgular` headers in 15.7%/7.9%/8.5%
respectively. A substantial fraction are free prose with no headers at all, so a parser
that depends on finding an impression section will not generalize.

---

## 5. Series and imaging metadata

Each study contains **3–14 series** (median 5, mean 5.5, p95 9):

| Series per study | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Studies | 1 | 675 | 2,299 | 698 | 310 | 145 | 176 | 74 | 21 | 5 | 2 | 1 |

**Every single study has all three anatomical planes present** (all 4,407 studies have
at least one Sagittal, one Coronal, and one Axial series). That is a strong structural
guarantee — a multi-plane architecture will never have to handle a missing plane on the
training set, though robustness on the hidden test set is unverified.

| Plane | Series |
|---|---:|
| Sagittal | 9,864 |
| Coronal | 8,609 |
| Axial | 5,898 |

### `Fluid_Sensitive` and `Fat_Suppression` are the same column

These two flags are **identical for every row** in both `train_series.csv` and
`test_series.csv` (14,010 ones and 10,361 zeros in train, matching exactly). They carry
one bit of information between them, not two. Whether this is intentional (fluid-
sensitive sequences in this dataset are always fat-suppressed) or a data-generation
artifact is unknown, but for modeling purposes it is **one feature, not two**.

The resulting plane × contrast combinations:

| Plane | Fluid-sensitive / fat-sat | Series |
|---|---|---:|
| Sagittal | no | 5,197 |
| Axial | yes | 4,719 |
| Sagittal | yes | 4,667 |
| Coronal | yes | 4,624 |
| Coronal | no | 3,985 |
| Axial | no | 1,179 |

Every study has at least two fluid-sensitive series (mode 3, range 2–7). The typical
study is therefore roughly one fluid-sensitive and one non-fluid-sensitive series in
each of sagittal and coronal, plus a fluid-sensitive axial.

This matters clinically: fluid-sensitive (fat-suppressed) sequences are where effusion,
bone marrow edema, and contusion are conspicuous, while non-fat-suppressed sequences
better show anatomy and meniscal morphology. So the twelve labels are unlikely to be
equally visible on all series — which is an argument that series selection (§7) is a
real decision with per-label consequences, not a detail.

---

## 6. What is still unknown

- **Hidden test set size, prevalence, and composition.** The 3-study stub tells us
  nothing beyond schema.
- **Whether reports are absent at inference** is strongly implied by the `test.csv`
  schema but not yet read off the official data description.
- **Slice counts, in-plane resolution, and slice spacing per series** — requires opening
  DICOMs, which is a job for the `00_env_data_check` Kaggle kernel, not a local pull.
- **Site identity.** The competition describes 16–19 contributing sites, but no site
  column is exposed. If site is recoverable from DICOM metadata it becomes relevant to
  the validation split (§14), since site-level leakage is a real risk.
- **How well reports actually determine the labels.** Open question, actively being
  probed — see §7 below.

---

## 7. Where the labels come from — and why it matters

**Confirmed by the competition organizers: the twelve labels are produced by an
independent read of the images, not extracted from the reports.**

This is the single most consequential fact about the dataset, and it is not visible from
the CSVs alone. It means the reports are a *proxy* for the target, not the target itself.
A perfect report-reader — one that handles negation, severity, and all three writing
systems flawlessly — would still produce wrong labels wherever the reporting radiologist
and the annotating radiologist disagreed.

### Direct evidence in the gold set

Reading the English-language gold studies against their labels shows disagreement in
**both** directions, which is the signature of two independent readers rather than a
lossy extraction process:

| Study | Report says | Label |
|---|---|---|
| 1 | "Moderate joint effusion... are observed" | `Effusion = 0` |
| 1 | "Osteochondral fracture at lateral patellar facet" | `Fracture = 0` |
| 2 | *(synovitis never mentioned)* | `Synovitis = 1` |
| 3 | "Synovitis of left knee and massive joint effusion" | `Synovitis = 0`, `Effusion = 1` |
| 5 | "complex tear... posterior horn of the lateral meniscus" | `Lateral Meniscus = 0` |
| 5 | *(synovitis never mentioned)* | `Synovitis = 1` |
| 4 | every finding stated explicitly | all 12 match exactly |

A pure text-extraction process can only ever produce false *negatives* relative to the
report (missing what is stated). Labels that are **positive when the report is silent**
— studies 2 and 5 — can only come from someone looking at the images.

`Effusion` does show a coherent severity threshold, which is a text-extractable pattern:

| Report | Label |
|---|---|
| "No knee effusion" | 0 |
| "Small joint effusion" | 0 |
| "Moderate joint effusion" | 0 |
| "Some amount... with hemarthrosis" | 1 |
| "massive joint effusion" | 1 |

So the gap is not uniform across labels — part of it is a learnable calibration problem,
and part of it is irreducible reader disagreement.

### The keyword-matching floor

A crude multilingual regex labeller scored against the 58 gold studies gives mean
precision **0.47**, mean recall **0.45**, mean agreement **63%**. Two caveats keep this
from being a clean measurement: the probe's own patterns were demonstrably broken for
`Medial OA` and `Lateral OA` (zero true positives — the regex missed Spanish *artrosis*
and English *tricompartmental osteoarthritis* entirely), and n=58 is small. Treat 63% as
a loose floor, not an estimate.

### Consequence

This is a **weak-supervision problem with a known, structured label-noise mechanism** —
which is a well-studied setting. See [`LITERATURE.md`](LITERATURE.md): the same
report-vs-image gap is documented in chest radiography at kappa 0.31–0.43, with
"non-actionable findings omitted from reports" identified as the dominant cause. That
mechanism explains our `Synovitis = 1` cases exactly.

The design consequence is that the 58 gold studies are best understood not as training
data but as **the only instrument we have for measuring how good a labeller is**.
`DESIGN.md` §4 has to decide how to spend them.
