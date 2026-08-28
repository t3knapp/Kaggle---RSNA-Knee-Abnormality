# The Mathematics of Learning from Noisy Labels — and Whether It Applies to Us

**Status:** 2026-08-28. Companion to [`LITERATURE.md`](LITERATURE.md) §3. This document
works through the theory behind the claim *"training on report-derived labels may cost us
little or no AUC"*, states the theorems and the assumptions they require, and then audits
those assumptions honestly against our actual situation.

**Bottom line up front:** the encouraging result is real and has a proper citation, but it
rests on an assumption about the *structure* of the label noise that our situation
partially violates. There is a weaker result covering part of the violation. The remaining
part is a genuine risk we can measure but have not yet measured.

---

## 1. Setup and notation

Fix one of the twelve findings; everything below is per-label, because macro-averaged AUC
treats the twelve columns independently.

- $x$ — an MRI study (the images; whatever the model sees).
- $y \in \{0,1\}$ — the **clean** label: the competition target, produced by an independent
  expert read of the images. This is what we are scored against and what we never observe
  outside the 58.
- $\tilde y \in \{0,1\}$ — the **corrupted** label: what our labelling pipeline produces
  from the report. This is what we would train on.
- $\eta(x) = P(y = 1 \mid x)$ — the clean class-probability function.
- $\bar\eta(x) = P(\tilde y = 1 \mid x)$ — the corrupted class-probability function.
- $\pi = P(y=1)$, $\bar\pi = P(\tilde y = 1)$ — clean and corrupted base rates.
- $s : \mathcal{X} \to \mathbb{R}$ — a *scorer*, the thing our network outputs.

The **AUC** of a scorer against clean labels is the probability that a random positive
outranks a random negative:

$$\text{AUC}_D(s) = \mathbb{E}_{X \sim P,\, X' \sim Q}\Big[\mathbb{1}[s(X) > s(X')] + \tfrac12 \mathbb{1}[s(X) = s(X')]\Big]$$

where $P = P(X \mid Y=1)$ and $Q = P(X \mid Y = -1)$.

The question this document answers: **if we minimise a loss against $\tilde y$, what
happens to our AUC against $y$?**

---

## 2. Building block 1 — what cross-entropy actually estimates

**Proposition A.** *Let $\ell$ be the binary cross-entropy loss
$\ell(p, y) = -\big(y\log p + (1-y)\log(1-p)\big)$. Over all measurable
$f : \mathcal{X} \to [0,1]$, the population risk $\mathbb{E}_{(X,Y)}[\ell(f(X), Y)]$ is
minimised at $f^\star(x) = P(Y=1 \mid x)$.*

*Proof.* The risk decomposes pointwise, so fix $x$ and write $\eta = P(Y=1\mid x)$,
$p = f(x)$. We minimise

$$g(p) = -\big(\eta \log p + (1-\eta)\log(1-p)\big), \qquad p \in (0,1).$$

Differentiating, $g'(p) = -\dfrac{\eta}{p} + \dfrac{1-\eta}{1-p}$. Setting $g'(p) = 0$:

$$\eta(1-p) = (1-\eta)p \;\Longrightarrow\; \eta - \eta p = p - \eta p \;\Longrightarrow\; p = \eta.$$

And $g''(p) = \dfrac{\eta}{p^2} + \dfrac{1-\eta}{(1-p)^2} > 0$, so this is the unique
minimum. $\blacksquare$

This is the property called being a **strictly proper** loss: the loss is minimised by
reporting the true conditional probability, so a model trained with it is estimating a
conditional expectation rather than memorising individual labels.

**Why this matters intuitively.** A single label $\tilde y_i$ is one Bernoulli draw — high
variance. A model fit with a proper loss approximates $\mathbb{E}[\tilde y \mid x]$ —
averaged over everything the model can't distinguish. Averaging destroys the component of
the noise that is independent of $x$. *This is the entire reason a model can be better than
the labels it was trained on*, and why VisualCheXbert's DenseNet reaches AUROC 0.875–0.883
against clean image ground truth while the report labels it trained on agree with that
ground truth at only kappa 0.31–0.43.

**Assumptions this needs, all of which are idealisations:**

| Assumption | Reality |
|---|---|
| Function class rich enough to attain the pointwise minimiser | Finite-capacity CNN — approximation error |
| Population risk, i.e. infinite data | We have ~4,349 studies |
| Global optimum actually reached | SGD on a non-convex objective |
| No regularisation pulling $f$ away from $\eta$ | Weight decay, augmentation, early stopping all do |

There is a sharper problem hiding in the first row, addressed in §8.

---

## 3. Building block 2 — AUC only sees the ordering

**Proposition B.** *If $\varphi : \mathbb{R} \to \mathbb{R}$ is strictly increasing, then
$\text{AUC}_D(\varphi \circ s) = \text{AUC}_D(s)$ for any scorer $s$.*

*Proof.* Strict monotonicity gives, for all $a, b$: $\varphi(a) > \varphi(b) \iff a > b$,
and $\varphi(a) = \varphi(b) \iff a = b$. So the events
$\{\varphi(s(X)) > \varphi(s(X'))\}$ and $\{s(X) > s(X')\}$ are the *same event*, likewise
for ties. Equal events have equal probability, and AUC is a sum of two such
probabilities. $\blacksquare$

**Corollary.** AUC is entirely blind to calibration. A model whose predicted probabilities
are systematically compressed, inflated, or passed through any strictly increasing warp
scores identically. This is the property the whole argument exploits.

For completeness, the reason $\eta$ is the right thing to estimate at all: the AUC-optimal
scorer is any function order-preserving for $\eta$ (Clémençon et al. 2008, cited in Menon
et al. 2016). Ranking by $\eta$ is optimal; so is ranking by anything monotonically related
to it.

---

## 4. Class-conditional noise: the encouraging case

**Definition (CCN).** Label noise is *class-conditional* if the flip probability depends
only on the true label, not otherwise on the instance:

$$\rho_+ = P(\tilde y = 0 \mid y = 1), \qquad \rho_- = P(\tilde y = 1 \mid y = 0),$$

with $\tilde y \perp x \mid y$.

**Proposition C.** *Under CCN,*
$$\bar\eta(x) = \rho_- + (1 - \rho_+ - \rho_-)\,\eta(x).$$

*Proof.* By the law of total probability, conditioning on $y$:

$$P(\tilde y = 1 \mid x) = P(\tilde y = 1 \mid y=1, x)\,\eta(x) + P(\tilde y = 1 \mid y = 0, x)\,(1-\eta(x)).$$

Conditional independence lets us drop $x$ from the noise terms, giving
$(1-\rho_+)\eta(x) + \rho_-(1-\eta(x))$, which rearranges to the stated form.
$\blacksquare$

This is **affine** in $\eta(x)$ with slope $1 - \rho_+ - \rho_-$, positive exactly when
$\rho_+ + \rho_- < 1$ — i.e. when the labeller beats a coin flip. Combining Propositions A,
B and C: training with cross-entropy on $\tilde y$ recovers $\bar\eta$, which is a strictly
increasing transform of $\eta$, which has the same AUC as $\eta$, which is AUC-optimal.

The published form of the result is stronger — it holds for *every* scorer, not just the
optimum:

> **Corollary 3, Menon, van Rooyen, Ong & Williamson, ICML 2015**
> ([*Learning from Corrupted Binary Labels via Class-Probability Estimation*](https://proceedings.mlr.press/v37/menon15.pdf))
> $$\text{AUC}_{D_{\text{corr}}}(s) = (1-\alpha-\beta)\cdot \text{AUC}_D(s) + \frac{\alpha+\beta}{2}$$

Corrupted AUC is an affine, increasing function of clean AUC, so **the scorer maximising
AUC on corrupted data is the scorer maximising AUC on clean data** — and, notably, you do
not need to know the noise rates to exploit this. Their paper frames it as: *"one can
optimise balanced error and AUC without knowledge of the corruption parameters."* For CCN
the result traces back to Blum & Mitchell (1998, §5).

*(Their $\alpha,\beta$ parameterise the more general mutually-contaminated model;
CCN reduces to it via $\alpha = \bar\pi^{-1}(1-\pi)\rho_-$ and
$\beta = (1-\bar\pi)^{-1}\pi\rho_+$.)*

### 4.1 The price you still pay

AUC-optimality is an *asymptotic* statement. The finite-sample cost appears in their regret
bound:

> **Corollary 4 (ibid.).** For a strongly proper composite loss $\ell$ with modulus
> $\lambda$,
> $$\text{regret}^D_{\text{AUC}}(s) \;\le\; \frac{C(\bar\pi)}{1-\alpha-\beta}\cdot\sqrt{\frac{2}{\lambda}}\cdot\sqrt{\text{regret}^{D_{\text{corr}}}_\ell(s)}, \qquad C(\bar\pi) = \big(\bar\pi(1-\bar\pi)\big)^{-1}$$

Two multipliers matter, and both are bad news for us:

1. **$\dfrac{1}{1-\alpha-\beta}$** — the noise penalty. Noisier labels mean you need more
   of them. The paper is explicit: *"for high-noise regimes we need more corrupted samples
   to learn effectively."* Noise costs **data efficiency, not attainable ranking**.
2. **$C(\bar\pi) = \big(\bar\pi(1-\bar\pi)\big)^{-1}$** — blows up as the corrupted base
   rate approaches 0 or 1. **Rare findings are punished quadratically hard.** Several of
   our twelve labels are likely rare, and equal-weight macro-averaging means each of them
   counts as much as Effusion.

Note also (their §2.2): the noise parameters $\alpha,\beta$ are **non-identifiable** in
general, and the clean base rate $\pi$ cannot be recovered from corrupted data at all. We
cannot infer true prevalence from our derived labels — only from the 58.

Cross-entropy qualifies as a strongly proper composite loss (they name logistic loss as a
canonical example), so the bound applies to standard training.

---

## 5. Instance-dependent noise: where it breaks

Our noise is *not* obviously class-conditional. The mechanism from the radiology literature
— radiologists omit findings that aren't clinically actionable *for that particular study* —
makes the flip probability depend on the study, not just on the true label.

The general model, from [Menon et al. 2016, *Learning from Binary Labels with
Instance-Dependent Corruption*](https://arxiv.org/abs/1605.00751), lets the flip
probabilities be functions $\rho_{+1}(x), \rho_{-1}(x)$.

> **Assumption 1 (bounded total noise).** $\;\rho_{1}(x) + \rho_{-1}(x) < 1$ for all $x$ —
> *"there is always some signal to learn from for each instance."*

> **Lemma 1 (ibid.).** $\;\bar\eta(x) = (1-\rho_1(x))\,\eta(x) + \rho_{-1}(x)\,(1-\eta(x))$

which generalises Proposition C above (and Natarajan et al. 2013, Lemma 7).

**And here is the negative result.** On whether the AUC guarantee survives, the paper is
blunt: *"without further assumptions, this is not possible... there is no necessary
relationship between $\rho_{\pm1}$ and $\eta$."* Because $\bar\eta$ need not be
order-preserving for $\eta$, the corrupted-optimal and clean-optimal rankings can differ.

### 5.1 A concrete order inversion

Two studies, and note both satisfy Assumption 1:

| | $\eta(x)$ (truth) | $\rho_{-1}(x)$ (false pos.) | $\rho_{1}(x)$ (false neg.) | $\bar\eta(x)$ |
|---|---:|---:|---:|---:|
| $x_1$: finding **is** the clinical question — over-called, hedged, always mentioned | 0.40 | 0.30 | 0.00 | **0.58** |
| $x_2$: finding is **incidental** — true but not actionable, so omitted | 0.60 | 0.00 | 0.50 | **0.30** |

Using Lemma 1: $\bar\eta(x_1) = 1.0(0.40) + 0.30(0.60) = 0.58$ and
$\bar\eta(x_2) = 0.5(0.60) + 0 = 0.30$.

Truth says $\eta(x_1) < \eta(x_2)$. The corrupted labels say the opposite. A model trained
perfectly on these labels ranks $x_1$ above $x_2$ — and **loses AUC that no amount of data
recovers**, because the bias is in the target, not the sample.

This is not a contrived example. It is a direct encoding of the two mechanisms the
radiology literature names: borderline findings get mentioned with caveats (inflating
$\rho_{-1}$ when the finding is the clinical question), and non-actionable findings get
omitted (inflating $\rho_{1}$ when it isn't).

### 5.2 The partial rescue: boundary-consistent noise

There is a structured subclass where ranking *is* preserved.

> **Definition (BCN+, ibid.).** $\rho_y = f_y \circ s$ where (a) $s$ is order-preserving for
> $\eta$; (b) $f_{\pm1}$ are non-decreasing when $\eta \le 1/2$ and non-increasing when
> $\eta \ge 1/2$; (c) $\Delta(z) = f_1(z) - f_{-1}(z)$ is non-increasing.

Condition (b) is the substance: **noise is highest where the truth is most ambiguous**
($\eta \approx 1/2$) and lowest where it is clear-cut. The authors motivate it exactly as
we would want: *"the more intrinsically 'hard' an instance, the higher noise we expect for
it"* — a reasonable model *"in problems involving human annotation."*

> **Proposition 7 (ibid.).** Under BCN+-admissibility,
> $\eta(x) < \eta(x') \implies \bar\eta(x) < \bar\eta(x')$; so $\eta = \varphi \circ \bar\eta$
> for some non-decreasing $\varphi$.

> **Proposition 8 (ibid.).** With $\rho_{\max} = \tfrac12 \max_x (\rho_1(x) + \rho_{-1}(x)) < \tfrac12$,
> $$\text{reg}_{\text{rank}}(s; D) \;\le\; \frac{\bar\pi(1-\bar\pi)}{\pi(1-\pi)}\cdot\frac{1}{1-2\rho_{\max}}\cdot \text{reg}_{\text{rank}}(s; \bar D)$$

So under BCN+, maximising AUC on corrupted data remains consistent for clean AUC. Condition
(c) is load-bearing — the paper gives a counterexample where dropping it forfeits order
preservation.

Worth noting as a curiosity that also warns against over-generalising: under
instance-dependent noise, **balanced error does *not* get an analogous regret bound**, which
the authors call "perhaps surprising" since AUC is an average of BER across thresholds. The
implication only runs one way. Results here do not transfer between metrics by intuition.

---

## 6. Composing two noise stages

Our pipeline has **two** corruptions in series, not one:

$$y_{\text{img}} \xrightarrow{\;\text{radiologist writes a report}\;} \text{report} \xrightarrow{\;\text{our LLM labeller}\;} \tilde y$$

**Proposition D.** *If both stages are CCN, the composition is CCN, and the slopes
multiply.*

*Proof.* Let stage 1 have rates $(\rho_+^{1}, \rho_-^{1})$ and stage 2
$(\rho_+^{2}, \rho_-^{2})$, applied to the output of stage 1. Applying Proposition C twice:

$$\bar\eta_2 = \rho_-^2 + (1-\rho_+^2-\rho_-^2)\,\bar\eta_1 = \rho_-^2 + (1-\rho_+^2-\rho_-^2)\Big[\rho_-^1 + (1-\rho_+^1-\rho_-^1)\eta\Big]$$

which is affine in $\eta$ with slope $(1-\rho_+^2-\rho_-^2)(1-\rho_+^1-\rho_-^1)$.
$\blacksquare$

Good news and bad news. Good: monotonicity survives composition, so the AUC-immunity
argument is not destroyed by having two stages. Bad: the slope is a **product**, so the
regret penalty $\frac{1}{1-\alpha-\beta}$ compounds multiplicatively. Two independently
mild corruptions make one severe one.

Concretely, if each stage has $\rho_+ + \rho_- = 0.3$, the combined slope is
$0.7 \times 0.7 = 0.49$ — roughly a $2\times$ data-efficiency penalty from the composition
alone.

**This gives a clean design implication:** effort spent reducing *extraction* noise (stage
2 — better LLM prompting, better multilingual handling) is worth it not because it fixes
the report-image gap (it cannot) but because it stops multiplying against it.

---

## 7. Honest audit: does any of this apply to us?

| Requirement | Our situation | Verdict |
|---|---|---|
| Proper loss, rich function class, enough data | Standard CE training on ~4,349 studies, 12 labels | Approximately — §8 |
| $\rho_+ + \rho_- < 1$ per label | Almost certainly for most labels. Unverified per-label; `Synovitis` looked close to uninformative in our spot checks | **Measure it** |
| **Noise is CCN** ($\tilde y \perp x \mid y$) | **Violated.** Omission depends on clinical indication | **No** |
| Noise is BCN+ (hardest cases noisiest) | *Partially.* The "borderline findings" mechanism fits BCN+ well | **Partially** |
| Metric is AUC | Confirmed: macro-averaged AUC ROC | **Yes** |
| Base rate not extreme | Unknown — the 58 are enriched and cannot estimate prevalence | **Risk** |

The crux is the third and fourth rows. Our two documented noise mechanisms behave
differently under this theory:

- **Borderline/severity disagreement** — *"Moderate joint effusion"* → `Effusion = 0`.
  Noise concentrated where the truth is genuinely ambiguous. This is **BCN+-shaped**, so
  Proposition 7 applies and ranking is preserved. Benign.
- **Non-actionable omission** — `Synovitis = 1` with the report silent. Here $\rho_1(x)$ is
  driven by *the clinical indication*, which need not be monotone in $\eta$: a florid,
  unambiguous synovitis (high $\eta$) is omitted just as readily as a subtle one when the
  study was ordered for something else. **This violates BCN+ condition (b) directly** —
  noise is not lowest where $\eta$ is extreme. This is the §5.1 inversion, and it is
  malignant.

**So the honest position is:** our noise is a mixture of a benign component the theory
covers and a malignant component it does not, in unknown proportion. The optimistic reading
("AUC is immune to label noise") is *not* licensed. The pessimistic reading ("noisy labels
ruin everything") is also wrong — much of the noise is provably harmless under our metric.

---

## 8. The gap between this theory and actual deep networks

Proposition A assumes the function class is rich enough to reach the pointwise minimiser.
Deep networks are *too* rich: they can fit arbitrary random labels
(Zhang et al. 2017, *Understanding deep learning requires rethinking generalization*). A
model with enough capacity does not converge to $\bar\eta$ — it memorises individual
$\tilde y_i$, including the instance-level noise, and the averaging argument in §2
collapses entirely.

What rescues it in practice is the empirical **memorisation effect**: networks tend to fit
generalisable structure before they fit noise (Arpit et al. 2017). That makes early stopping
and capacity control not merely hygiene but *the mechanism by which the theory applies at
all*. A model trained to convergence on noisy labels forfeits the protection this whole
document describes.

**Design consequence for §16:** regularisation and early stopping are load-bearing here in a
way they are not in clean-label problems. The stopping criterion should probably be
validated against the 58 clean studies rather than against held-out noisy labels — otherwise
we are early-stopping on the same corrupted signal we are trying to average out.

*(Zhang et al. and Arpit et al. are cited from background knowledge, not read for this
document — verify before relying on specifics.)*

---

## 9. What this tells us to measure

The theory converts directly into a measurement plan. All of it uses the 58 gold studies,
and none of it needs the images:

1. **Per-label $\rho_+$ and $\rho_-$** — run a candidate labeller on the 58 reports and
   build the $2\times2$ table against the gold labels. Gives the slope
   $1 - \rho_+ - \rho_-$ per label, which is the single number governing how much data each
   label needs.
2. **Is any label below the $\rho_+ + \rho_- < 1$ floor?** If a label's derived signal is
   worse than a coin flip, it carries no usable information and should be handled
   separately rather than trained on.
3. **Test for instance-dependence.** The sharp question: *is labeller error predictable from
   features of the report?* If disagreement correlates with, say, the clinical indication or
   report length, we have instance-dependent noise and §5.1 applies. If errors look
   unpredictable given $y$, CCN is a defensible approximation. **This is the measurement
   that decides which regime we are in**, and n=58 makes it underpowered — worth doing
   anyway, with wide error bars stated.
4. **Estimate the ceiling.** $\text{AUC}$ achievable is bounded by how well $\bar\eta$ ranks
   $y$. On the 58 we can compute the AUC of a labeller's soft scores against gold labels
   directly — a rough upper bound on what any image model trained on those labels can reach.

---

## 10. Summary

- Cross-entropy estimates a conditional expectation, so a model can be **better than its own
  training labels**. This is not a paradox; it is averaging.
- AUC is blind to monotone transformations of the score.
- Under **class-conditional** noise the corrupted class-probability is an affine, increasing
  function of the clean one, so **AUC-optimality transfers exactly** (Menon et al. 2015,
  Cor. 3). The price is data efficiency, scaling as $\frac{1}{1-\alpha-\beta}$ and
  $\frac{1}{\bar\pi(1-\bar\pi)}$ — the latter punishing rare labels hard.
- Under **general instance-dependent** noise this guarantee **fails**, and order inversions
  are constructible from realistic radiology mechanisms.
- Under the structured **BCN+** subclass (noise concentrated on ambiguous cases) ranking
  consistency is restored (Menon et al. 2016, Props. 7–8).
- Our noise mixes a BCN+-shaped component (severity disagreement — benign) with a component
  that violates BCN+ (non-actionable omission — malignant), in unmeasured proportion.
- Two noise stages compose multiplicatively, so reducing extraction noise pays off even
  though it cannot touch the report-image gap.
- All of it is conditional on the network not memorising the noise, which makes
  regularisation and early stopping structurally important.
