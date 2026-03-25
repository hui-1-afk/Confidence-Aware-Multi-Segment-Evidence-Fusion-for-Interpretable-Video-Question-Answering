# Confidence-Aware Top-2 Temporal Hypothesis Fusion for Long-Video Question Answering

This repository accompanies the paper **“Confidence-Aware Top-2 Temporal Hypothesis Fusion for Long-Video Question Answering”**.

Long-video question answering (VideoQA) requires both accurate temporal grounding and reliable reasoning over evidence that may be temporally dispersed. Existing modular pipelines typically collapse the verifier output to a single top-ranked segment for answer generation. While efficient, this strategy becomes fragile when residual temporal ambiguity remains after verification.

To address this limitation, we propose a **confidence-aware top-2 temporal hypothesis fusion method** for long-video VideoQA. Instead of collapsing the verifier output to a single surviving segment, the proposed method preserves the **two highest-ranked verified moments** as a bounded competing pair and introduces a lightweight **Fusioner** that performs:

- **overlap-aware deduplication**
- **confidence-aware feature construction**
- **question-guided interaction**

before answer decoding.

In this way, residual competition between plausible temporal evidences is resolved at the **evidence level** rather than through brittle top-1 commitment.

**Code, models, and data:**  
https://github.com/hui-1-afk/Confidence-Aware-Multi-Segment-Evidence-Fusion-for-Interpretable-Video-Question-Answering.git

---

## 1. Overview

In long-video VideoQA, a common design is **single-segment reasoning**, where the temporally localized segment with the highest confidence score is selected as the sole evidence for answer generation. This strategy is efficient, but it is fundamentally limited in videos containing:

- multi-stage actions
- temporally separated causal events
- repeated or visually similar moments
- partially complementary evidence distributed across time

The top-ranked segment may be locally salient yet semantically incomplete, because prerequisite context or later outcomes may appear elsewhere in the video.

Our method addresses this problem with a **bounded multi-hypothesis design**. Rather than keeping a large candidate pool, we retain only the **top-2 verified segments** and treat them as a structured pair of **competing temporal hypotheses**. The retained pair is then fused by a lightweight post-verification module before answer generation.

---

## 2. Main Contributions

The main contributions of this work are:

1. **Bounded post-verification uncertainty modeling**  
   We reformulate residual post-verification uncertainty as a bounded competition between the two most plausible temporal hypotheses, moving ambiguity handling from post-hoc answer selection to evidence-level reasoning.

2. **Lightweight top-2 fusion mechanism**  
   We introduce a lightweight top-2 fusion mechanism that refines and fuses the retained evidence pair through overlap-aware processing, confidence-aware feature construction, and question-guided interaction.

3. **Plug-in post-verification upgrade**  
   We present the proposed design as a plug-in post-verification upgrade for modular long-video QA pipelines and show, under matched within-backbone comparison, that bounded evidence-level fusion yields consistent gains on multiple evaluated benchmarks.

---

## 3. Method

### 3.1 Inherited VideoMind-style backbone

The overall system contains five components:

1. **Planner**  
   Determines the execution route.

2. **Grounder**  
   Generates candidate temporal moments relevant to the question.

3. **Verifier**  
   Re-scores and ranks candidate temporal moments by confidence.

4. **Fusioner**  
   Operates on the retained top-2 verified segments to perform confidence-aware evidence fusion.

5. **Answerer**  
   Produces the final response from the fused representation.

The main modification of this work is focused on the **post-verification transition from ranking to reasoning**.

---

### 3.2 Why top-2?

A straightforward extension to single-segment reasoning is to retain multiple temporal candidates after verification. However, the key question is not simply whether more segments should be kept, but what constitutes the **smallest useful evidence set** for resolving residual temporal ambiguity.

We therefore retain only:

- **Top-1**: the dominant evidence branch
- **Top-2**: the strongest remaining competing temporal hypothesis

This retained pair is treated not as a generic multi-segment pool, but as the **smallest non-degenerate hypothesis set** in which evidence-level competition can still be explicitly represented and resolved.

This design is motivated by both:

- **structure**: the most informative residual ambiguity is often concentrated near the top of the verifier ranking
- **efficiency**: adding more segments increases cross-segment interaction cost and brings in lower-confidence candidates that are more likely to be noisy, redundant, or semantically marginal

---

### 3.3 Fusioner

The Fusioner transforms the retained pair into a single global evidence representation for answer generation. The post-verification workflow contains four stages:

1. **Top-2 Retention**  
   Keep the two highest-ranked verified segments.

2. **Overlap-Aware Deduplication**  
   Preserve the verifier-preferred segment and remove duplicated temporal support from the competing branch.

3. **Confidence-Aware Cross-Hypothesis Interaction**  
   Inject verifier confidence into the retained segment features and perform lightweight question-guided interaction.

4. **Question-Guided Aggregation**  
   Aggregate the interacted pair into a fused global evidence representation for the Answerer.

This design generalizes single-segment reasoning as a special case: when the verifier strongly favors one segment, the fused representation naturally collapses toward the dominant branch; when the retained pair is close in confidence, both branches remain active and complementary temporal cues can be combined before decoding.

---

### 3.4 Efficiency

Although the Fusioner introduces explicit cross-hypothesis reasoning, its cost remains bounded because it operates on only two retained segment representations.

If the feature dimension is \(D\), the cross-hypothesis interaction complexity is:

\[
O(N'^2 D), \quad N' = 2
\]

By restricting interaction to the retained top-2 pair, the proposed design preserves the practical efficiency of top-ranked evidence selection while still enabling explicit hypothesis-level comparison under ambiguity.

---

## 4. Experimental Settings

### Benchmarks

We evaluate the proposed method on three representative benchmarks:

- **MVBench**  
  A benchmark covering 20 diverse video understanding tasks, with an average video duration of **15 seconds**.  
  Metric: **Average Accuracy**

- **NExT-QA**  
  A VideoQA benchmark focusing on causal and temporal reasoning, with an average video duration of **39.5 seconds**.  
  Metric: **Acc@QA**

- **MLVU**  
  A multi-task long-video understanding benchmark with an average duration of **15.5 minutes**.  
  Metric: **Macro-average Accuracy (M-Avg)**

### Implementation Details

- Backbone: **Qwen2-VL 2B**
- LoRA adapters: independent LoRA adapters with **rank 64** for Planner, Grounder, Verifier, Fusioner, and Answerer
- Grounder decoder hidden dimension: **256**
- Video sampling rate: **1 FPS**
- Verifier temporal boundary expansion: **50%**
- Special tokens: `<SEG_START>` and `<SEG_END>`
- Retained clips after verification: **Top-2** (\(N' = 2\))
- Fusion type: **query-guided cross-segment evidence fusion**
- Frozen modules during training: **Planner, Grounder, Verifier**
- Trainable modules during training: **Fusioner and Answerer LoRA parameters**
- Optimizer: **AdamW**
- Initial learning rate: **2 × 10⁻⁵**
- Batch size: **32**
- Training epochs: **1**
- Temperature parameter in Eq. (12): fixed across experiments

### Hardware Environment

All experiments are conducted on a server with:

- **8 × NVIDIA H100 80GB HBM3 GPUs**

This hardware setup is used for throughput efficiency rather than being a strict requirement of the framework.

---

## 5. Main Results

### 5.1 Comprehensive performance comparison

Under the reported setting, the proposed fusion model improves all three reported benchmarks relative to the 2B VideoMind baseline.

| Method | Size | MVBench | MLVU | NExT-QA |
|---|---:|---:|---:|---:|
| Video-LLaVA | 7B | 43.0 | 47.3 | 51.4 |
| Otter | 9B | 40.5 | 41.2 | 59.1 |
| TimeChat | 7B | 38.5 | 30.9 | 50.6 |
| LongVA | 7B | - | 56.3 | 68.3 |
| TinyLLaVA | 3B | 45.5 | 44.8 | 58.1 |
| ST-LLM | 7B | 54.9 | - | - |
| VideoGPT+ | 3.8B | 58.7 | - | - |
| VideoChat2 | 7B | 60.4 | 47.9 | 68.6 |
| VILA | 2.7B | 49.2 | 51.0 | 60.5 |
| VideoMind | 2B | 48.8 | 56.7 | 66.6 |
| **VideoMind (Fusion)** | **2B** | **61.5** | **58.9** | **74.5** |

Compared with the 2B VideoMind baseline, the proposed fusion model improves:

- **MVBench**: 48.8 → 61.5 (**+12.7**)
- **MLVU**: 56.7 → 58.9 (**+2.2**)
- **NExT-QA**: 66.6 → 74.5 (**+7.9**)

These results suggest that bounded top-2 fusion is particularly effective when answer generation depends on stronger temporal disambiguation and complementary evidence integration.

> Note: publicly reported results from other methods are provided for broader context only. The central empirical claim of this work is the **matched within-backbone comparison** between the inherited 2B VideoMind baseline and its fusion-based variant under aligned implementation settings.

---

### 5.2 Key module ablation

Results on the NExT-QA validation set:

| Setting | Accuracy (%) |
|---|---:|
| Frozen-score Fusion | 74.03 |
| w/o Verifier | 74.15 |
| w/o Planner | 74.33 |
| **Full (Ours)** | **74.45** |

The full model achieves the highest accuracy. Removing the Verifier reduces performance, indicating that confidence re-scoring remains important for the quality of the retained evidence pair. Removing the Planner causes only a small drop, which is consistent with the fact that the main contribution of this work lies after verification rather than in routing itself.

---

### 5.3 Fusion strategy ablation

Different fusion strategies on the NExT-QA validation set:

| Fusion Strategy | Accuracy (%) |
|---|---:|
| Top-1 Only | 73.2 |
| Hard Selection | 73.5 |
| Equal Weight | 73.8 |
| **Ours** | **74.1** |

Dynamic confidence-aware fusion achieves the best accuracy, outperforming both hard selection and simple averaging.

---

### 5.4 Retained candidate count analysis

To validate the bounded top-2 design, we compare Top-1+Top-2, Top-1+Top-3, Top-1+Top-4, and Top-1+Top-5 on the NExT-QA validation set.

| Task | Samples | Baseline | Ours | Top1+Top3 | Top1+Top4 | Top1+Top5 |
|---|---:|---:|---:|---:|---:|---:|
| CH | 683 | 65.59 | 73.36 | 72.77 | 68.10 | 64.79 |
| CW | 1924 | 67.34 | 75.33 | 75.27 | 67.19 | 66.66 |
| DC | 177 | 56.96 | 63.72 | 61.90 | 58.53 | 56.27 |
| DL | 295 | 82.57 | 92.37 | 92.51 | 91.83 | 81.27 |
| DO | 305 | 72.25 | 80.83 | 80.65 | 75.44 | 71.10 |
| TC | 663 | 67.28 | 75.27 | 74.66 | 72.26 | 66.88 |
| TN | 895 | 60.22 | 67.36 | 67.22 | 60.21 | 59.88 |
| TP | 54 | 62.79 | 70.24 | 68.24 | 60.87 | 58.77 |
| **ACC** | **4996** | **66.60** | **74.50** | **74.20** | **68.32** | **65.90** |

The proposed **Top-1+Top-2** setting achieves the best overall accuracy and performs best on **7 out of 8** task categories. This supports the claim that the most useful residual ambiguity is usually concentrated near the top of the verifier ranking, while lower-ranked candidates are more likely to introduce redundant or weakly relevant evidence.

---

## 6. Qualitative Observations

The paper further provides qualitative case studies showing that the second-ranked segment can remain semantically useful rather than redundant. Representative cases include:

- **transition-to-outcome continuity**
- **direct evidence followed by later confirmation**
- **interference-sensitive cases**, where the retained pair helps suppress distractors and aggregate semantically consistent clues

These examples support the motivation for retaining the top-2 verified segments under temporal ambiguity.

---

## 7. Conclusion

This work focuses on the post-verification transition in long-video question answering, where existing pipelines often commit too early to a single top-ranked segment for answer generation. We show that this hard top-1 decision is brittle when meaningful temporal ambiguity remains near the top of the verifier ranking.

To address this issue, we introduce a confidence-aware top-2 fusion framework that preserves the strongest competing segment and resolves the retained ambiguity before decoding through a lightweight post-verification Fusioner. Under the matched within-backbone setting, the proposed design consistently improves over the inherited VideoMind baseline on MVBench, MLVU, and NExT-QA, with especially clear gains on MVBench and NExT-QA.

These results support bounded top-2 fusion as an effective compromise between ambiguity preservation and reasoning efficiency in temporally dispersed VideoQA.

---

## 8. Paper Information

**Title**  
Confidence-Aware Top-2 Temporal Hypothesis Fusion for Long-Video Question Answering

**Authors**  
Mangang Xie, Hui Zhang, Quan Wan, Jisheng Dang, Bimei Wang, Jie Xu, Juanjuan Jing

---

## 9. Citation

If you find this repository useful, please cite the corresponding paper:

**Confidence-Aware Top-2 Temporal Hypothesis Fusion for Long-Video Question Answering**  
Mangang Xie, Hui Zhang, Quan Wan, Jisheng Dang, Bimei Wang, Jie Xu, Juanjuan Jing
