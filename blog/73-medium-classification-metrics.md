# Why a 94.5% Accuracy Model Missed Every Minority Sample

*Precision, Recall, and F1 reveal what accuracy hides in imbalanced classification*

---

I trained a logistic regression on data where 5% of samples were class 1. The model got 94.5% accuracy. It also predicted every single sample as class 0. The minority class? Zero precision, zero recall, zero F1.

Accuracy said the model was great. The model was useless.

## The Problem with Accuracy

Accuracy counts correct predictions and divides by total predictions. For balanced data this works fine. For imbalanced data, a model that always predicts the majority class achieves high accuracy without ever learning anything.

Three scenarios — same logistic regression, same 2-dimensional features (N(±1, 1)), same overlap, different class ratios:

### Scenario 1: Balanced (50/50)

```
Confusion matrix:
  class 0: [72,  8]  72 correct, 8 predicted as 1
  class 1: [ 4, 66]  66 correct, 4 predicted as 0

Accuracy:      0.9200
Precision:     0.9196
Recall:        0.9214
F1 score:      0.9199
```

All four metrics cluster around 0.92. When classes are balanced, they all tell the same story.

### Scenario 2: Moderate Imbalance (80/20)

```
Confusion matrix:
  class 0: [152,  9]
  class 1: [  3, 36]

Accuracy:      0.9400    ← 94% — looks good
Precision:     0.8903
Recall:        0.9336
F1 score:      0.9096    ← F1 says 0.91
```

Accuracy edges up to 0.94, F1 sits at 0.91. The gap is small because the model still captures 36 of 39 class-1 samples. A 0.03 gap is noticeable but not alarming.

### Scenario 3: Severe Imbalance (95/5)

```
Confusion matrix:
  class 0: [189,  0]  189 correct, 0 predicted as 1
  class 1: [ 11,  0]    0 correct, 11 predicted as 0

Accuracy:      0.9450    ← 94.5% — same as balanced!
Precision:     0.4725
Recall:        0.5000
F1 score:      0.4859
```

Accuracy stays at 0.945 — virtually unchanged from the balanced case. But the model learned nothing about class 1. Every single class-1 sample was misclassified. The macro-averaged F1 of 0.49 reveals the truth:

- Class 1 precision: 0.0000 (no predictions for class 1)
- Class 1 recall:    0.0000 (no class-1 samples found)
- Class 1 F1:        0.0000

The model is a dummy classifier that always predicts the majority class. Accuracy can't see this — it's dominated by the 95% majority.

## Summary

```
      scenario         accuracy  precision     recall         f1
  -----------------------------------------------------------------
   balanced 50/50        0.9200     0.9196     0.9214     0.9199
   moderate 80/20        0.9400     0.8903     0.9336     0.9096
    severe 95/5          0.9450     0.4725     0.5000     0.4859
```

As imbalance increases, accuracy stays flat around 0.92–0.95. F1 drops from 0.92 to 0.49.

## What Each Metric Measures

**Accuracy** answers: what fraction of predictions are correct? Every prediction counts equally. When class 0 has 95% of samples, getting class 0 right gives 95% accuracy. Getting class 1 right gives an additional 5%. A model that only predicts class 0 reaches 95% by default.

**Precision** (macro-average) answers: when the model predicts a class, how often is it right? For each class, precision = TP / (TP + FP). Macro-precision averages these per-class values. In scenario 3, class-0 precision is 0.945 (189/200 predictions correct), class-1 precision is 0.000 (no predictions). The macro average is (0.945 + 0.000) / 2 = 0.4725.

**Recall** answers: of all actual samples of a class, how many did the model find? For each class, recall = TP / (TP + FN). Class-0 recall is 1.000 (all 189 found), class-1 recall is 0.000 (none of 11 found). Macro average: (1.000 + 0.000) / 2 = 0.5000.

**F1** answers: what is the harmonic mean of precision and recall? For each class, F1 = 2 × P × R / (P + R). Macro-F1 averages these per-class values. In scenario 3: (0.9717 + 0.0000) / 2 = 0.4859.

The macro average treats every class equally, regardless of its frequency. One bad minority class cuts the macro score in half, even when the majority class performs perfectly.

## Practical Guidelines

**For balanced data, accuracy is fine.** All metrics tell the same story. Use whichever is most interpretable for your audience.

**For imbalanced data, use macro-averaged F1.** It penalizes models that ignore minority classes. A high accuracy with low F1 is a warning sign.

**Always check per-class metrics.** The summary number hides which classes the model does well on. A classification report (per-class precision, recall, F1) reveals the full picture.

**Don't use accuracy for threshold selection.** If you optimize accuracy on imbalanced data, the optimal threshold will be the majority-class rate. You'll end up with a model that never predicts the minority class.

## Try It

```bash
cargo add datarust
```

```rust
use datarust::metrics::classification::{
    precision_score, recall_score, f1_score, accuracy_score,
};

let y_true = vec![0.0, 0.0, 0.0, 1.0, 1.0];
let y_pred = vec![0.0, 0.0, 0.0, 0.0, 0.0];

let acc = accuracy_score(&y_true, &y_pred)?;      // 0.6
let prec = precision_score(&y_true, &y_pred)?;     // 0.3  (macro)
let rec = recall_score(&y_true, &y_pred)?;         // 0.5  (macro)
let f1 = f1_score(&y_true, &y_pred)?;              // 0.375 (macro)
```

Accuracy says 60%. F1 says 0.375. The model found zero class-1 samples. Accuracy hides this. F1 does not.

A model scoring 94.5% accuracy isn't always a good model. When classes are imbalanced, F1 tells the real story. Check per-class metrics — accuracy alone won't save you.
