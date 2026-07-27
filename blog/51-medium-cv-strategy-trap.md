---
title: "KFold Crashed on Fold 4. StratifiedKFold Worked on All 5."
subtitle: "The cross-validation strategy that silently lies about your model"
author: "Murat Genc"
date: "2026-07-26"
tags: ["machine-learning", "rust", "datarust", "cross-validation", "model-selection"]
series: "datarust-v06"
---

You split your data, train a model, get 95% accuracy, and ship it. Then it fails in production. The problem wasn't the model — it was the split. KFold assumes your data is randomly ordered. When it isn't, you get training sets with a single class, and your model learns nothing.

## The Three Experiments

### Experiment 1: Balanced Data (50/50)

100 samples, 50 per class, features separated by 2 standard deviations:

```
KFold                        per-fold: 0.8500  0.8000  0.7000  0.8500  0.6500
                              mean=0.7700  std=0.0812

StratifiedKFold              per-fold: 0.6000  0.8000  0.8000  0.8500  0.8000
                              mean=0.7700  std=0.0872
```

Both strategies give the same mean accuracy (0.77). With balanced, shuffled data, it doesn't matter which you use. This is the happy path.

### Experiment 2: Imbalanced Data (90/10)

100 samples, 90 class 0, 10 class 1:

```
KFold fold class distributions:
  fold 0: 18 class 0, 2 class 1 (total 20)
  fold 1: 16 class 0, 4 class 1 (total 20)
  fold 2: 18 class 0, 2 class 1 (total 20)
  fold 3: 19 class 0, 1 class 1 (total 20)
  fold 4: 19 class 0, 1 class 1 (total 20)

StratifiedKFold fold class distributions:
  fold 0: 18 class 0, 2 class 1 (total 20)
  fold 1: 18 class 0, 2 class 1 (total 20)
  fold 2: 18 class 0, 2 class 1 (total 20)
  fold 3: 18 class 0, 2 class 1 (total 20)
  fold 4: 18 class 0, 2 class 1 (total 20)
```

KFold gives fold 3 only 1 class-1 sample and fold 4 only 1 class-1 sample. StratifiedKFold gives every fold exactly 2 class-1 samples. The accuracy scores tell the story:

```
KFold           mean=0.9000  std=0.0548
StratifiedKFold mean=0.9000  std=0.0000
```

Same mean, but KFold's standard deviation is 0.055 — five percentage points of variance that exists only because of how the data was split. With StratifiedKFold, every fold sees the same class distribution, so the variance drops to zero.

The trap: you'd never notice this if you only look at the mean. Your confidence interval is lying to you.

### Experiment 3: Ordered Data, No Shuffle

80 class 0 samples first, then 20 class 1 samples. No shuffling:

```
KFold fold class distributions (no shuffle):
  fold 0: 20 class 0, 0 class 1
  fold 1: 20 class 0, 0 class 1
  fold 2: 20 class 0, 0 class 1
  fold 3: 20 class 0, 0 class 1
  fold 4:  0 class 0, 20 class 1

StratifiedKFold fold class distributions (no shuffle):
  fold 0: 16 class 0, 4 class 1
  fold 1: 16 class 0, 4 class 1
  fold 2: 16 class 0, 4 class 1
  fold 3: 16 class 0, 4 class 1
  fold 4: 16 class 0, 4 class 1
```

KFold's first four folds contain only class 0. Fold 4 contains only class 1. When LogisticRegression tries to train on fold 4's training set (all class 0), it crashes:

```
LogisticRegression requires at least 2 distinct classes
```

This isn't a rare edge case. Any dataset where samples are ordered by label — time-series data, sorted CSVs, database exports — triggers this.

StratifiedKFold handles it correctly: round-robin assignment ensures every fold has both classes.

## Why KFold Silently Misleads

The ordered data experiment is the dramatic case — it crashes. But the imbalanced data experiment is the dangerous one. KFold doesn't crash; it just gives you unreliable variance estimates. Your model might be worse than you think, or better — you can't tell because the folds aren't comparable.

Here's the math: with 90/10 imbalance and KFold, some folds test on 2 class-1 samples while others test on 4. The accuracy on a fold with 2 class-1 samples is dominated by class-0 performance. The accuracy on a fold with 4 class-1 samples is more sensitive to class-1 performance. You're measuring different things in different folds.

## When Each Strategy Is Appropriate

**KFold** is correct when:
- Data is randomly ordered (you shuffled it)
- Classes are balanced (±10%)
- You're doing regression, not classification

**StratifiedKFold** is correct when:
- Classes are imbalanced (>10% difference)
- Data is ordered by label (time-series, sorted exports)
- You're doing classification and want reliable estimates

**The safe default**: Always use StratifiedKFold for classification. The overhead is negligible, and it never makes things worse.

## The Code

```rust
use datarust::model_selection::{KFold, StratifiedKFold};

// KFold — fine for balanced, shuffled data
let kf = KFold::new()
    .with_n_splits(5)
    .with_shuffle(true)
    .with_random_state(42);

// StratifiedKFold — safe for any classification task
let skf = StratifiedKFold::new()
    .with_n_splits(5)
    .with_shuffle(true)
    .with_random_state(42);
```

The `split()` methods differ: KFold takes `n_samples`, StratifiedKFold takes `y` (the labels). This is the API telling you which one knows about class balance.

## Tradeoffs

StratifiedKFold is slightly slower because it groups samples by class before splitting. For datasets with many classes or extreme imbalance, the round-robin assignment can produce folds where some classes have only 1 sample — sklearn warns about this, and datarust propagates the same behavior.

KFold is simpler and faster, but only correct under the assumptions above. If you're wrong about those assumptions, your error bars are wrong.

The real cost of using StratifiedKFold when you don't need it: ~1ms of additional computation. The real cost of using KFold when you do need it: shipped a model that doesn't work.

## Try It

```bash
cargo add datarust
```

```rust
use datarust::model_selection::{KFold, StratifiedKFold};

// Check what KFold does to your data
let kf = KFold::new().with_n_splits(5);
for (i, (train, test)) in kf.split(100).unwrap().enumerate() {
    println!("fold {i}: train={}, test={}", train.len(), test.len());
}

// Check what StratifiedKFold does
let y: Vec<f64> = vec![0.0; 90].into_iter()
    .chain(vec![1.0; 10]).collect();
let skf = StratifiedKFold::new().with_n_splits(5);
for (i, (train, test)) in skf.split(&y).unwrap().enumerate() {
    let n1 = test.iter().filter(|&&j| y[j] == 1.0).count();
    println!("fold {i}: {} class-1 in test", n1);
}
```

If you're using KFold for classification, you're one ordered dataset away from shipping a broken model. Switch to StratifiedKFold. The API difference is one word. The correctness difference is everything.
