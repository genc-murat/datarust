# I Added One Row. Seed 42 Replaced 14 of My 25 Test Cases.

*A practical datarust guide to deterministic shuffles, changing dataset snapshots, stable identity-based holdouts, and the experiment that stopped comparing the same rows.*

---

The model had not changed.

The features had not changed. The hyperparameters had not changed. The train/test code still used `random_state = 42`.

Only one new row had arrived overnight.

Yet the evaluation score moved much more than that row could explain.

My first suspicion was nondeterminism. I ran the job again against the same snapshot and got the same score. The seed was doing its job.

Then I printed the record IDs in the test set.

Before the refresh, there were 25 test records. After appending one row, there were still 25. But only 11 of the original test IDs remained. Fourteen had moved into training, and fourteen former training records had taken their places.

```text
old-ID overlap: 11/25
```

Seed `42` had reproduced the shuffle for each input. It had never promised to preserve membership when the input changed.

That distinction sounds small until a weekly model report compares two scores as if they came from the same exam. If both the model and the exam changed, the difference is difficult to interpret.

Let's reproduce the surprise with [datarust](https://crates.io/crates/datarust), then build a holdout whose membership follows stable record identity rather than the current number and order of rows.

## A tiny data refresh with a large evaluation change

The controlled dataset is intentionally boring. Every row contains one number: its immutable ID.

The first snapshot contains IDs `0..122`, or 123 rows. The refreshed snapshot adds ID `123`, bringing the total to 124.

I use the ID as a feature only so we can recover test membership from the returned `Matrix`. We will not fit a model with it. The alternating targets merely satisfy the train/test API and play no role in the result.

Both snapshots use:

- a 20% test fraction
- shuffled splitting
- random seed `42`
- identical order for every existing row

The only input difference is one row appended at the end.

## The complete Rust experiment

Create a small binary and add datarust:

```sh
cargo new split_stability
cd split_stability
cargo add datarust
```

Replace `src/main.rs` with this:

```rust
use std::collections::BTreeSet;

use datarust::model_selection::TrainTestSplit;
use datarust::Matrix;

fn data(n: usize) -> (Matrix, Vec<f64>) {
    let rows = (0..n)
        .map(|id| vec![id as f64])
        .collect();
    let targets = (0..n)
        .map(|id| (id % 2) as f64)
        .collect();
    (Matrix::new(rows).unwrap(), targets)
}

fn test_ids(
    n: usize,
) -> Result<BTreeSet<usize>, Box<dyn std::error::Error>> {
    let (x, y) = data(n);
    let (_, x_test, _, _) = TrainTestSplit::new()
        .with_test_size(0.20)
        .with_random_state(42)
        .split(&x, &y)?;

    Ok((0..x_test.nrows())
        .map(|row| x_test.get(row, 0) as usize)
        .collect())
}

fn stable_bucket(id: u64, seed: u64) -> u64 {
    let mut z = id ^ seed;
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30))
        .wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27))
        .wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn stable_test_ids(n: usize) -> BTreeSet<usize> {
    (0..n)
        .filter(|&id| {
            stable_bucket(id as u64, 42) % 10_000 < 2_000
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let original = test_ids(123)?;
    let appended = test_ids(124)?;

    let retained: BTreeSet<_> = original
        .intersection(&appended)
        .copied()
        .collect();
    let moved_out: BTreeSet<_> = original
        .difference(&appended)
        .copied()
        .collect();
    let moved_in: BTreeSet<_> = appended
        .difference(&original)
        .copied()
        .collect();

    println!("Seeded shuffle");
    println!("  original test size: {}", original.len());
    println!("  updated test size:  {}", appended.len());
    println!("  retained old IDs:   {retained:?}");
    println!("  moved out:          {moved_out:?}");
    println!("  moved in:           {moved_in:?}");
    println!(
        "  old-ID overlap:     {}/{}",
        retained.len(),
        original.len(),
    );

    let stable_original = stable_test_ids(123);
    let stable_appended = stable_test_ids(124);
    let stable_retained = stable_original
        .intersection(&stable_appended)
        .count();
    let stable_new: BTreeSet<_> = stable_appended
        .difference(&stable_original)
        .copied()
        .collect();

    println!();
    println!("Stable ID hash");
    println!(
        "  original test size: {}",
        stable_original.len(),
    );
    println!(
        "  updated test size:  {}",
        stable_appended.len(),
    );
    println!("  new test IDs:       {stable_new:?}");
    println!(
        "  old-ID overlap:     {stable_retained}/{}",
        stable_original.len(),
    );
    Ok(())
}
```

Run it:

```sh
cargo run --release
```

The output is:

```text
Seeded shuffle
  original test size: 25
  updated test size:  25
  retained old IDs:   {2, 5, 8, 11, 21, 26, 42, 70, 99, 104, 110}
  moved out:          {25, 32, 34, 49, 58, 69, 71, 75, 95, 97, 103, 111, 115, 117}
  moved in:           {37, 47, 50, 59, 60, 86, 94, 96, 100, 101, 108, 116, 120, 122}
  old-ID overlap:     11/25

Stable ID hash
  original test size: 25
  updated test size:  26
  new test IDs:       {123}
  old-ID overlap:     25/25
```

The seeded split is perfectly repeatable. It is also a substantially different split after one row is appended.

The identity-based split behaves differently. Every old ID keeps its assignment. The new row happens to hash into the test bucket, so the test set grows from 25 to 26.

That is the contract I wanted for this evaluation.

## Reproducible is not the same as stable

A fixed seed makes a pseudo-random operation deterministic for the same inputs.

Conceptually, a shuffled train/test split does this:

```text
indices = [0, 1, 2, ..., n - 1]
shuffle(indices, seed)
test = first round(n × test_fraction) indices
train = everything else
```

If the seed, row count, and row order are the same, datarust produces the same permutation and therefore the same split.

Change the length, however, and the shuffle is operating on a different collection. Fisher–Yates begins at the last position and works backward. With 124 items rather than 123, the sequence of swap ranges changes immediately. The same random stream is now mapped into different bounds, and those changed swaps cascade through the permutation.

The old records do not carry a memory of their earlier assignment. They are simply positions in this run's new shuffle.

This is not a datarust defect, and it is not unusual behavior. It follows from asking a deterministic shuffle to reshuffle a different input.

The precise contract is:

```text
same seed + same ordered data = same split
```

It is not:

```text
same seed + related data snapshot = same record membership
```

## Why the score comparison becomes ambiguous

Suppose Monday's model scores `0.84` and Tuesday's scores `0.81`.

If Tuesday added fresh training data while retaining the same test cohort, the comparison mostly asks whether the new training snapshot and model changed performance on a familiar benchmark.

If the test membership also changed, the three-point movement combines at least two effects:

```text
model/data change + evaluation-cohort change
```

Perhaps the new model is worse. Perhaps Tuesday's test rows are harder. Perhaps both happened. A single aggregate score cannot separate them.

This matters even when the test sets are statistically exchangeable in theory. Real datasets contain rare categories, large customers, unusual devices, edge cases, and delayed labels. Moving a handful of those rows can visibly change a metric, especially on small evaluation sets.

It also affects error analysis. A regression that disappeared from Tuesday's report may not have been fixed; its record may have moved into the training set.

## Appending is not the only way to trigger it

The example preserves the order of all 123 existing rows and adds one at the end. Production pipelines are often less polite.

A database query without an explicit `ORDER BY` can return the same records in a different order. A parallel ingestion job can merge partitions differently. Deduplication can remove an early row and shift every later position. Backfilled records may arrive in the middle of a sorted export.

If membership is derived from shuffled row positions, each of those operations can change the test set even when the seed remains fixed.

This is why I now audit split membership by stable IDs, not just split sizes:

```text
test rows yesterday: 25
test rows today:     25
```

Those counts look reassuring. They say nothing about whether the rows are the same.

## Assign the entity, not the row position

The second half of the example gives each immutable ID a deterministic bucket:

```rust
stable_bucket(id, 42) % 10_000
```

Buckets `0..1999` go to test, which represents a 20% rule:

```rust
stable_bucket(id, 42) % 10_000 < 2_000
```

The decision depends only on the ID, seed, and fixed hash algorithm. Adding ID `123` cannot change the hash of ID `25`, so ID `25` keeps its previous assignment.

The mixing function in the example is deliberately small and dependency-free. It is suitable for demonstrating deterministic bucketing, not for passwords, signatures, or adversarial security. In a production pipeline, use a documented stable hash or keyed hash, and version that choice as part of the data contract.

Do not rely on a language's unspecified or process-randomized default hash behavior. If a library update silently changes the hashing algorithm, the supposedly permanent holdout can move all over again.

## A stable fraction is approximate

The shuffled version computes an exact test count after rounding:

```text
round(123 × 0.20) = 25
round(124 × 0.20) = 25
```

Hash bucketing makes an independent membership decision for each ID. Across a large, ordinary population, roughly 20% should land below the threshold. It does not promise exactly 20% in every small snapshot.

Our first hash-based snapshot contains 25 test rows. The appended ID also lands in test, so the next snapshot contains 26. Nothing is wrong: existing membership stability and exact cardinality are different goals.

If an exact test count is mandatory, store a fixed manifest of chosen IDs. If an approximately sized cohort that grows naturally is acceptable, deterministic bucketing is often simpler.

## Hash the unit that must stay together

An event ID is not always the right identity.

Imagine that one customer produces 200 transactions. Hashing each transaction independently may place 160 in training and 40 in test. The model can then learn customer-specific behavior from the training rows and appear to generalize when it sees the same customer in evaluation.

If the deployment question is “How will this work for a customer not seen during training?”, hash the customer ID:

```text
bucket(customer_id) -> train or test
```

Every transaction for that customer follows the same assignment.

The same reasoning applies to:

- patient IDs for medical visits
- device IDs for telemetry events
- account IDs for payments
- document IDs for extracted chunks
- household IDs for individual members

Choose the identity at the boundary where information leakage would become unrealistic.

Composite identities need a canonical representation. `"12" + "34"` and `"1" + "234"` must not collapse to the same key. Normalize casing and whitespace, encode field boundaries, define how missing values behave, and keep that serialization stable across languages.

## Stable hashing does not replace chronological evaluation

Identity bucketing solves one specific problem: stable membership across changing unordered snapshots.

It does not make a random split appropriate for every dataset.

If the production task predicts the future from the past, evaluation should normally respect time. Training on December and testing on October can leak future patterns backward even if the entity IDs were partitioned perfectly.

For forecasting, churn, fraud drift, pricing, and many operational systems, a more honest design may be:

```text
training:   events before cutoff
validation: next time window
test:       later untouched window
```

If the same entity appears repeatedly and both time and grouping matter, enforce both constraints. For example, reserve customers by identity, then evaluate their events only after a cutoff. The correct split follows the production question, not a preference for one splitting trick.

## Sometimes a frozen manifest is clearer

For a formal benchmark, I often prefer an explicit artifact:

```text
evaluation-v3.txt
```

It contains the exact record or entity IDs allowed in the test set. Every training run loads that manifest, verifies that the IDs still resolve, and records its checksum beside the metrics.

That gives the comparison a simple interpretation:

```text
same exam, new model
```

Newly collected rows can feed training, a separate rolling evaluation window, or the next deliberately versioned benchmark. They do not silently rewrite the current exam.

A manifest also makes exclusions reviewable. If a label was corrected, an entity withdrew consent, or a corrupted batch must be removed, the benchmark version changes explicitly rather than through an accidental query result.

The cost is maintenance. A permanently reused holdout can become stale, and repeated human tuning against it gradually turns it into another validation set.

## Keep both a stable benchmark and a fresh window

I do not want stability to become fossilization.

A practical evaluation setup can contain multiple views:

1. A development validation set for normal iteration.
2. A sealed, stable benchmark for comparable release-to-release reporting.
3. A recent rolling window for drift and current production conditions.

The stable benchmark answers, “Did the new model improve on the same cases?”

The rolling window answers, “Does it still work on what is arriving now?”

If they disagree, that is useful evidence. Improvement on the frozen cohort paired with decline on fresh data may indicate drift, benchmark overfitting, or a population change. One test set rarely answers every question well.

## What I record with every split now

A seed is useful metadata, but it is only one line of the receipt. For a reproducible evaluation, I record:

- the source snapshot, query version, or immutable dataset checksum
- row ordering rules
- split algorithm and implementation version
- random seed, when randomness is used
- test fraction and rounding rule
- stable ID fields and their normalization
- grouping and time-cutoff policies
- a checksum of the final train and test ID lists

The last item catches the problem immediately. Two runs can have the same seed and the same test count while producing different membership checksums.

For debugging, I also compare set overlap:

```text
retention = |old_test ∩ new_test| / |old_test|
```

Our seeded-shuffle refresh retained:

```text
11 / 25 = 44%
```

The stable-ID refresh retained:

```text
25 / 25 = 100%
```

That percentage describes evaluation continuity more directly than the seed does.

## Fixed seeds are still valuable

None of this is an argument against seeded randomness.

A fixed seed is excellent for:

- reproducing a bug against one dataset snapshot
- comparing estimators on one shared random split
- making tests deterministic
- preventing run-to-run noise during development

The mistake was giving the seed a stronger meaning than it had.

Seed `42` did not identify my test cases. It initialized a random-number stream. When I supplied a different collection, I asked that stream a different sequence of questions and got a different permutation.

Once I needed membership to survive data refreshes, record identity had to become part of the splitting rule.

## The quiet question behind every model score

When a dashboard says a model improved from `0.81` to `0.84`, I now ask one question before discussing algorithms:

```text
Was it the same exam?
```

Sometimes the answer should be no. A fresh time window or new population is exactly what we need to measure. But that change should be intentional and visible.

In this experiment, adding one row replaced 14 of 25 test cases even though the seed stayed at `42`. datarust behaved deterministically. My mental model of determinism was incomplete.

The repair was not a more memorable seed. It was choosing a split contract that matched the question:

- seeded shuffle for repeatability on an unchanged snapshot
- stable identity hashing for persistent cohort membership
- frozen ID manifests for exact benchmarks
- chronological boundaries for future-facing evaluation

Reproducibility begins with the random seed. Comparable evaluation requires knowing who sat the exam.
