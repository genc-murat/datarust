# The Pipeline Loaded From JSON. Its Custom Function Did Not.

*A practical datarust guide to FunctionTransformer, executable code, explicit rebinding, and the model artifact that was fitted but not ready to serve.*

---

The model file loaded without an error.

The preprocessing pipeline said it was fitted.

Then the first request failed:

```text
invalid input: FunctionTransformer: func not set
```

This was not corrupted JSON. It was not a missing coefficient or a forgotten scaler mean. The pipeline contained a custom Rust function that applied `ln(1 + x)` to two skewed features. datarust serialized the transformer's learned state and position in the pipeline, but deliberately did not serialize the function pointer.

JSON can preserve data. It cannot carry a portable, executable Rust function into another process.

Once I rebound the function after loading, the predictions returned and matched the original process to floating-point precision:

```text
max prediction difference: 3.55e-15
```

That distinction matters whenever a convenient custom preprocessing closure becomes part of a deployed model. The code may feel like a small helper during training. In production, it is an executable dependency of every prediction.

Let's reproduce the failure and repair it explicitly with [datarust](https://crates.io/crates/datarust).

## A useful custom transformation

Our small regression problem has two raw features:

```text
visits
spend
```

Both are non-negative and right-skewed. The target is generated from their `ln(1 + x)` values:

```text
target = 12 + 4.5 × ln(1 + visits) + 2 × ln(1 + spend)
```

That makes a log transform an honest part of the feature definition rather than decorative preprocessing.

datarust's `FunctionTransformer` lets us wrap an ordinary function pointer:

```rust
fn log1p(x: &Matrix) -> datarust::Result<Matrix> {
    Matrix::new(
        x.rows_ref()
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| value.ln_1p())
                    .collect()
            })
            .collect(),
    )
}
```

We also provide the inverse, `expm1`, so transformed values can be mapped back when a caller needs that operation.

The `serde` feature is required for JSON persistence:

```toml
[dependencies]
datarust = { version = "0.6", features = ["serde"] }
```

Here is the complete experiment:

```rust
use datarust::function_transformer::FunctionTransformer;
use datarust::linear_model::Ridge;
use datarust::pipeline::Pipeline;
use datarust::serialize::{from_json, to_json};
use datarust::traits::{Predictor, Transformer};
use datarust::transformer_kind::TransformerKind;
use datarust::Matrix;

fn log1p(x: &Matrix) -> datarust::Result<Matrix> {
    Matrix::new(
        x.rows_ref()
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| value.ln_1p())
                    .collect()
            })
            .collect(),
    )
}

fn expm1(x: &Matrix) -> datarust::Result<Matrix> {
    Matrix::new(
        x.rows_ref()
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| value.exp_m1())
                    .collect()
            })
            .collect(),
    )
}

fn max_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    let mut targets = Vec::new();

    for i in 0..80 {
        let visits = ((i * 17) % 60) as f64;
        let spend = ((i * i * 31) % 20_000) as f64;
        let row = vec![visits, spend];
        let target = 12.0
            + 4.5 * visits.ln_1p()
            + 2.0 * spend.ln_1p();

        rows.push(row);
        targets.push(target);
    }

    let training = Matrix::new(rows)?;
    let requests = Matrix::new(vec![
        vec![3.0, 120.0],
        vec![18.0, 2_500.0],
        vec![50.0, 15_000.0],
    ])?;

    let mut preprocess = Pipeline::new().push(
        "log1p",
        TransformerKind::FunctionTransformer(
            FunctionTransformer::new(log1p)
                .with_inverse(expm1),
        ),
    );
    let transformed_training =
        preprocess.fit_transform(&training)?;

    let mut model = Ridge::new().with_alpha(0.01);
    model.fit(&transformed_training, &targets)?;
    let original = model.predict(
        &preprocess.transform(&requests)?,
    )?;

    let preprocess_json = to_json(&preprocess)?;
    let model_json = to_json(&model)?;

    let mut restored_preprocess: Pipeline =
        from_json(&preprocess_json)?;
    let restored_model: Ridge = from_json(&model_json)?;

    println!(
        "loaded preprocessing fitted: {}",
        restored_preprocess.is_fitted()
    );
    println!(
        "function stored in JSON: {}",
        preprocess_json.contains("\"func\"")
    );

    match restored_preprocess.transform(&requests) {
        Ok(_) => println!(
            "transform before rebind: unexpectedly succeeded"
        ),
        Err(error) => println!(
            "transform before rebind: {error}"
        ),
    }

    let step = restored_preprocess
        .get_step_mut("log1p")
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                "missing log1p step",
            )
        })?;

    match step {
        TransformerKind::FunctionTransformer(function) => {
            function.set_func(log1p);
            function.set_inv_func(expm1);
        }
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "log1p has the wrong type",
            )
            .into())
        }
    }

    let restored = restored_model.predict(
        &restored_preprocess.transform(&requests)?,
    )?;

    println!("\npredictions");
    for (index, (before, after)) in
        original.iter().zip(&restored).enumerate()
    {
        println!(
            "request {}: before restart = {:8.3} | \
             after rebind = {:8.3}",
            index + 1,
            before,
            after,
        );
    }
    println!(
        "max prediction difference: {:.2e}",
        max_difference(&original, &restored)
    );

    Ok(())
}
```

Run it with:

```sh
cargo run --release --features serde
```

Against the current datarust codebase, I measured:

```text
loaded preprocessing fitted: true
function stored in JSON: false
transform before rebind: invalid input: FunctionTransformer: func not set (call set_func after deserialization)

predictions
request 1: before restart =   27.831 | after rebind =   27.831
request 2: before restart =   40.899 | after rebind =   40.899
request 3: before restart =   48.924 | after rebind =   48.924
max prediction difference: 3.55e-15
```

The JSON round trip succeeded. The loaded preprocessing state was still incomplete for execution.

## “Fitted” answered a narrower question

The surprising line is this one:

```text
loaded preprocessing fitted: true
```

`FunctionTransformer` remembers that `fit` was called and how many input features it saw. Those fields are serializable data. The enclosing `Pipeline` asks whether each step is fitted, so it also reports `true` after loading.

That status does not promise that every external runtime dependency is attached.

For an ordinary `StandardScaler`, fitted state includes the values needed to run: training means and standard deviations. For PCA, it includes learned components. For an encoder, it includes the vocabulary.

A custom function is different. Its behavior lives in compiled application code. The serialized transformer can remember its place, name, fitted flag, and feature count, but the code performing the transformation must already exist in the new binary.

I now read `is_fitted()` as:

> The object's learnable state has been fitted.

I do not read it as:

> Every resource required by this application is ready.

Readiness is a larger deployment check.

## A function pointer is not a portable recipe

Rust can call a function pointer inside the current executable. Serializing the pointer value would not make its code portable.

An address meaningful in one process may be meaningless after a restart. A different build can arrange code differently. Another machine may run a different architecture or application version. JSON has no standard representation for “copy this compiled function and safely execute it over there.”

datarust therefore skips both the forward and inverse function pointers during serde serialization. After deserialization they are `None`, and attempting to transform returns a clear error instead of guessing.

That failure is useful. The dangerous alternative would be a pipeline that silently substitutes identity behavior and produces plausible predictions in the wrong feature space.

The model in this experiment learned coefficients for:

```text
ln(1 + visits)
ln(1 + spend)
```

Passing raw `visits` and `spend` to those coefficients would preserve the two-column shape while changing the meaning and scale of both features. A loud first-request failure is much easier to diagnose than quiet semantic corruption.

## Rebinding is dependency injection for feature code

The repair is intentionally explicit:

```rust
let step = restored_preprocess
    .get_step_mut("log1p")
    .ok_or_else(/* configuration error */)?;

match step {
    TransformerKind::FunctionTransformer(function) => {
        function.set_func(log1p);
        function.set_inv_func(expm1);
    }
    _ => return Err(/* wrong step type */),
}
```

The artifact says that a step named `log1p` belongs at that position. The serving binary supplies the implementation associated with that contract.

I verify three things before accepting the artifact:

- The expected step name exists.
- It is actually a `FunctionTransformer`.
- The application has an implementation for the artifact's expected transformation version.

The example also restores `expm1`. Prediction only needs the forward function, so forgetting the inverse would not affect these three outputs. It would fail later when another code path called `inverse_transform`. Rebinding both sides together avoids a delayed surprise.

## The step name became part of the artifact interface

`"log1p"` looks like a friendly label in the training code:

```rust
Pipeline::new().push("log1p", ...)
```

After deployment uses that name to locate and rebind behavior, it becomes part of the artifact interface.

Renaming it to `log_features` in a later binary is no longer cosmetic. An older artifact still asks for `log1p`. Changing the formula while keeping the old name is worse: the artifact loads, the new function binds, and the fitted estimator receives a different feature definition without an obvious structural error.

For production custom steps, I prefer explicit versioned identifiers:

```text
log1p_nonnegative_v1
currency_normalization_v3
legacy_ratio_policy_v2
```

The identifier is not a substitute for artifact metadata, but it makes accidental incompatibility harder to hide. A registry can reject an unknown version instead of falling back to whatever implementation has a similar name.

## Rebinding is not refitting

Calling `set_func(log1p)` restores executable behavior. It does not call `fit`, recalculate model coefficients, or let serving data influence training state.

That is the correct boundary:

```text
training process
    fit preprocessing state
    fit Ridge coefficients
    serialize both

serving process
    deserialize fitted state
    bind approved custom functions
    run golden predictions
    accept traffic
```

Refitting after load would be a different operation. It could make predictions depend on the first request batch, leak evaluation data, or create coordinates inconsistent with the stored estimator.

The custom function in this example happens to be stateless, so binding it is enough. If custom preprocessing needs learned parameters—a lookup table, selected threshold, exchange-rate snapshot, or feature vocabulary—those values must also be represented as serializable state with a clearly defined lifecycle. Hiding them in global variables or network calls makes the model artifact incomplete again.

## Why the example keeps preprocessing and Ridge separate

datarust can serialize a fitted `SupervisedPipeline` containing built-in transformers and an estimator as one convenient object. That is the preferred path when every component is fully reconstructible from serialized state.

This example persists the numeric `Pipeline` and `Ridge` separately because it needs mutable access to the loaded preprocessing step before prediction. The current `Pipeline` API exposes `get_step_mut`, making the rebind explicit.

That separation adds an operational obligation: the two JSON values must be versioned, validated, promoted, and rolled back as one logical model. Loading preprocessing from release A with Ridge from release B can be just as wrong as losing the function.

For many applications, the simpler choice is to avoid a custom function in the persisted graph when a concrete datarust transformer already expresses the operation. Built-in transformers carry typed configuration and fitted state without requiring application code to search by name and reattach behavior.

`FunctionTransformer` is most useful when the transformation is genuinely application-specific and the deployment contract is worth owning.

## A golden prediction test catches more than JSON validity

Successful deserialization proved only that the JSON matched the Rust data structure. It did not prove the artifact could transform a request.

Before serving, I run a small golden set through the complete raw-input path:

```text
known raw rows
    -> loaded preprocessing
    -> rebound functions
    -> loaded estimator
    -> predictions
```

The predictions must match values recorded during model promotion within an explicit tolerance. For this experiment, the largest difference is `3.55e-15`.

That check catches:

- a missing function binding,
- a wrong custom-function version,
- reordered preprocessing steps,
- a mismatched estimator file,
- changed feature order or units,
- damaged fitted parameters,
- numerical changes large enough to matter.

I also execute every supported inverse path if the application uses it. A forward-only health check will not notice that `set_inv_func` was forgotten.

## Custom code deserves the same review as model code

The transformation is short enough to look harmless:

```rust
value.ln_1p()
```

It still defines model semantics. A future edit from `ln_1p()` to `ln()` changes behavior at every value and becomes invalid at zero. Applying the function to an unexpected negative number can produce `NaN`. Swapping column order can apply a mathematically valid transform to the wrong feature.

I treat the function like a versioned piece of the model:

- Validate its input domain.
- Test exact boundary cases.
- Record its version with the artifact.
- Keep the implementation deterministic.
- Avoid hidden I/O and mutable global state.
- Compare training and serving binaries for compatibility.
- Fail startup if a required binding is unavailable.

Rebinding arbitrary functions named by an untrusted artifact would also be a poor plugin system. The serving application should expose a small allowlisted registry of approved transformations, not dynamically execute code described by JSON.

The artifact selects from code already compiled into the trusted binary. It does not provide the code.

## The restart exposed the real model boundary

Before persistence, it was easy to think of `log1p` as a line of preprocessing and Ridge as the model.

After restart, the boundary was clearer:

```text
raw feature schema
+ custom transformation implementation
+ fitted preprocessing state
+ estimator parameters
+ compatible application version
= executable prediction system
```

The JSON contained the serializable parts. The binary contained the function. The deployment contract had to join them deliberately.

Once the binding was restored, all three predictions matched the originals:

```text
27.831
40.899
48.924
```

The problem was never that serialization failed.

It serialized exactly what data serialization can preserve.

The missing piece was code—and code has to arrive with the application.

---

*The complete example and its reported output were run against the current datarust codebase with the `serde` feature enabled. You can find the crate on [crates.io](https://crates.io/crates/datarust).*
