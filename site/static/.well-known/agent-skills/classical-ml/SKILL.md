---
name: classical-ml
description: Scikit-learn-style preprocessing, feature engineering, and classical machine learning estimators in Rust with datarust. Use this skill when scaling features, encoding categorical columns, fitting classical estimators, or building ML pipelines.
compatibility: Claude, Cursor, OpenAI, Antigravity, LLM agents
when_to_use: When an agent or developer needs scikit-learn-style preprocessing, feature scaling, encoding, missing-value imputation, dimensionality reduction (PCA), regression, classification, or clustering in pure Rust.
---

# Classical Machine Learning & Preprocessing with datarust

## When to use this skill
Use this skill when:
- Preprocessing numerical and categorical features: StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder, SimpleImputer.
- Dimensionality reduction and matrix decomposition: PCA, TruncatedSVD.
- Fitting and evaluating classical estimators: LinearRegression, Ridge, LogisticRegression, KMeans.
- Composing multi-step machine learning workflows with Pipeline and ColumnTransformer.
- Running fast, pure-Rust machine learning with zero external dependencies and exact scikit-learn bit-level parity.

## Features
- Preprocessing: StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder, SimpleImputer.
- Decomposition: PCA, TruncatedSVD.
- Estimators: LinearRegression, Ridge, LogisticRegression, KMeans.
- Composition: Pipeline and ColumnTransformer.

## Quick Start
```rust
use datarust::prelude::*;

let scaler = StandardScaler::default().fit(&x)?;
let x_scaled = scaler.transform(&x)?;
```
