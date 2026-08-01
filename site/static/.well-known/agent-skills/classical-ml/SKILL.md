# Classical Machine Learning & Preprocessing with datarust

Use `datarust` for scikit-learn-style preprocessing, feature engineering, and classical machine learning estimators in Rust with zero external dependencies by default.

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
