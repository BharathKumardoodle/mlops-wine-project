from __future__ import annotations
import os
import argparse
import math
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import mlflow
import mlflow.sklearn


def parse_args():
    p = argparse.ArgumentParser("Simple MLflow demo (wine prediction)")
    p.add_argument("--csv", default="data/wine_sample.csv", help="Path to CSV")
    p.add_argument("--target", default="quality", help="Target column name")
    p.add_argument("--experiment", default="wine-prediction", help="MLflow experiment name")
    p.add_argument("--run", default="run-2", help="MLflow run name")
    p.add_argument("--n-estimators", type=int, default=50)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # MLflow setup
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:7006")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    # Load CSV
    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    # ---- CRITICAL FIXES ----

    # 1. Strip column name spaces
    df.columns = df.columns.str.strip()

    if args.target not in df.columns:
        raise SystemExit(
            f"Target column '{args.target}' not found. Columns: {list(df.columns)}"
        )

    # 2. Split features / target
    X = df.drop(columns=[args.target])
    y = df[args.target]

    # 3. Remove bullets & force numeric (FIXES: '• 6.4')
    X = X.apply(lambda col: (
        col.astype(str)
           .str.replace(r"[^\d\.\-]", "", regex=True)
           .replace("", np.nan)
           .astype(float)
    ))

    # 4. Handle missing values
    X = X.fillna(X.median())

    # 5. Ensure target numeric
    y = pd.to_numeric(y, errors="coerce").fillna(y.median()).astype(int)

    # Safety checks
    assert not X.isna().any().any(), "NaN values present in X"
    assert not y.isna().any(), "NaN values present in y"

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # ---- TRAIN & LOG ----
    with mlflow.start_run(run_name=args.run):

        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        })

        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("mse", float(mse))
        mlflow.log_metric("rmse", float(rmse))
        mlflow.log_metric("r2", float(r2))

        mlflow.sklearn.log_model(model, "model")

        print("✅ Training completed successfully")
        print(f"RMSE: {rmse:.4f} | R2: {r2:.4f}")


if __name__ == "__main__":
    main()
