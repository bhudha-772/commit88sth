#!/usr/bin/env python3
# timeseries_pipeline.py
"""
Time-series pipeline + simple backtester for H.E.R.O.

This version is robust to missing heavy deps (numpy, pandas). If those libs
aren't installed the module still imports cleanly; methods that require
numpy/pandas will raise a clear RuntimeError if invoked.
"""
from __future__ import annotations
import math
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

# Optional heavy deps (import lazily / safely)
try:
    import numpy as np  # type: ignore
    HAS_NUMPY = True
except Exception:
    np = None  # type: ignore
    HAS_NUMPY = False

try:
    import pandas as pd  # type: ignore
    HAS_PANDAS = True
except Exception:
    pd = None  # type: ignore
    HAS_PANDAS = False

# scikit-learn optional (only used in training)
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # type: ignore
    from sklearn.metrics import accuracy_score, mean_squared_error  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:
    RandomForestClassifier = None  # type: ignore
    RandomForestRegressor = None  # type: ignore
    accuracy_score = None
    mean_squared_error = None
    SKLEARN_AVAILABLE = False

try:
    import joblib  # type: ignore
    HAS_JOBLIB = True
except Exception:
    joblib = None  # type: ignore
    HAS_JOBLIB = False


@dataclass
class BacktestResult:
    accuracy: float
    mse: Optional[float]
    cumulative_return: float
    sharpe: float
    max_drawdown: float
    trades: int
    metadata: dict


class DependencyError(RuntimeError):
    pass


class TimeSeriesPipeline:
    def __init__(self, model_store: str = "ts_models"):
        self.model_store = Path(model_store)
        self.model_store.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Feature engineering utils
    # -------------------------
    def create_features(
        self,
        df,
        price_col: str = "price",
        lags = (1, 2, 3, 5, 10),
        rolling_windows = (5, 10, 20),
    ):
        """
        Accepts df with df[price_col] and returns df with numeric features.
        Requires pandas.
        """
        if not HAS_PANDAS:
            raise DependencyError("create_features requires pandas. Install pandas to use pipeline features.")
        data = df.copy().sort_index()
        # returns
        data["return_1"] = data[price_col].pct_change().fillna(0)
        # lag features
        for lag in lags:
            data[f"lag_{lag}"] = data["return_1"].shift(lag).fillna(0)
        # rolling stats on returns
        for w in rolling_windows:
            data[f"roll_mean_{w}"] = data["return_1"].rolling(w, min_periods=1).mean().fillna(0)
            data[f"roll_std_{w}"] = data["return_1"].rolling(w, min_periods=1).std().fillna(0)
        # momentum
        data["momentum_5"] = data[price_col].pct_change(5).fillna(0)
        # drop rows with NaNs if any
        data = data.dropna()
        return data

    # -------------------------
    # Build labelled dataset
    # -------------------------
    def build_dataset(
        self,
        df_features,
        price_col: str = "price",
        horizon: int = 1,
        task: str = "classification",
    ):
        """
        Create target for next-period return: classification (direction) or regression (future return).
        horizon: number of periods ahead to predict.
        task: 'classification' or 'regression'
        """
        if not HAS_PANDAS:
            raise DependencyError("build_dataset requires pandas.")
        df = df_features.copy()
        df["future_return"] = df[price_col].pct_change(periods=horizon).shift(-horizon)
        df = df.dropna()
        if task == "classification":
            # direction (1 for up, 0 for down/flat)
            y = (df["future_return"] > 0).astype(int)
        else:
            y = df["future_return"]
        X = df.drop(columns=[price_col, "future_return"])
        return X, y

    # -------------------------
    # Walk-forward training
    # -------------------------
    def walk_forward_train(
        self,
        X,
        y,
        initial_train_size: int = 200,
        step: int = 50,
        model_type: str = "classifier",
        model_params: Optional[dict] = None,
    ):
        """
        Perform expanding-window (walk-forward) training and return list of fitted models + validation predictions.
        This trains models on increasing slices and predicts on the next 'step' chunk each time.
        Requires scikit-learn.
        """
        if not SKLEARN_AVAILABLE:
            raise DependencyError("walk_forward_train requires scikit-learn (RandomForest).")
        n = X.shape[0]
        if HAS_NUMPY:
            idx = np.arange(n)
        else:
            idx = list(range(n))
        models: List[Any] = []
        val_preds: List[Any] = []
        val_trues: List[Any] = []
        val_index: List[Any] = []

        model_params = model_params or {}
        pos = initial_train_size
        while pos + step <= n:
            train_idx = idx[:pos]
            test_idx = idx[pos: pos + step]
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            if model_type == "classifier":
                model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, **model_params)
            else:
                model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42, **model_params)

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            models.append(model)
            val_preds.extend(preds.tolist())
            val_trues.extend(y_test.tolist())
            val_index.extend(X_test.index.tolist())

            pos += step

        # return predictions aligned with val_index as a DataFrame if possible
        if HAS_PANDAS:
            val_df = pd.DataFrame({"pred": val_preds, "true": val_trues}, index=val_index).sort_index()
        else:
            val_df = {"pred": val_preds, "true": val_trues, "index": val_index}
        return models, val_df

    # -------------------------
    # Backtester (improved)
    # -------------------------
    def simple_backtest(
        self,
        price_series,
        pred_series,
        trade_cost: float = 0.001,
        max_leverage: float = 1.0,
    ):
        """
        Backtest requires pandas Series for price_series and pred_series.
        """
        if not HAS_PANDAS:
            raise DependencyError("simple_backtest requires pandas.")
        df = pd.DataFrame({"price": price_series}).sort_index()
        df["pred"] = pred_series.reindex(df.index).fillna(0).astype(float)
        df["return"] = df["price"].pct_change().fillna(0)

        # position: clip to [0, max_leverage] (long-only). Use pred as signal to be long.
        df["position"] = df["pred"].shift(0).fillna(0).clip(0, max_leverage)

        # compute trade cost whenever position changes (abs delta * trade_cost)
        df["trade_cost"] = df["position"].diff().abs() * trade_cost

        # strategy returns
        df["strategy_ret"] = df["position"] * df["return"] - df["trade_cost"]
        df["cum_ret"] = (1 + df["strategy_ret"]).cumprod() - 1

        # baseline buy-and-hold: always long at max_leverage
        bh_returns = df["return"] * max_leverage
        bh_cum = (1 + bh_returns).cumprod() - 1

        returns = df["strategy_ret"].dropna()
        if len(returns) == 0:
            return {
                "cumulative_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "trades": 0,
                "returns_series": df["strategy_ret"].to_list(),
                "baseline_cumulative_return": float(bh_cum.iloc[-1]) if len(bh_cum) else 0.0,
            }

        avg = returns.mean()
        std = returns.std(ddof=0) if returns.std(ddof=0) > 0 else 1e-9
        sharpe = (avg / std) * (math.sqrt(252))
        cum_ret = df["cum_ret"].iloc[-1]

        # max drawdown
        running_val = (1 + df["strategy_ret"]).cumprod()
        running_max = running_val.cummax()
        drawdown = (running_val / running_max - 1)
        max_dd = float(drawdown.min())
        trades = int((df["position"].diff().abs() > 0).sum())

        return {
            "cumulative_return": float(cum_ret),
            "sharpe": float(sharpe),
            "max_drawdown": max_dd,
            "trades": trades,
            "returns_series": df["strategy_ret"].tolist(),
            "baseline_cumulative_return": float(bh_cum.iloc[-1]) if len(bh_cum) else 0.0,
        }

    # -------------------------
    # Train and backtest wrapper
    # -------------------------
    def train_and_backtest(
        self,
        price_df,
        price_col: str = "price",
        horizon: int = 1,
        task: str = "classification",
        initial_train_size: int = 200,
        step: int = 50,
        model_type: str = "classifier",
        save_model_id: Optional[str] = None,
    ) -> BacktestResult:
        """
        High-level flow:
          - create features
          - build dataset (X,y)
          - walk-forward train to get val predictions
          - backtest predictions
          - compute metrics & optionally save final model
        """
        if not HAS_PANDAS:
            raise DependencyError("train_and_backtest requires pandas (for price_df argument).")
        df_feat = self.create_features(price_df, price_col=price_col)
        X, y = self.build_dataset(df_feat, price_col=price_col, horizon=horizon, task=task)

        # keep aligned price series for backtest (use index of X)
        price_series = price_df[price_col].reindex(X.index)

        models, val_df = self.walk_forward_train(
            X,
            y,
            initial_train_size=initial_train_size,
            step=step,
            model_type=model_type,
        )

        # predictions for backtest: use val_df['pred'] (might be floats for regression)
        # for classification ensure binary 0/1
        if isinstance(val_df, dict):
            # fallback simple summarization when sklearn/pandas not available
            acc = 0.0
            mse = None
            bt = {"cumulative_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "trades": 0, "returns_series": []}
        else:
            if task == "classification":
                pred_col = val_df["pred"]
            else:
                pred_col = (pd.Series(val_df["pred"]) > 0).astype(int).values
            if task == "classification":
                pred_series = pd.Series(val_df["pred"].astype(int), index=val_df.index).sort_index()
            else:
                pred_series = pd.Series((pd.Series(val_df["pred"]) > 0).astype(int).values, index=val_df.index).sort_index()

            bt = self.simple_backtest(price_series, pred_series)

            # evaluation metrics
            acc = None
            mse = None
            try:
                if task == "classification" and accuracy_score:
                    acc = accuracy_score(val_df["true"], val_df["pred"])
                elif mean_squared_error:
                    mse = mean_squared_error(val_df["true"], val_df["pred"])
            except Exception:
                pass

        # optionally save the last model
        if save_model_id and models and HAS_JOBLIB:
            last_model = models[-1]
            joblib.dump(last_model, str(self.model_store / f"{save_model_id}.joblib"))

        baseline = bt.get("baseline_cumulative_return", None)
        metadata = {"n_models_trained": len(models) if isinstance(models, list) else 0, "task": task, "horizon": horizon, "baseline_return": baseline}
        return BacktestResult(
            accuracy=float(acc) if acc is not None else 0.0,
            mse=float(mse) if mse is not None else None,
            cumulative_return=float(bt.get("cumulative_return", 0.0)),
            sharpe=float(bt.get("sharpe", 0.0)),
            max_drawdown=float(bt.get("max_drawdown", 0.0)),
            trades=int(bt.get("trades", 0)),
            metadata=metadata,
        )


# Quick synthetic demo left as-is but requires numpy/pandas to run
def generate_synthetic_price(n: int = 1000, seed: int = 42, drift: float = 0.0):
    if not HAS_PANDAS or not HAS_NUMPY:
        raise DependencyError("generate_synthetic_price requires numpy and pandas.")
    np.random.seed(seed)
    t = np.arange(n)
    seasonal = 0.002 * np.sin(2 * np.pi * t / 50)
    noise = 0.01 * np.random.randn(n)
    trend = drift * t
    returns = seasonal + noise + trend
    price = 100 * (1 + returns).cumprod()
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({"price": price}, index=dates)
    return df


if __name__ == "__main__":
    print("Running synthetic demo for TimeSeriesPipeline...")
    if not HAS_PANDAS:
        print("pandas not installed; demo unavailable.")
    else:
        pipeline = TimeSeriesPipeline()
        df = generate_synthetic_price(800)
        result = pipeline.train_and_backtest(
            df,
            price_col="price",
            horizon=1,
            task="classification",
            initial_train_size=300,
            step=50,
            save_model_id="demo_classifier_v1",
        )
        print("Backtest summary:")
        print(" Accuracy:", result.accuracy)
        print(" MSE:", result.mse)
        print(" Cumulative return:", result.cumulative_return)
        print(" Sharpe:", result.sharpe)
        print(" Max drawdown:", result.max_drawdown)
        print(" Trades:", result.trades)
        print(" Metadata:", json.dumps(result.metadata))
