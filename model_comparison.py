import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "molecule_data_preprocessed.csv")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
MODELS_DIR = os.path.join(SCRIPT_DIR, "saved_models")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "model_comparison_results.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

EMBEDDING_DIM = 32
EMBEDDING_COLS = [f"emb_{i}" for i in range(EMBEDDING_DIM)]

TARGET_COLS = [
    "dipole_moment", "polarizability", "homo_energy", "lumo_energy",
    "homo_lumo_gap", "spatial_extent", "zero_point_energy",
    "internal_energy_0K", "heat_capacity",
]

TARGET_UNITS = {
    "dipole_moment": "Debye", "polarizability": "Bohr^3",
    "homo_energy": "Hartree", "lumo_energy": "Hartree",
    "homo_lumo_gap": "Hartree", "spatial_extent": "Bohr^2",
    "zero_point_energy": "Hartree", "internal_energy_0K": "Hartree",
    "heat_capacity": "cal/mol*K",
}

SHORT_NAMES = {
    "dipole_moment": "Dipole", "polarizability": "Polar.",
    "homo_energy": "HOMO", "lumo_energy": "LUMO",
    "homo_lumo_gap": "Gap", "spatial_extent": "Spatial",
    "zero_point_energy": "ZPVE", "internal_energy_0K": "U0",
    "heat_capacity": "Cv",
}


def mape_score(y_true, y_pred, epsilon=1e-10):
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

def get_models():
    return {
        "Random Forest": MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=100,
                max_depth=18,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features=0.5,
                max_samples=0.8,
                n_jobs=-1,
                random_state=42,
            )
        ),

        "Gradient Boosting": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=150,
                max_depth=8,
                learning_rate=0.1,
                min_samples_leaf=20,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=42,
            )
        ),

        "MLP Neural Network": MLPRegressor(
            hidden_layer_sizes=(256, 256, 128),
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
            batch_size=1024,
        ),

        "SVR (Linear)": MultiOutputRegressor(
            LinearSVR(
                C=1.0,
                max_iter=1000,
                random_state=42,
            )
        ),

        "K-Nearest Neighbors": MultiOutputRegressor(
            KNeighborsRegressor(
                n_neighbors=7,
                weights="distance",
                algorithm="ball_tree",
                leaf_size=50,
                n_jobs=-1,
            )
        ),
    }


def evaluate_model(y_true, y_pred, target_cols):
    per_property = []

    for i, col in enumerate(target_cols):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        per_property.append({
            "property": col,
            "MAE": mean_absolute_error(yt, yp),
            "RMSE": np.sqrt(mean_squared_error(yt, yp)),
            "R2": r2_score(yt, yp),
            "MAPE": mape_score(yt, yp),
        })

    overall = {
        "MAE": np.mean([p["MAE"] for p in per_property]),
        "RMSE": np.mean([p["RMSE"] for p in per_property]),
        "R2": r2_score(y_true, y_pred, multioutput="uniform_average"),
        "MAPE": np.mean([p["MAPE"] for p in per_property]),
    }

    return per_property, overall


MODEL_COLORS = {
    "Random Forest": "#2ecc71",
    "Gradient Boosting": "#3498db",
    "MLP Neural Network": "#e74c3c",
    "SVR (Linear)": "#9b59b6",
    "K-Nearest Neighbors": "#f39c12",
}


def plot_overall_comparison(all_results):
    models = list(all_results.keys())
    colors = [MODEL_COLORS[m] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Model Comparison - Overall Performance", fontsize=16, fontweight="bold")
    r2_vals = [all_results[m]["overall"]["R2"] for m in models]
    bars = axes[0].bar(models, r2_vals, color=colors, edgecolor="white", linewidth=1.2)
    axes[0].set_title("R2 Score", fontsize=12)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("R2")
    for bar, val in zip(bars, r2_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=25)

    mae_vals = [all_results[m]["overall"]["MAE"] for m in models]
    bars = axes[1].bar(models, mae_vals, color=colors, edgecolor="white", linewidth=1.2)
    axes[1].set_title("MAE", fontsize=12)
    axes[1].set_ylabel("MAE")
    for bar, val in zip(bars, mae_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=25)

    rmse_vals = [all_results[m]["overall"]["RMSE"] for m in models]
    bars = axes[2].bar(models, rmse_vals, color=colors, edgecolor="white", linewidth=1.2)
    axes[2].set_title("RMSE", fontsize=12)
    axes[2].set_ylabel("RMSE")
    for bar, val in zip(bars, rmse_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[2].tick_params(axis="x", rotation=25)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "09_model_comparison_overall.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


def plot_r2_per_property(all_results):
    models = list(all_results.keys())
    properties = [SHORT_NAMES[c] for c in TARGET_COLS]
    n_models = len(models)
    n_props = len(properties)

    x = np.arange(n_props)
    width = 0.15

    fig, ax = plt.subplots(figsize=(18, 7))

    for i, model_name in enumerate(models):
        r2_vals = [p["R2"] for p in all_results[model_name]["per_property"]]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, r2_vals, width, label=model_name,
                      color=MODEL_COLORS[model_name], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Property", fontsize=12)
    ax.set_ylabel("R2 Score", fontsize=12)
    ax.set_title("R2 Score per Property - All Models", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(properties, fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="R2=0.8 threshold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "10_r2_per_property.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


def plot_mae_per_property(all_results):
    models = list(all_results.keys())
    properties = [SHORT_NAMES[c] for c in TARGET_COLS]
    n_models = len(models)
    n_props = len(properties)

    x = np.arange(n_props)
    width = 0.15

    fig, ax = plt.subplots(figsize=(18, 7))

    for i, model_name in enumerate(models):
        mae_vals = [p["MAE"] for p in all_results[model_name]["per_property"]]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, mae_vals, width, label=model_name,
               color=MODEL_COLORS[model_name], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Property", fontsize=12)
    ax.set_ylabel("MAE", fontsize=12)
    ax.set_title("MAE per Property - All Models", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(properties, fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "11_mae_per_property.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


def plot_radar_chart(all_results):
    models = list(all_results.keys())
    properties = [SHORT_NAMES[c] for c in TARGET_COLS]
    n_props = len(properties)

    angles = np.linspace(0, 2 * np.pi, n_props, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for model_name in models:
        r2_vals = [max(0, p["R2"]) for p in all_results[model_name]["per_property"]]
        r2_vals += r2_vals[:1]
        ax.plot(angles, r2_vals, linewidth=2, label=model_name,
                color=MODEL_COLORS[model_name])
        ax.fill(angles, r2_vals, alpha=0.08, color=MODEL_COLORS[model_name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(properties, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("R2 Score Radar - Model Comparison", fontsize=14,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "12_radar_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


def plot_training_time(timing):
    models = list(timing.keys())
    times = [timing[m] for m in models]
    colors = [MODEL_COLORS[m] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(models, times, color=colors, edgecolor="white", linewidth=1.2)

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{t:.1f}s", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Training Time (seconds)", fontsize=12)
    ax.set_title("Training Time Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "13_training_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Model Comparison Benchmark")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Use only N molecules (sample 10000 for a quick run). "
             "Default: use entire dataset."
    )
    args = parser.parse_args()


    df = pd.read_csv(DATASET_PATH)
    print(f"   Full dataset: {df.shape[0]:,} molecules, {df.shape[1]} columns")

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"   [SAMPLE] Using {args.sample:,} molecules for quick benchmark")
    else:
        print(f"   [FULL] Using all {len(df):,} molecules")

    X = df[EMBEDDING_COLS].values
    y = df[TARGET_COLS].values
    print(f"   X: {X.shape}  |  y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[SPLIT] Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train)

    models = get_models()
    all_results = {}
    timing = {}

    for idx, (model_name, model) in enumerate(models.items(), 1):
        print(f"\n  [{idx}/{len(models)}] MODEL: {model_name}")
        t_start = time.time()

        uses_scaled_y = "MLP" in model_name or "SVR" in model_name
        if uses_scaled_y:
            model.fit(X_train_s, y_train_s)
        else:
            model.fit(X_train_s, y_train)

        t_elapsed = time.time() - t_start
        timing[model_name] = t_elapsed
        print(f"  [DONE] Trained in {t_elapsed:.1f} seconds")

        # Save the trained model
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        model_path = os.path.join(MODELS_DIR, f"{safe_name}.joblib")
        joblib.dump(model, model_path)
        print(f"  [SAVE] Model saved: {model_path}")

        y_pred_s = model.predict(X_test_s)

        if uses_scaled_y:
            y_pred = scaler_y.inverse_transform(y_pred_s)
        else:
            y_pred = y_pred_s
        per_property, overall = evaluate_model(y_test, y_pred, TARGET_COLS)
        all_results[model_name] = {
            "per_property": per_property,
            "overall": overall,
        }

        print(f"\n  {'Property':<22s} {'MAE':>9s} {'RMSE':>9s} {'R2':>9s} {'MAPE%':>9s}")
        print(f"  {'-' * 60}")
        for p in per_property:
            print(f"  {p['property']:<22s} {p['MAE']:9.4f} {p['RMSE']:9.4f} "
                  f"{p['R2']:9.4f} {p['MAPE']:8.2f}%")

        print(f"  {'-' * 60}")
        print(f"  {'OVERALL':<22s} {overall['MAE']:9.4f} {overall['RMSE']:9.4f} "
              f"{overall['R2']:9.4f} {overall['MAPE']:8.2f}%")


    print(f"\n  {'Model':<25s} {'R2':>8s} {'MAE':>10s} {'RMSE':>10s} {'MAPE%':>8s} {'Time':>8s}")

    summary_rows = []
    for model_name in all_results:
        o = all_results[model_name]["overall"]
        t = timing[model_name]
        print(f"  {model_name:<25s} {o['R2']:8.4f} {o['MAE']:10.4f} "
              f"{o['RMSE']:10.4f} {o['MAPE']:7.2f}% {t:7.1f}s")
        summary_rows.append({
            "Model": model_name,
            "R2": round(o["R2"], 4),
            "MAE": round(o["MAE"], 4),
            "RMSE": round(o["RMSE"], 4),
            "MAPE (%)": round(o["MAPE"], 2),
            "Training Time (s)": round(t, 1),
        })

    best_model = max(all_results, key=lambda m: all_results[m]["overall"]["R2"])
    best_r2 = all_results[best_model]["overall"]["R2"]
    print(f"\n  >> Best Model: {best_model} (R2 = {best_r2:.4f})")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[SAVE] Summary table: {RESULTS_CSV}")

    detail_rows = []
    for model_name in all_results:
        for p in all_results[model_name]["per_property"]:
            detail_rows.append({
                "Model": model_name,
                "Property": p["property"],
                "MAE": round(p["MAE"], 6),
                "RMSE": round(p["RMSE"], 6),
                "R2": round(p["R2"], 6),
                "MAPE (%)": round(p["MAPE"], 2),
            })
    detail_df = pd.DataFrame(detail_rows)
    detail_path = os.path.join(SCRIPT_DIR, "model_comparison_detail.csv")
    detail_df.to_csv(detail_path, index=False)
    print(f"[SAVE] Detail table: {detail_path}")

    joblib.dump(scaler_X, os.path.join(MODELS_DIR, "scaler_X.joblib"))
    joblib.dump(scaler_y, os.path.join(MODELS_DIR, "scaler_y.joblib"))
    print(f"[SAVE] Scalers saved to {MODELS_DIR}")

    plot_overall_comparison(all_results)
    plot_r2_per_property(all_results)
    plot_mae_per_property(all_results)
    plot_radar_chart(all_results)
    plot_training_time(timing)

if __name__ == "__main__":
    main()