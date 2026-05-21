import os
import time
import glob
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "molecule_data_preprocessed.csv")
MODELS_DIR   = os.path.join(SCRIPT_DIR, "saved_models")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "plots")
RESULTS_CSV  = os.path.join(SCRIPT_DIR, "inference_benchmark_results.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

EMBEDDING_DIM = 32
EMBEDDING_COLS = [f"emb_{i}" for i in range(EMBEDDING_DIM)]

TARGET_COLS = [
    "dipole_moment", "polarizability", "homo_energy", "lumo_energy",
    "homo_lumo_gap", "spatial_extent", "zero_point_energy",
    "internal_energy_0K", "heat_capacity",
]

FILE_TO_NAME = {
    "random_forest":       "Random Forest",
    "gradient_boosting":   "Gradient Boosting",
    "mlp_neural_network":  "MLP Neural Network",
    "svr_linear":          "SVR (Linear)",
    "k-nearest_neighbors": "K-Nearest Neighbors",
}

MODEL_COLORS = {
    "Random Forest":       "#2ecc71",
    "Gradient Boosting":   "#3498db",
    "MLP Neural Network":  "#e74c3c",
    "SVR (Linear)":        "#9b59b6",
    "K-Nearest Neighbors": "#f39c12",
}


def load_models():
    models = {}
    pattern = os.path.join(MODELS_DIR, "*.joblib")
    for path in sorted(glob.glob(pattern)):
        basename = os.path.splitext(os.path.basename(path))[0]
        if basename.startswith("scaler"):
            continue
        display = FILE_TO_NAME.get(basename, basename)
        models[display] = joblib.load(path)
        print(f"  Loaded: {display}  ({path})")
    return models


def benchmark_single(model, X, runs=20):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(X)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = np.array(times)
    mean_t = times.mean()
    std_t  = times.std()
    per_sample_ms = (mean_t / X.shape[0]) * 1000
    return mean_t, std_t, per_sample_ms, times


def plot_inference_bar(results):
    names  = [r["Model"] for r in results]
    means  = [r["Mean Time (s)"] for r in results]
    stds   = [r["Std Time (s)"] for r in results]
    colors = [MODEL_COLORS.get(n, "#888888") for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(names, means, xerr=stds, color=colors,
                   edgecolor="white", linewidth=1.2, capsize=4)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_width() + s + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{m:.4f} ± {s:.4f} s",
                va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Prediction Time (seconds)", fontsize=12)
    ax.set_title("Inference Time — Full Test Set", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "14_inference_time_total.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


def plot_per_sample(results):
    names   = [r["Model"] for r in results]
    per_ms  = [r["Per Sample (ms)"] for r in results]
    colors  = [MODEL_COLORS.get(n, "#888888") for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(names, per_ms, color=colors,
                   edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, per_ms):
        ax.text(bar.get_width() * 1.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f} ms",
                va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Time per Sample (ms)", fontsize=12)
    ax.set_title("Inference Time — Per Molecule", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "15_inference_time_per_sample.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


def plot_box(all_times):
    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(all_times.keys())
    data  = [all_times[n] for n in names]
    colors = [MODEL_COLORS.get(n, "#888888") for n in names]

    bp = ax.boxplot(data, vert=False, patch_artist=True, labels=names)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_xlabel("Prediction Time (seconds)", fontsize=12)
    ax.set_title("Inference Time Distribution (multiple runs)",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "16_inference_time_boxplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inference Benchmark")
    parser.add_argument("--sample", type=int, default=None,
                        help="Predict on N test samples (default: full test set)")
    parser.add_argument("--runs", type=int, default=20,
                        help="Number of repeated runs for timing (default: 20)")
    args = parser.parse_args()
    df = pd.read_csv(DATASET_PATH)
    print(f"   Full dataset: {df.shape[0]:,} molecules")

    X = df[EMBEDDING_COLS].values
    y = df[TARGET_COLS].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if args.sample and args.sample < len(X_test):
        idx = np.random.RandomState(42).choice(len(X_test), args.sample, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]
        print(f"   [SAMPLE] Using {args.sample:,} test samples")
    else:
        print(f"   [FULL] Using all {len(X_test):,} test samples")

    scaler_X_path = os.path.join(MODELS_DIR, "scaler_X.joblib")
    scaler_y_path = os.path.join(MODELS_DIR, "scaler_y.joblib")

    if not os.path.exists(scaler_X_path):
        print("\n[ERROR] Scalers not found!  Run model_comparison.py first.")
        return

    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    X_test_s = scaler_X.transform(X_test)

    models = load_models()
    if not models:
        print("[ERROR] No models found in saved_models/. Run model_comparison.py first.")
        return

    print(f"\n[BENCH] Benchmarking inference ({args.runs} runs each)...\n")
    results = []
    all_times = {}

    header = f"  {'Model':<25s} {'Mean (s)':>10s} {'Std (s)':>10s} {'Per Sample':>12s} {'Samples':>8s}"
    print(header)
    print(f"  {'-' * 70}")

    for name, model in models.items():
        mean_t, std_t, per_ms, raw_times = benchmark_single(
            model, X_test_s, runs=args.runs
        )
        all_times[name] = raw_times
        row = {
            "Model": name,
            "Mean Time (s)": round(mean_t, 6),
            "Std Time (s)": round(std_t, 6),
            "Per Sample (ms)": round(per_ms, 6),
            "Samples": X_test_s.shape[0],
            "Runs": args.runs,
        }
        results.append(row)
        print(f"  {name:<25s} {mean_t:10.4f} {std_t:10.4f} {per_ms:10.4f} ms {X_test_s.shape[0]:>8,}")

    fastest = min(results, key=lambda r: r["Mean Time (s)"])
    print(f"\n  >> Fastest Model: {fastest['Model']} "
          f"({fastest['Mean Time (s)']:.4f}s, "
          f"{fastest['Per Sample (ms)']:.4f} ms/sample)")
    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[SAVE] Results: {RESULTS_CSV}")
    plot_inference_bar(results)
    plot_per_sample(results)
    plot_box(all_times)

if __name__ == "__main__":
    main()