# PUBLIC REVIEW: computational expressions retained; provenance comments added.
# Formula explanations and package boundaries: docs/CALCULATIONS.md.
# Full function/source index: docs/FUNCTION_INDEX.md.
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser(description="Reproduce the manuscript's NSSP fit-only K=2..8 diagnostics; does not train supervised models or replace reference results.")
parser.add_argument("--repo", type=Path, required=True, help="Repository containing src/ and data/raw/NSSP.csv")
parser.add_argument("--output", type=Path, help="New diagnostics directory (default: results/reproduced/manuscript_kmeans)")
args = parser.parse_args()
PROJECT = args.repo.resolve()
OUT = (args.output or PROJECT / "results/reproduced/manuscript_kmeans").resolve()
if OUT == (PROJECT / "results/reference").resolve() or (PROJECT / "results/reference").resolve() in OUT.parents:
    raise ValueError("Refusing to overwrite preserved reference results")
FIG = OUT / "figure_kmeans_diagnostics.png"
OUT.mkdir(parents=True, exist_ok=True)
FIG.parent.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT / "src"))

import common_config
from common_features import (
    add_time_series_features,
    choose_feature_columns,
    chronological_split,
    chronological_validation_split,
)
from common_io import load_nssp


TARGET = "percent_visits_combined"
GROUP_COLS = ["geography", "county"]
LAGGED_COVARIATES = {column: [1] for column in common_config.NSSP_COMPONENT_COLUMNS}
K_VALUES = list(range(2, 9))


# Calculation provenance:
#   add_time_series_features -> common_features.add_time_series_features (src/common_features.py)
#   choose_feature_columns -> common_features.choose_feature_columns (src/common_features.py)
#   chronological_split -> common_features.chronological_split (src/common_features.py)
#   chronological_validation_split -> common_features.chronological_validation_split (src/common_features.py)
#   load_nssp -> common_io.load_nssp (src/common_io.py)
def prepare_frames():
    raw = load_nssp(PROJECT / "data" / "raw" / "NSSP.csv", target=TARGET)
    engineered = add_time_series_features(
        raw,
        target=TARGET,
        group_cols=GROUP_COLS,
        lagged_covariates=LAGGED_COVARIATES,
    )
    features = choose_feature_columns(engineered, target=TARGET, dataset_type="nssp")
    modeling = engineered.dropna(subset=[TARGET] + features).copy()
    train, test, test_cutoff = chronological_split(modeling)
    fit, validation, validation_cutoff = chronological_validation_split(train)
    return features, modeling, train, test, fit, validation, test_cutoff, validation_cutoff


# Calculation provenance:
#   KMeans -> sklearn.cluster.KMeans (third-party package)
#   StandardScaler -> sklearn.preprocessing.StandardScaler (third-party package)
#   calinski_harabasz_score -> sklearn.metrics.calinski_harabasz_score (third-party package)
#   davies_bouldin_score -> sklearn.metrics.davies_bouldin_score (third-party package)
#   np.bincount -> numpy.bincount (third-party package)
#   np.bincount(labels).max -> numpy.bincount(labels).max (third-party package)
#   np.bincount(labels).min -> numpy.bincount(labels).min (third-party package)
#   pd.DataFrame -> pandas.DataFrame (third-party package)
#   silhouette_score -> sklearn.metrics.silhouette_score (third-party package)
def candidate_metrics(fit: pd.DataFrame, features: list[str]):
    scaler = StandardScaler()
    x = scaler.fit_transform(fit[features])
    rows = []
    for k in K_VALUES:
        km = KMeans(
            n_clusters=k,
            random_state=common_config.RANDOM_STATE,
            n_init=common_config.KMEANS_N_INIT,
        )
        labels = km.fit_predict(x)
        sample_size = min(common_config.SILHOUETTE_SAMPLE_SIZE, len(x))
        rows.append({
            "k": k,
            "silhouette": silhouette_score(
                x,
                labels,
                sample_size=sample_size if sample_size < len(x) else None,
                random_state=common_config.RANDOM_STATE,
            ),
            "inertia_per_row": km.inertia_ / len(x),
            "calinski_harabasz": calinski_harabasz_score(x, labels),
            "davies_bouldin": davies_bouldin_score(x, labels),
            "min_cluster_n": int(np.bincount(labels).min()),
            "max_cluster_n": int(np.bincount(labels).max()),
        })
    metrics = pd.DataFrame(rows)
    xnorm = (metrics["k"] - metrics["k"].min()) / (metrics["k"].max() - metrics["k"].min())
    ynorm = (metrics["inertia_per_row"] - metrics["inertia_per_row"].min()) / (
        metrics["inertia_per_row"].max() - metrics["inertia_per_row"].min()
    )
    metrics["elbow_distance"] = (1.0 - xnorm) - ynorm
    metrics["elbow_selected"] = metrics["elbow_distance"] == metrics["elbow_distance"].max()
    metrics["configured_k"] = metrics["k"] == common_config.N_CLUSTERS
    return metrics, scaler, x


# Calculation provenance:
#   KMeans -> sklearn.cluster.KMeans (third-party package)
#   StandardScaler -> sklearn.preprocessing.StandardScaler (third-party package)
#   np.arange -> numpy.arange (third-party package)
#   np.arctan2 -> numpy.arctan2 (third-party package)
#   np.bincount -> numpy.bincount (third-party package)
#   np.mod -> numpy.mod (third-party package)
#   np.where -> numpy.where (third-party package)
#   pd.DataFrame -> pandas.DataFrame (third-party package)
def final_profiles(train: pd.DataFrame, features: list[str]):
    scaler = StandardScaler()
    x = scaler.fit_transform(train[features])
    km = KMeans(
        n_clusters=common_config.N_CLUSTERS,
        random_state=common_config.RANDOM_STATE,
        n_init=common_config.KMEANS_N_INIT,
    )
    labels = km.fit_predict(x)
    counts = np.bincount(labels, minlength=common_config.N_CLUSTERS)
    centroids = pd.DataFrame(km.cluster_centers_, columns=features)

    level_cols = ["lag_1", "lag_2", "lag_3", "rolling_mean_3"]
    centroids["burden_level_z"] = centroids[level_cols].mean(axis=1)
    centroids["variability_z"] = centroids["rolling_std_3"]
    raw_centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=features)
    angle = np.arctan2(raw_centers["week_sin"], raw_centers["week_cos"])
    phase_week = np.mod(angle, 2 * np.pi) * 52.0 / (2 * np.pi)
    phase_week = np.where(phase_week < 1, phase_week + 52, phase_week)

    profile = pd.DataFrame({
        "Cluster": np.arange(common_config.N_CLUSTERS),
        "Rows": counts,
        "Share_Percent": counts / counts.sum() * 100.0,
        "Burden_Level_z": centroids["burden_level_z"],
        "Variability_z": centroids["variability_z"],
        "COVID_Lag1_z": centroids["percent_visits_covid_lag_1"],
        "Influenza_Lag1_z": centroids["percent_visits_influenza_lag_1"],
        "RSV_Lag1_z": centroids["percent_visits_rsv_lag_1"],
        "Approx_Phase_Week": phase_week,
    })
    return scaler, km, x, labels, centroids, profile


# Calculation provenance:
#   PCA -> sklearn.decomposition.PCA (third-party package)
#   np.abs -> numpy.abs (third-party package)
#   np.abs(data).max -> numpy.abs(data).max (third-party package)
#   np.asarray -> numpy.asarray (third-party package)
#   np.flatnonzero -> numpy.flatnonzero (third-party package)
#   np.random.default_rng -> numpy.random.default_rng (third-party package)
#   plt.close -> matplotlib.pyplot.close (third-party package)
#   plt.rcParams.update -> matplotlib.pyplot.rcParams.update (third-party package)
#   plt.subplots -> matplotlib.pyplot.subplots (third-party package)
def make_figure(metrics, x, labels, centroids, profile):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.2,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.2,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.4,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), dpi=220)
    navy = "#17365D"
    orange = "#D97904"
    colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#9467BD"]

    ax = axes[0, 0]
    ax.plot(metrics["k"], metrics["silhouette"], marker="o", lw=2.0, color=navy)
    ax.axvline(common_config.N_CLUSTERS, color=orange, lw=1.4, ls="--")
    selected = metrics.loc[metrics["configured_k"]].iloc[0]
    ax.scatter([selected["k"]], [selected["silhouette"]], s=65, color=orange, zorder=4)
    ax.annotate(f"configured k=4\nsilhouette={selected['silhouette']:.3f}",
                (selected["k"], selected["silhouette"]), xytext=(18, 26),
                textcoords="offset points", arrowprops=dict(arrowstyle="->", color=orange))
    ax.set_xticks(metrics["k"])
    ax.set_xlabel("Candidate cluster count, k")
    ax.set_ylabel("Mean silhouette coefficient")
    ax.set_title("(a) Fit-only separation sensitivity")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    ax.plot(metrics["k"], metrics["inertia_per_row"], marker="s", lw=2.0, color=navy)
    ax.axvline(common_config.N_CLUSTERS, color=orange, lw=1.4, ls="--")
    elbow = metrics.loc[metrics["elbow_selected"]].iloc[0]
    ax.scatter([elbow["k"]], [elbow["inertia_per_row"]], s=65, color=orange, zorder=4)
    ax.annotate(f"maximum elbow distance: k={int(elbow['k'])}",
                (elbow["k"], elbow["inertia_per_row"]), xytext=(12, 18),
                textcoords="offset points", arrowprops=dict(arrowstyle="->", color=orange))
    ax.set_xticks(metrics["k"])
    ax.set_xlabel("Candidate cluster count, k")
    ax.set_ylabel("Within-cluster sum of squares per row")
    ax.set_title("(b) Fit-only compactness sensitivity")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    pca = PCA(n_components=2, random_state=common_config.RANDOM_STATE)
    projected = pca.fit_transform(x)
    rng = np.random.default_rng(common_config.RANDOM_STATE)
    selected_indices = []
    for cluster in range(common_config.N_CLUSTERS):
        idx = np.flatnonzero(labels == cluster)
        take = min(2500, len(idx))
        selected_indices.extend(rng.choice(idx, size=take, replace=False).tolist())
    selected_indices = np.asarray(selected_indices, dtype=int)
    for cluster in range(common_config.N_CLUSTERS):
        mask = labels[selected_indices] == cluster
        pts = projected[selected_indices][mask]
        ax.scatter(pts[:, 0], pts[:, 1], s=7, alpha=0.25, color=colors[cluster],
                   label=f"Cluster {cluster} (n={profile.loc[cluster, 'Rows']:,})", rasterized=True)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title("(c) Two-dimensional PCA view of full pre-test assignments")
    ax.legend(loc="best", frameon=False, fontsize=7.5, markerscale=2)
    ax.grid(alpha=0.18)

    ax = axes[1, 1]
    heat_cols = [
        "burden_level_z", "variability_z",
        "percent_visits_covid_lag_1", "percent_visits_influenza_lag_1",
        "percent_visits_rsv_lag_1",
    ]
    labels_x = ["Burden\nlevel", "Rolling\nvariability", "COVID\nlag 1", "Influenza\nlag 1", "RSV\nlag 1"]
    data = centroids[heat_cols].to_numpy()
    limit = max(1.0, float(np.abs(data).max()))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(labels_x)), labels_x)
    ax.set_yticks(range(common_config.N_CLUSTERS), [f"Cluster {i}" for i in range(common_config.N_CLUSTERS)])
    ax.set_title("(d) Standardized cluster-centroid profiles")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(data[i, j]) > limit * 0.55 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Standardized centroid (z score)")

    fig.suptitle("NSSP K-Means Diagnostics and Four-Cluster Profiles", fontsize=14, weight="bold", color=navy)
    fig.text(0.5, 0.008,
             "Candidate-k diagnostics use only the chronological fit segment. PCA and profiles use the full pre-test history; the final test is excluded.",
             ha="center", fontsize=8.5, style="italic", color="#444444")
    fig.tight_layout(rect=[0.02, 0.035, 0.98, 0.95])
    fig.savefig(FIG, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# Calculation provenance:
#   candidate_metrics -> local function in this file
#   final_profiles -> local function in this file
#   make_figure -> local function in this file
#   pd.DataFrame -> pandas.DataFrame (third-party package)
#   pd.DataFrame([{'Modeling_Rows': len(modeling), 'Fit_Rows_For_K_Sensitivity': len(fit), 'Validation_Rows_Excluded_From_K_Sensitivity': len(validation), 'Full_Pretest_Rows_For_Final_Profiles': len(train), 'Final_Test_Rows_Excluded': len(test), 'Validation_Cutoff': validation_cutoff, 'Test_Cutoff': test_cutoff, 'Configured_K': common_config.N_CLUSTERS, 'Configured_K_Fit_Silhouette': float(metrics.loc[metrics['configured_k'], 'silhouette'].iloc[0]), 'Elbow_Selected_K': int(metrics.loc[metrics['elbow_selected'], 'k'].iloc[0]), 'Final_Pretest_Silhouette': silhouette_score(x_train, labels, sample_size=min(common_config.SILHOUETTE_SAMPLE_SIZE, len(x_train)), random_state=common_config.RANDOM_STATE)}]).to_csv -> pandas.DataFrame([{'Modeling_Rows': len(modeling), 'Fit_Rows_For_K_Sensitivity': len(fit), 'Validation_Rows_Excluded_From_K_Sensitivity': len(validation), 'Full_Pretest_Rows_For_Final_Profiles': len(train), 'Final_Test_Rows_Excluded': len(test), 'Validation_Cutoff': validation_cutoff, 'Test_Cutoff': test_cutoff, 'Configured_K': common_config.N_CLUSTERS, 'Configured_K_Fit_Silhouette': float(metrics.loc[metrics['configured_k'], 'silhouette'].iloc[0]), 'Elbow_Selected_K': int(metrics.loc[metrics['elbow_selected'], 'k'].iloc[0]), 'Final_Pretest_Silhouette': silhouette_score(x_train, labels, sample_size=min(common_config.SILHOUETTE_SAMPLE_SIZE, len(x_train)), random_state=common_config.RANDOM_STATE)}]).to_csv (third-party package)
#   pd.DataFrame({'Feature': features}).to_csv -> pandas.DataFrame({'Feature': features}).to_csv (third-party package)
#   prepare_frames -> local function in this file
#   silhouette_score -> sklearn.metrics.silhouette_score (third-party package)
def main():
    features, modeling, train, test, fit, validation, test_cutoff, validation_cutoff = prepare_frames()
    metrics, _, _ = candidate_metrics(fit, features)
    scaler, km, x_train, labels, centroids, profile = final_profiles(train, features)
    make_figure(metrics, x_train, labels, centroids, profile)

    metrics.to_csv(OUT / "nssp_kmeans_candidate_metrics.csv", index=False)
    profile.to_csv(OUT / "nssp_kmeans_cluster_profiles.csv", index=False)
    pd.DataFrame({"Feature": features}).to_csv(OUT / "nssp_kmeans_input_features.csv", index=False)
    centroids.to_csv(OUT / "nssp_kmeans_standardized_centroids.csv", index_label="Cluster")
    pd.DataFrame([{
        "Modeling_Rows": len(modeling),
        "Fit_Rows_For_K_Sensitivity": len(fit),
        "Validation_Rows_Excluded_From_K_Sensitivity": len(validation),
        "Full_Pretest_Rows_For_Final_Profiles": len(train),
        "Final_Test_Rows_Excluded": len(test),
        "Validation_Cutoff": validation_cutoff,
        "Test_Cutoff": test_cutoff,
        "Configured_K": common_config.N_CLUSTERS,
        "Configured_K_Fit_Silhouette": float(metrics.loc[metrics["configured_k"], "silhouette"].iloc[0]),
        "Elbow_Selected_K": int(metrics.loc[metrics["elbow_selected"], "k"].iloc[0]),
        "Final_Pretest_Silhouette": silhouette_score(
            x_train,
            labels,
            sample_size=min(common_config.SILHOUETTE_SAMPLE_SIZE, len(x_train)),
            random_state=common_config.RANDOM_STATE,
        ),
    }]).to_csv(OUT / "nssp_kmeans_analysis_metadata.csv", index=False)

    print(metrics.to_string(index=False))
    print()
    print(profile.to_string(index=False))
    print(f"Saved figure: {FIG}")


if __name__ == "__main__":
    main()
