from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _get_plotting_libs():
    mpl_config_dir = Path.cwd() / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    return plt, sns



def build_shared_projection(df, pca_components, umap_neighbors, umap_min_dist, random_state):
    embedding_columns = sorted([col for col in df.columns if col.startswith("embedding_")])
    if not embedding_columns:
        raise ValueError("No embedding columns found in input dataframe.")

    matrix = df[embedding_columns].to_numpy()
    n_components = min(pca_components, matrix.shape[0], matrix.shape[1])
    if n_components < 1:
        raise ValueError("PCA requires at least one component.")

    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(matrix)

    if reduced.shape[0] > max(umap_neighbors, 4):
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=umap_neighbors,
                min_dist=umap_min_dist,
                random_state=random_state,
            )
            coords = reducer.fit_transform(reduced)
        except ImportError:
            coords = reduced[:, :2] if reduced.shape[1] >= 2 else np.pad(reduced, ((0, 0), (0, 2 - reduced.shape[1])))
    else:
        coords = reduced[:, :2] if reduced.shape[1] >= 2 else np.pad(reduced, ((0, 0), (0, 2 - reduced.shape[1])))

    projected = df.copy()
    projected["x"] = coords[:, 0]
    projected["y"] = coords[:, 1]
    return projected


def compactness_by_state(df):
    summary_rows = []
    for (state, class_name), group in df.groupby(["state", "class_name"]):
        center_x = group["x"].mean()
        center_y = group["y"].mean()
        radius = (
            ((group["x"] - center_x) ** 2 + (group["y"] - center_y) ** 2) ** 0.5
        ).mean()
        summary_rows.append(
            {"state": state, "class_name": class_name, "mean_radius": radius}
        )
    return pd.DataFrame(summary_rows)


def save_state_scatter(df, state_name, class_name, output_path, palette, alpha, point_size, dpi):
    subset = df[(df["state"] == state_name) & (df["class_name"] == class_name)]
    if subset.empty:
        return
    plt, sns = _get_plotting_libs()  # type: ignore[assignment]
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=subset,
        x="x",
        y="y",
        color=palette.get(class_name, "#333"),
        alpha=alpha,
        s=point_size,
        edgecolor=None,
    )
    plt.title(f"{class_name} - {state_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_state_density(df, state_name, class_name, output_path, palette, dpi):
    subset = df[(df["state"] == state_name) & (df["class_name"] == class_name)]
    if subset.empty:
        return
    plt, sns = _get_plotting_libs()
    plt.figure(figsize=(6, 5))
    sns.kdeplot(
        data=subset,
        x="x",
        y="y",
        fill=True,
        cmap="Reds" if class_name == "lesion" else "Blues",
        levels=5,
    )
    plt.title(f"{class_name} density - {state_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_joint_scatter(df, output_path, palette, alpha, point_size, dpi):
    if df.empty:
        return
    plt, sns = _get_plotting_libs()
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="class_name",
        style="state",
        palette=palette,
        alpha=alpha,
        s=point_size,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def run_reduction_and_plot(
    before_csv,
    after_csv,
    artifacts_dir,
    pca_components,
    umap_neighbors,
    umap_min_dist,
    random_state,
    alpha,
    point_size,
    dpi,
):
    before_df = pd.read_csv(before_csv)
    after_df = pd.read_csv(after_csv)
    combined = pd.concat([before_df, after_df], ignore_index=True)

    projected = build_shared_projection(
        combined, pca_components, umap_neighbors, umap_min_dist, random_state
    )

    plots_dir = Path(artifacts_dir) / "plots"
    metrics_dir = Path(artifacts_dir) / "metrics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    palette = {"lesion": "#d84b4b", "background": "#4f83cc"}
    save_state_scatter(
        projected,
        "before",
        "lesion",
        plots_dir / "lesion_scatter_before.png",
        palette,
        alpha,
        point_size,
        dpi,
    )
    save_state_scatter(
        projected,
        "after",
        "lesion",
        plots_dir / "lesion_scatter_after.png",
        palette,
        alpha,
        point_size,
        dpi,
    )
    save_state_density(
        projected,
        "before",
        "lesion",
        plots_dir / "lesion_density_before.png",
        palette,
        dpi,
    )
    save_state_density(
        projected,
        "after",
        "lesion",
        plots_dir / "lesion_density_after.png",
        palette,
        dpi,
    )
    save_joint_scatter(
        projected,
        plots_dir / "lesion_background_scatter_before_after.png",
        palette,
        alpha,
        point_size,
        dpi,
    )

    compactness = compactness_by_state(projected)
    compactness.to_csv(metrics_dir / "compactness_summary.csv", index=False)
    projected.to_csv(metrics_dir / "shared_projection_points.csv", index=False)
    return plots_dir, metrics_dir
