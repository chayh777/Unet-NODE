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


def _project_csv_pair(
    first_csv,
    second_csv,
    pca_components,
    umap_neighbors,
    umap_min_dist,
    random_state,
):
    first_df = pd.read_csv(first_csv)
    second_df = pd.read_csv(second_csv)
    combined = pd.concat([first_df, second_df], ignore_index=True)
    return build_shared_projection(
        combined, pca_components, umap_neighbors, umap_min_dist, random_state
    )


def _state_title(state_name):
    return str(state_name).replace("_", " ").title()


def _color_map_for_class(class_name):
    return "Reds" if class_name == "lesion" else "Blues"


def _scatter_axis(ax, subset, palette, alpha, point_size, title):
    if subset.empty:
        ax.set_title(f"{title}\n(no data)")
        ax.set_axis_off()
        return

    _, sns = _get_plotting_libs()
    sns.scatterplot(
        data=subset,
        x="x",
        y="y",
        hue="class_name",
        palette=palette,
        alpha=alpha,
        s=point_size,
        edgecolor=None,
        ax=ax,
    )
    ax.set_title(title)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def _density_axis(ax, subset, palette, title):
    if subset.empty:
        ax.set_title(f"{title}\n(no data)")
        ax.set_axis_off()
        return

    _, sns = _get_plotting_libs()
    plotted = False
    for class_name in subset["class_name"].dropna().unique():
        class_subset = subset[subset["class_name"] == class_name]
        if len(class_subset) < 2:
            continue
        sns.kdeplot(
            data=class_subset,
            x="x",
            y="y",
            fill=True,
            cmap=_color_map_for_class(class_name),
            levels=5,
            alpha=0.45,
            ax=ax,
        )
        plotted = True

    if not plotted:
        sns.scatterplot(
            data=subset,
            x="x",
            y="y",
            hue="class_name",
            palette=palette,
            s=20,
            edgecolor=None,
            ax=ax,
        )

    ax.set_title(title)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def save_side_by_side_scatter(
    df,
    state_names,
    output_path,
    palette,
    alpha,
    point_size,
    dpi,
):
    if df.empty:
        return

    plt, _ = _get_plotting_libs()
    fig, axes = plt.subplots(1, len(state_names), figsize=(6 * len(state_names), 5), sharex=True, sharey=True)
    if len(state_names) == 1:
        axes = [axes]

    for ax, state_name in zip(axes, state_names):
        subset = df[df["state"] == state_name]
        _scatter_axis(ax, subset, palette, alpha, point_size, _state_title(state_name))

    handles, labels = axes[0].get_legend_handles_labels()
    if handles and labels:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_side_by_side_density(df, state_names, output_path, palette, dpi):
    if df.empty:
        return

    plt, _ = _get_plotting_libs()
    fig, axes = plt.subplots(1, len(state_names), figsize=(6 * len(state_names), 5), sharex=True, sharey=True)
    if len(state_names) == 1:
        axes = [axes]

    for ax, state_name in zip(axes, state_names):
        subset = df[df["state"] == state_name]
        _density_axis(ax, subset, palette, f"{_state_title(state_name)} Density")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)



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
    projected = _project_csv_pair(
        before_csv,
        after_csv,
        pca_components,
        umap_neighbors,
        umap_min_dist,
        random_state,
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


def run_low_data_geometry_plot(
    pre_csv,
    post_csv,
    output_dir,
    pca_components,
    umap_neighbors,
    umap_min_dist,
    random_state,
    alpha,
    point_size,
    dpi,
):
    projected = _project_csv_pair(
        pre_csv,
        post_csv,
        pca_components,
        umap_neighbors,
        umap_min_dist,
        random_state,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    palette = {"lesion": "#d84b4b", "background": "#4f83cc"}
    state_names = ("pre_adapter", "post_adapter")

    save_side_by_side_scatter(
        projected,
        state_names,
        output_dir / "bottleneck_before_after_scatter.png",
        palette,
        alpha,
        point_size,
        dpi,
    )
    save_side_by_side_density(
        projected,
        state_names,
        output_dir / "bottleneck_before_after_density.png",
        palette,
        dpi,
    )

    projected.to_csv(output_dir / "shared_projection_points.csv", index=False)
    compactness_by_state(projected).to_csv(output_dir / "geometry_metrics.csv", index=False)
    return output_dir
