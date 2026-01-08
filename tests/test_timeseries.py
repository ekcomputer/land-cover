from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# old_csv = Path(
#     "/Volumes/metis/ABOVE3/land_cover_joins/out/glakes_green_abovelc25/xlsx/archive/greennessx2_albers_landCoverBuffers_og.csv"
# )
old_csv = Path(
    "/Volumes/metis/ABOVE3/land_cover_joins/out/glakes_green_abovelc25/xlsx/archive/greennessx2_albers_landCoverBuffers_parallel.csv"
)
new_csv = Path(
    "/Volumes/metis/ABOVE3/land_cover_joins/out/glakes_green_abovelc25/xlsx/archive/greennessx2_albers_landCoverBuffers_vectorized.csv"
)
rast_csv = Path(
    "/Volumes/metis/ABOVE3/land_cover_joins/out/glakes_green_abovelc25/xlsx/archive/greennessx2_albers_landCoverBuffers_rasterize.csv"
)


def load_old_xlsx_dir(d: Path) -> pd.DataFrame:
    xlsx_files = sorted(d.glob("*.csv"))
    if not xlsx_files:
        raise FileNotFoundError(f"No .csv files found in {d}")
    return pd.concat((pd.read_csv(p) for p in xlsx_files), ignore_index=True)

if __name__ == "__main__":
    # Load
    df_old = pd.read_csv(old_csv)
    df_new = pd.read_csv(new_csv)
    df_rast = pd.read_csv(rast_csv)

    # Harmonize & tag
    for df, tag in [(df_old, "parallel"), (df_new, "vectorized"), (df_rast, "rasterized")]:
        # Ensure expected columns exist
        assert {"Year", "Join_idx", "Wetland"}.issubset(df.columns), f"Missing cols in {tag}"
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df["Join_idx"] = pd.to_numeric(df["Join_idx"], errors="coerce").astype("Int64")
        df["Wetland"] = pd.to_numeric(df["Wetland"], errors="coerce")
        df["method"] = tag

    df = pd.concat([df_old, df_new, df_rast], ignore_index=True)

    # Filter requested indices
    keep = {0, 1, 2, 3, 4}
    plot_df = df[df["Join_idx"].isin(keep)].dropna(subset=["Year", "Wetland"])

    # Plot
    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=plot_df.sort_values(["method", "Join_idx", "Year"]),
        x="Year",
        y="Wetland",
        hue="Join_idx",
        col="method",
        kind="line",
        height=4,
        aspect=1.4,
        marker="o",
        facet_kws={"sharey": True, "sharex": True},
    )
    g.set_axis_labels("Year", "Wetland area")
    g.set_titles("{col_name}")
    plt.tight_layout()
    plt.show()
    pass
