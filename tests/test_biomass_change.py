"""Functionality tests for biomass_change.extractTimeSeriesForLakes.

Tests the main workflow using a subset of real TopoCAT data with actual
biomass raster data.
"""

import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from land_cover.biomass_change import extractTimeSeriesForLakes
from land_cover.load import biomass_30m_pth, biomass_300m_pth, topocat_subset_aea_pth


@pytest.fixture(scope="module")
def test_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="module")
def test_polygons(test_data_dir):
    """Load first 400 rows from TopoCAT and save to temp directory."""
    # Load data
    gdf_subset = gpd.read_file(topocat_subset_aea_pth, rows=slice(1679300, 1679900))

    # Save to temp directory
    test_shp_path = test_data_dir / "test_catchments.gpkg"
    gdf_subset.to_file(test_shp_path, driver="GPKG")

    return test_shp_path, gdf_subset


def test_extractTimeSeriesForLakes_basic(test_data_dir, test_polygons):
    """Test basic functionality with single buffer and small dataset."""
    test_shp_path, gdf_subset = test_polygons

    # Run extraction with single buffer (no buffer, just lake polygons)
    years = list(range(1984, 2023))
    buffer_lengths = [0]

    csv_out_path = test_data_dir / "output_basic_test.csv"
    extractTimeSeriesForLakes(
        pth_shp_in=test_shp_path,
        buffer_lengths=buffer_lengths,
        csv_out_pth=csv_out_path,
        pth_lc_in=biomass_30m_pth,
        pth_lc_in_coarse=biomass_300m_pth,
        years=years,
        n_workers=2,  # Use fewer workers for testing
        join_index="Outlet_id",
    )

    # Verify output
    assert csv_out_path.exists(), "Output CSV not created"

    # Load and check results
    df = pd.read_csv(csv_out_path)

    # Check expected columns
    expected_cols = ["Year", "Buffer_m", "agb_mean", "agb_std", "Outlet_id", "Area_m2", "Perim_m2"]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"

    # Check year range
    assert df["Year"].min() == 1984
    assert df["Year"].max() == 2022
    assert len(df["Year"].unique()) == 39

    # Check buffer
    assert df["Buffer_m"].unique().tolist() == [0]

    # Check number of lakes processed (should match unique Outlet_id)
    n_lakes_processed = df["Outlet_id"].nunique()
    assert n_lakes_processed > 0, "No lakes processed"
    assert n_lakes_processed <= 600, "More lakes than expected"

    # Check that mean values are reasonable (should be in biomass range)
    non_nan_means = df["agb_mean"].dropna()
    assert len(non_nan_means) > 0, "All mean values are NaN"
    assert non_nan_means.min() >= 0, "Negative biomass values found"
    assert non_nan_means.max() <= 250, "Unreasonably high biomass values"

    print(f"✓ Processed {n_lakes_processed} lakes")
    print(f"✓ Output shape: {df.shape}")
    print(f"✓ Mean biomass range: {non_nan_means.min():.1f} - {non_nan_means.max():.1f}")


def test_extractTimeSeriesForLakes_resume(test_data_dir, test_polygons):
    """Test resume functionality when CSV already exists."""
    test_shp_path, gdf_subset = test_polygons

    # Output CSV
    csv_out_path = test_data_dir / "output_resume_test.csv"

    # First run: process subset
    years = list(range(1984, 2023))
    buffer_lengths = [0]

    # Create subset with only first 50 catchments
    gdf_small = gpd.read_file(test_shp_path)
    gdf_small_subset = gdf_small.head(50)
    test_shp_small_path = test_data_dir / "test_catchments_small.gpkg"
    gdf_small_subset.to_file(test_shp_small_path, driver="GPKG")

    extractTimeSeriesForLakes(
        pth_shp_in=test_shp_small_path,
        buffer_lengths=buffer_lengths,
        csv_out_pth=csv_out_path,
        pth_lc_in=biomass_30m_pth,
        pth_lc_in_coarse=biomass_300m_pth,
        years=years,
        n_workers=1,
        join_index="Outlet_id",
    )

    # Check first run results
    df_first = pd.read_csv(csv_out_path)
    n_lakes_first = df_first["Outlet_id"].nunique()

    # Second run: should detect already processed lakes
    extractTimeSeriesForLakes(
        pth_shp_in=test_shp_small_path,
        buffer_lengths=buffer_lengths,
        csv_out_pth=csv_out_path,
        pth_lc_in=biomass_30m_pth,
        pth_lc_in_coarse=biomass_300m_pth,
        years=years,
        n_workers=1,
        join_index="Outlet_id",
    )

    # Check that no additional rows were added (resume worked)
    df_second = pd.read_csv(csv_out_path)
    n_lakes_second = df_second["Outlet_id"].nunique()

    assert n_lakes_first == n_lakes_second, "Resume failed: duplicate processing occurred"
    assert len(df_first) == len(df_second), "Resume failed: rows were added"

    print(f"✓ Resume test passed: {n_lakes_first} lakes, no duplicates")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
