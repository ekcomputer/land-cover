"""Tests for harmonize_doc_dataset and load_harmonized_doc functions."""

import geopandas as gpd
import pandas as pd
import pytest

from land_cover.load import (
    harmonize_doc_dataset,
    load_harmonized_doc,
    loadDranga17,
    loadKurek,
    loadShahabinia25,
    loadStolpmann21,
)


class TestHarmonizeDocDataset:
    """Test harmonize_doc_dataset function."""

    def test_harmonize_dataframe_basic(self):
        """Test harmonizing a basic DataFrame with required columns."""
        df = pd.DataFrame(
            {
                "latitude": [60.0, 61.0],
                "longitude": [-130.0, -131.0],
                "sample_id": ["S1", "S2"],
                "doc_conc": [10.5, 12.3],
            }
        )

        result = harmonize_doc_dataset(
            df,
            lat_var="latitude",
            lon_var="longitude",
            sample_id_var="sample_id",
            area_var="",
            doc_var="doc_conc",
            dic_var="",
        )

        # Check output type and CRS
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs == "EPSG:4326"

        # Check renamed columns
        assert "lat" in result.columns
        assert "lon" in result.columns
        assert "sample_id" in result.columns
        assert "doc" in result.columns
        assert "geometry" in result.columns

        # Check original columns removed
        assert "latitude" not in result.columns
        assert "longitude" not in result.columns
        assert "doc_conc" not in result.columns

        # Check geometry creation
        assert len(result) == 2
        assert result.geometry[0].x == -130.0
        assert result.geometry[0].y == 60.0

    def test_harmonize_with_optional_fields(self):
        """Test harmonizing with optional area and DIC fields."""
        df = pd.DataFrame(
            {
                "lat_col": [60.0, 61.0],
                "lon_col": [-130.0, -131.0],
                "id_col": ["S1", "S2"],
                "area_col": [100.0, 200.0],
                "doc_col": [10.5, 12.3],
                "dic_col": [5.0, 6.0],
            }
        )

        result = harmonize_doc_dataset(
            df,
            lat_var="lat_col",
            lon_var="lon_col",
            sample_id_var="id_col",
            area_var="area_col",
            doc_var="doc_col",
            dic_var="dic_col",
        )

        # Check optional columns are present
        assert "area_km2" in result.columns
        assert "dic" in result.columns
        assert result.loc[0, "area_km2"] == 100.0
        assert result.loc[1, "dic"] == 6.0

    def test_harmonize_without_optional_fields(self):
        """Test harmonizing when optional fields are not provided (empty strings)."""
        df = pd.DataFrame(
            {
                "lat_col": [60.0, 61.0],
                "lon_col": [-130.0, -131.0],
                "id_col": ["S1", "S2"],
                "doc_col": [10.5, 12.3],
            }
        )

        result = harmonize_doc_dataset(
            df,
            lat_var="lat_col",
            lon_var="lon_col",
            sample_id_var="id_col",
            area_var="",  # Empty string for missing
            doc_var="doc_col",
            dic_var="",  # Empty string for missing
        )

        # Check optional columns are NOT present
        assert "area" not in result.columns
        assert "dic" not in result.columns
        assert len(result.columns) == 5  # lat, lon, sample_id, doc, geometry

    def test_harmonize_geodataframe(self):
        """Test harmonizing a GeoDataFrame with existing geometry."""
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            {
                "latitude": [60.0, 61.0],
                "longitude": [-130.0, -131.0],
                "sample_id": ["S1", "S2"],
                "doc_conc": [10.5, 12.3],
                "geometry": [Point(-130.0, 60.0), Point(-131.0, 61.0)],
            },
            crs="EPSG:4326",
        )

        result = harmonize_doc_dataset(
            gdf,
            lat_var="latitude",
            lon_var="longitude",
            sample_id_var="sample_id",
            area_var="",
            doc_var="doc_conc",
            dic_var="",
        )

        # Should preserve geometry
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs == "EPSG:4326"
        assert len(result) == 2


class TestLoadHarmonizedDoc:
    """Test load_harmonized_doc function."""

    def test_load_harmonized_doc_returns_geodataframe(self):
        """Test that load_harmonized_doc returns a GeoDataFrame."""
        result = load_harmonized_doc()

        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs == "EPSG:4326"

    def test_harmonized_doc_has_required_columns(self):
        """Test that harmonized doc has all required standard columns."""
        result = load_harmonized_doc()

        required_cols = ["lat", "lon", "sample_id", "doc", "source", "geometry"]
        for col in required_cols:
            assert col in result.columns, f"Missing required column: {col}"
        nan_ratios = result[required_cols].isna().mean()
        assert (nan_ratios < 0.1).all(), f"More than 10% NaN values found in required columns: {nan_ratios[nan_ratios >= 0.1]}"
        
    def test_harmonized_doc_has_source_tags(self):
        """Test that each row has a source tag."""
        result = load_harmonized_doc()

        expected_sources = {"Kurek23", "Dranga17", "Stolpmann21", "Shahabinia25"}
        actual_sources = set(result["source"].unique())

        # Check all expected sources are present
        assert expected_sources.issubset(
            actual_sources
        ), f"Missing sources. Expected {expected_sources}, got {actual_sources}"

    def test_harmonized_doc_has_valid_coordinates(self):
        """Test that rows have valid lat/lon coordinates where present."""
        result = load_harmonized_doc()

        # Check reasonable ranges for non-null values
        valid_lats = result["lat"].dropna()
        valid_lons = result["lon"].dropna()

        assert len(valid_lats) > 0, "No valid latitude values found"
        assert len(valid_lons) > 0, "No valid longitude values found"

        # After normalization, all longitudes should be in -180 to 180
        assert (valid_lons >= -180).all() and (
            valid_lons <= 180
        ).all(), f"Longitude out of range: min={valid_lons.min()}, max={valid_lons.max()}"
        assert (valid_lats >= -90).all() and (valid_lats <= 90).all()

    def test_harmonized_doc_has_doc_values(self):
        """Test that DOC column has values."""
        result = load_harmonized_doc()

        # At least some non-null DOC values
        assert result["doc"].notna().sum() > 0, "No DOC values found"

    def test_harmonized_doc_geometry_matches_coordinates(self):
        """Test that geometry points match lat/lon coordinates for Point geometries."""
        result = load_harmonized_doc()

        # Filter to Point geometries only (some sources may have Polygons)
        point_mask = result.geometry.type == "Point"
        point_gdf = result[point_mask]

        # Check first few point rows
        for idx in point_gdf.head(10).index:
            geom = result.loc[idx, "geometry"]
            lon = result.loc[idx, "lon"]
            lat = result.loc[idx, "lat"]

            # Skip if coordinates are null
            if pd.isna(lon) or pd.isna(lat):
                continue

            assert geom.x == lon, f"Longitude mismatch at {idx}: geometry={geom.x}, coord={lon}"
            assert geom.y == lat, f"Latitude mismatch at {idx}: geometry={geom.y}, coord={lat}"

    def test_harmonized_doc_source_counts(self):
        """Test that we get reasonable counts from each source."""
        result = load_harmonized_doc()

        source_counts = result["source"].value_counts()

        # Each source should have at least some records
        for source in ["Kurek23", "Dranga17", "Stolpmann21", "Shahabinia25"]:
            assert source_counts[source] > 0, f"No records from {source}"

    def test_harmonized_doc_optional_columns(self):
        """Test that optional columns (area, dic) are present where expected."""
        result = load_harmonized_doc()

        # Dranga17 and Shahabinia25 should have area and dic
        dranga = result[result["source"] == "Dranga17"]
        shahabinia = result[result["source"] == "Shahabinia25"]

        # These sources should have some non-null values for area/dic
        if "area" in result.columns:
            assert dranga["area"].notna().sum() > 0, "Dranga17 should have area data"
            assert shahabinia["area"].notna().sum() > 0, "Shahabinia25 should have area data"

        if "dic" in result.columns:
            assert dranga["dic"].notna().sum() > 0, "Dranga17 should have DIC data"
            assert shahabinia["dic"].notna().sum() > 0, "Shahabinia25 should have DIC data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
