# AI Coding Agent Instructions for land-cover

## Project Overview
Research codebase analyzing Arctic/boreal lake land cover change using remote sensing, GIS, and spatial statistics. Integrates multiple datasets (HydroLAKES, Liu aquatic vegetation, GLAKES, ABoVE land cover) to correlate lake greenness trends with surrounding vegetation and hydrological changes.

## Command line usage
You may use tools from the gdal/ogr libraries such as gdalinfo and ogrinfo and linux utilities head, ls, wc to preview files. You may not add any data files or run any programs that would cause data on the hard drive to be edited.

## Architecture & Data Flow

### Core Module Organization
- `land_cover/load.py`: Central data loader with 20+ `load*()` functions for different datasets. Each loader handles dataset-specific quirks (e.g., namespace column renaming with `_glakes`, `_gswl` suffixes to prevent conflicts)
- `land_cover/distance.py`: Boundary distance calculations using Dask parallelization
- `land_cover/plotting.py`: Cartopy-based mapping (transitioning from contextily), regression hexplots, choropleth maps
- `land_cover/gee.py`: Google Earth Engine asset downloads and lake polygon digitization workflows
- `land_cover/geocode.py`: Multi-provider geocoding (Nominatim, Google, GeoNames) with water feature filtering
- `land_cover/joins.py`: Spatial join utilities with many-to-one reduction logic
- `land_cover/land_cover_change_buffer_from_csv.py`: Main analysis script (866 lines) for zonal statistics in buffer zones around lakes

### Workflow Pattern
1. Load lake geometries via `load.py` functions (e.g., `loadGreenness()`, `loadEffluxShp()`)
2. Join with external datasets using spatial operations in `joins.py`
3. Calculate land cover metrics in buffer zones using `land_cover_change_buffer_from_csv.py`
4. Analyze time series features (stored as `*_tsFeatures.csv` variants)
5. Visualize with `plotting.py` functions in Jupyter notebooks

## Critical Conventions

### Coordinate Reference Systems
- **Default projected CRS**: `ESRI:102001` (Canada Albers Equal Area Conic) - use for area calculations and distance operations
- **Input data CRS**: `EPSG:4326` (WGS84) - typical for raw data imports
- Always reproject to Albers before geometric operations: `gdf.to_crs("ESRI:102001")`

### Namespace Management
All loaders apply dataset-specific suffixes to avoid column name collisions:
```python
# Example from loadGLAKES()
new_names = [name + "_glakes" for name in old_names]
gdf = gdf.rename(columns=dict(zip(old_names, new_names)))
```
Common suffixes: `_glakes`, `_gswl`, `_hylak`, `_gswl`

### File Path Conventions
- External data mounts: `/Volumes/metis/`, `/Volumes/thebe/`
- Outputs: `/Volumes/metis/ABOVE3/land_cover_joins/out/`
- Plots: `/Volumes/metis/ABOVE3/fig` (referenced as `plot_dir`)
- File naming variants: `_norm`, `_core`, `_tsFeatures`, `_short_tsFeatures` indicate processing stages

### Data Loading Best Practices
- Use `engine='pyogrio'` for faster shapefile reading when needed
- Apply `bbox` filtering early to reduce memory: `gpd.read_file(path, bbox=(-170, 51, -125, 72))`
- Prefer cached working files (e.g., `GLAKES_gswl_abz_pth`) over re-clipping large datasets

## Parallelization with Dask

### Current Setup in distance.py
```python
# Uses scheduler="processes" for CPU-bound tasks
with ProgressBar():
    results = dask.compute(*tasks, scheduler="processes")
```

### Dask Client Usage (Currently Commented Out)
When uncommenting `Client(n_workers=8)` in `distance.py`:
- Client instance is global - Dask automatically uses it for all `compute()` calls
- No need to pass client explicitly to functions
- Access dashboard: `client.dashboard_link` for monitoring
- Processes scheduler bypasses GIL for true parallelism in NumPy/shapely operations

## Common Gotchas

### Spatial Joins Performance
`joins.merge_left_one_one()` can hang on large datasets (>10 min). See `notebooks/joins/join_greennessx2.ipynb` for working implementation with manual groupby operations instead of sjoin.

### GEE Asset Handling
`gee.py` requires authentication: `ee.Authenticate()` then `ee.Initialize(project="ee-ekyzivat")`. The `clean_gee_results()` function prioritizes manual digitizations over PLD matches using `savetype_priority`.

### Buffer Zone Calculations
`extractBufferZonalHist()` in `land_cover_change_buffer_from_csv.py`:
- Rasterizes buffers once, then bins per band for efficiency
- Returns `None` if polygon doesn't overlap raster (handle gracefully)
- Multi-buffer output shape: `(n_bands, n_buffers, nclasses)`

### Percent Change Calculations
Use `utils.pct_change()` with explicit denominator choice:
```python
df["water_pchange_p1p3"] = pct_change(df.area_1984_1999, df.area_2010_2019, denom="old")
```
Default multiplies by 100 for percentage (set `multiply=False` for fraction).

### Data inconsistencies and error catching
- rather than using excessive try/accept blocks, add comments stating your assumptions. For example, before performing a join, you don't need to check that the join columns exist in the data sets. For really obvious checks, you can ignore them. For example, before loading a file, you don't need to check if it exists.

## Jupyter Notebook Patterns

### Standard Imports Block
```python
from land_cover.load import loadGreenness, plot_dir, load_reconstruct_time_series_above_boreal
from land_cover.utils import pct_change
from land_cover.plotting import reg_hexplot, boxplots_by_group
%load_ext autoreload
%autoreload 2
```

### Analysis Workflow (from greenness_vs_lc_change_hydrol.ipynb)
1. Load: `df = load_reconstruct_time_series_above_boreal()`
2. Compute changes: NDVI differences, percent changes with `pct_change()`
3. Group analysis: `boxplots_by_group(df, yvar, group_col="Trees_trend")`
4. Iterate over variable lists for systematic visualization

## External Dependencies
- Google Earth Engine API requires project initialization
- NEON data uses `neonutilities` for stacking downloaded archives
- Cartopy basemaps need internet for tile fetching (`img_tiles.GoogleTiles(style='satellite')`)

## Testing & Validation
Limited test coverage (`land_cover/tests/test_timeseries.py` exists but minimal). Validation typically done via:
- Visual inspection in notebooks with `plot_basemap(gdf)` 
- Data shape/count assertions in loader functions
- Progress monitoring with `tqdm` and Dask `ProgressBar`

## Key Regional Filters
- `kurek_bounds`: `[-156.9, 58.4, -111.0, 71.2]` - Kurek study area
- ABOVE region: `bbox=(-170, 51, -127, 72)` - Western North America boreal
- Olson biomes filter: `(np.isin(gdf.BIOME, [6, 11])) & (gdf.REALM == "NA")` for NA boreal/tundra

---
applyTo: "**/*.py"
---
## Project coding standards for Python
- Follow the PEP 8 style guide for Python.
- Always prioritize readability and clarity.
- Write clear and concise comments for each function.
- Use numpy docstring syntax 
- Ensure functions have descriptive names and include type hints when it is clear which type to hint.
- Maintain proper indentation (use 4 spaces for each level of indentation).
