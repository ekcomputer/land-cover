# land-cover
Scripts for analyzing land cover raster data and comparing to lake biogeochemical trends.

## Installation
```shell
mamba create -n landcover python=3.11
mamba activate landcover   
pip install -e . 
```

## Verify installation
Prepare AGB raster mosaics
```shell
bash mosaic_vrt.sh [path to directory containing tifs]
```

Update paths to vrt files in [`load`](land_cover/load.py) module:
* `biomass_30m_pth`
* `biomass_300m_pth`
* `topocat_subset_aea_pth`

Run tests
```shell
python -m pytest tests/test_biomass_change.py
```

## Wanwan instructions

Edit the biomass averaging [script](notebooks/joins/land-cover-buffer-biomass-hpc.py) to specify `csv_out_pth` and `n_workers`.

Run:
```shell
python notebooks/joins/land-cover-buffer-biomass-hpc.py
```

## Biomass Time Series

Extract values from a continuous raster dataset (aboveground biomass) instead of categorical land cover classifications.

### Raster Specifications

#### Input Data
- **High-res (30m)**: `/Volumes/metis/ABOVE3/Liang26_AGB/AGB_Bh014v011.tif`
- **Coarse (300m)**: `/Volumes/metis/ABOVE3/Liang26_AGB/Res300m/AGB_Bh014v011.tif`

#### Raster Properties
- **Data Type**: Int32
- **Bands**: 39 (one per year, 1984-2022)
- **NoData Value**: -999
- **CRS**: ESRI:102001 (Canada Albers Equal Area Conic)
- **Pixel Size**: 30m or 300m

### Usage Example

```python
from land_cover.biomass_change import extractTimeSeriesForLakes
from land_cover.load import loadBiomassContinuousTimeSeries

# Extract biomass statistics for all lakes and buffers
years = list(range(1986, 2025))  # 39 bands
extractTimeSeriesForLakes(
    pth_shp_in="path/to/lakes.shp",
    buffer_lengths=[0, 90, 300, 1000],
    csv_out_pth="agb_buffers.csv",
    pth_lc_in="/Volumes/metis/ABOVE3/Liang26_AGB/AGB_Bh014v011.tif",
    pth_lc_in_coarse="/Volumes/metis/ABOVE3/Liang26_AGB/Res300m/AGB_Bh014v011.tif",
    years=years,
    n_workers=8
)

```
### Output Format

Example CSV output:
```
Year,Buffer_m,mean,std,Lake_name,Lake_id_glakes,Area_m2,Perim_m2
1986,90,145.23,34.56,lake_1,1,5000000.0,15000.0
1987,90,148.75,33.21,lake_1,1,5000000.0,15000.0
...
2024,90,167.89,38.45,lake_1,1,5000000.0,15000.0
```

## Notes
- Multiprocessing support uses file locking for thread-safe CSV writes
- Resume functionality: checks for already-processed lakes in output CSV
- Large lake handling: uses coarsened raster 