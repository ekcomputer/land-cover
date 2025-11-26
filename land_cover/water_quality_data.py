import os, glob, dask
import dask.dataframe as dd
import neonutilities as nu
import pandas as pd

NEON_AK_CHEM_SURFACE_DIR = "/Volumes/metis/ABOVE3/NEON/NEON_chem-surfacewater"
NEON_AK_CHEM_SURFACE_DIR_ZIP = "/Volumes/metis/ABOVE3/NEON/NEON_chem-surfacewater.zip"

NEON_AK_WQ_DIR = "/Volumes/metis/ABOVE3/NEON/NEON_water-quality"
NEON_AK_WQ_DIR_ZIP = "/Volumes/metis/ABOVE3/NEON/NEON_water-quality.zip"

# Toolik lake 2017-2023
NEON_AK_RIP_STRUCT_DIR = "/Volumes/metis/ABOVE3/NEON/NEON_comp-struct-riparian"
NEON_AK_RIP_STRUCT_DIR_ZIP = "/Volumes/metis/ABOVE3/NEON/NEON_comp-struct-riparian.zip"

# Only rivers
NEON_AK_RIP_COVER_DIR = "/Volumes/metis/ABOVE3/NEON/NEON_cover-veg-riparian"
NEON_AK_RIP_COVER_DIR_ZIP = "/Volumes/metis/ABOVE3/NEON/NEON_cover-veg-riparian.zip"


NEON_OUTPUT_DIR = "/Volumes/metis/ABOVE3/NEON/edk_out"

# Just run once!
# nu.stack_by_table(NEON_AK_CHEM_SURFACE_DIR_ZIP)
# nu.stack_by_table(NEON_AK_WQ_DIR_ZIP)
nu.stack_by_table(NEON_AK_RIP_STRUCT_DIR_ZIP)
nu.stack_by_table(NEON_AK_RIP_COVER_DIR_ZIP)

# def load_NEON_chem_surface():
#     return pd.read_csv(os.path.join(NEON_AK_DIR, "stackedFiles/swc_externalLabDataByAnalyte.csv"))

# df = load_NEON_chem_surface()
pass
