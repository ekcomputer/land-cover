"""Version of lake buffers script for TopoCat lake catchment dataset; run on HPC.
"""

from land_cover.biomass_change import extractTimeSeriesForLakes
from land_cover.load import (biomass_30m_pth, 
                             biomass_300m_pth,
                             topocat_subset_aea_pth)


## RUN for entire TopoCAT dataset in domain
extractTimeSeriesForLakes(
    pth_shp_in=topocat_subset_aea_pth,
    buffer_lengths=[0],
    csv_out_pth="output.csv",
    pth_lc_in=biomass_30m_pth,
    pth_lc_in_coarse=biomass_300m_pth,
    years=list(range(1984, 2023)),
    n_workers=8,
    join_index="Outlet_id",
)


print("DONE.")
