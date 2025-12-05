import numpy as np

def pct_change(old, new, denom="old", multiply=True, zero_replace=np.nan):
    """
    Compute percent (or relative) change between two dataframe columns and add as a new column.

    Parameters
    - col_old: name of "baseline" column (string)
    - col_new: name of "new" column (string)
    - denom: denominator choice for relative change: 'old' (default), 'new', or 'mean'
    - multiply: if True, multiply by 100 to return percent (default True). If False, returns fraction.
    - zero_replace: what to do when denominator == 0. If np.nan (default) leaves result NaN.
                    If 'eps', replaces zeros with a tiny epsilon to avoid inf.
                    Or provide a numeric value to use instead.

    Returns
    - Series added to df at out_col and also returned.
    """

    a = old.astype("float64")
    b = new.astype("float64")

    if denom == "old":
        denom_s = a
    elif denom == "new":
        denom_s = b
    elif denom == "mean":
        denom_s = (a + b) / 2.0
    else:
        raise ValueError("denom must be one of 'old', 'new', 'mean'")

    # handle zeros in denominator
    if zero_replace == "eps":
        denom_s = denom_s.replace(0, np.finfo(float).eps)
    elif np.isfinite(zero_replace) and not np.isnan(zero_replace):
        denom_s = denom_s.replace(0, zero_replace)
    else:
        # leave zeros as-is so division yields inf/NaN which can be inspected/filtered
        pass

    res = (b - a) / denom_s

    if multiply:
        res = res * 100.0

    return res


# Example usage:
# df['NDVI_pct_84_11'] = pct_change(df, 'NDVI8499', 'NDVI1121')

def create_unique_index(A, B):
    """Note: the int version of the index returned by np.unique is dependent on the input dataset
    size
    """
    # create unique integer index for each (Latitude, Longitude) pair by viewing float64 bit patterns
    lat = A.to_numpy(dtype=np.float64)
    lon = B.to_numpy(dtype=np.float64)

    bits = np.column_stack((lat.view(np.int64), lon.view(np.int64)))
    row_view = bits.view(np.dtype((np.void, bits.dtype.itemsize * bits.shape[1])))

    _, loc_idx = np.unique(row_view, return_inverse=True)

    return loc_idx