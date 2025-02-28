import dask
from dask.diagnostics import ProgressBar
import time
from tqdm import tqdm
import numpy as np
from affine import Affine

# add write_crs
def export_stac(stac, output, crs, transform):

    if isinstance(transform, np.ndarray):
        if transform.size == 0:
            transform = None  # Avoid passing empty arrays
        else:
            transform = Affine(*transform.ravel()[:6])
    """
    # Force overwrite: Remove existing transform before writing a new one
    if hasattr(stac.rio, "_obj") and "transform" in stac.rio._obj.attrs:
        del stac.rio._obj.attrs["transform"]
    """
    
    stac.rio.write_crs(crs, inplace=True)
    stac.rio.write_transform(transform, inplace=True)
    stac.attrs['crs'] = crs
    stac.attrs['transform'] = transform
      
    if dask.is_dask_collection(stac):
        with ProgressBar():
            img = stac.compute()
    else:
        img = stac

    img.to_netcdf(output)

    print(f"Export is done: {output}")