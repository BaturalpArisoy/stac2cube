import os
import dask
from dask.diagnostics import ProgressBar
import time
from tqdm import tqdm
import numpy as np
from affine import Affine

def export_stac(stac, output, crs=None, transform=None):

    if isinstance(transform, np.ndarray):
        if transform.size == 0:
            transform = None  # Avoid passing empty arrays
        else:
            transform = Affine(*transform.ravel()[:6])
    
    if crs is None:
        crs = stac.crs
        stac.rio.write_crs(crs, inplace=True)
    else:
        stac.rio.write_crs(crs, inplace=True)
    if transform is None:
        transform = stac.transform
        transform = Affine(*transform.ravel()[:6])
        stac.rio.write_transform(transform, inplace=True)
    else:
        stac.rio.write_transform(transform, inplace=True)
    stac.attrs['crs'] = crs
    stac.attrs['transform'] = transform
      
    if dask.is_dask_collection(stac):
        with ProgressBar():
            img = stac.compute()
    else:
        img = stac

    # Unfortunately, no access to overwite netcdf files if netcdf is read as dataset. For now, it deletes the previous one and export the new one.
    # .close() does not work.
    # This approach is sadly not optimal because in case of a processing error during exporting, the previous netcdf file will be already deleted.
    # Look for an alternative!
    if os.path.exists(output):
        os.remove(output)

    img.to_netcdf(output)

    print(f"Export is done: {output}")

    return img