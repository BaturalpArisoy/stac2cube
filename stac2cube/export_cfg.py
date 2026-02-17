import os
import numpy as np
import dask
from dask.diagnostics import ProgressBar
from affine import Affine
import xarray as xr
from pyproj import CRS
from pathlib import Path


def _is_dask_backed(obj) -> bool:
    if isinstance(obj, xr.DataArray):
        return dask.is_dask_collection(obj.data)
    if isinstance(obj, xr.Dataset):
        return any(dask.is_dask_collection(da.data) for da in obj.data_vars.values())
    raise TypeError(f"Unsupported type: {type(obj)}")


def export_stac(
    stac,
    output,
    crs=None,
    transform=None,
    var_name=None,
    overwrite=True,
):
    if not isinstance(stac, (xr.DataArray, xr.Dataset)):
        raise TypeError(
            f"export_stac expects xarray.DataArray or xarray.Dataset, got {type(stac)}"
        )

    crs = crs or stac.crs
    transform = transform or stac.transform

    stac.attrs["transform"] = transform
    stac = stac.rio.write_crs(crs, inplace=True)
    stac.attrs["crs"] = crs

    if _is_dask_backed(stac):
        with ProgressBar():
            stac = stac.compute()

    if overwrite and os.path.exists(output):
        try:
            os.remove(output)
        except PermissionError:
            from datetime import datetime

            suffix = datetime.now().strftime("_%Y%m%d_%H%M%S")
            output = str(Path(output).with_stem(Path(output).stem + suffix))

    if isinstance(stac, xr.DataArray):
        name = var_name or stac.name or "Spectral_Temporal_Stack"
        ds = stac.to_dataset(name=name)
        ds.to_netcdf(output)
    else:
        stac.to_netcdf(output)

    print(f"Export is done: {output}")
    return stac


def export_to_cogs(
    stac: xr.DataArray, output_dir: str, prefix: str = "", dtype="float32"
):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not isinstance(stac, xr.DataArray):
        raise TypeError(f"Expected xarray.DataArray, got {type(stac)}")

    if "band" not in stac.dims:
        raise ValueError(f"Expected a 'band' dimension. Found dims: {stac.dims}")

    ds = stac.to_dataset(dim="band")

    if "time" in ds.dims:
        for i in range(ds.sizes["time"]):
            single = ds.isel(time=i)
            t = single["time"].values
            date_str = (
                np.datetime_as_string(t, unit="D")
                if np.issubdtype(type(t), np.datetime64)
                or np.issubdtype(np.asarray(t).dtype, np.datetime64)
                else str(t)[:10]
            )
            out_file = outdir / f"{prefix}{date_str}.tif"
            print(f"Writing {out_file.name}")
            single.rio.to_raster(
                out_file, driver="COG", dtype=dtype, compress="deflate"
            )
    else:
        out_file = outdir / f"{prefix}cog.tif"
        print(f"Writing {out_file.name}")
        ds.rio.to_raster(out_file, driver="COG", dtype=dtype, compress="deflate")
