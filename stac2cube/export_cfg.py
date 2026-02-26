import os
import numpy as np
import dask
from dask.diagnostics import ProgressBar
from affine import Affine
import xarray as xr
from pyproj import CRS
from pathlib import Path
import re
import itertools




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
    stac: xr.DataArray | xr.Dataset, output_dir: str, prefix: str = "", dtype="float32"
):

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not isinstance(stac, (xr.DataArray, xr.Dataset)):
        raise TypeError(f"Expected xarray.DataArray or xarray.Dataset, got {type(stac)}")

    # -----------------------------
    # Helpers
    # -----------------------------
    def _safe_name(s):
        s = str(s)
        s = s.strip()
        s = re.sub(r'[<>:"/\\|?*]+', "_", s)  # windows-safe
        s = re.sub(r"\s+", "_", s)
        return s

    def _format_coord_value(val, dim_name=None):
        arr = np.asarray(val)

        # datetime-like
        try:
            if np.issubdtype(arr.dtype, np.datetime64):
                return np.datetime_as_string(arr.astype("datetime64[ns]"), unit="D")
        except Exception:
            pass

        # scalar conversion
        try:
            v = arr.item() if arr.shape == () else val
        except Exception:
            v = val

        # prettier month formatting
        if dim_name == "month":
            try:
                return f"{int(v):02d}"
            except Exception:
                return _safe_name(v)

        return _safe_name(v)

    def _write_raster(da: xr.DataArray, out_file: Path):
        """
        Write a single xarray.DataArray slice as COG.
        Keeps 'band' as multiband if present.
        """
        print(f"Writing {out_file.name}")

        if "band" in da.dims:
            ds = da.to_dataset(dim="band")
            ds.rio.to_raster(out_file, driver="COG", dtype=dtype, compress="deflate")
        else:
            da.rio.to_raster(out_file, driver="COG", dtype=dtype, compress="deflate")

    def _export_dataarray(da: xr.DataArray, base_name: str | None = None, date_only_for_time_series=False):
        """
        Export a DataArray by iterating over all non-spatial dims except 'band'.
        """
        if not isinstance(da, xr.DataArray):
            raise TypeError(f"_export_dataarray expects DataArray, got {type(da)}")

        if "y" not in da.dims or "x" not in da.dims:
            raise ValueError(f"Expected spatial dims 'y' and 'x'. Found dims: {da.dims}")

        # Dimensions to iterate over (keep 'band' together as multiband)
        iter_dims = [d for d in da.dims if d not in ("y", "x", "band")]

        # No extra dims -> one file
        if len(iter_dims) == 0:
            name = base_name or "cog"
            out_file = outdir / f"{prefix}{_safe_name(name)}.tif"
            _write_raster(da, out_file)
            return

        # Build index combinations for all iter dims
        dim_sizes = [da.sizes[d] for d in iter_dims]

        for idx_tuple in itertools.product(*[range(n) for n in dim_sizes]):
            isel_dict = dict(zip(iter_dims, idx_tuple))
            single = da.isel(**isel_dict)

            # Build filename
            # Special case: Spectral_Temporal_Stack with only time dim -> use date only (old behavior)
            if date_only_for_time_series and iter_dims == ["time"]:
                t = single["time"].values if "time" in single.coords else da["time"].values[idx_tuple[0]]
                date_str = _format_coord_value(t, "time")
                filename = f"{prefix}{date_str}.tif"
            else:
                parts = []
                for d, i in zip(iter_dims, idx_tuple):
                    coord_val = da[d].values[i] if d in da.coords else i
                    parts.append(f"{d}-{_format_coord_value(coord_val, d)}")

                stem = _safe_name(base_name or da.name or "cog")
                suffix = "_".join(parts)
                filename = f"{prefix}{stem}_{suffix}.tif"

            out_file = outdir / filename
            _write_raster(single, out_file)

    # -----------------------------
    # Main logic
    # -----------------------------
    if isinstance(stac, xr.DataArray):
        # Backward-compatible behavior
        if "band" not in stac.dims:
            raise ValueError(f"Expected a 'band' dimension. Found dims: {stac.dims}")

        # If it looks like a time series stack, keep old date filenames
        is_time_series_stack = ("time" in stac.dims)
        _export_dataarray(
            stac,
            base_name=(stac.name or "cog"),
            date_only_for_time_series=is_time_series_stack,
        )
        return

    # Dataset case: export all data variables
    if isinstance(stac, xr.Dataset):
        if len(stac.data_vars) == 0:
            raise ValueError("Dataset contains no data variables to export.")

        # 1) Export Spectral_Temporal_Stack first (if present) using date filenames
        if "Spectral_Temporal_Stack" in stac.data_vars:
            da_main = stac["Spectral_Temporal_Stack"]
            if "band" not in da_main.dims:
                raise ValueError(
                    "Spectral_Temporal_Stack is missing required 'band' dimension "
                    f"(dims: {da_main.dims})."
                )
            _export_dataarray(
                da_main,
                base_name="Spectral_Temporal_Stack",
                date_only_for_time_series=("time" in da_main.dims),
            )

        # 2) Export remaining variables with variable-based filenames
        for var_name, da in stac.data_vars.items():
            if var_name == "Spectral_Temporal_Stack":
                continue

            if not isinstance(da, xr.DataArray):
                continue

            # Skip non-raster-like vars if any
            if "y" not in da.dims or "x" not in da.dims:
                print(f"Skipping {var_name}: no spatial dims ('y','x') found (dims: {da.dims})")
                continue

            _export_dataarray(
                da,
                base_name=var_name,
                date_only_for_time_series=False,
            )