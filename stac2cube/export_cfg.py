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


def _set_compression(ds, compress, complevel=4):
    """Make each spatial variable's on-disk layout deterministic, in place.

    xarray keeps the storage settings of the file a variable was READ from in
    ``.encoding`` and silently reuses them on the next ``to_netcdf`` - so a
    cube opened from a compressed file would be re-written compressed even
    when compression was not requested (and derived variables, which lose
    their encoding, would not be). To avoid that ambiguity, any inherited
    layout keys are stripped first, then zlib is requested only when
    ``compress`` is True: the flag alone decides the output layout.

    The encoding is merged into the variable's existing ``.encoding`` instead
    of being passed to ``to_netcdf(encoding=...)``, which would override the
    whole dict and drop ``grid_mapping`` (the CRS link GIS tools need).
    Chunking is one spatial slice per chunk: NaN-masked scenes compress well
    and single-date reads stay cheap.
    """
    for da in ds.data_vars.values():
        if "y" not in da.dims or "x" not in da.dims:
            continue
        for stale in (
            "contiguous", "chunksizes", "zlib", "complevel",
            "shuffle", "fletcher32", "preferred_chunks", "compression",
        ):
            da.encoding.pop(stale, None)
        if compress:
            da.encoding.update(
                {
                    "zlib": True,
                    "complevel": complevel,
                    "chunksizes": tuple(
                        da.sizes[d] if d in ("y", "x") else 1 for d in da.dims
                    ),
                }
            )


def _strip_netcdf_encoding(ds):
    """Remove NetCDF/HDF5-only layout keys from every variable's encoding.

    Needed before a Zarr write: encoding inherited from a NetCDF source (or set
    by :func:`_set_compression`) carries keys like ``chunksizes``/``zlib`` that
    Zarr does not understand and that would raise on ``to_zarr``.
    """
    for var in ds.variables.values():
        for stale in (
            "contiguous", "chunksizes", "zlib", "complevel", "shuffle",
            "fletcher32", "preferred_chunks", "compression", "_FillValue",
        ):
            var.encoding.pop(stale, None)


def _write_zarr(ds, output, overwrite):
    """Write ``ds`` to a Zarr store, streaming chunks from dask.

    Unlike the NetCDF path this does NOT materialize the whole cube first: the
    array is (re)chunked to one spatial slice per (time, band) and dask writes
    those chunks straight to the store, so peak memory is a few chunks rather
    than the full cube - the reason Zarr is offered for very large cubes.

    Compression: Zarr applies its own default lossless codec, so a store is
    always compressed. The ``compress`` (zlib) flag is a NetCDF-only knob and
    is intentionally not plumbed here - forcing a specific Zarr v3 codec object
    is version-fragile, and the default already compresses well.
    """
    import shutil
    import warnings

    _strip_netcdf_encoding(ds)

    # Uniform per-slice chunks: valid for Zarr (which needs equal chunk sizes
    # bar the last) and good for partial reads. Daskifies an in-memory cube so
    # the write streams in both the lazy and already-computed cases.
    chunk_spec = {d: (-1 if d in ("y", "x") else 1) for d in ds.dims}
    ds = ds.chunk(chunk_spec)

    if overwrite and os.path.isdir(output):
        shutil.rmtree(output, ignore_errors=True)

    with warnings.catch_warnings():
        # Both are informational and do not affect data read back by xarray
        # (verified separately): (1) fixed-length unicode coords (e.g. band
        # names) have no stable Zarr-v3 spec yet - other Zarr libraries may not
        # read those *labels*, though values are fine; (2) consolidated
        # metadata is an xarray convenience outside the v3 core spec.
        warnings.filterwarnings("ignore", message=".*does not have a Zarr V3 specification.*")
        warnings.filterwarnings("ignore", message=".*[Cc]onsolidated metadata.*")
        ds.to_zarr(output, mode="w")


def export_stac(
    stac,
    output,
    crs=None,
    transform=None,
    var_name=None,
    overwrite=True,
    compress=False,
):
    """Write a cube to disk. Output format is chosen by the file extension:

    - ``*.zarr`` -> a chunked, streamed Zarr store (always compressed with
      Zarr's default codec; low peak memory - good for very large cubes).
    - anything else -> a single NetCDF file (the long-standing default).

    ``compress=True`` -> lossless zlib compression (level 4) on all spatial
    variables. NetCDF only (Zarr compresses by default). Values are
    bit-identical on read-back; the write is slower. Shrinks cloud-masked
    cubes especially well (NaN runs compress strongly).
    """
    if not isinstance(stac, (xr.DataArray, xr.Dataset)):
        raise TypeError(
            f"export_stac expects xarray.DataArray or xarray.Dataset, got {type(stac)}"
        )

    crs = crs or stac.crs
    transform = transform or stac.transform

    stac.attrs["transform"] = transform
    stac = stac.rio.write_crs(crs, inplace=True)
    stac.attrs["crs"] = crs

    is_zarr = str(output).lower().endswith(".zarr")

    # NetCDF is written from an in-memory array (long-standing behavior). Zarr
    # is kept lazy so dask can stream it chunk-by-chunk (see _write_zarr).
    if not is_zarr and _is_dask_backed(stac):
        with ProgressBar():
            stac = stac.compute()

    if isinstance(stac, xr.DataArray):
        name = var_name or stac.name or "Spectral_Temporal_Stack"
        ds = stac.to_dataset(name=name)
    else:
        ds = stac

    if is_zarr:
        _write_zarr(ds, output, overwrite)
        print(f"Export is done: {output}")
        return stac

    if overwrite and os.path.exists(output):
        try:
            os.remove(output)
        except PermissionError:
            from datetime import datetime

            suffix = datetime.now().strftime("_%Y%m%d_%H%M%S")
            output = str(Path(output).with_stem(Path(output).stem + suffix))

    _set_compression(ds, compress)
    ds.to_netcdf(output)

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
    def _reorder_rgb_first(da: xr.DataArray) -> xr.DataArray:
        """
        GeoTIFF/QGIS interpret band 1->Red, 2->Green, 3->Blue by default.
        Our cube stores bands in wavelength order (e.g. blue, green, red, nir),
        which makes QGIS swap red and blue. So for GeoTIFF export only, if all
        three RGB bands are present, move them to the front as red, green, blue
        and keep every other band after them in its original order.
        The NetCDF path (export_stac) keeps the wavelength order untouched.
        """
        if "band" not in da.dims or "band" not in da.coords:
            return da

        bands = list(da.coords["band"].values)
        lookup = {str(b).lower(): b for b in bands}

        aliases = {
            "red": ("red", "r", "b04"),
            "green": ("green", "g", "b03"),
            "blue": ("blue", "b", "b02"),
        }

        def _find(names):
            for n in names:
                if n in lookup:
                    return lookup[n]
            return None

        rgb = [_find(aliases[c]) for c in ("red", "green", "blue")]
        if any(b is None for b in rgb):
            return da  # not a full RGB set -> leave order as is

        rest = [b for b in bands if b not in rgb]
        new_order = rgb + rest
        if new_order == bands:
            return da
        return da.sel(band=new_order)

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

        # GeoTIFF-only: present RGB bands as red, green, blue (see helper).
        da = _reorder_rgb_first(da)

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