import contextlib
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




# Name of the time-series variable every cube is built around. Cubes written
# before the rename carry the old name; they are migrated on read (see
# normalize_stack_name / open_cube), so nothing else in the codebase - and no
# legacy file on disk - needs to know the old spelling.
STACK_VAR = "Time_Series"
LEGACY_STACK_VARS = ("Spectral_Temporal_Stack",)


def normalize_stack_name(obj):
    """Rename a legacy time-series variable to :data:`STACK_VAR`.

    Applied to everything read through :func:`open_cube`, so a cube written
    before the rename is indistinguishable from a fresh one everywhere
    downstream. Also usable on an in-memory Dataset/DataArray handed straight
    to a public function (those bypass ``open_cube``). Returns the object
    unchanged when there is nothing to rename, and never renames onto an
    existing ``STACK_VAR`` variable.
    """
    if isinstance(obj, xr.DataArray):
        if str(obj.name) in LEGACY_STACK_VARS:
            return obj.rename(STACK_VAR)
        return obj
    if not isinstance(obj, xr.Dataset):
        return obj
    if STACK_VAR in obj.data_vars:
        return obj
    for legacy in LEGACY_STACK_VARS:
        if legacy in obj.data_vars:
            out = obj.rename({legacy: STACK_VAR})
            # rename() drops the dataset's closer (verified: _close becomes
            # None), which would turn every `with open_cube(...) as ds:` on a
            # legacy cube into a leaked file handle - and on Windows a locked
            # file the pipeline cannot overwrite. Re-attach it, exactly as the
            # zarr coord-materialization above has to.
            try:
                out.set_close(obj._close)
            except AttributeError:
                out._close = obj._close
            return out
    return obj


def resolve_stack_var(obj, var_name=None):
    """The time-series variable name to use on ``obj``.

    Accepts a legacy name passed by an older script and maps it forward, so
    ``var_name="Spectral_Temporal_Stack"`` keeps working after the rename.
    ``None`` resolves to whichever of the new/legacy names the object carries.
    """
    names = list(getattr(obj, "data_vars", []) or [])
    if var_name is not None:
        if str(var_name) in LEGACY_STACK_VARS and STACK_VAR in names:
            return STACK_VAR
        return var_name
    if STACK_VAR in names:
        return STACK_VAR
    for legacy in LEGACY_STACK_VARS:
        if legacy in names:
            return legacy
    return STACK_VAR


def is_zarr_path(path) -> bool:
    """True when ``path`` names a Zarr store (by the ``.zarr`` extension)."""
    return str(path).lower().rstrip("/\\").endswith(".zarr")


def is_vrt_path(path) -> bool:
    """True when ``path`` names a QGIS band-mapping sidecar."""
    return str(path).lower().rstrip("/\\").endswith(".vrt")


def _reject_vrt(path, action, hint=None):
    """Refuse a ``.vrt`` anywhere a cube is expected.

    A VRT written by :func:`write_qgis_vrt` is a GIS view OF a cube, not a cube:
    it flattens (time, band) into a numbered stack and holds no coordinates. It
    is also a snapshot - the labels stop matching as soon as the cube's dates
    change - so letting one into the pipeline can only produce confusing
    results or silently wrong ones.

    Left to itself the failure is unhelpful: reading a VRT half-succeeds
    through rasterio and dies with "conflicting sizes for dimension 'time':
    length 24 on the data but length 4 on coordinate 'time'", which says
    nothing about what the user actually did.
    """
    if not is_vrt_path(path):
        return
    if hint is None:
        sibling = Path(str(path)).with_suffix(".nc")
        hint = (
            f"Load {sibling.name!r} instead."
            if sibling.exists()
            else "Use the .nc cube it was made from."
        )
    hint = f" {hint}" if hint else ""
    raise ValueError(
        f"Cannot {action} {os.path.basename(str(path))!r}: a .vrt is a QGIS "
        f"band-mapping sidecar, not a data cube.{hint}"
    )


def resolve_cube_path(path):
    """Normalize a user-picked cube path.

    A Zarr store is a directory, but file choosers select files - so a user
    browsing "into" ``cube.zarr`` picks an inner file like ``zarr.json``.
    If any parent directory of ``path`` ends in ``.zarr``, return that store
    root; otherwise return ``path`` unchanged.
    """
    p = Path(str(path))
    for parent in [p] + list(p.parents):
        if parent.name.lower().endswith(".zarr"):
            return str(parent)
    return str(path)


def _frame_chunked_netcdf(path):
    """Open a NetCDF cube with one dask chunk per (time, band) image.

    This is the chunking a viewer/animation wants: every redraw asks for a
    single scene, so a chunk should hold a single scene and nothing more.

    ``chunks="auto"`` does NOT do that. It sizes chunks by dask's ~128 MB
    heuristic, which on a real cube (355 dates x 6 bands x 648 x 618, stored
    contiguously) comes out as (174, 2, 317, 301): one RGB frame then spans 8
    of those chunks, so drawing it reads ~1 GB off disk to keep 4.8 MB. Measured
    on that cube: 0.96 s per frame with ``"auto"`` vs 0.012 s per frame here
    (~80x). The compressed layout is hit the same way - each chunk must be
    inflated in full - measured 0.28 s vs 0.014 s (~20x) on a 100-date cube.

    A NetCDF written elsewhere may genuinely be chunked across several dates or
    bands on disk. Splitting those stored chunks finer would make every frame
    inflate the same block again, so that layout is followed as stored
    (``chunks={}``) instead.
    """
    import warnings

    frame_spec = {"time": 1, "band": 1}
    with warnings.catch_warnings():
        # Emitted when the spec cuts across stored chunks; that case is
        # detected below and re-opened, so the warning has nothing to add.
        warnings.filterwarnings(
            "ignore", message=".*separate the stored chunks.*"
        )
        ds = xr.open_dataset(path, chunks=frame_spec)

    # preferred_chunks is the on-disk chunking (absent for contiguous storage,
    # where any hyperslab is cheap and per-image chunks are simply right).
    spans_disk_chunk = False
    for var in ds.data_vars.values():
        if "y" not in var.dims or "x" not in var.dims:
            continue
        pref = var.encoding.get("preferred_chunks") or {}
        if any(int(pref.get(d, 1)) > 1 for d in frame_spec):
            spans_disk_chunk = True
            break
    if not spans_disk_chunk:
        return ds

    ds.close()
    return xr.open_dataset(path, chunks={})


def open_cube(path, chunks=None):
    """Open a cube from disk, dispatching on the path extension.

    ``*.zarr`` -> :func:`xarray.open_zarr` (always lazy/dask-backed, chunked
    as stored); anything else -> :func:`xarray.open_dataset` (NetCDF), with
    ``chunks`` passed through unchanged so existing eager/lazy behavior is
    preserved.

    ``chunks="frames"`` is a NetCDF-only shorthand for "lazy, one chunk per
    scene" (see :func:`_frame_chunked_netcdf`) - what every interactive path
    reading a cube from disk wants. Zarr stores are already written one scene
    per chunk, so the flag needs no equivalent there.

    This is the single entry point every pipeline component uses to read a
    cube from a path, so NetCDF and Zarr cubes are interchangeable everywhere.
    Note one storage-level difference: Zarr attributes are JSON, so attrs
    written as numpy arrays/scalars (e.g. ``bbox``, ``transform``,
    ``y.resolution``) come back as plain Python lists/floats. Consumers must
    not rely on numpy-only methods of attr values (use ``np.asarray(...)`` /
    ``float(...)``).
    """
    import warnings

    _reject_vrt(path, "open")

    if is_zarr_path(path):
        with warnings.catch_warnings():
            # Same two informational zarr-v3 warnings suppressed in
            # _write_zarr; values read back identically (verified).
            warnings.filterwarnings(
                "ignore", message=".*does not have a Zarr V3 specification.*"
            )
            warnings.filterwarnings("ignore", message=".*[Cc]onsolidated metadata.*")
            ds = xr.open_zarr(path)
        # open_zarr leaves non-dimension coordinates (e.g. cloud_percentage)
        # dask-backed; a dask boolean coord cannot be used as a drop-indexer
        # (cloud_filter's .where(..., drop=True) raises). Coordinates are tiny,
        # so materialize them - matching how NetCDF coords arrive in memory -
        # while the data variables stay lazy.
        out = ds.assign_coords(
            {name: coord.compute() for name, coord in ds.coords.items()}
        )
        # assign_coords drops the dataset's closer (verified: _close becomes
        # None), which would turn every `with open_cube(...) as ds:` into a
        # handle leak. Re-attach it so close() keeps working.
        try:
            out.set_close(ds._close)
        except AttributeError:
            out._close = ds._close

        # NetCDF stores the numeric-array attrs (bbox, transform) as ndarrays;
        # Zarr's JSON attrs return plain lists with identical values. Convert
        # back so downstream behavior - including attr text written into
        # exports (e.g. COG metadata tags) - is identical for both containers.
        def _restore_array_attrs(attrs):
            for key in ("bbox", "transform"):
                v = attrs.get(key)
                if (
                    isinstance(v, list)
                    and v
                    and all(isinstance(x, (int, float)) for x in v)
                ):
                    attrs[key] = np.asarray(v)

        _restore_array_attrs(out.attrs)
        for var in out.variables.values():
            _restore_array_attrs(var.attrs)
        return normalize_stack_name(out)
    if isinstance(chunks, str) and chunks == "frames":
        return normalize_stack_name(_frame_chunked_netcdf(path))
    return normalize_stack_name(xr.open_dataset(path, chunks=chunks))


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

    ``_FillValue`` is deliberately NOT in that list even though xarray's Zarr
    backend rejects it, because the backend never sees it: ``to_zarr`` runs
    ``encode_cf_variable`` first, which moves ``_FillValue`` out of encoding and
    into attrs (and the error is raised only afterwards, for whatever survives).
    Dropping it here instead defeated that encode step - and for a cube read
    from a PACKED NetCDF, where ``_FillValue`` is the only record of the
    no-data value and ``encoding["dtype"]`` is still int16, that silently lost
    data: with no fill value to substitute, the float NaNs were cast straight
    to int16 and came back as arbitrary numbers rather than NaN (xarray warns
    "saving variable with floating point data as an integer dtype without any
    _FillValue to use for NaNs", easy to miss in a batch export). Keeping the
    key lets the NaNs be encoded as -32768 and the CF attribute be written, so
    the store round-trips.
    """
    for var in ds.variables.values():
        for stale in (
            "contiguous", "chunksizes", "zlib", "complevel", "shuffle",
            "fletcher32", "preferred_chunks", "compression",
        ):
            var.encoding.pop(stale, None)


def _set_zarr_shuffle(ds):
    """Give int16-on-disk variables a byte-shuffling codec.

    zarr-python 3's default codec chain is ``bytes`` + ``zstd(level=0)`` with no
    shuffle filter. For int16 data packed with CF ``scale_factor``/``add_offset``
    (what the super-resolution export writes) that is a poor default: the high
    byte of a packed reflectance value is near-constant across a scene, so
    grouping the bytes before compressing turns it into long runs. Measured on a
    real SR cube, shuffling lifts the ratio from 1.20 to 1.61 - the NetCDF path
    gets this for free because netCDF4-python enables the HDF5 shuffle filter
    whenever zlib is on, which is why an SR cube used to be ~33% *larger* as
    Zarr than as NetCDF.

    Blosc is the portable way to ask for shuffle: it is a registered Zarr v3
    codec that other implementations read (verified against GDAL's independent
    C++ driver). The ``numcodecs.shuffle`` codec also works in zarr-python but
    is outside the v3 spec and GDAL rejects it, so it is deliberately not used.

    Only int16 is touched. Shuffling *hurts* the float32 cubes measurably
    (ratio 1.52 -> 1.30) because their mantissa low bytes are noise, so those
    keep the library default.
    """
    try:
        from zarr.codecs import BloscCodec, BloscShuffle
    except ImportError:
        return  # older/newer zarr without this API: keep the default codec
    for var in ds.variables.values():
        if np.dtype(var.encoding.get("dtype", var.dtype)) != np.int16:
            continue
        var.encoding["compressors"] = (
            BloscCodec(cname="zstd", clevel=4, shuffle=BloscShuffle.shuffle),
        )


def _set_zarr_fill_value(ds):
    """Make each array's native Zarr ``fill_value`` match its CF ``_FillValue``.

    Zarr stores the no-data value twice over: once as the array's own
    ``fill_value`` in ``zarr.json``, and once as the CF ``_FillValue``
    attribute. xarray keeps them in sync only for Zarr *v2*, where the writer
    reads ``attrs["_FillValue"]`` directly; on *v3* - the format zarr-python 3
    writes by default - it instead reads ``encoding["fill_value"]``, defaults a
    missing one to NaN for floats, and leaves INTEGER arrays to zarr's own
    default of 0.

    That default is actively wrong for the int16-packed cubes
    (``pack_to_int16``): their CF ``_FillValue`` is -32768, so a native
    ``fill_value`` of 0 declares the packed integer 0 - which decodes to
    ``add_offset``, an ordinary mid-range reflectance - to be no-data. xarray
    reads such a store back correctly (it honors the CF attribute), but readers
    that go by the native value do not: GDAL/QGIS report ``NoDataValue = 0``
    and punch every pixel that happens to pack to 0 out of the image, scattered
    across the scene. The equivalent NetCDF has only the CF attribute, so it
    shows no such holes - the cubes differ purely in declared metadata, not in
    a single pixel value.

    Setting the encoding key fixes both readers at once and costs nothing: on
    v3 the CF attribute is written alongside it regardless, so xarray's own
    decoding is unchanged. Left alone on v2, where ``fill_value`` is not a
    valid encoding key (xarray raises) and the attrs path already does the
    right thing.

    The CF ``_FillValue`` is treated as authoritative and overwrites any
    ``fill_value`` already in ``.encoding``. That is not paranoia: reading a
    Zarr store back stores its native fill in ``encoding["fill_value"]``, so a
    cube that came from a Zarr input carries the SOURCE store's value - which
    is stale the moment the pipeline changes the array (super-resolving a
    float32 cube, whose inherited fill is NaN, into an int16-packed one). Like
    the NetCDF keys :func:`_strip_netcdf_encoding` drops, an inherited fill
    describes where the data was read from, not where it is going.
    """
    try:
        import zarr

        if int(zarr.config.get("default_zarr_format", 3)) != 3:
            return
    except Exception:
        return
    for var in ds.variables.values():
        fill = var.attrs.get("_FillValue", var.encoding.get("_FillValue"))
        dtype = np.dtype(var.encoding.get("dtype", var.dtype))
        if fill is None:
            # No CF no-data value to honor. Drop an inherited fill that the
            # outgoing dtype cannot represent (a NaN carried onto an integer
            # array), which zarr would reject, and let xarray default it.
            stale = var.encoding.get("fill_value")
            if (
                stale is not None
                and dtype.kind in "iu"
                and not np.isfinite(stale)
            ):
                var.encoding.pop("fill_value")
            continue
        # Cast so the value written matches the array's own dtype exactly.
        var.encoding["fill_value"] = (
            dtype.type(fill) if dtype.kind in "iufc" else fill
        )


def _progress(q=False):
    """The dask progress bar, or nothing at all when quiet.

    The bar redraws several times a second. On a terminal that is a progress
    indicator; in a SLURM log it is thousands of control-character writes that
    bury the actual output, which is exactly where `q` is used.
    """
    return contextlib.nullcontext() if q else ProgressBar()


def _write_zarr(ds, output, overwrite, q=False):
    """Write ``ds`` to a Zarr store, streaming chunks from dask.

    Unlike the NetCDF path this does NOT materialize the whole cube first: the
    array is (re)chunked to one spatial slice per (time, band) and dask writes
    those chunks straight to the store, so peak memory is a few chunks rather
    than the full cube - the reason Zarr is offered for very large cubes.

    Compression: Zarr applies its own default lossless codec, so a store is
    always compressed. The ``compress`` (zlib) flag is a NetCDF-only knob and is
    intentionally not plumbed here. The one place the default is not good enough
    is int16-packed data, which :func:`_set_zarr_shuffle` handles.
    """
    import shutil
    import warnings

    _set_zarr_fill_value(ds)
    _strip_netcdf_encoding(ds)
    _set_zarr_shuffle(ds)

    # Uniform per-slice chunks: valid for Zarr (which needs equal chunk sizes
    # bar the last) and good for partial reads. Daskifies an in-memory cube so
    # the write streams in both the lazy and already-computed cases.
    chunk_spec = {d: (-1 if d in ("y", "x") else 1) for d in ds.dims}
    ds = ds.chunk(chunk_spec)

    if overwrite and os.path.isdir(output):
        shutil.rmtree(output, ignore_errors=True)

    with warnings.catch_warnings():
        # Both are informational for xarray, which reads everything back
        # correctly: (1) fixed-length unicode coords (e.g. band names) have no
        # stable Zarr-v3 spec yet; (2) consolidated metadata is an xarray
        # convenience outside the v3 core spec.
        #
        # (1) costs OTHER readers the cube STRUCTURE, but not the data. The
        # `band` coord goes out as v3 "fixed_length_utf32" (or "string"), which
        # GDAL 3.12 cannot parse, so its MULTIDIMENSIONAL api - the one that
        # would expose named time/band dims and their labels - fails on the
        # whole group. Its CLASSIC raster path is unaffected: GDAL/QGIS open the
        # store fine and show Time_Series as one flat stack of time x band
        # numbered bands, correctly georeferenced (verified). The NetCDF export
        # behaves the SAME way there (24 unlabelled bands, empty descriptions),
        # so this is not a Zarr-vs-NetCDF difference - export COGs when band
        # identity in a GIS matters. Independent of the codec choice.
        warnings.filterwarnings("ignore", message=".*does not have a Zarr V3 specification.*")
        warnings.filterwarnings("ignore", message=".*[Cc]onsolidated metadata.*")
        # Same progress bar the NetCDF path shows. The chunk() above always
        # leaves ds dask-backed, so to_zarr always drives the dask scheduler
        # ProgressBar hooks and the bar is never a no-op.
        with _progress(q):
            ds.to_zarr(output, mode="w")


def _stream_chunks(ds, budget_bytes=128 * 1024**2):
    """Re-group the non-spatial dims so one streamed write carries ~``budget``.

    A dask-backed ``to_netcdf`` writes one dask chunk per call. With the
    per-slice chunking a STAC load produces (one (time, band) image per chunk)
    that is thousands of tiny HDF5 writes, which the *compressed* path pays for
    twice over: measured on a real 40-date (404 MB) cube, per-slice streaming
    took 89 s and produced a 379 MB file where one big write took 8 s and
    309 MB - same values, same filters, same netCDF chunk layout, so the extra
    bytes are HDF5 per-write allocation overhead and the extra time is per-call
    overhead.

    Grouping whole images together buys most of that back while keeping peak
    memory at roughly one group per dask worker instead of the whole cube. The
    128 MB budget is where the curve flattens on that cube: peak 386 MB
    (vs 822 MB materialized), wall 15.7 s, file 316.7 MB (+2.6%). Bigger
    budgets get closer to the materialized speed and size but give the memory
    back; see the compress notes in export_stac.

    Only the non-spatial dims are merged; ``y``/``x`` are left alone so the
    dask chunks stay whole images and never straddle a netCDF chunk.
    """
    if not _is_dask_backed(ds):
        return ds
    spatial_vars = [
        v for v in ds.data_vars.values() if "y" in v.dims and "x" in v.dims
    ]
    if not spatial_vars:
        return ds
    # Take the dim ORDER from the variable itself (Dataset.dims is a mapping and
    # does not preserve it), so "fastest varying" below really is the innermost
    # non-spatial dim - the one netCDF chunks run contiguously along.
    ref = max(spatial_vars, key=lambda v: v.size)
    slice_bytes = int(ds.sizes["y"]) * int(ds.sizes["x"]) * max(
        v.dtype.itemsize for v in spatial_vars
    )
    if slice_bytes <= 0:
        return ds
    per_group = max(1, int(budget_bytes // slice_bytes))
    other = [d for d in ref.dims if d not in ("y", "x")]
    if not other:
        return ds
    # Fill the budget from the fastest-varying non-spatial dim outwards, so a
    # group is a run of whole images rather than a stripe across many dates.
    spec = {}
    remaining = per_group
    for d in reversed(other):
        take = max(1, min(int(ds.sizes[d]), remaining))
        spec[d] = take
        remaining = max(1, remaining // take)
    return ds.chunk(spec)


def write_qgis_vrt(cube_path, var_name=None, vrt_path=None):
    """Write a small ``.vrt`` beside a NetCDF cube so QGIS shows named bands.

    GDAL flattens the cube's non-spatial dims into one numbered band stack and
    cannot recover their labels, so a 4-date x 6-band cube opens in QGIS as
    "Band 1 ... Band 24" with no band names and time shown only as a raw CF
    offset. A VRT is a tiny XML file that points AT the cube (no data is
    copied) and carries a description per band, so the same stack opens as
    "2024-04-01 blue", "2024-04-01 green", ...

    The reference is written relative to the VRT, so moving the ``.nc`` and the
    ``.vrt`` together keeps it working. The cube itself is not touched: open the
    ``.vrt`` in a GIS, keep using the ``.nc`` everywhere else.

    NetCDF only. The same trick applies cleanly to a Zarr store's *labels*, but
    reading pixels back through the VRT fails on the string ``band`` coordinate
    (GDAL: "Invalid or unsupported format for data_type: fixed_length_utf32"),
    so it is refused rather than silently producing an unreadable file.

    The XML is generated directly from the cube's own metadata rather than via
    ``gdal.Translate``. That is deliberate: on conda, GDAL's netCDF driver is a
    separate plugin (``gdal_netCDF.dll``) that a Python process started without
    full environment activation cannot load - the same trap that costs the JP2
    driver - and a writer that cannot open its own input is useless. Nothing
    here needs to READ the cube through GDAL; QGIS supplies its own netCDF
    driver when it opens the result.

    Returns the path written.
    """
    if is_zarr_path(cube_path):
        raise ValueError(
            "write_qgis_vrt supports NetCDF cubes only - a VRT over a Zarr "
            "store carries the band names but cannot read pixels back."
        )

    from xml.sax.saxutils import escape, quoteattr

    # numpy -> GDAL type names. Anything not listed is not something the cube
    # writer produces; fall back to Float32 rather than emitting a broken file.
    GDAL_TYPES = {
        "uint8": "Byte", "int8": "Int8", "uint16": "UInt16", "int16": "Int16",
        "uint32": "UInt32", "int32": "Int32", "float32": "Float32",
        "float64": "Float64",
    }

    cube_path = os.path.abspath(str(cube_path))
    if vrt_path is None:
        vrt_path = str(Path(cube_path).with_suffix(".vrt"))
    vrt_path = os.path.abspath(str(vrt_path))

    ds = open_cube(cube_path)
    try:
        if var_name is None:
            var_name = STACK_VAR if STACK_VAR in ds.data_vars else next(
                (n for n, v in ds.data_vars.items()
                 if "y" in v.dims and "x" in v.dims), None
            )
        if var_name is None:
            raise ValueError(f"No spatial data variable found in {cube_path}")
        da = ds[var_name]

        # GDAL flattens the non-spatial dims outer-first, i.e. the band index
        # runs over the LAST dim fastest - verified against the data for
        # (time, band). itertools.product below reproduces exactly that order.
        stack_dims = [d for d in da.dims if d not in ("y", "x")]
        per_dim = []
        for dim in stack_dims:
            if dim in ds.coords:
                vals = np.asarray(ds[dim].values)
                if np.issubdtype(vals.dtype, np.datetime64):
                    labels = list(
                        np.datetime_as_string(vals.astype("datetime64[ns]"),
                                              unit="D")
                    )
                else:
                    labels = [str(v) for v in vals]
            else:
                labels = [str(i) for i in range(da.sizes[dim])]
            per_dim.append(labels)
        names = [" ".join(combo) for combo in itertools.product(*per_dim)]

        nx, ny = int(da.sizes["x"]), int(da.sizes["y"])
        # On-disk dtype, which is what GDAL will see - not the decoded one. A
        # packed cube stores int16 but decodes to float.
        dtype = np.dtype(da.encoding.get("dtype", da.dtype))
        gdal_type = GDAL_TYPES.get(dtype.name, "Float32")
        nodata = da.encoding.get("_FillValue", da.attrs.get("_FillValue"))
        if nodata is None and dtype.kind == "f":
            nodata = float("nan")

        wkt = CRS.from_user_input(ds.rio.crs or ds.attrs["crs"]).to_wkt("WKT1_GDAL")
        gt = ds.rio.transform().to_gdal()
    finally:
        ds.close()

    # The source is written RELATIVE to the VRT when the two sit in the same
    # directory, so moving the pair together keeps it working.
    same_dir = os.path.dirname(vrt_path) == os.path.dirname(cube_path)
    ref = os.path.basename(cube_path) if same_dir else cube_path
    src_name = f'NETCDF:"{ref}":{var_name}'

    out = [
        f'<VRTDataset rasterXSize="{nx}" rasterYSize="{ny}">',
        f'  <SRS dataAxisToSRSAxisMapping="1,2">{escape(wkt)}</SRS>',
        '  <GeoTransform>' + ', '.join(repr(float(v)) for v in gt)
        + '</GeoTransform>',
    ]
    for i, name in enumerate(names, start=1):
        out.append(f'  <VRTRasterBand dataType="{gdal_type}" band="{i}">')
        out.append(f'    <Description>{escape(str(name))}</Description>')
        if nodata is not None:
            out.append(f'    <NoDataValue>{float(nodata)!r}</NoDataValue>')
        out.append('    <SimpleSource>')
        out.append('      <SourceFilename relativeToVRT='
                   f'{quoteattr("1" if same_dir else "0")}>'
                   f'{escape(src_name)}</SourceFilename>')
        out.append(f'      <SourceBand>{i}</SourceBand>')
        out.append(f'      <SourceProperties RasterXSize="{nx}" '
                   f'RasterYSize="{ny}" DataType="{gdal_type}" '
                   f'BlockXSize="{nx}" BlockYSize="1"/>')
        out.append(f'      <SrcRect xOff="0" yOff="0" xSize="{nx}" ySize="{ny}"/>')
        out.append(f'      <DstRect xOff="0" yOff="0" xSize="{nx}" ySize="{ny}"/>')
        out.append('    </SimpleSource>')
        out.append('  </VRTRasterBand>')
    out.append('</VRTDataset>')

    with open(vrt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    return vrt_path


def _resolve_geo_attr(stac, name):
    """Find ``crs`` / ``transform`` for an object that may not carry it itself.

    A cube built by :func:`get_stac_layers` is a DataArray with both on its own
    ``attrs``, which is why this used to be a plain attribute lookup. Several
    products are not: :func:`calculate_statistics` returns a Dataset whose
    ``attrs`` are EMPTY because the georeferencing rode along on the variables
    (``ds["Time_Series"].attrs``), so exporting one raised
    ``AttributeError: 'Dataset' object has no attribute 'crs'`` and had to be
    worked around by copying the attrs across by hand.

    Order: the object's own attrs, then (for a Dataset) the first variable that
    has the attribute, then the rio accessor as a last resort - it reads the
    CF grid-mapping variable / coordinates, so it also covers a cube whose
    attrs were dropped by an xarray operation. Raises with the reason when
    nothing supplies it, rather than writing a cube with no georeferencing.
    """
    value = stac.attrs.get(name)
    if value is not None:
        return value

    if isinstance(stac, xr.Dataset):
        for var in stac.data_vars:
            value = stac[var].attrs.get(name)
            if value is not None:
                return value

    try:
        if name == "crs":
            # Kept as a STRING, like the attrs form: a rasterio/pyproj CRS
            # object is not a serializable NetCDF attribute, and this value is
            # written straight back into attrs["crs"] below.
            crs_obj = stac.rio.crs
            if crs_obj is None:
                value = None
            else:
                code = crs_obj.to_epsg()
                value = f"EPSG:{code}" if code else crs_obj.to_wkt()
        else:
            # recalc=True derives it from the x/y coordinates. It falls back to
            # the identity affine when there is nothing to derive it from,
            # which is not a georeference - treat that as "not found".
            affine = stac.rio.transform(recalc=True)
            value = None if affine == Affine.identity() else affine
    except Exception:
        value = None
    if value is not None:
        return value

    raise ValueError(
        f"export_stac could not determine the '{name}' of this cube: it is not "
        f"in its attrs, in any of its variables' attrs, nor derivable from its "
        f"coordinates. Pass {name}=... explicitly."
    )


def export_stac(
    stac,
    output,
    crs=None,
    transform=None,
    var_name=None,
    overwrite=True,
    compress=False,
    vrt=False,
    q=False,
):
    """Write a cube to disk. Output format is chosen by the file extension:

    - ``*.zarr`` -> a chunked, streamed Zarr store (always compressed with
      Zarr's default codec; low peak memory - good for very large cubes).
    - anything else -> a single NetCDF file (the long-standing default).

    ``compress=True`` -> lossless zlib compression (level 4) on all spatial
    variables. NetCDF only (Zarr compresses by default). Values are
    bit-identical on read-back; the write is slower. Shrinks cloud-masked
    cubes especially well (NaN runs compress strongly).

    ``vrt=True`` -> also write a small ``<output>.vrt`` beside the NetCDF that
    labels every band with its date and band name for QGIS (see
    :func:`write_qgis_vrt`). NetCDF only; ignored for Zarr. The cube itself is
    unchanged - the VRT is an extra ~40 KB pointer file.

    Both containers now STREAM a dask-backed cube to disk instead of
    materializing it first, so peak memory is a few chunks rather than the
    whole cube. One consequence: the returned object is whatever was passed in,
    so for a lazy input it stays lazy (this was already true of the Zarr path).
    Callers that want the values in memory should ``.compute()`` the result -
    or, cheaper, re-open the file that was just written.
    """
    if not isinstance(stac, (xr.DataArray, xr.Dataset)):
        raise TypeError(
            f"export_stac expects xarray.DataArray or xarray.Dataset, got {type(stac)}"
        )
    # Writing a cube to a .vrt would produce a file that is neither: the name
    # promises a sidecar while the bytes are NetCDF, and the real sidecar would
    # then overwrite it.
    _reject_vrt(
        output, "write a cube to",
        hint="Use a .nc or .zarr path; tick the band-mapping option to get the "
             "sidecar alongside it.",
    )

    # Georeferencing: what the caller passed wins, otherwise it is read off the
    # object. `is None`, NOT `or`: the stored transform is a 9-element ndarray,
    # whose truthiness raises "The truth value of an array with more than one
    # element is ambiguous", so `transform or ...` rejected a perfectly valid
    # explicit transform.
    if crs is None:
        crs = _resolve_geo_attr(stac, "crs")
    if transform is None:
        transform = _resolve_geo_attr(stac, "transform")

    stac.attrs["transform"] = transform
    stac = stac.rio.write_crs(crs, inplace=True)
    stac.attrs["crs"] = crs

    is_zarr = str(output).lower().endswith(".zarr")

    if isinstance(stac, xr.DataArray):
        name = var_name or stac.name or "Time_Series"
        ds = stac.to_dataset(name=name)
    else:
        ds = stac

    if is_zarr:
        _write_zarr(ds, output, overwrite, q=q)
        if not q:
            print(f"Export is done: {output}")
        return stac

    if overwrite and os.path.exists(output):
        try:
            os.remove(output)
        except PermissionError:
            from datetime import datetime

            suffix = datetime.now().strftime("_%Y%m%d_%H%M%S")
            output = str(Path(output).with_stem(Path(output).stem + suffix))

    # NetCDF used to be written from a fully materialized array: the whole cube
    # was .compute()d first, which peaks at ~2x its size (dask holds every
    # finished chunk, then CONCATENATES them into one new contiguous array).
    # It does not have to be. xarray hands a dask-backed variable to
    # dask.array.store, which pulls one chunk at a time and routes every HDF5
    # call through the backend's process-global HDF5 lock, so the write streams
    # and peak memory is a few chunks. Measured on a real 3.59 GB cube:
    # +7183 MB -> +57 MB peak RSS, and faster; the file reads back bit-
    # identical (values, dtypes, dims, coords, attrs, CRS, _FillValue, netCDF
    # chunk layout and filters all compared). The HDF5 lock is also what makes
    # the graph safe when it READS from other open NetCDF files - the cloud
    # masking / update pattern - which is why this is not the old "NetCDF: HDF
    # error" trap.
    _set_compression(ds, compress)
    if compress:
        # Only the compressed path needs coarser write chunks (see
        # _stream_chunks); uncompressed per-slice streaming is already both
        # cheaper and faster than materializing.
        ds = _stream_chunks(ds)
    if _is_dask_backed(ds):
        # Keeps the familiar progress bar: the compute now happens inside
        # to_netcdf, on the same dask scheduler ProgressBar hooks.
        with _progress(q):
            ds.to_netcdf(output)
    else:
        ds.to_netcdf(output)

    if not q:
        print(f"Export is done: {output}")
    # After the write, so the VRT describes the file that now exists.
    #
    # A sidecar that is already there is REFRESHED even when vrt=False. Its band
    # labels are a snapshot of the cube's dates, so any write that changes the
    # date list - a cloud filter, a time slice, an update, or the "Generate and
    # Overwrite" buttons that add mask bands to the input file - leaves the old
    # labels pointing at the wrong scenes, with no error anywhere. Rewriting is
    # cheap and keeps the pair honest; the alternative is silently wrong dates
    # in someone's GIS.
    stale = None if vrt else Path(str(output)).with_suffix(".vrt")
    if vrt or (stale is not None and stale.exists()):
        try:
            written = write_qgis_vrt(output)
            if not q:
                print(
                    f"QGIS band-labelled VRT: {written}" if vrt
                    else f"Refreshed the existing QGIS VRT: {written}"
                )
        except Exception as exc:
            # A failure here must not lose the cube that was just exported.
            print(f"Note: could not write the QGIS VRT ({exc}). The NetCDF is fine.")
    return stac







def export_to_cogs(
    stac: xr.DataArray | xr.Dataset, output_dir: str, prefix: str = "",
    dtype="float32", q=False,
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
        if not q:
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
            # Special case: Time_Series with only time dim -> use date only (old behavior)
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

        # 1) Export Time_Series first (if present) using date filenames
        if "Time_Series" in stac.data_vars:
            da_main = stac["Time_Series"]
            if "band" not in da_main.dims:
                raise ValueError(
                    "Time_Series is missing required 'band' dimension "
                    f"(dims: {da_main.dims})."
                )
            _export_dataarray(
                da_main,
                base_name="Time_Series",
                date_only_for_time_series=("time" in da_main.dims),
            )

        # 2) Export remaining variables with variable-based filenames
        for var_name, da in stac.data_vars.items():
            if var_name == "Time_Series":
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