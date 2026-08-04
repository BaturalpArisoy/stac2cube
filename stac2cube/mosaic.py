"""Mosaic several data cubes into one.

The spatial counterpart of :func:`stac2cube.update_stac`: that one concatenates
along TIME (new dates onto an existing cube), this one along SPACE (neighbouring
or overlapping areas onto one grid). Its first purpose is to reassemble the
pieces produced by ``aoi_splitter.py`` / ``river_splitter.py``, which cut a
sprawling AOI into cheap chunks precisely so the whole thing never has to be
built at once.

What it does NOT do
-------------------
It does not re-download anything and it does not re-composite. A mosaic of the
cubes you have is exactly that: the pixels each cube already holds, placed on a
common grid. Two consequences worth stating out loud, because both are easy to
forget once the result looks like one seamless cube:

* **Overlapping pixels can legitimately disagree.** Each piece was built from
  its own STAC query, so the same pixel can be backed by a different set of
  scenes in each cube - a different tile of a multi-tile day, a different set of
  scenes dropped by the footprint prefilter, a different native CRS warped in.
  Measured on two adjacent Naryn reach cubes (naryn_reaches 1 and 2, both
  median composites of the same date range): every pixel of their 101x262 shared
  window is valid in BOTH, and they differ by up to 0.0168 in red reflectance
  and 0.397 in NDVI. That is why ``report=True`` measures the disagreement and
  prints it instead of quietly picking a side - see :func:`_overlap_report`.

* **A mosaic of composites is not a composite of the mosaic.** If one piece's
  median is over 40 dates and its neighbour's over 12, the merged median layer
  has spatially varying temporal support. Nothing here can fix that (the dates
  are gone - only the composite was stored); it is recorded in
  ``attrs["mosaic_composite_warning"]`` so the result is not mistaken for a
  uniformly composited product. When the pieces still carry their time series,
  mosaic THAT and recompute the composite from the result instead.

Size
----
The union grid is rectangular, so mosaicking pieces that were cut apart to avoid
a sparse cube reconstructs exactly that sparse cube. All 47 naryn_reaches cubes
union to 31931 x 5874 = 187.6 Mpx at 10.7% fill - 4.50 GB as a 6-band composite,
and roughly 450 GB if they carried 100 timesteps. The estimate is printed before
any pixel moves, and the whole path stays lazy so a result that only fits on
disk can still be written with ``output=``.
"""

import os
import numpy as np
import xarray as xr

from .export_cfg import open_cube, export_stac
from .get_data import crs_attr_string, validate_target_crs

# How far a cube's pixel centres may sit from the target grid, as a fraction of
# one pixel, and still count as "the same grid". Not zero because coordinates
# come back from NetCDF/Zarr as float64 that has been through a transform, so
# bit-equality is too strict; far below half a pixel because anything approaching
# that is a genuinely different grid that must be resampled, not waved through.
GRID_TOLERANCE = 0.01

# Overlap policies. "first" is the only one that never produces a value no input
# cube contained, which is why it is the default.
_OVERLAP_POLICIES = ("first", "mean", "median", "min", "max")

# Attributes that MUST agree across the inputs: each one changes what the pixel
# values mean, so merging cubes that disagree on any of them produces a cube
# whose own attrs are a lie about part of its extent.
_HARD_ATTRS = ("mission", "cloud_status")

# Attributes worth reporting when they differ but which do not invalidate the
# merge - they describe how each piece was built, not what its numbers mean.
_SOFT_ATTRS = (
    "resampling",
    "stac_api",
    "tile_handling",
    "partial_scene_handling",
    "min_footprint_coverage",
    "nir_dark_threshold",
    "shadow_proj_distance",
)

# Attributes that describe ONE cube's extent or provenance and cannot describe a
# mosaic of several. Recomputed, unioned, or dropped - never inherited.
_PER_CUBE_ATTRS = (
    "crs", "transform", "bbox", "estimated_size",
    # A CF pointer to the grid-mapping variable. It belongs on the raster
    # variable (where rio.write_crs puts it back), never on the Dataset.
    "grid_mapping",
    "native_crs", "native_crs_share", "tile_id",
    "footprint_prefilter_skipped", "footprint_prefilter",
    "scene_metadata_multiday", "solar_azimuth_by_day",
)


# --- inputs -------------------------------------------------------------------


def _open_inputs(cubes, chunks="frames"):
    """Open every input, returning ``[(label, Dataset), ...]``.

    Paths go through :func:`open_cube`, so NetCDF and Zarr pieces can be mixed
    freely and a legacy ``Spectral_Temporal_Stack`` cube is renamed on the way
    in. Already-open objects are taken as they are (and never mutated).
    """
    if isinstance(cubes, (str, os.PathLike, xr.Dataset, xr.DataArray)):
        raise TypeError(
            "mosaic_cubes expects a LIST of cubes (paths or xarray objects), "
            "not a single cube. Pass e.g. [cube_a, cube_b]."
        )
    cubes = list(cubes)
    if len(cubes) < 2:
        raise ValueError(
            f"mosaic_cubes needs at least 2 cubes to merge, got {len(cubes)}."
        )

    out = []
    for i, c in enumerate(cubes, start=1):
        if isinstance(c, (str, os.PathLike)):
            label = os.path.basename(str(c).rstrip("/\\")) or str(c)
            ds = open_cube(c, chunks=chunks)
        elif isinstance(c, xr.DataArray):
            label = c.name or f"cube {i}"
            ds = c.to_dataset(name=c.name or "Time_Series")
        elif isinstance(c, xr.Dataset):
            label = f"cube {i}"
            ds = c
        else:
            raise TypeError(
                f"Input {i} is a {type(c).__name__}; mosaic_cubes takes file "
                "paths, xarray.Dataset or xarray.DataArray."
            )
        for dim in ("y", "x"):
            if dim not in ds.dims:
                raise ValueError(
                    f"{label} has no '{dim}' dimension - it is not a stac2cube "
                    "raster cube."
                )
        out.append((label, ds))
    return out


def _raster_layers(ds):
    """Names of the real raster variables, i.e. everything with both y and x.

    Excludes the scalar ``spatial_ref`` grid-mapping variable, which
    :func:`open_cube` leaves in ``data_vars`` (it opens without
    ``decode_coords="all"``), and would otherwise be mosaicked as if it were a
    band.
    """
    return [
        str(v) for v in ds.data_vars
        if "y" in ds[v].dims and "x" in ds[v].dims
    ]


def _cube_attrs(ds):
    """A cube's metadata, wherever the writer happened to put it.

    Cubes do NOT all carry their attributes in the same place, and the
    difference is invisible until something reads the wrong one:

    * a TIME-SERIES cube is exported from a DataArray, and
      ``export_stac`` turns that into a Dataset with ``to_dataset(name=...)``,
      which hangs the attributes on the VARIABLE - the file's global attrs are
      empty;
    * a COMPOSITE cube is exported as a Dataset and carries them in both places.

    Reading only ``ds.attrs`` therefore sees nothing at all for a time-series
    cube, which would make the compatibility check below compare ``None`` with
    ``None`` for every cube and wave through inputs built with different
    missions or different cloud masking - exactly what it exists to prevent.
    The Dataset's own attrs still win where both are set.
    """
    attrs = dict(ds.attrs)
    for name in _raster_layers(ds):
        for key, value in ds[name].attrs.items():
            attrs.setdefault(key, value)
    return attrs


def _cube_crs(ds, label):
    """The cube's CRS as an ``"EPSG:<code>"`` string (or WKT)."""
    crs = _cube_attrs(ds).get("crs")
    if crs is None:
        crs = getattr(ds.rio, "crs", None)
    if crs is None:
        raise ValueError(
            f"{label} carries no CRS (neither attrs['crs'] nor a written "
            "spatial_ref), so it cannot be placed on a common grid."
        )
    return crs_attr_string(crs)


# --- grid ---------------------------------------------------------------------


def _grid_of(ds, label):
    """Pixel size, origin and extent of a cube's grid.

    Also verifies the grid is REGULAR: everything below - the phase test, the
    integer offset onto the union grid - assumes a constant step, and an
    irregular axis would otherwise be snapped onto a grid it does not lie on,
    shifting pixels silently.
    """
    x = np.asarray(ds["x"].values, dtype="float64")
    y = np.asarray(ds["y"].values, dtype="float64")
    if x.size < 2 or y.size < 2:
        raise ValueError(
            f"{label} is {y.size}x{x.size} pixels; a mosaic needs at least 2 "
            "pixels per axis to know its resolution."
        )

    def _step(a, axis):
        d = np.diff(a)
        step = float(np.median(d))
        if not np.allclose(d, step, rtol=0, atol=abs(step) * GRID_TOLERANCE):
            raise ValueError(
                f"{label} has an irregular {axis} axis (steps from "
                f"{d.min():.6g} to {d.max():.6g}). Mosaicking assumes a regular "
                "grid."
            )
        return step

    return {
        "res_x": _step(x, "x"),
        "res_y": _step(y, "y"),
        "x0": float(x[0]), "x1": float(x[-1]),
        "y0": float(y[0]), "y1": float(y[-1]),
        "nx": int(x.size), "ny": int(y.size),
        "xmin": float(min(x[0], x[-1])), "xmax": float(max(x[0], x[-1])),
        "ymin": float(min(y[0], y[-1])), "ymax": float(max(y[0], y[-1])),
    }


def _same_phase(a, b, res, tol=GRID_TOLERANCE):
    """Do two on-grid coordinates belong to the SAME grid of step ``res``?

    Compares ``(a - b) mod res`` against both 0 and ``res``, because a phase
    difference of 0.999 pixels is 0.001 pixels the other way round - testing
    only the remainder would reject an aligned pair whose float dust happens to
    fall just below the modulus.
    """
    r = abs(a - b) % abs(res)
    return min(r, abs(res) - r) <= abs(res) * tol


def _choose_target_crs(infos, crs, q=False):
    """Pick the mosaic's single CRS: the one covering the largest ground area.

    A mosaic has ONE grid, so one CRS. Defaulting to the CRS of the cube with
    the biggest footprint is the same "least reprojection" principle
    :func:`stac2cube.get_data._choose_target_crs` applies when picking a target
    CRS for scenes - the choice that warps the fewest pixels. Ties break on the
    order the cubes were given, so the result is deterministic.

    Deliberately NOT ``estimate_utm_crs`` on the union: it takes the bbox CENTRE
    as a point and follows the nominal 6-degree zone rule, which disagrees with
    MGRS delivery in the Norway and Svalbard exception zones - it would pick a
    CRS none of the inputs is in.
    """
    if crs is not None:
        return validate_target_crs(crs), "requested by the caller"

    area = {}
    for info in infos:
        g = info["grid"]
        area[info["crs"]] = area.get(info["crs"], 0.0) + abs(
            (g["xmax"] - g["xmin"]) * (g["ymax"] - g["ymin"])
        )
    if len(area) == 1:
        return next(iter(area)), "the only projection present"

    best = max(area, key=lambda k: area[k])
    share = 100.0 * area[best] / sum(area.values())
    if not q:
        others = ", ".join(k for k in sorted(area) if k != best)
        print(
            f"The inputs are in {len(area)} projections. Mosaicking into "
            f"{best} ({share:.0f}% of the combined area); {others} will be "
            "reprojected into it.",
            flush=True,
        )
    return best, f"largest area ({share:.0f}% of the total)"


def _target_resolution(infos, target_crs, resolution):
    """Pixel size of the mosaic grid: the finest present, unless overridden.

    Preferring the finest never throws detail away - the coarse cube is
    upsampled, which invents no information but loses none either. Measured only
    on cubes ALREADY in the target CRS when there are any, since a foreign
    cube's step is in its own CRS's metres and only approximately comparable.
    """
    if resolution is not None:
        r = abs(float(resolution))
        if r <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}.")
        return r, -r

    native = [i for i in infos if i["crs"] == target_crs] or infos
    rx = min(abs(i["grid"]["res_x"]) for i in native)
    ry = min(abs(i["grid"]["res_y"]) for i in native)
    return rx, -ry


def _union_grid(infos, target_crs, res_x, res_y, q=False):
    """Build the x/y coordinates of the mosaic, covering every input.

    The grid PHASE is taken from a reference cube (the largest one already in
    the target CRS) rather than from the union's own corner, so every cube that
    already shares that phase lands on the grid exactly, by integer offset, and
    needs no resampling at all. Cubes in another CRS have their bounds
    transformed into the target and the grid is extended outward to cover them.
    """
    ref = max(
        (i for i in infos if i["crs"] == target_crs),
        key=lambda i: i["grid"]["nx"] * i["grid"]["ny"],
        default=None,
    )
    if ref is not None:
        phase_x = ref["grid"]["x0"]
        phase_y = ref["grid"]["y0"]
    else:
        # No input is in the target CRS - it was requested explicitly - so every
        # cube is being resampled and there is no phase worth preserving. Align
        # to a whole multiple of the pixel size (phase 0). Inheriting the first
        # cube's phase instead was tempting and wrong: an easting of 300605 in
        # UTM 43N carries a phase of 5 m that means nothing in, say, EPSG:3035,
        # and it would silently put every mosaic built this way on a grid offset
        # by half a pixel from the obvious one.
        phase_x = 0.0
        phase_y = 0.0

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for info in infos:
        g = info["grid"]
        if info["crs"] == target_crs:
            b = (g["xmin"], g["ymin"], g["xmax"], g["ymax"])
        else:
            from pyproj import Transformer

            tr = Transformer.from_crs(info["crs"], target_crs, always_xy=True)
            b = tr.transform_bounds(
                g["xmin"], g["ymin"], g["xmax"], g["ymax"]
            )
        info["bounds_target"] = b
        xmins.append(b[0]); ymins.append(b[1]); xmaxs.append(b[2]); ymaxs.append(b[3])

    step_x, step_y = abs(res_x), abs(res_y)
    # floor/ceil onto the reference phase so the grid covers every input whole.
    x_start = phase_x + step_x * np.floor((min(xmins) - phase_x) / step_x)
    nx = int(np.floor((max(xmaxs) - x_start) / step_x)) + 1
    y_start = phase_y - step_y * np.floor((phase_y - max(ymaxs)) / step_y)
    ny = int(np.floor((y_start - min(ymins)) / step_y)) + 1

    xs = x_start + np.arange(nx, dtype="float64") * step_x
    ys = y_start - np.arange(ny, dtype="float64") * step_y
    if not q:
        print(
            f"Mosaic grid: {ny} x {nx} pixels at {step_x:g} m in {target_crs}.",
            flush=True,
        )
    return xs, ys


def _snap_indices(coord, grid, res, label, axis):
    """Integer offset of ``coord`` into ``grid``, or None if it is off-grid.

    Returns the start index when the cube's pixel centres coincide with grid
    positions to within :data:`GRID_TOLERANCE` of a pixel. Being an INTEGER
    offset is the whole point: the cube's own coordinates are then replaced by
    the exact grid values, so the later reindex matches on identical floats and
    places every pixel without touching a single value.
    """
    i0 = int(np.round((coord[0] - grid[0]) / res))
    if i0 < 0 or i0 + coord.size > grid.size:
        return None
    window = grid[i0:i0 + coord.size]
    if np.max(np.abs(coord - window)) > abs(res) * GRID_TOLERANCE:
        return None
    return i0


# --- placing one cube on the mosaic grid --------------------------------------


def _resampling_enum(name):
    from rasterio.enums import Resampling

    try:
        return getattr(Resampling, str(name))
    except AttributeError:
        raise ValueError(
            f"resampling={name!r} is not a rasterio resampling method. Use one "
            "of: nearest, bilinear, cubic, cubic_spline, lanczos, average, "
            "mode, max, min, med, q1, q3."
        )


def _reproject_layer(da, src_crs, target_crs, xs, ys, res_x, res_y, resampling, q):
    """Warp one layer onto the mosaic grid.

    Time series are warped ONE DATE AT A TIME. rioxarray warps the last two
    dimensions and treats a single leading dimension as bands, so a
    ``(time, band, y, x)`` array has one dimension too many; slicing time off
    also bounds the memory, which matters because this path is eager - rasterio
    needs the pixels in memory, so a reprojected input is materialized whatever
    the rest of the pipeline does.
    """
    from affine import Affine

    step_x, step_y = abs(res_x), abs(res_y)
    # Pixel CORNER of the top-left pixel; the coordinates are pixel centres.
    dst_transform = Affine(
        step_x, 0.0, float(xs[0]) - step_x / 2.0,
        0.0, -step_y, float(ys[0]) + step_y / 2.0,
    )
    shape = (int(ys.size), int(xs.size))
    method = _resampling_enum(resampling)

    def _warp(arr):
        out = (
            arr.rio.write_crs(src_crs)
            .rio.write_nodata(np.nan)
            .rio.reproject(
                target_crs,
                transform=dst_transform,
                shape=shape,
                resampling=method,
                nodata=np.nan,
            )
        )
        # rioxarray regenerates the axes from the transform; they equal xs/ys to
        # float dust, but the combine below matches coordinates exactly, so pin
        # them to the grid's own values.
        return out.assign_coords(x=xs, y=ys)

    if "time" in da.dims:
        frames = []
        n = int(da.sizes["time"])
        for i in range(n):
            frames.append(_warp(da.isel(time=[i]).squeeze("time", drop=False)))
            if not q and (i + 1) % 10 == 0:
                print(f"  reprojecting: {i + 1}/{n} dates", flush=True)
        return xr.concat(frames, dim="time")
    return _warp(da)


def _place(ds, info, layers, xs, ys, res_x, res_y, target_crs,
           on_grid_mismatch, resampling, q):
    """Put one cube's layers on the mosaic grid, aligned or resampled.

    The fast path - same CRS, same resolution, same phase - only relabels the
    coordinates with the grid's exact floats and reindexes, which moves no
    values and stays lazy. Anything else has to be warped, and that is refused
    by default: a silent sub-pixel snap is exactly the kind of change that never
    shows up in a plot but ruins a per-pixel comparison later.
    """
    g = info["grid"]
    aligned = (
        info["crs"] == target_crs
        and abs(abs(g["res_x"]) - abs(res_x)) <= abs(res_x) * GRID_TOLERANCE
        and abs(abs(g["res_y"]) - abs(res_y)) <= abs(res_y) * GRID_TOLERANCE
        and _same_phase(g["x0"], float(xs[0]), res_x)
        and _same_phase(g["y0"], float(ys[0]), res_y)
    )

    out = {}
    if aligned:
        x = np.asarray(ds["x"].values, dtype="float64")
        y = np.asarray(ds["y"].values, dtype="float64")
        ix = _snap_indices(x, xs, abs(res_x), info["label"], "x")
        iy = _snap_indices(y, ys, -abs(res_y), info["label"], "y")
        if ix is not None and iy is not None:
            snapped = ds.assign_coords(
                x=xs[ix:ix + x.size], y=ys[iy:iy + y.size]
            )
            for name in layers:
                out[name] = snapped[name].reindex(x=xs, y=ys)
            info["placement"] = "aligned"
            return out
        aligned = False  # phase matched but the window did not; fall through

    if on_grid_mismatch != "resample":
        raise ValueError(
            f"{info['label']} is not on the mosaic grid: it is in "
            f"{info['crs']} at {abs(g['res_x']):g} m "
            f"(mosaic: {target_crs} at {abs(res_x):g} m, "
            f"origin x={float(xs[0]):.3f} y={float(ys[0]):.3f}). Placing it "
            "means RESAMPLING, which changes its pixel values. Pass "
            'on_grid_mismatch="resample" to allow that (and choose '
            "resampling=), or rebuild the piece on the mosaic's grid."
        )

    if not q:
        print(
            f"Reprojecting {info['label']} from {info['crs']} into "
            f"{target_crs} ({resampling})...",
            flush=True,
        )
    for name in layers:
        out[name] = _reproject_layer(
            ds[name], info["crs"], target_crs, xs, ys, res_x, res_y,
            resampling, q,
        )
    info["placement"] = f"resampled ({resampling})"
    return out


# --- joining the non-spatial axes ---------------------------------------------


def _quiet_placement_warnings():
    """Silence dask's chunk-count PerformanceWarning during placement only.

    Reindexing a piece onto the union grid necessarily multiplies its chunk
    count - a 388x422 cube placed on a 4645x2278 grid gains a lot of NaN blocks,
    and dask warns that the count grew by 12x or 17x. Here that is the intended
    operation, not a mistake, and the extra blocks are empty. Scoped to the
    placement step so a genuine chunk explosion anywhere else still surfaces.
    """
    import warnings
    from contextlib import contextmanager

    try:
        from dask.array.core import PerformanceWarning
    except ImportError:  # dask moved it in some versions
        PerformanceWarning = None

    @contextmanager
    def _ctx():
        with warnings.catch_warnings():
            if PerformanceWarning is not None:
                warnings.filterwarnings(
                    "ignore",
                    category=PerformanceWarning,
                    message=".*Increasing number of chunks.*",
                )
            yield

    return _ctx()


def _common_chunks(da, block=2048):
    """A chunk layout every placed cube can share.

    Each input arrives with its OWN chunk boundaries, offset onto the union grid
    by its own pixel offset, and reindexing adds more at the edges. Combining
    them then forces dask to align on the union of every boundary, which
    fragments the graph - measured on 5 reach cubes, dask warned that the chunk
    count grew by 12x and 17x. Putting every cube on the same regular block grid
    first removes the problem at the source: the boundaries coincide, so the
    combine is block-for-block.

    One scene per chunk along time (the layout every stac2cube writer uses) and
    all bands together, so a single date stays cheap to read.
    """
    spec = {}
    for dim in ("y", "x"):
        if dim in da.dims:
            spec[dim] = min(block, int(da.sizes[dim]))
    if "time" in da.dims:
        spec["time"] = 1
    if "band" in da.dims:
        spec["band"] = -1
    return spec


def _join_values(seqs, how, what):
    """Union or intersection of a coordinate's values, order preserved."""
    if how == "union":
        seen, out = set(), []
        for s in seqs:
            for v in s:
                if v not in seen:
                    seen.add(v)
                    out.append(v)
        return out
    keep = set(seqs[0])
    for s in seqs[1:]:
        keep &= set(s)
    out = [v for v in seqs[0] if v in keep]
    if not out:
        raise ValueError(
            f"The cubes share no {what}. Their {what} sets are: "
            + " | ".join(", ".join(map(str, s)) for s in seqs)
            + f'. Use {what[:-1] if what.endswith("s") else what}'
            '_join="union" to keep them all instead.'
        )
    return out


def _drop_time_coords(ds, label, dropped):
    """Remove per-scene (time,) coordinates - they cannot describe a mosaic.

    ``cloud_percentage`` and ``scene_coverage`` are measured against ONE cube's
    own AOI: coverage divides by that AOI's pixel count and the cloud percentage
    counts clouds inside it. The same date therefore carries different values in
    each piece, and neither of them describes the merged extent. There is
    nothing to recompute them from either - the union grid includes area no
    input covers, which would be counted as unobserved and inflate both.

    A second reason to drop rather than pick a side: the coverage DENOMINATOR
    changed (it used to be the imaged union, now it is the AOI), so a value from
    a cube built before that change is not comparable with one built after, and
    nothing in the file says which it is.

    Same principle as :func:`stac2cube.get_update._reconcile_time_coords`: drop
    what cannot honestly describe the result rather than fill it in.
    """
    names = [
        str(c) for c in ds.coords
        if str(c) != "time" and ds[c].dims == ("time",)
    ]
    if names:
        dropped.update(names)
        ds = ds.drop_vars(names)
    return ds


# --- combining ----------------------------------------------------------------


def _combine(arrays, overlap):
    """Reduce the per-cube layers, already on one grid, to a single layer."""
    if overlap == "first":
        # Priority order = the order the cubes were given. `where` keeps the
        # left value wherever it is finite and takes the next cube's only in the
        # holes, so no pixel is ever a blend and every value came from exactly
        # one input.
        out = arrays[0]
        for nxt in arrays[1:]:
            out = out.where(out.notnull(), nxt)
        return out

    stacked = xr.concat(arrays, dim="_source", coords="minimal", compat="override")

    # NaN means "this cube does not reach here", so the reduction skips it.
    #
    # The complication is that the union grid is rectangular while the cubes are
    # not, so pixels NO cube reaches are NaN in every input - and a skipna
    # reduction over an all-NaN slice makes numpy warn ("All-NaN slice
    # encountered" / "Mean of empty slice") once per chunk, for exactly the case
    # this function is built around, where NaN out is the right answer.
    #
    # Rather than suppress the warning, the all-NaN slice is never created: those
    # pixels are filled with zeros before the reduction (any finite value would
    # do - the reduction of a constant slice is that constant) and masked back to
    # NaN afterwards. Suppression was tried first and is not reliable here:
    # warnings filters are process-global state and dask runs these kernels on a
    # thread pool, so one worker's catch_warnings block is undone by another's.
    n_valid = stacked.notnull().sum(dim="_source")
    safe = stacked.where(n_valid > 0, 0.0)
    out = getattr(safe, overlap)(dim="_source", skipna=True)
    return out.where(n_valid > 0)


def _source_layers(arrays, overlap):
    """Provenance rasters: which cube a pixel came from, and how many had it.

    ``mosaic_source`` is the 1-based index of the first cube holding a finite
    value (0 = no cube did), which for ``overlap="first"`` is literally where
    the value came from. ``mosaic_n_sources`` counts how many cubes covered the
    pixel, so an overlap is visible whatever the policy. Both are reduced over
    ``band`` (a pixel is either imaged or not, not per-band), so a time series
    yields ``(time, y, x)`` and a composite ``(y, x)``.
    """
    valid = []
    for a in arrays:
        v = a.notnull()
        if "band" in v.dims:
            v = v.any(dim="band")
        valid.append(v)
    stacked = xr.concat(valid, dim="_source", coords="minimal", compat="override")
    any_valid = stacked.any(dim="_source")
    # argmax over booleans returns the first True, which is the first cube that
    # covers the pixel; masked to 0 where none of them does.
    first = stacked.argmax(dim="_source") + 1
    src = xr.where(any_valid, first, 0).astype("uint8").rename("mosaic_source")
    n = stacked.sum(dim="_source").astype("uint8").rename("mosaic_n_sources")
    return src, n


# --- reporting ----------------------------------------------------------------


def _overlap_report(placed, infos, layer, max_pairs=50, q=False):
    """Measure, and print, how much overlapping cubes actually disagree.

    The question this answers is the one the overlap policy cannot: for pixels
    more than one cube holds, do the cubes AGREE? If they do, the policy is
    irrelevant and the mosaic is unambiguous. If they do not, the size of the
    disagreement is the uncertainty in the seams - and it is real, because each
    piece was built from its own STAC query (different tiles per day, different
    prefiltered scenes, different warp paths).

    Measured pairwise on the overlapping window only, on ONE band and ONE date,
    which keeps it far cheaper than the mosaic itself. It is a diagnostic, not
    the merge - nothing here feeds the result.

    The sample date is chosen PER PAIR from the dates both cubes actually hold,
    never positionally. Taking index 0 of the merged time axis looked right and
    was not: on two across-track cubes sharing 4 of 8 dates, the first date
    belonged to one cube alone, so every pixel of the other was NaN and the
    report announced "no pixel in common" while the mosaic's own provenance
    layer counted 105740 shared pixel-dates.

    Returns ``(n_pairs_overlapping, worst_abs_diff, summary_lines)``.
    """
    def _times_of(info):
        ds = info["ds"]
        if "time" not in ds.dims:
            return None
        # The raw datetime64 scalars, NOT .tolist(): tolist() on a
        # datetime64[ns] array yields plain integers (nanoseconds since the
        # epoch), which then fail to convert back to a date.
        return set(np.asarray(ds["time"].values))

    lines, n_overlap, worst = [], 0, 0.0
    n = len(infos)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    checked = 0
    for i, j in pairs:
        if checked >= max_pairs:
            lines.append(
                f"(stopped after {max_pairs} pairs; {len(pairs) - checked} more "
                "not measured)"
            )
            break
        a, b = infos[i]["label"], infos[j]["label"]
        da, db = placed[i].get(layer), placed[j].get(layer)
        if da is None or db is None:
            continue

        if "time" in da.dims and "time" in db.dims:
            ta, tb = _times_of(infos[i]), _times_of(infos[j])
            shared = sorted(ta & tb) if (ta and tb) else []
            if not shared:
                lines.append(
                    f"  {a} vs {b}: no date in common, so their pixels never "
                    "meet and the overlap policy cannot apply to this pair."
                )
                continue
            when = shared[0]
            da, db = da.sel(time=when), db.sel(time=when)
        if "band" in da.dims:
            da = da.isel(band=0)
        if "band" in db.dims:
            db = db.isel(band=0)
        both = (da.notnull() & db.notnull())
        # Reduced without skipna, over arrays whose non-overlap pixels are
        # filled rather than NaN, for the same reason as _combine: a skipna
        # reduction over the all-NaN majority of this grid warns once per chunk.
        diff = np.abs(da - db)
        n_arr = both.sum()
        total = diff.where(both, 0.0).sum()
        dmax_arr = diff.where(both, -np.inf).max()
        import dask

        n, total, dmax = dask.compute(n_arr, total, dmax_arr)
        n = int(n)
        if n == 0:
            continue
        checked += 1
        n_overlap += n
        dmax = float(dmax)
        dmean = float(total) / n
        worst = max(worst, dmax)
        lines.append(
            f"  {a} vs {b}: {n} shared pixels, mean |diff| {dmean:.6g}, "
            f"max |diff| {dmax:.6g}"
        )

    if not q:
        if not lines:
            print(
                "Overlap check: the cubes hold no pixel in common (they only "
                "abut or are disjoint), so the overlap policy never applies.",
                flush=True,
            )
        else:
            print(
                f"Overlap check on '{layer}' (first band; per pair, the first "
                "date both cubes hold):",
                flush=True,
            )
            for line in lines:
                print(line, flush=True)
            if worst == 0.0:
                print(
                    "  The overlapping cubes agree exactly, so the overlap "
                    "policy makes no difference to this mosaic.",
                    flush=True,
                )
            else:
                print(
                    f"  They DISAGREE by up to {worst:.6g}. Each piece was "
                    "built from its own query, so overlapping pixels can rest "
                    "on different scenes. The policy decides which value the "
                    "mosaic keeps.",
                    flush=True,
                )
    return n_overlap, worst, lines


# --- attributes ---------------------------------------------------------------


def _attr_equal(a, b):
    try:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return np.array_equal(np.asarray(a), np.asarray(b))
        return bool(a == b)
    except Exception:
        return False


def _check_compatibility(infos, strict, q):
    """Refuse (or report) inputs whose pixels do not mean the same thing.

    A cube's attrs describe the WHOLE cube, so the merged attrs can only be
    honest if the inputs agree on what a value means. ``mission`` and
    ``cloud_status`` decide exactly that - merging an SCL-masked piece with an
    unmasked one gives a cube that claims one masking for an extent where two
    were used. The rest are reported and carried as divergences.

    Returns the list of human-readable divergence strings (empty when the
    inputs are uniform).
    """
    divergences = []
    for key in _HARD_ATTRS + _SOFT_ATTRS:
        values = [(i["label"], i["attrs"].get(key)) for i in infos]
        first = values[0][1]
        if all(_attr_equal(v, first) for _, v in values):
            continue
        text = f"{key}: " + ", ".join(f"{lab}={v!r}" for lab, v in values)
        divergences.append(text)
        if key not in _HARD_ATTRS:
            # Soft divergences are printed whatever `strict` is. They used to be
            # recorded in attrs and NOT printed under the default strict=True,
            # so the one setting most likely to explain a seam - the pieces
            # having been built from different scene sets - was invisible unless
            # someone thought to inspect the merged cube's attributes.
            if not q:
                print(f"Note: the cubes were not built identically - {text}", flush=True)
            continue
        if key in _HARD_ATTRS:
            if strict:
                raise ValueError(
                    f"The cubes disagree on '{key}', which changes what their "
                    f"pixel values mean:\n  {text}\nMerging them would produce "
                    "a cube whose attributes describe only part of its extent. "
                    "Rebuild the pieces consistently, or pass strict=False to "
                    "merge anyway (the divergence is then recorded in "
                    "attrs['mosaic_divergences'])."
                )
            if not q:
                print(f"Warning: {text}", flush=True)
    if divergences and not q:
        print(
            "Every difference above is recorded in "
            "attrs['mosaic_divergences'].",
            flush=True,
        )
    return divergences


def _mosaic_attrs(infos, target_crs, crs_reason, xs, ys, res_x, res_y,
                  divergences, options, layer_dims):
    """Attributes for the merged cube: shared, recomputed, and provenance."""
    from affine import Affine

    shared = {}
    first = infos[0]["attrs"]
    for key, value in first.items():
        if key in _PER_CUBE_ATTRS:
            continue
        if all(_attr_equal(i["attrs"].get(key), value) for i in infos[1:]):
            shared[key] = value

    step_x, step_y = abs(res_x), abs(res_y)
    transform = Affine(
        step_x, 0.0, float(xs[0]) - step_x / 2.0,
        0.0, -step_y, float(ys[0]) + step_y / 2.0,
    )

    from pyproj import Transformer

    tr = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
    west, south, east, north = tr.transform_bounds(
        float(xs.min()) - step_x / 2.0, float(ys.min()) - step_y / 2.0,
        float(xs.max()) + step_x / 2.0, float(ys.max()) + step_y / 2.0,
    )

    shared["crs"] = target_crs
    # An Affine, not the flat 9-float array a cube read back from disk carries.
    # That is what a freshly built cube holds in memory (main.py passes
    # rio.transform() straight through), and what export_stac needs: its
    # `transform or stac.transform` fallback raises "truth value of an array
    # with more than one element is ambiguous" on an ndarray. NetCDF/Zarr
    # serialise the Affine to the same 9 floats, so the file is unchanged.
    shared["transform"] = transform
    shared["bbox"] = np.array([west, south, east, north], dtype="float64")

    # Union of the per-cube provenance lists, so nothing an input recorded is
    # lost even though the per-cube shares no longer apply to the whole.
    for key in ("tile_id", "native_crs"):
        merged = []
        for info in infos:
            for v in np.atleast_1d(info["attrs"].get(key, [])).tolist():
                if v not in merged:
                    merged.append(v)
        if merged:
            shared[key] = merged

    shared["mosaic_n_inputs"] = len(infos)
    shared["mosaic_sources"] = [i["label"] for i in infos]
    # Q: what happened to the CRS of the cubes that were NOT chosen? Recorded
    # here, one entry per input, so a reprojected piece can always be traced
    # back to the projection its pixels were actually acquired and built in.
    shared["mosaic_source_crs"] = [i["crs"] for i in infos]
    shared["mosaic_source_placement"] = [
        i.get("placement", "unknown") for i in infos
    ]
    shared["mosaic_target_crs_reason"] = crs_reason
    for key, value in options.items():
        shared[f"mosaic_{key}"] = value
    if divergences:
        shared["mosaic_divergences"] = divergences

    # A mosaic of stored composites carries the composites' own limitation: the
    # pieces were reduced over whatever dates each one had.
    if any("time" not in dims for dims in layer_dims.values()):
        shared["mosaic_composite_warning"] = (
            "Temporal composites were mosaicked as stored. Each piece was "
            "reduced over its OWN set of dates, so the temporal support of "
            "these layers varies across the extent; they are not equivalent to "
            "compositing the merged time series."
        )
    return shared


# --- public API ---------------------------------------------------------------


def mosaic_layers(cubes):
    """What a set of cubes offers a mosaic - without merging anything.

    ``mosaic_cubes(layers=...)`` is a FILTER, not a way to find out what is
    there: leaving it at None merges everything common, and naming a layer only
    some cubes hold is an error. That is fine headless, where the error text
    lists the common layers, but useless for a caller that has to OFFER the
    choice before anything runs - a GUI cannot populate a layer picker, or a
    projection picker, by provoking an exception.

    Reads metadata only: :func:`open_cube` is lazy, so no pixels are fetched or
    decoded, and paths opened here are closed again.

    Returned in ONE call, from ONE pass over the cubes, because the alternative
    is every caller reimplementing "what counts as a common layer" beside the
    real definition and the two drifting apart.

    Returns
    -------
    dict
        ``common``     - layers every cube has, sorted (what a mosaic would
                         merge by default);
        ``per_cube``   - ``{label: [layer, ...]}`` for each input;
        ``only_some``  - layers at least one cube has and at least one lacks,
                         i.e. exactly what would be dropped with a note;
        ``crs``        - ``{label: "EPSG:..."}``, each cube's own projection
                         (None where a cube declares none);
        ``crs_counts`` - ``{crs: how many cubes are in it}``, most common first,
                         for offering a target projection.
    """
    opened = _open_inputs(cubes)
    per_cube, crs, to_close = {}, {}, []
    try:
        for (label, ds), original in zip(opened, list(cubes)):
            per_cube[label] = sorted(_raster_layers(ds))
            try:
                crs[label] = _cube_crs(ds, label)
            except Exception:
                crs[label] = None
            if isinstance(original, (str, os.PathLike)):
                to_close.append(ds)
        sets = [set(v) for v in per_cube.values()]
        common = sorted(set.intersection(*sets))
        only_some = sorted(set.union(*sets) - set(common))
        counts = {}
        for value in crs.values():
            if value:
                counts[value] = counts.get(value, 0) + 1
        crs_counts = dict(
            sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        )
    finally:
        for ds in to_close:
            try:
                ds.close()
            except Exception:
                pass
    return {
        "common": common,
        "per_cube": per_cube,
        "only_some": only_some,
        "crs": crs,
        "crs_counts": crs_counts,
    }


def mosaic_cubes(
    cubes,
    overlap="first",
    time_join="outer",
    band_join="inner",
    layers=None,
    crs=None,
    resolution=None,
    resampling="nearest",
    on_grid_mismatch="raise",
    source_map=False,
    strict=True,
    report=True,
    output=None,
    compress=False,
    q=False,
):
    """Merge several data cubes into one covering their combined extent.

    Parameters
    ----------
    cubes : list
        Two or more cubes: file paths (``.nc`` / ``.zarr``, opened with
        :func:`open_cube`), ``xarray.Dataset`` or ``xarray.DataArray``. **Order
        is priority** for ``overlap="first"``.
    overlap : {"first", "mean", "median", "min", "max"}
        What to do where more than one cube holds a valid pixel. ``"first"``
        (default) takes the value from the earliest cube in the list that has
        one - the only policy that never produces a number no input contained.
        The others reduce across the cubes that cover the pixel. Cubes do NOT
        have to overlap: where none of them reaches, the mosaic is NaN.
    time_join : {"outer", "inner"}
        ``"outer"`` (default) keeps every timestamp any cube has, so a cube
        contributes NaN over its own area on dates it lacks. ``"inner"`` keeps
        only timestamps every cube has. Ignored for layers without a time
        dimension (temporal composites).
    band_join : {"inner", "union"}
        ``"inner"`` (default) keeps the bands common to all cubes; ``"union"``
        keeps every band, NaN where a cube lacks it.
    layers : list of str, optional
        Which raster variables to merge (e.g. ``["Time_Series"]``). Default:
        every variable present in all inputs.
    crs : str, optional
        Target CRS. Default: the CRS of the inputs covering the largest area.
        A mosaic has one grid, so one CRS; every input's own CRS is recorded in
        ``attrs["mosaic_source_crs"]``.
    resolution : float, optional
        Target pixel size in CRS units. Default: the finest among the inputs.
    resampling : str
        rasterio resampling method used only when a cube has to be warped
        (default ``"nearest"``, which preserves values but shifts them by up to
        half a pixel).
    on_grid_mismatch : {"raise", "resample"}
        What to do with a cube that is not already on the mosaic grid.
        ``"raise"`` (default) refuses, because resampling changes pixel values
        and doing it silently is not something a merge should decide.
    source_map : bool
        Add ``mosaic_source`` (which cube each pixel came from, 1-based, 0 =
        none) and ``mosaic_n_sources`` (how many covered it) rasters.
    strict : bool
        Refuse inputs that disagree on ``mission`` or ``cloud_status``.
    report : bool
        Measure and print how much the overlapping cubes disagree.
    output : str, optional
        Write the result (``.nc`` or ``.zarr``); the cube is streamed, so the
        peak memory stays a few chunks. The mosaic is returned either way.
    compress, q
        Passed to :func:`export_stac` / silence the progress messages.

    Returns
    -------
    xarray.Dataset
        The merged cube, lazy unless an input had to be reprojected.

    Notes
    -----
    Per-scene coordinates (``cloud_percentage``, ``scene_coverage``) are dropped
    - they are measured against one cube's own AOI and cannot describe a merged
    extent. See :func:`_drop_time_coords`.
    """
    if overlap not in _OVERLAP_POLICIES:
        raise ValueError(
            f"overlap={overlap!r} is not one of {_OVERLAP_POLICIES}."
        )
    if time_join not in ("outer", "inner"):
        raise ValueError(f'time_join must be "outer" or "inner", got {time_join!r}.')
    if band_join not in ("inner", "union"):
        raise ValueError(f'band_join must be "inner" or "union", got {band_join!r}.')
    if on_grid_mismatch not in ("raise", "resample"):
        raise ValueError(
            f'on_grid_mismatch must be "raise" or "resample", got '
            f"{on_grid_mismatch!r}."
        )

    # Writing the mosaic onto one of its own inputs would read and overwrite the
    # same store: for Zarr that silently NaNs the result mid-write, and for
    # NetCDF the file is removed before the lazy read that still needs it.
    if output is not None:
        out_real = os.path.realpath(str(output))
        for c in cubes:
            if isinstance(c, (str, os.PathLike)) and os.path.realpath(str(c)) == out_real:
                raise ValueError(
                    f"output={output!r} is also one of the inputs. The mosaic is "
                    "read lazily from its inputs, so writing onto one of them "
                    "would destroy the data mid-write. Write to a new path."
                )

    opened = _open_inputs(cubes)
    infos = []
    for label, ds in opened:
        infos.append({
            "label": label,
            "ds": ds,
            "attrs": _cube_attrs(ds),
            "crs": _cube_crs(ds, label),
            "grid": _grid_of(ds, label),
        })

    divergences = _check_compatibility(infos, strict, q)

    # ---- which layers -------------------------------------------------------
    per_cube = [set(_raster_layers(i["ds"])) for i in infos]
    common = set.intersection(*per_cube)
    if layers is not None:
        missing = [name for name in layers if name not in common]
        if missing:
            raise ValueError(
                f"layers={layers} names variable(s) not present in every cube: "
                f"{', '.join(missing)}. Common layers: "
                f"{', '.join(sorted(common)) or '(none)'}."
            )
        chosen = list(layers)
    else:
        chosen = sorted(common)
    if not chosen:
        raise ValueError(
            "The cubes share no raster variable to merge. They hold: "
            + " | ".join(
                f"{i['label']}: {', '.join(sorted(s)) or '(none)'}"
                for i, s in zip(infos, per_cube)
            )
        )
    dropped_layers = sorted(set.union(*per_cube) - set(chosen))
    if dropped_layers and not q:
        print(
            "Note: " + ", ".join(dropped_layers) + " is not present in every "
            "cube, so it is not part of the mosaic.",
            flush=True,
        )

    # ---- target grid --------------------------------------------------------
    target_crs, crs_reason = _choose_target_crs(infos, crs, q=q)
    res_x, res_y = _target_resolution(infos, target_crs, resolution)
    xs, ys = _union_grid(infos, target_crs, res_x, res_y, q=q)

    # ---- per-scene coords, band and time axes -------------------------------
    dropped_coords = set()
    for info in infos:
        info["ds"] = _drop_time_coords(info["ds"], info["label"], dropped_coords)
        # One time precision across the inputs. A freshly built cube carries
        # datetime64[us] while the same cube read back from NetCDF is [ns], and
        # timestamps of different units hash differently - so the set
        # intersections that drive time_join="inner" and the per-pair sample
        # date would find nothing in common between two cubes that hold exactly
        # the same dates. Day-floored timestamps are exact in both units, so
        # this normalisation moves no date.
        if "time" in info["ds"].dims:
            info["ds"] = info["ds"].assign_coords(
                time=info["ds"]["time"].values.astype("datetime64[ns]")
            )
    if dropped_coords and not q:
        print(
            "Note: the per-scene coordinate(s) "
            + ", ".join(f"'{c}'" for c in sorted(dropped_coords))
            + " were dropped. Each was measured against a single cube's own "
            "area, so no value of it describes the merged extent, and there is "
            "nothing here to recompute them from.",
            flush=True,
        )

    bands = None
    if all("band" in i["ds"].dims for i in infos):
        bands = _join_values(
            [[str(v) for v in i["ds"]["band"].values] for i in infos],
            "union" if band_join == "union" else "inner",
            "bands",
        )

    times = None
    if any("time" in i["ds"].dims for i in infos):
        with_time = [i for i in infos if "time" in i["ds"].dims]
        if len(with_time) != len(infos) and not q:
            print(
                "Note: "
                + ", ".join(i["label"] for i in infos if "time" not in i["ds"].dims)
                + " has no time dimension, so only the layers without one can "
                "be merged with it.",
                flush=True,
            )
        stamps = [list(np.asarray(i["ds"]["time"].values)) for i in with_time]
        if time_join == "union" or time_join == "outer":
            times = np.unique(np.concatenate([np.asarray(s) for s in stamps]))
        else:
            keep = set(stamps[0])
            for s in stamps[1:]:
                keep &= set(s)
            if not keep:
                raise ValueError(
                    "The cubes share no acquisition timestamp, so "
                    'time_join="inner" keeps nothing. Their ranges are: '
                    + " | ".join(
                        f"{i['label']}: "
                        f"{np.datetime_as_string(np.min(s), unit='D')}..."
                        f"{np.datetime_as_string(np.max(s), unit='D')} "
                        f"({len(s)} dates)"
                        for i, s in zip(with_time, stamps)
                    )
                    + '. Use time_join="outer" to keep every date instead.'
                )
            times = np.sort(np.array(sorted(keep)))
        if not q:
            print(
                f"Time axis: {len(times)} date(s) ({time_join} join of "
                + ", ".join(f"{len(s)}" for s in stamps)
                + ").",
                flush=True,
            )

    # ---- place every cube on the grid ---------------------------------------
    placed = []
    with _quiet_placement_warnings():
        for info in infos:
            avail = [name for name in chosen if name in info["ds"].data_vars]
            arrs = _place(
                info["ds"], info, avail, xs, ys, res_x, res_y, target_crs,
                on_grid_mismatch, resampling, q,
            )
            # Band and time are reindexed AFTER the spatial placement so a cube
            # that had to be warped is warped once, on the dates it holds.
            for name, da in list(arrs.items()):
                if bands is not None and "band" in da.dims:
                    da = da.reindex(band=bands)
                if times is not None and "time" in da.dims:
                    da = da.reindex(time=times)
                # Last, so every cube - aligned or reprojected, whatever its
                # stored chunking - reaches the combine on one shared block grid.
                if getattr(da.data, "chunks", None) is not None:
                    da = da.chunk(_common_chunks(da))
                arrs[name] = da
            placed.append(arrs)

    # ---- report before merging ----------------------------------------------
    if report:
        _overlap_report(placed, infos, chosen[0], q=q)

    # ---- merge --------------------------------------------------------------
    merged, layer_dims = {}, {}
    for name in chosen:
        arrays = [p[name] for p in placed if name in p]
        if len(arrays) == 1:
            out = arrays[0]
        else:
            out = _combine(arrays, overlap)
        out.name = name
        # `where` and the reductions drop the variable's own attributes; the
        # source cube's are still the right description of what the values are.
        out.attrs = dict(infos[0]["ds"][name].attrs) if name in infos[0]["ds"] else {}
        merged[name] = out
        layer_dims[name] = tuple(out.dims)

    if source_map:
        ref = chosen[0]
        arrays = [p[ref] for p in placed if ref in p]
        src, n_src = _source_layers(arrays, overlap)
        src.attrs["description"] = (
            "1-based index into attrs['mosaic_sources'] of the first cube "
            "holding a valid pixel here; 0 = no cube covers it."
        )
        n_src.attrs["description"] = (
            "How many input cubes hold a valid pixel here."
        )
        merged["mosaic_source"] = src
        merged["mosaic_n_sources"] = n_src

    ds_out = xr.Dataset(merged)
    ds_out.attrs = _mosaic_attrs(
        infos, target_crs, crs_reason, xs, ys, res_x, res_y, divergences,
        {
            "overlap": overlap,
            "time_join": time_join,
            "band_join": band_join,
            "resampling": resampling if any(
                "resampled" in i.get("placement", "") for i in infos
            ) else "none (no cube needed resampling)",
        },
        layer_dims,
    )
    ds_out = ds_out.rio.write_crs(target_crs)
    # write_crs STRIPS attrs["crs"] (it treats a CRS attribute as ambiguous once
    # the grid-mapping variable carries the authoritative one), so it has to be
    # put back afterwards. Every stac2cube cube carries it - get_stac_parameters
    # reads it to pin a cube's grid on update, and export_stac's own
    # `crs or stac.crs` fallback needs it - so a mosaic without it would be the
    # one cube in the package that cannot be updated or re-exported on its own.
    ds_out.attrs["crs"] = target_crs

    nbytes = sum(int(v.nbytes) for v in ds_out.data_vars.values())
    from .main import _human_size

    ds_out.attrs["estimated_size"] = _human_size(nbytes)
    if not q:
        print(f"Mosaic: {_human_size(nbytes)} logical size.", flush=True)

    # Copied onto every merged layer as well as the Dataset, because that is
    # what a stac2cube cube looks like on disk and what its readers expect: a
    # cube exported from a DataArray keeps its metadata on the VARIABLE (the
    # file's global attrs are empty), so a mosaic that stored them only at
    # Dataset level would come back from a round-trip through export_stac
    # looking metadata-free to anything reading the variable. See _cube_attrs.
    for name in chosen:
        ds_out[name].attrs.update(
            {k: v for k, v in ds_out.attrs.items() if k != "grid_mapping"}
        )

    if output is not None:
        export_stac(
            ds_out, output, crs=target_crs,
            transform=ds_out.attrs["transform"], compress=compress, q=q,
        )
    return ds_out
