from .vector_refiner import polygon_2_gdf
import re
import math
import numpy as np
import xarray as xr


def _aoi_mask_from_geometries(stac, geometries):
    """Rasterize AOI geometries onto stac's grid. Returns (y, x) bool DataArray,
    True = inside the AOI."""
    from rasterio.features import geometry_mask

    transform = stac.rio.transform()
    out_shape = (stac.sizes["y"], stac.sizes["x"])
    mask = geometry_mask(
        [getattr(g, "__geo_interface__", g) for g in geometries],
        out_shape=out_shape,
        transform=transform,
        invert=True,        # True INSIDE the geometries
        all_touched=False,  # match rio.clip's pixel selection (center-in), so the
                            # boundary pixels rio.clip drops aren't counted as cloud
    )
    return xr.DataArray(
        mask, dims=("y", "x"), coords={"y": stac["y"], "x": stac["x"]}
    )


def compute_cloud_percentage(
    stac, aoi_mask=None, cloud_mask=None, lazy=False, imaged_mask=None
):
    """
    Per-time cloud percentage computed against the AOI footprint.

    Semantics:
      * AOI footprint = imaged land, intersected with ``aoi_mask`` if given.
      * Genuine-missing pixels -- never imaged inside the AOI -- are treated as
        no-data and excluded from BOTH numerator and denominator.
      * Numerator(t) = observable AOI pixels that are clouds at time t.

    Two ways to count clouds:
      * ``cloud_mask=None`` (default, masked cubes): clouds are the NaN pixels
        left behind by masking. Counted per band, denominator scaled by nbands.
      * ``cloud_mask`` given (keep-clouds AND masked cubes in the build/update
        path): the per-pixel cloud boolean (time, y, x) is supplied directly and
        aligned to the (possibly clipped) cube grid by label.

    Two ways to tell an IMAGED pixel from a no-data one:
      * ``imaged_mask=None`` (cubes read back from disk): a pixel counts as
        imaged when it holds data (``~isnull``) or is flagged cloud. This reads
        the cube's pixels - every band - which on a lazy build means
        downloading the whole cube just to produce this one number.
      * ``imaged_mask`` given (build/update path, needs ``cloud_mask`` too): the
        per-pixel "was imaged" boolean from the scene-classification band, i.e.
        the SAME SCL/QA read the cloud boolean comes from. No spectral band is
        touched, so a lazy build stays lazy. Measured on a 2711x3129 AOI,
        4 bands + 2 indices, element84: 12 scenes 38.0 s -> 3.1 s (peak RSS
        1.7 GB -> 0.4 GB), 36 scenes 88.3 s -> 7.4 s (2.3 GB -> 0.7 GB), with
        identical percentages.

        It is also the more reliable signal. element84 and terrabyte declare
        nodata=0, so an across-track swath gap arrives as 0, NOT NaN, and
        ``isnull`` cannot see it: those unimaged pixels then sit in the
        denominator and dilute the percentage. Measured on a bbox straddling a
        real swath edge (scene coverage 0.37): 31 % via ``isnull`` vs 84 % here.
        The isnull path only got it right when a normalized-difference index
        happened to be requested, because 0/0 turns the gap into NaN - i.e. the
        old percentage depended on the band list. See
        :func:`compute_scene_coverage_from_imaged` for the same reasoning.

    The observable footprint is "imaged at least once" where a pixel counts as
    imaged on a date if it holds data OR is flagged cloud by ``cloud_mask``. The
    cloud term is what makes the metric correct for MASKED cubes and independent
    of the date range: a masked cloud pixel is NaN, so without it a pixel cloudy
    on *every* date in the cube would be indistinguishable from no-data and be
    dropped from the usable-land denominator, undercounting cloud% and making it
    shift as dates are added/removed. Counting it via the cloud boolean keeps
    persistently-cloudy land in the denominator. For keep-clouds cubes cloud
    pixels keep their values, so the cloud term changes nothing there.

    For single-time cubes cloud and missing cannot be separated temporally, so
    all in-AOI NaN are counted as cloud (missing pixels assumed negligible).
    This caveat applies to the ``imaged_mask=None`` path only: the class band
    labels no-data per pixel per date, so one timestep is enough there.

    ``lazy=True`` returns the percentage without the final ``.compute()`` so the
    caller can materialize it together with another SCL/QA-derived reduction
    (e.g. scene coverage) in a single ``dask.compute`` - the shared SCL read is
    then done once instead of twice. Default False keeps the eager behaviour.

    Returns an int DataArray indexed by ``time`` (or None if there is no time dim).
    """
    if "time" not in stac.dims:
        return None

    reduce_dims = [d for d in ("band", "y", "x") if d in stac.dims]
    nbands = int(stac.sizes.get("band", 1))

    # Align the cloud boolean to the cube grid up front - it is also needed to
    # build an honest footprint below. Clip may have dropped rows/cols; time is
    # 1:1 in order, so re-stamp the cube's (floored) time.
    if cloud_mask is not None:
        cm = cloud_mask.sel(y=stac["y"], x=stac["x"])
        cm = cm.assign_coords(time=stac["time"]).astype(bool)
    else:
        cm = None

    # ---- SCL/QA fast path: everything from the classification band ----------
    # The per-date observability comes from imaged_mask, so the cube's own
    # pixels are never read. Requires the cloud boolean too (the NaN-counting
    # fallback below has no per-date cloud signal to pair with an imaged mask).
    if imaged_mask is not None and cm is not None:
        im = imaged_mask.sel(y=stac["y"], x=stac["x"]).astype(bool)
        im = im.assign_coords(time=stac["time"])
        if aoi_mask is not None:
            im = im & aoi_mask.astype(bool)
        # observed(t) = AOI pixels imaged on THIS date. Genuine no-data (never
        # imaged, clipped away, or a swath gap on this date) is excluded from
        # numerator and denominator alike, exactly as in the isnull path - just
        # measured on the class band instead.
        #
        # No `footprint = im.any("time")` term: intersecting it with im(t) is a
        # no-op, since a pixel imaged on date t is by definition imaged at least
        # once (the same identity holds for the isnull path's
        # `footprint & ~missing`). Leaving it out is not just tidier - it makes
        # every timestep independent, so dask streams one scene at a time
        # instead of having to hold the whole SCL series in memory until the
        # cross-time reduction completes.
        observed = im                             # (time, y, x)
        denom_t = observed.sum(dim=("y", "x"))
        nan_in = (cm & observed).sum(dim=("y", "x"))
        frac = xr.where(denom_t > 0, (nan_in / denom_t) * 100.0, 0.0)
        pct = np.floor(frac + 0.5).astype("int16")
        if not lazy and getattr(pct, "chunks", None) is not None:
            pct = pct.compute()
        return pct

    isnull = stac.isnull()

    if stac.sizes["time"] > 1:
        # Imaged = holds data (~isnull) OR is flagged cloud. Cloud-flagged land
        # is imaged land that merely happens to be NaN in a masked cube, so
        # counting it keeps persistently-cloudy land in the footprint instead of
        # mistaking it for no-data. Genuine no-data (never imaged, incl.
        # clipped-away pixels) stays dropped.
        imaged = (~isnull).any(dim="band") if "band" in isnull.dims else (~isnull)
        if cm is not None:
            imaged = imaged | cm
        footprint = imaged.any(dim="time")  # (y, x)
    else:
        footprint = xr.DataArray(
            np.ones((stac.sizes["y"], stac.sizes["x"]), dtype=bool),
            dims=("y", "x"),
            coords={"y": stac["y"], "x": stac["x"]},
        )

    if aoi_mask is not None:
        footprint = footprint & aoi_mask.astype(bool)

    if cm is not None:
        # Count clouds from the SCL/QA boolean rather than from NaN holes. This is
        # what keeps a *partial* scene honest: when a polygon straddles a swath /
        # tile edge, the satellite images only part of the AOI on a given date and
        # the rest is genuine NO_DATA. Those gap pixels are NOT clouds - SCL tags
        # them class 0 - so counting NaN would wrongly read a half-empty scene as
        # ~50% cloud.
        sp_dims = [d for d in ("y", "x") if d in cm.dims]

        # Per-date genuine no-data: NaN in the cube but NOT flagged as cloud by cm.
        # (In remove-clouds mode a cloud pixel is also NaN, but cm marks it, so cm
        # and `missing` stay disjoint.) Exclude it from BOTH numerator and
        # denominator so the percentage reflects only the pixels actually imaged
        # that date.
        null_t = isnull.any(dim="band") if "band" in isnull.dims else isnull
        missing = null_t & (~cm)              # (time, y, x): unimaged this date
        observed = footprint & (~missing)     # per-time observable AOI footprint
        denom_t = observed.sum(dim=sp_dims)    # per-time denominator
        nan_in = (cm & observed).sum(dim=sp_dims)  # per-time cloud pixel count

        frac = xr.where(denom_t > 0, (nan_in / denom_t) * 100.0, 0.0)
        pct = np.floor(frac + 0.5).astype("int16")
        if not lazy and getattr(pct, "chunks", None) is not None:
            pct = pct.compute()
        return pct
    else:
        denom = int(footprint.sum()) * nbands
        if denom == 0:
            return xr.zeros_like(stac["time"], dtype="int16")

        nan_in = (isnull & footprint).sum(dim=reduce_dims)  # per-time NaN count in AOI
    # Integer percent, rounded to NEAREST (round half up): removes the systematic
    # +1 bias of ceiling, where a sub-1% masked sliver (present in almost every
    # scene because the SCL mask drops a few scattered pixels) wrongly read 1%.
    # Now 0% means "< 0.5% of the observable AOI masked" and 100% means
    # ">= 99.5% masked". floor(frac + 0.5) is deterministic regardless of float
    # noise, so exact integer percentages are never nudged off.
    frac = (nan_in / denom) * 100.0
    pct = np.floor(frac + 0.5).astype("int16")
    if not lazy and getattr(pct, "chunks", None) is not None:
        pct = pct.compute()
    return pct


def compute_scene_coverage(stac, cloud_mask=None, compute=True, aoi_mask=None):
    """Per-time fraction (0..1) of the AOI a scene images.

    Across-track / swath-edge scenes cover only part of the AOI; the missing
    part loads as NaN. This returns, per timestep, how much of the AOI that
    scene actually holds:

      coverage(t) = (pixels imaged at t) / (AOI pixels)

    ``aoi_mask`` (rasterized AOI polygon, y/x bool) restricts both sides, so
    pixels outside a non-rectangular clip - NaN in every scene - are excluded
    from the denominator and a scene covering the whole polygon reads 1.0. Omit
    it for a plain bbox, where the AOI is the grid. See
    :func:`_aoi_pixel_count` for why the denominator is the AOI rather than the
    imaged union.

    "Imaged" is measured on a single representative band (``band=0``, a spectral
    band since the cube is built spectral-bands-first): a genuine swath/tile gap
    is NaN across every band, so one band identifies it, and using a spectral
    band avoids miscounting a pixel that is merely NaN in a derived index.

    ``cloud_mask`` (per-pixel cloud boolean, time/y/x, True = cloud) makes the
    measure cloud-aware, which is what keeps a cloud-MASKED cube honest: masked
    clouds are NaN too, so without this a fully-imaged but cloudy scene would
    read as low coverage and be wrongly dropped. A pixel is counted as IMAGED
    when it holds data OR is flagged cloud (observed, just masked out), exactly
    like the cloud term in :func:`compute_cloud_percentage`; only NaN-and-not-
    cloud (genuine swath/orbit no-data) reduces coverage. Omit it for keep-
    clouds / unmasked cubes, where NaN already means no-data.

    ``compute=True`` (default) materializes the result eagerly (one band read);
    ``compute=False`` returns a fully lazy dask-backed DataArray so a build can
    attach the coverage as a coordinate without reading any data until the coord
    is first used (export, a filter, or the GUI warning).

    Returns a float DataArray indexed by ``time`` (or None if no time dim).
    """
    if "time" not in stac.dims:
        return None

    da = stac
    if "band" in da.dims:
        da = da.isel(band=0)

    observed = da.notnull()               # (time, y, x): holds data
    if cloud_mask is not None:
        # Align the cloud boolean to the (possibly clipped) cube grid and time,
        # then count cloud-flagged pixels as imaged: they were observed, only
        # masked. Same treatment as compute_cloud_percentage's footprint term.
        cm = cloud_mask.sel(y=stac["y"], x=stac["x"]).astype(bool)
        cm = cm.assign_coords(time=stac["time"])
        observed = observed | cm
    if aoi_mask is not None:
        observed = observed & aoi_mask.astype(bool)

    per_time = observed.sum(dim=("y", "x")).astype("float64")
    denom = _aoi_pixel_count(stac, aoi_mask)

    if not compute:
        # Fully deferred: nothing is read until the coord is actually used. This
        # is what lets get_stac_layers attach scene_coverage to every cube
        # without forcing an eager band read on an otherwise-lazy build. Now
        # that the denominator is the AOI, this reduces per timestep only, so
        # the read also streams one scene at a time.
        return per_time / denom

    if float(denom) == 0.0:
        return xr.zeros_like(stac["time"], dtype="float64")

    cov = per_time / denom
    if getattr(cov, "chunks", None) is not None:
        cov = cov.compute()
    return cov


def _aoi_pixel_count(stac, aoi_mask=None):
    """Denominator for scene coverage: how many pixels the AOI has.

    ``aoi_mask`` given -> the rasterized polygon's pixel count, so a
    non-rectangular AOI is respected. None (a plain bbox, where the AOI IS the
    grid) -> the full grid.

    This is deliberately NOT "pixels imaged in any scene", which is what the
    coverage used to divide by. That denominator was defined by the cube's own
    contents, which made the metric say different things about the same scene
    depending on which dates sat next to it: on a range holding only partial
    scenes it collapsed onto the partial footprint and every scene read 1.00 -
    "covers the whole area" for a scene imaging a quarter of it (measured on a
    2-day range: 1.0000 against a true 0.2529). It also made the number
    incomparable between cubes and unstable under update and date filtering.

    Measured against the old denominator on real AOIs over a normal date range
    (aktal and test_clip, June 2024): the imaged union reached 100.0% of the AOI
    in both cases, so every per-scene value is unchanged. The two only part
    company where the old one was wrong.
    """
    if aoi_mask is not None:
        return aoi_mask.astype(bool).sum().astype("float64")
    return float(int(stac.sizes["y"]) * int(stac.sizes["x"]))


def compute_scene_coverage_from_imaged(stac, imaged_mask, aoi_mask=None):
    """Per-time AOI coverage fraction (0..1) from the SCL/QA "imaged" boolean.

    ``imaged_mask`` (time, y, x; True where the pixel was imaged) is the reliable
    per-scene signal derived from the scene-classification band: a swath / orbit
    gap is SCL class 0 (No Data), so it is caught even when the cube loads gaps
    as 0 rather than NaN (which a band's ``notnull`` would miss). It is also
    cloud-aware for free - clouds are imaged classes, so a cloudy-but-complete
    scene reads ~1.0 with no extra handling.

    Preferred over :func:`compute_scene_coverage` when cloud detection ran,
    because it needs no separate band read: ``imaged_mask`` comes from the SAME
    SCL/QA fetch as the cloud boolean, and returning a lazy result lets the
    caller compute it together with the cloud percentage in one pass (one read).

      coverage(t) = (imaged AOI pixels at t) / (AOI pixels)

    ``aoi_mask`` (rasterized AOI polygon, y/x bool) restricts both sides so a
    non-rectangular clip is respected: without it, in-bbox-but-outside-polygon
    pixels (imaged in the raw scene, clipped away in the cube) would count. For
    a plain bbox the AOI is the whole grid, so it may be omitted.

    The denominator is the AOI, not the imaged union - see
    :func:`_aoi_pixel_count` for the measurements behind that. It makes the
    value a property of the scene and the AOI alone, so it means the same thing
    in every cube, does not move when dates are added, filtered or updated, and
    can never report a partial scene as complete.

    Returns a LAZY float DataArray indexed by ``time`` (or None if no time dim).
    Nothing here reduces over time, so dask streams one scene at a time instead
    of having to hold the whole class-band series until a cross-time reduction
    completes.
    """
    if "time" not in stac.dims:
        return None

    # Align the raw (pre-clip) boolean to the cube grid and re-stamp the cube's
    # (floored) time positionally - same treatment as the cloud boolean in
    # compute_cloud_percentage; time is still 1:1 at this point in the build.
    im = imaged_mask.sel(y=stac["y"], x=stac["x"]).astype(bool)
    im = im.assign_coords(time=stac["time"])
    if aoi_mask is not None:
        im = im & aoi_mask.astype(bool)

    per_time = im.sum(dim=("y", "x")).astype("float64")
    return per_time / _aoi_pixel_count(stac, aoi_mask)


def drop_partial_scenes(stac, min_coverage=0.9, cloud_mask=None, q=False, coverage=None):
    """Remove partially-imaged (across-track / swath-edge) scenes.

    A scene is kept when it images at least ``min_coverage`` (fraction 0..1) of
    the AOI footprint, per :func:`compute_scene_coverage`. Survivors gain a
    ``scene_coverage`` (time,) coordinate (fraction 0..1) so the kept cube is
    self-describing. Filtering is positional (``isel``) so the data stays lazy.

    ``cloud_mask`` (per-pixel cloud boolean) makes the coverage cloud-aware so a
    cloudy-but-complete scene is NOT mistaken for a partial one on a masked cube
    (see :func:`compute_scene_coverage`).

    ``coverage`` (an already-computed per-time coverage DataArray) is reused as
    is when supplied, so the single band-0 read is not repeated by callers that
    have already measured it (e.g. get_stac_layers attaches scene_coverage to
    every cube before deciding whether to drop). Omit it to measure here.

    Raises if the threshold would drop every scene. Returns the cube unchanged
    when there is no time dimension.
    """
    if "time" not in stac.dims:
        return stac

    cov = coverage if coverage is not None else compute_scene_coverage(stac, cloud_mask=cloud_mask)
    if cov is None:
        return stac

    cov_vals = np.asarray(cov.values, dtype="float64")
    thr = float(min_coverage)
    keep = cov_vals >= thr
    n_total = int(keep.size)

    if not keep.any():
        _mx = float(np.nanmax(cov_vals)) if cov_vals.size else 0.0
        raise ValueError(
            f"Partial-scene removal (min_coverage={thr:.0%}) keeps no scenes: "
            f"the most complete scene covers only {_mx:.0%} of the AOI. Lower "
            "the coverage threshold (or widen the date range) and rerun."
        )

    if not keep.all():
        stac = stac.isel(time=np.flatnonzero(keep))
        cov_vals = cov_vals[keep]
        if not q:
            print(
                f"Partial-scene removal (min_coverage={thr:.0%}): kept "
                f"{int(keep.sum())}/{n_total} scenes.",
                flush=True,
            )

    stac = stac.assign_coords(scene_coverage=("time", cov_vals))
    return stac


def _reproject_grid(stac, src_crs, target_crs, resolution=None):
    """Destination grid (transform, width, height) for a whole-cube warp.

    Computed ONCE from the cube's own extent so every timestep lands on the
    same grid. Letting ``rio.reproject`` pick a grid per timestep would give
    each date a slightly different transform (it derives one from that slice's
    bounds), and the dates could then not be concatenated back into one cube.

    ``resolution=None`` keeps rasterio's default output pixel size, which is
    chosen so the warped raster has roughly the same pixel count as the source -
    i.e. approximately the input resolution, NOT exactly. Pass a number (CRS
    units, metres here) to pin it.

    The origin is snapped to a whole multiple of the pixel size, the same rule
    the build path uses (see get_data.grid_snap_unit). Without it
    ``calculate_default_transform`` puts the corner on wherever the warped
    bounds happen to land, so reprojecting two cubes of the same area into the
    same CRS gave two grids offset by a fraction of a pixel, and they no longer
    overlaid each other. Only whole multiples of the pixel size are used here -
    a reprojected cube has no provider grid left to align to, so there is
    nothing coarser worth snapping to.
    """
    from affine import Affine
    from rasterio.warp import calculate_default_transform

    left, bottom, right, top = stac.rio.bounds()
    kwargs = {}
    if resolution is not None:
        res = float(resolution)
        if res <= 0:
            raise ValueError(f"resolution={resolution!r} must be a positive number.")
        kwargs["resolution"] = (res, res)

    transform, width, height = calculate_default_transform(
        src_crs,
        target_crs,
        int(stac.sizes["x"]),
        int(stac.sizes["y"]),
        left,
        bottom,
        right,
        top,
        **kwargs,
    )

    # Grow outward to the snapped origin so nothing is cropped, then re-derive
    # the size from the snapped corner.
    px, py = abs(transform.a), abs(transform.e)
    if px > 0 and py > 0:
        x0 = math.floor(transform.c / px) * px
        y0 = math.ceil(transform.f / py) * py
        width += int(round((transform.c - x0) / px))
        height += int(round((y0 - transform.f) / py))
        transform = Affine(transform.a, transform.b, x0, transform.d, transform.e, y0)

    return transform, width, height


def _reproject_band_groups(stac, resampling):
    """Split the band axis into (bands, resampling) groups.

    Categorical layers (SCL, QA) are pinned to "nearest" for the same reason
    the loader pins them (see ``get_data._CATEGORICAL_BANDS``): interpolating
    class codes or bit-packed words produces meaningless values. Binary cloud
    masks (``cloud_mask_*``) are categorical too. Everything else uses the
    requested method.
    """
    from .get_data import _CATEGORICAL_BANDS

    if "band" not in stac.dims:
        return [(None, resampling)]

    names = [str(b) for b in stac["band"].values]
    categorical = [
        n for n in names
        if n.lower() in _CATEGORICAL_BANDS or n.lower().startswith("cloud_mask")
    ]
    if not categorical or resampling == "nearest":
        return [(None, resampling)]

    spectral = [n for n in names if n not in categorical]
    groups = []
    if spectral:
        groups.append((spectral, resampling))
    groups.append((categorical, "nearest"))
    return groups


def reproject_stac(stac, crs, resolution=None, resampling="nearest", q=False):
    """Warp a data cube into another CRS.

    :param stac: cube to reproject (``xarray.DataArray`` with y/x dims; a time
        and/or band dimension is optional).
    :param crs: target CRS - EPSG code, WKT or PROJ string. Validated with
        :func:`stac2cube.get_data.validate_target_crs`, so it must be a
        projected, metre-based CRS (the same rule the builder applies).
    :param resolution: output pixel size in target-CRS units (metres). ``None``
        (default) lets rasterio keep approximately the input resolution.
    :param resampling: rasterio resampling method. Default ``"nearest"``, which
        is the only method that invents no new values; categorical bands (SCL /
        QA / ``cloud_mask_*``) stay on "nearest" whatever is chosen here.
    :param q: quiet - suppress the progress lines.

    Time series are warped ONE DATE AT A TIME onto a single pre-computed grid
    (see :func:`_reproject_grid`), because rioxarray warps the last two
    dimensions and treats one leading dimension as bands - a
    ``(time, band, y, x)`` cube has one dimension too many. This path is EAGER:
    rasterio needs the pixels in memory, so a lazy cube is materialized here.

    Caveats worth stating rather than hiding:

    * Reprojection RESAMPLES: pixel values and the pixel grid both change, and
      the operation is not losslessly reversible. Reproject once, from the
      cube in its native projection, rather than chaining warps.
    * The warped cube is the axis-aligned bounding box of the rotated source
      footprint, so the corners outside that footprint become NaN.
    * ``cloud_percentage`` and ``scene_coverage`` coords are CARRIED OVER
      unchanged, not recomputed. They were measured on the source grid; the new
      corner NaNs are not clouds and not missing scene data, so recomputing them
      here would report wrong numbers.

    Returns the reprojected DataArray with ``attrs["crs"]``, ``["transform"]``
    and ``["pixel_resolution"]`` updated to the new grid, and the source
    projection recorded in ``attrs["reprojected_from"]``.
    """
    from .get_data import crs_attr_string, validate_target_crs
    from rasterio.enums import Resampling

    if not isinstance(stac, xr.DataArray):
        raise TypeError(
            f"reproject_stac expects a DataArray, got {type(stac).__name__}. "
            "Pass the cube's data variable (e.g. ds['Time_Series'])."
        )
    if "x" not in stac.dims or "y" not in stac.dims:
        raise ValueError(
            "reproject_stac needs a cube with 'y' and 'x' dimensions; got dims "
            f"{tuple(stac.dims)}."
        )

    target = validate_target_crs(crs)

    method = str(resampling).lower()
    if method not in Resampling.__members__:
        raise ValueError(
            f"Unknown resampling method '{resampling}'. Valid options: "
            f"{sorted(Resampling.__members__)}."
        )

    # Source CRS: the written spatial_ref first, then the cube's own attr (which
    # is what an exported/loaded stac2cube cube always carries).
    src_crs = stac.rio.crs
    if src_crs is None:
        crs_attr = stac.attrs.get("crs")
        if crs_attr is None:
            raise ValueError(
                "The cube declares no CRS (no spatial_ref coordinate and no "
                "attrs['crs']), so it cannot be reprojected. Its current "
                "projection has to be known before it can be warped."
            )
        stac = stac.rio.write_crs(crs_attr_string(crs_attr))
        src_crs = stac.rio.crs

    src_name = crs_attr_string(src_crs)
    if src_name == target and resolution is None:
        if not q:
            print(f"Cube is already in {target}; nothing to reproject.", flush=True)
        return stac

    # rioxarray's reproject fills everything outside the source footprint with
    # the nodata value. On a float cube that is NaN; an integer cube cannot hold
    # NaN, so it needs a declared fill - guessing one (0, 255, ...) would be
    # indistinguishable from a real class value in a mask cube.
    if np.issubdtype(stac.dtype, np.floating):
        nodata = np.nan
    else:
        nodata = stac.rio.nodata
        if nodata is None:
            nodata = stac.attrs.get("_FillValue", stac.attrs.get("nodata"))
        if nodata is None:
            raise ValueError(
                f"This cube is {stac.dtype} (integer) and declares no no-data "
                "value. Warping adds pixels outside the source footprint, and "
                "there is no honest value to fill them with: an integer cube "
                "cannot hold NaN, and picking 0 or 255 would be "
                "indistinguishable from a real class code. Convert it to float "
                "first, or set a no-data value on it."
            )

    dst_transform, width, height = _reproject_grid(
        stac, src_crs, target, resolution=resolution
    )
    shape = (int(height), int(width))
    groups = _reproject_band_groups(stac, method)

    attrs_ref = dict(getattr(stac, "attrs", {}) or {})
    band_order = (
        [str(b) for b in stac["band"].values] if "band" in stac.dims else None
    )

    def _warp(arr, how):
        out = arr.rio.write_nodata(nodata).rio.reproject(
            target,
            transform=dst_transform,
            shape=shape,
            resampling=Resampling[how],
            nodata=nodata,
        )
        return out

    def _warp_slice(arr):
        """Warp one (band, y, x) - or (y, x) - slice, per band group."""
        if len(groups) == 1:
            return _warp(arr, groups[0][1])
        parts = [_warp(arr.sel(band=names), how) for names, how in groups]
        return xr.concat(parts, dim="band").sel(band=band_order)

    if "time" in stac.dims:
        n = int(stac.sizes["time"])
        if not q:
            print(
                f"Reprojecting {n} date(s) from {src_name} to {target} "
                f"({method})...",
                flush=True,
            )
        frames = []
        for i in range(n):
            frames.append(_warp_slice(stac.isel(time=[i]).squeeze("time", drop=False)))
            if not q and (i + 1) % 10 == 0:
                print(f"  reprojected: {i + 1}/{n} dates", flush=True)
        # coords/compat are xarray's current defaults, pinned explicitly: the
        # per-date coords (cloud_percentage, scene_coverage, ...) are scalars on
        # each frame and have to be compared and stacked back into (time,)
        # coords. Pinning them also silences xarray's pending default-change
        # warning, whose future defaults would reject this combination outright.
        out = xr.concat(frames, dim="time", coords="different", compat="equals")
    else:
        if not q:
            print(f"Reprojecting from {src_name} to {target} ({method})...", flush=True)
        out = _warp_slice(stac)

    # Per-time coords (cloud_percentage, scene_coverage, tile, scene metadata)
    # ride along as scalar coords through the per-date warp and are rebuilt by
    # the concat, but re-attach anything that got lost so the cube stays
    # self-describing. Values are carried over, NOT recomputed - see the
    # docstring for why.
    #
    # A per-date coord whose value is the SAME on every date (e.g. scene_coverage
    # 1.0 throughout) is collapsed back to a SCALAR coord by concat's
    # coords="different" comparison, which silently makes the cube one-sided: it
    # no longer lines up with the time axis, and a later concat/update would trip
    # over the mismatch. Restore anything that is not a (time,) coord again.
    if "time" in stac.dims:
        for name, coord in stac.coords.items():
            if coord.dims != ("time",):
                continue
            existing = out.coords.get(name)
            if existing is None or existing.dims != ("time",):
                out = out.assign_coords({name: ("time", np.asarray(coord.values))})

    out.attrs.update(attrs_ref)
    # On a float cube NaN is the only honest no-data; a stale numeric nodata
    # attr makes a later rio.clip fill the outside-polygon area AND every NaN
    # hole with that value (see clip_stac).
    if np.issubdtype(out.dtype, np.floating):
        for _k in ("_FillValue", "missing_value", "fill_value", "nodata"):
            out.attrs.pop(_k, None)

    out.attrs["crs"] = target
    out.attrs["transform"] = dst_transform
    # The warp puts the cube on a new grid, so the inherited pixel_resolution
    # would be stale; overwrite it with the grid actually written.
    out.attrs["pixel_resolution"] = float(abs(dst_transform.a))
    out.attrs["reprojected_from"] = src_name
    out.attrs["reprojection_resampling"] = method

    if not q:
        print(
            f"Reprojected grid: {shape[0]} x {shape[1]} pixels at "
            f"{abs(dst_transform.a):g} m in {target}.",
            flush=True,
        )
    return out


_S2CLOUDLESS_STATUS = re.compile(r"^cloud_mask_\d+$")


def _clouds_were_removed(stac):
    """True when this cube's cloud pixels are NaN, so NaN identifies them.

    Only then can a cloud percentage be recounted from the cube's own pixels.
    On a KEEP-CLOUDS cube (``cloud_status='clouds_detected'``) the clouds are
    still there as ordinary reflectance and nothing marks them, so counting NaN
    returns 0 for every scene - which is what the clip refresh below used to
    write over the real percentages (measured: a cube whose scenes were 100%
    cloudy came back as 0% across the board).

    ``externally_masked`` is deliberately excluded: something removed pixels,
    but not necessarily clouds (a shadow-only mask removes shadows), so NaN
    does not mean cloud there either.
    """
    status = str(stac.attrs.get("cloud_status") or "").strip().lower()
    return status in ("scl_masked", "scl_shadow_masked") or bool(
        _S2CLOUDLESS_STATUS.match(status)
    )


def clip_stac(stac, polygon, crs=None, bbox_crs="EPSG:4326"):
    """
    polygon:
      - polygon-like input supported by polygon_2_gdf
      - OR bbox as [minx, miny, maxx, maxy]
    crs:
      - target CRS for clipping (defaults to stac.crs)
    bbox_crs:
      - CRS of bbox coordinates (defaults to EPSG:4326)
    """
    # Decide target CRS. attrs["crs"] first, then the written spatial_ref: this
    # used to be a bare `stac.crs` attribute lookup, which raised
    # "AttributeError: 'DataArray' object has no attribute 'crs'" on any cube
    # that carries its projection only as the grid-mapping coordinate. That is
    # reachable - rioxarray's write_crs() DELETES attrs["crs"] (verified), and
    # get_stac_layers' stats branch calls it after re-applying the cube attrs,
    # so a composite cube returned in memory has no attrs["crs"] at all.
    if crs is None:
        crs = stac.attrs.get("crs")
    if crs is None:
        crs = getattr(getattr(stac, "rio", None), "crs", None)
    if crs is None:
        raise ValueError(
            "This cube declares no projection (no attrs['crs'] and no written "
            "spatial_ref coordinate), so it cannot be clipped - the polygon "
            "has nothing to be projected into. Pass crs=... explicitly."
        )

    # If bbox list/tuple -> build a GeoDataFrame; else use your existing polygon loader
    is_bbox = (
        isinstance(polygon, (list, tuple))
        and len(polygon) == 4
        and all(isinstance(v, (int, float)) for v in polygon)
    )

    if is_bbox:
        import geopandas as gpd
        from shapely.geometry import box

        minx, miny, maxx, maxy = polygon
        if minx >= maxx or miny >= maxy:
            raise ValueError(f"Invalid bbox (min must be < max): {polygon}")

        gdf = gpd.GeoDataFrame(
            geometry=[box(minx, miny, maxx, maxy)],
            crs=bbox_crs,
        )
    else:
        gdf = polygon_2_gdf(polygon)

    # Reproject polygon/bbox to data CRS and clip
    pproj = gdf.to_crs(crs)

    # rioxarray's clip runs `fillna(rio.nodata)` over the WHOLE cropped cube
    # whenever the cube carries a non-NaN nodata value (rioxarray
    # _clip_xarray). odc-stac stamps the raw DN fill `nodata: 0` on every
    # loaded band and that attr survives scaling and the band concat, so on a
    # scaled float cube the fill would turn the outside-polygon area AND every
    # existing NaN hole (masked clouds, swath gaps) into 0.0 - a valid
    # reflectance value, indistinguishable from data. (Whether the attr is
    # still present at clip time depends on the exact build path - a temporal
    # composite drops attrs and so clipped to NaN, a plain build kept nodata=0
    # and clipped to zeros.) On a float cube NaN is the only honest no-data,
    # so drop the stale nodata declarations before clipping. Integer cubes
    # (e.g. a uint8 mask cube), which cannot hold NaN, keep their declared
    # nodata. Encoding _FillValue is left alone: an encoded nodata already
    # makes rio.clip fill with NaN, and exports still need it.
    if np.issubdtype(stac.dtype, np.floating):
        stac = stac.copy(deep=False)  # attrs dict is copied; caller unaffected
        for _k in ("_FillValue", "missing_value", "fill_value", "nodata"):
            stac.attrs.pop(_k, None)

    stac = stac.rio.clip(pproj.geometry.values, crs=crs, drop=True)

    # Record the grid the cube is NOW on. This used to re-attach the transform
    # captured BEFORE the clip, so a clipped cube advertised an origin it did
    # not sit on - measured 21 px east and 27 px north out on a middle-half
    # clip. The coordinates and the CF GeoTransform written into the file were
    # always correct (rioxarray derives those from x/y), so a GIS never saw it;
    # what was wrong is the self-describing attribute, and it travelled into
    # every export and into the shadow / cloud tools, which read
    # attrs["transform"] and stamp it onto their own products.
    stac.attrs["crs"] = crs
    stac.attrs["transform"] = stac.rio.transform()

    # If this cube was cloud-MASKED (carries a cloud_percentage coord and its
    # clouds are NaN), the value is now stale: clipping changed the extent.
    # Recompute it against the clipped AOI so the percentage reflects clouds
    # *inside the clip only*. The rasterized AOI mask keeps pixels outside a
    # non-rectangular polygon from being counted as cloud. Verified against the
    # SCL mask on both a plain and an already-clipped cube (6% -> 9% and 11%,
    # matching the truth for the clipped area in each case).
    #
    # A KEEP-CLOUDS cube cannot be recounted this way and keeps its stored
    # values, which then describe the area the cube was BUILT over rather than
    # the clip - recorded in cloud_percentage_scope and stated once, because a
    # stale real measurement is still a measurement while the alternative here
    # was a fabricated 0% for every scene. Getting the true number back needs
    # the scene classification, which the cube does not carry; that is a
    # re-query, not a clip.
    if "cloud_percentage" in stac.coords:
        if _clouds_were_removed(stac):
            try:
                aoi_mask = _aoi_mask_from_geometries(stac, pproj.geometry.values)
                pct = compute_cloud_percentage(stac, aoi_mask=aoi_mask)
                if pct is not None:
                    stac = stac.assign_coords(
                        cloud_percentage=("time", np.asarray(pct.data))
                    )
                    stac.attrs.pop("cloud_percentage_scope", None)
            except Exception:
                # Never let a percentage refresh break the actual clip result.
                pass
        else:
            stac.attrs["cloud_percentage_scope"] = "pre_clip_aoi"
            print(
                "Note: this cube's clouds were detected but kept, so the cloud "
                "percentages cannot be recounted from its pixels. They are left "
                "as measured over the area the cube was built for, not the clip.",
                flush=True,
            )

    return stac
