from .vector_refiner import polygon_2_gdf
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


def compute_scene_coverage(stac, cloud_mask=None, compute=True):
    """Per-time fraction (0..1) of the observable AOI footprint a scene images.

    Across-track / swath-edge scenes cover only part of the AOI; the missing
    part loads as NaN. This returns, per timestep, how much of the AOI that
    scene actually holds:

      coverage(t) = (pixels imaged at t) / (pixels imaged in ANY scene)

    The denominator is the footprint - pixels imaged at least once - so pixels
    that are NaN in *every* scene (outside a non-rectangular clip) are excluded
    and a scene that fully covers the AOI reads ~1.0. This mirrors the footprint
    logic of :func:`compute_cloud_percentage`.

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

    footprint = observed.any(dim="time")  # (y, x): imaged at least once
    per_time = observed.sum(dim=("y", "x")).astype("float64")

    if not compute:
        # Fully deferred: keep the denominator lazy too, so on a dask-backed
        # cube nothing is read until the coord is actually used. This is what
        # lets get_stac_layers attach scene_coverage to every cube without
        # forcing an eager band read on an otherwise-lazy build. A degenerate
        # all-empty footprint (denom==0) yields NaN here instead of the eager
        # path's explicit zeros - acceptable, as such a cube carries no scene
        # to judge.
        return per_time / footprint.sum().astype("float64")

    denom = int(footprint.sum())
    if denom == 0:
        return xr.zeros_like(stac["time"], dtype="float64")

    cov = per_time / denom
    if getattr(cov, "chunks", None) is not None:
        cov = cov.compute()
    return cov


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

      coverage(t) = (imaged AOI pixels at t) / (imaged AOI pixels in ANY scene)

    ``aoi_mask`` (rasterized AOI polygon, y/x bool) intersects the footprint so a
    non-rectangular clip is respected: without it, in-bbox-but-outside-polygon
    pixels (imaged in the raw scene, clipped away in the cube) would inflate the
    denominator. For a plain bbox the AOI is the whole grid, so it may be omitted.

    Returns a LAZY float DataArray indexed by ``time`` (or None if no time dim).
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

    footprint = im.any(dim="time")                    # (y, x) imaged in any scene
    per_time = im.sum(dim=("y", "x")).astype("float64")
    return per_time / footprint.sum().astype("float64")


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
    # Decide target CRS
    crs = stac.crs if crs is None else crs

    # Preserve original transform before clip (clip changes extent).
    # A freshly built cube hasn't had attrs["transform"] set yet (main.py writes
    # it AFTER this clip call), so fall back to the live grid transform. A
    # loaded/exported cube already carries the attr, so the Editor/ARD clip path
    # behaves exactly as before.
    transform = stac.attrs.get("transform")
    if transform is None:
        transform = stac.rio.transform()

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

    # Store CRS + original transform back as attrs
    stac.attrs["crs"] = crs
    stac.attrs["transform"] = transform

    # If this cube was cloud-masked (carries a cloud_percentage coord), the value
    # is now stale: clipping changed the extent. Recompute it against the clipped
    # AOI so the percentage reflects clouds *inside the clip only*. The rasterized
    # AOI mask keeps pixels outside a non-rectangular polygon from being counted
    # as cloud.
    if "cloud_percentage" in stac.coords:
        try:
            aoi_mask = _aoi_mask_from_geometries(stac, pproj.geometry.values)
            pct = compute_cloud_percentage(stac, aoi_mask=aoi_mask)
            if pct is not None:
                stac = stac.assign_coords(
                    cloud_percentage=("time", np.asarray(pct.data))
                )
        except Exception:
            # Never let a percentage refresh break the actual clip result.
            pass

    return stac
