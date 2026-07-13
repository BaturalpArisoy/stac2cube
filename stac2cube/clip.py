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


def compute_cloud_percentage(stac, aoi_mask=None, cloud_mask=None):
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

    Returns an int DataArray indexed by ``time`` (or None if there is no time dim).
    """
    if "time" not in stac.dims:
        return None

    reduce_dims = [d for d in ("band", "y", "x") if d in stac.dims]
    nbands = int(stac.sizes.get("band", 1))
    isnull = stac.isnull()

    # Align the cloud boolean to the cube grid up front - it is also needed to
    # build an honest footprint below. Clip may have dropped rows/cols; time is
    # 1:1 in order, so re-stamp the cube's (floored) time.
    if cloud_mask is not None:
        cm = cloud_mask.sel(y=stac["y"], x=stac["x"])
        cm = cm.assign_coords(time=stac["time"]).astype(bool)
    else:
        cm = None

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
        if getattr(pct, "chunks", None) is not None:
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
    if getattr(pct, "chunks", None) is not None:
        pct = pct.compute()
    return pct


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
