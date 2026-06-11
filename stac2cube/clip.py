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


def compute_cloud_percentage(stac, aoi_mask=None):
    """
    Per-time cloud percentage computed against the AOI footprint.

    Semantics:
      * AOI footprint = pixels inside ``aoi_mask`` (whole extent if None).
      * Genuine-missing pixels -- NaN in *every* time step inside the AOI -- are
        treated as no-data and excluded from BOTH numerator and denominator.
      * Numerator(t) = observable AOI pixels that are NaN at time t (clouds).

    For single-time cubes cloud and missing cannot be separated temporally, so
    all in-AOI NaN are counted as cloud (missing pixels assumed negligible).

    Returns an int DataArray indexed by ``time`` (or None if there is no time dim).
    """
    if "time" not in stac.dims:
        return None

    reduce_dims = [d for d in ("band", "y", "x") if d in stac.dims]
    nbands = int(stac.sizes.get("band", 1))
    isnull = stac.isnull()

    if stac.sizes["time"] > 1:
        # Observable = valid at least once -> drops pixels missing in every scene
        # (this also drops clipped-away pixels, which are NaN in every scene).
        non_reduce = [d for d in ("time", "band") if d in stac.dims]
        footprint = (~isnull).any(dim=non_reduce)  # (y, x)
    else:
        footprint = xr.DataArray(
            np.ones((stac.sizes["y"], stac.sizes["x"]), dtype=bool),
            dims=("y", "x"),
            coords={"y": stac["y"], "x": stac["x"]},
        )

    if aoi_mask is not None:
        footprint = footprint & aoi_mask.astype(bool)

    denom = int(footprint.sum()) * nbands
    if denom == 0:
        return xr.zeros_like(stac["time"], dtype="int16")

    nan_in = (isnull & footprint).sum(dim=reduce_dims)  # per-time NaN count in AOI
    # Integer percent, rounded UP: any cloud at all -> at least 1%, so that 0%
    # reliably means a cloud-free scene when filtering. The 1e-9 guard keeps an
    # exact integer percentage from being bumped up by floating-point noise.
    frac = (nan_in / denom) * 100.0
    pct = np.ceil(frac - 1e-9).astype("int16")
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
