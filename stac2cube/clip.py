from .vector_refiner import polygon_2_gdf


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

    # Preserve original transform before clip (clip changes extent)
    transform = stac.transform

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

    return stac
