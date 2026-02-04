from .vector_refiner import polygon_2_gdf


def clip_stac(stac, polygon, crs=None):
    
    gdf = polygon_2_gdf(polygon)
    crs = stac.crs if crs is None else crs
    transform = stac.transform
    pproj = gdf.to_crs(crs)
    #stac.rio.write_crs(crs, inplace = True) # delete later
    
    #transform = stac.transform if transform is None else transform
    
    stac = stac.rio.clip(pproj.geometry.values, crs = crs, drop=True)

    stac.attrs["crs"] = crs
    stac.attrs["transform"] = transform
    
    return stac