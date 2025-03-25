from .vector_refiner import polygon_2_gdf


def clip_stac(stac, polygon, crs):
    
    gdf = polygon_2_gdf(polygon)
    
    pproj = gdf.to_crs(crs)
    
    stac.rio.write_crs(crs, inplace = True) # delete later
    
    stac = stac.rio.clip(pproj.geometry.values, crs = crs)
    
    return stac