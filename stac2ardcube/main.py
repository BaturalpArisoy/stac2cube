from .get_data import get_stac
from .stac_processing import scale_factor
from .get_spectral_indices import calculate_spectral_index
from .export_cfg import export_stac
from .get_topo import calculate_topo
from .get_animation import generate_animation
from .clip import clip_stac
from .get_statistics import calculate_statistics

import xarray as xr
import rioxarray as rio

def get_stac_layers(
    mission,
    polygon,
    resolution = None,
    daterange = None,
    bands= None,
    max_cc= None,
    clip_raster= None,
    cloud_masking = None,
    indices = None ,
    output = None,
    aggregator = None,
    stats = None,
    topographic_features = None,
    animation = None):
    
    stac, baselines = get_stac(mission, polygon, resolution, daterange, bands, max_cc, cloud_masking)
    crs = stac.spatial_ref.crs_wkt
    transform = stac.rio.transform()
    
    # Cloud masking
#    if cloud_masking is True:
#        stac = cloud_mask(stac, mission)

   
    # Scale factor
    stac = scale_factor(stac, mission, baselines)
    #stac.rio.write_crs(crs, inplace=True)
    
    # Transform zeros to nan
    stac = stac.where(stac != 0)
    
    # Index calculation
    # Add code when only indices are asked without band selection
    if indices is not None:
        stac_indices = calculate_spectral_index(stac, mission, indices)

    # Add animation here
    if animation is True:
        generate_animation(stac)
    
    if mission == 'cop_dem':        
        dem = stac.isel(time=0).dem
        dem = dem.expand_dims(dim={'band': ['dem']})
        stac_topo_features = calculate_topo(dem, topographic_features)
        
    # Dataset -> DataArray
    if mission != 'cop_dem':  
        bands = list(stac.data_vars.keys())
        stac = xr.concat([stac[band] for band in bands], dim='band')
        stac = stac.assign_coords(band=bands)    

    # DataArray manipulation
    if indices is not None:
        stac = xr.concat([stac, stac_indices], dim='band')

    if mission == 'cop_dem':
        stac = xr.concat([dem, stac_topo_features], dim='band')
        stac = stac.rename('Topographic_Features')
    else:
        stac = stac.transpose('time', 'band', 'y', 'x')
        stac = stac.rename('Spectral_Temporal_Stack')

    # Calculate stats image (optional)
    if aggregator is not None:
        print(f"stac before {aggregator}:")
        print(stac)
        print("\n-------------------------------------")
        if aggregator == 'mean':
            stac = stac.mean(dim='time', skipna=True)
            
        elif aggregator == 'median':
            stac = stac.median(dim='time', skipna=True)
    
    # Clip netcdf as clip raster
    if clip_raster:
        stac = clip_stac(stac, polygon, crs) # delete write_crs in clip_stac
            
    # Finalizing
    stac['time'] = stac['time'].dt.strftime('%Y-%m-%d')

    if output is None:
        stac.attrs['crs'] = crs
        stac.attrs['transform'] = transform
        print(stac)
        return stac # returns lazy
    else:
        print(stac)
        if mission != 'cop_dem':
            if aggregator is None:
                print(stac.time.values)
                stac = calculate_statistics(stac, stats)
        print(stac.band.values)
        return export_stac(stac, output, crs, transform)
