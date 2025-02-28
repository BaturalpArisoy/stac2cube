from .main import get_stac_layers
from s2cloudless import S2PixelCloudDetector
import numpy as np
import xarray as xr
import sys
from .export_cfg import export_stac
import rioxarray as rio
import cv2
import os

import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def get_cloud_layers(polygon, daterange, output, cloud_layers=None, clip_raster=None,
                             threshold= None, masking= None):
    
    # Default to both layers if not provided.
    if cloud_layers is None:
        cloud_layers = ["cloud_prob", "cloud_mask"]

    if threshold:
        threshold = threshold
    else:
        threshold = 0.7
        
    # Variables for stac retrieval
    max_cc = 100
    mission = 's2_l1c'
    bands = ['coastal', 'blue', 'red', 'rededge1', 'nir',
             'nir08', 'nir09', 'cirrus', 'swir16', 'swir22']

    # Retrieve the lazy STAC DataArray.
    stac = get_stac_layers(mission=mission, polygon=polygon, daterange=daterange, bands=bands, max_cc=max_cc, clip_raster=clip_raster) #LAZY
    print(stac, flush=True)
    crs = stac.crs
    transform = stac.transform
    
    # Instantiate the cloud detector.
    average_over=4
    dilation_size=2
    cloud_detector = S2PixelCloudDetector(threshold=threshold, average_over=average_over,
                                          dilation_size=dilation_size, all_bands=False)

    # Prepare lists to collect cloud layer results.
    cloud_prob_results = []
    cloud_mask_results = []
    times = []  # To store time coordinate for each processed slice

    # Process each time slice lazily.
    total = len(stac.time)
    for i, t in enumerate(stac.time.values, start=1):
        # Select one time slice lazily and then compute it.
        img = stac.sel(time=t).compute()  # Compute only the current slice
        times.append(t)

        # Transpose to (y, x, band) for s2cloudless.
        img_transposed = img.transpose('y', 'x', 'band')
        # Convert to numpy array with a batch dimension: (1, y, x, band)
        img_np = img_transposed.to_numpy()[np.newaxis, ...]
        
        # Compute the cloud probability map only once.
        cp_3d = cloud_detector.get_cloud_probability_maps(img_np)
        cp = cp_3d[0]

        # Append the probability maps if requested.
        if "cloud_prob" in cloud_layers:
            cloud_prob_results.append(cp)

        # Generate the cloud mask using the provided helper function.
        if "cloud_mask" in cloud_layers:
            cm = cloud_detector.get_mask_from_prob(cp_3d, threshold=threshold)[0]
            cloud_mask_results.append(cm)
        
        
        print(f"Processed time slice: {i}/{total} (time: {t})", flush=True)
        
        # Drop temporary variables to help free memory.
        del img, img_transposed, img_np
        
    # Build DataArrays for each computed cloud layer.
    dataarrays = []
    band_names = []

    if "cloud_prob" in cloud_layers:
        cp_stack = np.stack(cloud_prob_results, axis=0)  # shape: (time, y, x)
        cp_da = xr.DataArray(cp_stack, dims=["time", "y", "x"],
                             coords={"time": times, "y": stac.y, "x": stac.x})
        cp_da = cp_da.expand_dims(dim={"band": ["cloud_prob"]})
        dataarrays.append(cp_da)
        band_names.append("cloud_prob")

    if "cloud_mask" in cloud_layers:
        cm_stack = np.stack(cloud_mask_results, axis=0)  # shape: (time, y, x)
        cm_da = xr.DataArray(cm_stack, dims=["time", "y", "x"],
                             coords={"time": times, "y": stac.y, "x": stac.x})
        cm_da = cm_da.expand_dims(dim={"band": ["cloud_mask"]})
        dataarrays.append(cm_da)
        band_names.append("cloud_mask")

    if dataarrays:
        # Concatenate along the 'band' dimension.
        cloud_only_stack = xr.concat(dataarrays, dim="band")
        # Reorder dimensions so that 'band' comes after 'time'
        cloud_only_stack = cloud_only_stack.transpose("time", "band", "y", "x")
        # Assign band names.
        cloud_only_stack = cloud_only_stack.assign_coords(band=band_names)
        cloud_only_stack.name = "Cloud_Stack"
        #cloud_only_stack.attrs['crs'] = crs
        #cloud_only_stack.attrs['transform'] = transform

        # Export the result.
        export_stac(cloud_only_stack, output, crs, transform)
        
        if masking:
            dirname, filename = os.path.split(masking)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_masked{ext}"
            output_mask = os.path.join(dirname, output_filename)

            mask_stac_clouds(masking, cloud_only_stack, output_mask)
        
    else:
        print("No cloud layers selected. Nothing to export.")


def mask_stac_clouds(stac, cloud, output):
    
    if isinstance(stac, (str, os.PathLike)):
        stac = xr.open_dataset(stac)
        stac = stac.Spectral_Temporal_Stack
    if isinstance(cloud, (str, os.PathLike)):
        cloud = xr.open_dataset(cloud)
        cloud = cloud.Cloud_Stack
    
    if isinstance(stac, xr.Dataset):
        stac = stac.Spectral_Temporal_Stack
    if isinstance(cloud, xr.Dataset):
        cloud = cloud.Cloud_Stack
        
    cloud_mask = cloud.sel(band='cloud_mask')
    masked_stac = stac.where(cloud_mask == 0)
    
    crs = stac.crs
    transform = stac.transform

    export_stac(masked_stac, output, crs, transform)