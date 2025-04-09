from .main import get_stac_layers
from .get_update import get_stac_parameters
from .get_update import find_missing_times
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


def get_cloud_layers(polygon=None, daterange=None, output=None, threshold=None, clip_raster=None, masking=None, update=None):
    """
    Retrieves Sentinel-2 imagery using STAC, computes cloud probabilities for each time slice,
    and (optionally) computes cloud masks based on the provided threshold(s). If no threshold is provided,
    the function returns only the cloud probability layer. Otherwise, it returns a combined DataArray
    that includes the cloud probability (as "cloud_prob") and the mask(s) for each provided threshold,
    with mask band names like "cloud_mask_70" (for threshold=70).

    Parameters:
        polygon: Geometry for data retrieval.
        daterange: Date range for filtering imagery.
        output: Output path for the exported product.
        threshold (None, float, or list): If provided, threshold(s) in the 0-100 range used to generate masks.
            If not provided, only cloud probability is returned.
        clip_raster: Optional clipping geometry.
        masking: Optional parameter for additional masking.

    Returns:
        Exported product (via export_stac).
    """

    if masking:
        stac_parameters = get_stac_parameters(masking)
        polygon = stac_parameters["polygon"]
        daterange = stac_parameters["daterange"]
    if update:
        stac_parameters = get_stac_parameters(update)
        polygon = stac_parameters["polygon"]
        output = update
    else:
        if not daterange:
            raise ValueError("Error: Please select a daterange.")
        if not polygon:
            raise ValueError("Error: Please select a polygon or bbox list with geographic coordinates.")

    # --- STAC Retrieval ---
    # Default maximum cloud cover and mission configuration.
    max_cc = 100
    mission = 'sentinel_2_l1c'
    bands = ['coastal', 'blue', 'red', 'rededge1', 'nir',
             'nir08', 'nir09', 'cirrus', 'swir16', 'swir22']

    # Retrieve the lazy STAC DataArray.
    stac = get_stac_layers(mission=mission, polygon=polygon, daterange=daterange,
                           bands=bands, max_cc=max_cc, clip_raster=clip_raster)
    crs = stac.crs
    transform = stac.transform
    bbox = stac.bbox

    if update:
        stac_existing = xr.open_dataset(update)
        stac_existing = stac_existing.Cloud_Stack
        stac, missing_times = find_missing_times(stac_existing, stac)
        if not missing_times:
            raise ValueError("The probability map is up to date. Nothing to update!")

    # --- Cloud Probability Calculation ---
    # Set the parameters for the cloud detector.
    # Default threshold (0.7) for computing cloud probability.
    average_over = 4
    dilation_size = 2
    default_threshold = 0.7
    cloud_detector = S2PixelCloudDetector(threshold=default_threshold, average_over=average_over,
                                          dilation_size=dilation_size, all_bands=False)

    cloud_prob_results = []
    times = []  # To store the time coordinate for each processed slice
    total = len(stac.time)

    for i, t in enumerate(stac.time.values, start=1):
        # Retrieve and compute the current time slice.
        img = stac.sel(time=t).compute()
        times.append(t)

        # Transpose to (y, x, band) for s2cloudless and add a batch dimension.
        img_transposed = img.transpose('y', 'x', 'band')
        img_np = img_transposed.to_numpy()[np.newaxis, ...]

        # Compute the cloud probability maps (3D with shape: (batch, y, x)).
        cp_3d = cloud_detector.get_cloud_probability_maps(img_np)
        cp = cp_3d[0]
        cloud_prob_results.append(cp)

        print(f"Processed time slice: {i}/{total} (time: {t})", flush=True)
        del img, img_transposed, img_np

    # Assemble the cloud probability DataArray.
    cp_stack = np.stack(cloud_prob_results, axis=0)  # shape: (time, y, x)
    cp_da = xr.DataArray(cp_stack, dims=["time", "y", "x"],
                         coords={"time": times, "y": stac.y, "x": stac.x})
    cp_da = cp_da.expand_dims(dim={"band": ["cloud_prob"]})

    cp_da.name = "Cloud_Stack"

    
    def update_prob_maps(stac_existing, cloud_only_stack):
        stac_existing = stac_existing.sel(band="cloud_prob")
        stac_existing = stac_existing.expand_dims(dim={"band": 1})
        cloud_only_stack = xr.concat([stac_existing, cloud_only_stack], dim="time")
        cloud_only_stack = cloud_only_stack.sortby("time")
        return cloud_only_stack

    # --- Determine Output Based on 'threshold' Parameter ---
    # If no threshold(s) are provided, return only the probability layer.
    if threshold is None:
        # Scale cloud probability (assumed to be [0,1]) to 0-100 and convert to uint8.
        cloud_prob_uint8 = (cp_da.sel(band="cloud_prob") * 100).astype(np.uint8)
        cloud_only_stack = cloud_prob_uint8.expand_dims(dim="band")
        cloud_only_stack = cloud_only_stack.assign_coords(band=["cloud_prob"])
        
        if update:
            cloud_only_stack = update_prob_maps(stac_existing, cloud_only_stack)
        
        cloud_only_stack = cloud_only_stack.transpose("time", "band", "y", "x")



    else:
        # Also convert cloud probability to uint8.
        cloud_prob_uint8 = (cp_da.sel(band="cloud_prob") * 100).astype(np.uint8)
        cloud_prob_uint8 = cloud_prob_uint8.expand_dims(dim="band")
        cloud_only_stack = cloud_prob_uint8.assign_coords(band=["cloud_prob"])

        if update:
            cloud_only_stack = update_prob_maps(stac_existing, cloud_only_stack)

        # If threshold(s) are provided, compute the cloud masks using mask_from_probability.
        # This function supports either a single threshold or a list of thresholds.
        mask_da = mask_from_probability(cloud_only_stack.sel(band="cloud_prob"),
                                        threshold=threshold,
                                        average_over=average_over,
                                        dilation_size=dilation_size)
        

        # Concatenate the probability layer with the generated mask(s) along the band dimension.
        cloud_only_stack = xr.concat([cloud_only_stack, mask_da], dim="band")
        cloud_only_stack = cloud_only_stack.transpose("time", "band", "y", "x")

    # Add data array attributes
    if not update:
        cloud_only_stack.attrs['bbox'] = bbox
        #cloud_only_stack.attrs['mission'] = mission
        #cloud_only_stack.attrs['spectral_bands'] = bands
        #cloud_only_stack.attrs['indices'] = []
    
    # --- Export and Optional Masking ---
    # Export the resulting stack.
    img = export_stac(cloud_only_stack, output, crs, transform)

    if masking:
        dirname, filename = os.path.split(masking)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_masked{str(threshold)}{ext}"
        output_mask = os.path.join(dirname, output_filename)
        mask_layer = f"cloud_mask_{str(threshold)}"
        img = mask_stac_clouds(masking, cloud_only_stack, mask_layer, output_mask)

    return img


def mask_stac_clouds(stac, cloud, mask_layer, output):
    """
    Applies the cloud mask to the provided STAC dataset and exports the masked result.
    """
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

    cloud_mask = cloud.sel(band=mask_layer)
    masked_stac = stac.where(cloud_mask == 0)

    # Calculate cloud percentage per time slice.
    null_count_per_time = masked_stac.isnull().sum(dim=['band', 'y', 'x'])
    total_elements = masked_stac.sizes['band'] * masked_stac.sizes['y'] * masked_stac.sizes['x']
    cloud_percentage_int = ((null_count_per_time / total_elements) * 100).astype(int)
    masked_stac = masked_stac.assign_coords(cloud_percentage=('time', cloud_percentage_int.data))

    crs = cloud.crs
    transform = cloud.transform

    img = export_stac(masked_stac, output, crs, transform)
    return img


def mask_from_probability(cloud_probability, threshold=0.7, average_over=4, dilation_size=2):
    """
    Generates binary cloud masks from a cloud probability DataArray.
    Accepts a single threshold or a list of thresholds (provided in the 0-100 range).
    Returns a DataArray with dimensions (time, band, y, x) where each mask band is named
    according to its threshold (e.g., "cloud_mask_70").
    """
    if not isinstance(threshold, list):
        thresholds = [threshold]
    else:
        thresholds = threshold

    # Normalize probabilities to [0, 1] if necessary.
    if cloud_probability.max() > 1:
        prob_da = cloud_probability / 100.0
    else:
        prob_da = cloud_probability

    band_dataarrays = []

    for t_val in thresholds:
        # Scale the threshold from 0-100 to 0-1.
        scaled_threshold = t_val / 100.0
        cloud_detector = S2PixelCloudDetector(threshold=scaled_threshold,
                                              average_over=average_over,
                                              dilation_size=dilation_size)
        mask_list = []

        for t in prob_da.time.values:
            prob_slice = prob_da.sel(time=t)
            prob_np = prob_slice.to_numpy()[np.newaxis, ...]
            cm = cloud_detector.get_mask_from_prob(prob_np, threshold=scaled_threshold)
            mask_da = xr.DataArray(cm[0], dims=["y", "x"],
                                   coords={"y": prob_slice.y, "x": prob_slice.x})
            mask_list.append(mask_da)

        threshold_mask_da = xr.concat(mask_list, dim="time")
        threshold_mask_da = threshold_mask_da.assign_coords(time=prob_da.time)
        band_label = f"cloud_mask_{int(t_val)}"
        threshold_mask_da = threshold_mask_da.expand_dims(dim={"band": [band_label]})
        band_dataarrays.append(threshold_mask_da)

    final_mask_da = xr.concat(band_dataarrays, dim="band")
    final_mask_da = final_mask_da.transpose("time", "band", "y", "x")
    final_mask_da.name = "Cloud_Stack"

    return final_mask_da
