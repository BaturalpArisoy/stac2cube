import numpy as np
import xarray as xr


def cloud_mask(stac, mission):
    cfg = _mission_cfg(mission)
    layer_name = cfg[0]
    classes = cfg[1]

    if mission == "sentinel_2_l2a":
        # For Sentinel-2, use the classification directly with isin().
        cloud_mask = getattr(stac, layer_name).isin(classes)
    elif mission == "landsat_c2_l2":
        # For Landsat, qa_pixel is bit-packed. Extract three flags:
        #   dilated_cloud is in bit offset 1,
        #   cirrus      is in bit offset 2,
        #   cloud       is in bit offset 3.
        mask_dilated = ((stac.qa_pixel >> 1) & 1).astype(bool)
        mask_cirrus = ((stac.qa_pixel >> 2) & 1).astype(bool)
        mask_cloud = ((stac.qa_pixel >> 3) & 1).astype(bool)
        # Combine the three flags: a pixel is cloud if any of the flags are True.
        cloud_mask = mask_dilated | mask_cirrus | mask_cloud
    else:
        # If no cloud masking is implemented for the mission, do nothing.
        print(f"No cloud masking configured for mission {mission}")
        return stac

    stac_masked = stac.where(~cloud_mask)
    # Optionally drop the cloud band if it exists.
    if layer_name is not None and layer_name in stac_masked:
        stac_masked = stac_masked.drop_vars(layer_name)

    return stac_masked


def scale_factor(stac, mission, baselines):
    if mission == "sentinel_2_l1c":
        baselines = baselines.astype(float)
        baselines_aligned = baselines.sel(time=stac.time)
        stac = xr.where(baselines_aligned >= 4.00, (stac - 1000) / 10000, stac / 10000)
        return stac
    else:
        cfg = _mission_cfg(mission)
        gain = cfg[2]
        offset = cfg[3]
        return (stac + offset) * gain


def _mission_cfg(mission):
    # relevant band (cloud), classifications (cloud), gain (scale), offset (scale)
    cfg = {
        "sentinel_2_l2a": ("scl", [8, 9, 10], 1e-4, 0),  # scl: 3 is cloud shadows
        "sentinel_2_l1c": (None, None, 1e-4, -1000),
        "landsat_c2_l2": (
            "qa_pixel",
            [1, 2, 3],
            0.0000275,
            -0.2,
        ),  # qa_pixel: bit-packed values
        "sentinel_1_rtc": (None, None, 1, 0),
        "cop_dem_glo_30": (None, None, 1, 0),
    }
    return cfg[mission]
