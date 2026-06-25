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


def scale_factor(stac, mission, baselines, source=None):
    # Normalize short source aliases (mirror get_data) so the per-provider
    # Sentinel-2 L2A offset logic below triggers regardless of how it was called.
    _source_aliases = {"e84": "element84", "tb": "terrabyte", "pc": "planetary_computer"}
    source = _source_aliases.get(source, source)

    # L1C: baseline-aware scaling. The -1000 DN radiometric offset was introduced
    # at processing baseline 04.00; reflectance = (DN - 1000)/10000 for >= 04.00,
    # else DN/10000. L1C carries no SCL and the per-scene baseline also matters
    # downstream (s2cloudless), so L1C stays baseline-driven.
    if mission == "sentinel_2_l1c":
        baselines = baselines.astype(float)
        baselines_aligned = baselines.sel(time=stac.time)
        # Cast to float32 up front: keeps the cube float32 (not float64, which
        # integer/float division would otherwise produce) and avoids uint16
        # underflow in (DN - 1000) for dark pixels (DN < 1000).
        stac = stac.astype("float32")
        stac = xr.where(baselines_aligned >= 4.00, (stac - 1000) / 10000, stac / 10000)
        return stac

    cfg = _mission_cfg(mission)
    gain = cfg[2]
    offset = cfg[3]

    # Sentinel-2 L2A: reflectance = DN * 1e-4, minus the PB>=04.00 BOA offset
    # (-1000 DN = -0.1) where it is actually baked into the served pixels. Whether
    # it is present depends ENTIRELY on the provider, verified against real pixels:
    #   * element84 == cdse to 4 dp, and element84 == PC - 1000.
    # Cast to float32 first (raw DNs are uint16; `*gain` would otherwise promote to
    # float64 -> doubled cube, and (DN-1000) would underflow uint16 for DN<1000).
    if mission == "sentinel_2_l2a":
        stac = stac.astype("float32")
        if source == "element84":
            # element84 builds its OWN L2A COGs and harmonizes them: the +1000
            # offset is already removed from the pixels -> nothing to subtract.
            return stac * gain
        if source in ("cdse", "terrabyte"):
            # Raw ESA Collection-1 archive: every scene is reprocessed to PB>=05,
            # so the +1000 offset is present on ALL scenes -> always subtract it.
            return (stac - 1000) * gain
        # planetary_computer: raw, NON-reprocessed rolling archive with mixed
        # baselines. The +1000 offset exists only for PB>=04.00, i.e. acquisitions
        # from 2022-01-25 onward; older scenes (e.g. PB 02.xx) carry no offset.
        cutoff = np.datetime64("2022-01-25")
        return xr.where(stac["time"] >= cutoff, (stac - 1000) * gain, stac * gain)

    # Other missions (Landsat / S1 / DEM): single gain/offset from _mission_cfg.
    return (stac.astype("float32") + offset) * gain


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
