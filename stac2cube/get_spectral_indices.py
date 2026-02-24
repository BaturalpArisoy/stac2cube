import numpy as np
import xarray as xr


def calculate_spectral_index(stac, mission, indices):

    mission_cfg = _resolve_mission_cfg(mission)
    mission_name = mission_cfg.get("name", str(mission))

    if not indices:
        raise ValueError("indices is empty/False. Example: indices=['ndvi','ndwi']")

    indices = [str(i).lower() for i in indices]

    bands_cfg = mission_cfg.get("bands", False)
    if not bands_cfg:
        raise ValueError(
            f"Mission '{mission_name}' has no bands configured (bands=False)."
        )

    bands_cfg = {str(b).lower() for b in bands_cfg}
    is_sar = {"vh", "vv"}.issubset(bands_cfg)

    is_s2 = mission_name.startswith("sentinel_2")
    is_ls = mission_name.startswith("landsat")

    out = []

    # ---------------- SAR (Sentinel-1 RTC) ----------------
    if is_sar:
        vh = _require_band(stac, mission_name, "vh")
        vv = _require_band(stac, mission_name, "vv")

        for idx in indices:
            if idx == "vh/vv":
                da = _safe_div(vh, vv)
            elif idx == "vv/vh":
                da = _safe_div(vv, vh)
            elif idx == "rvi":
                da = _safe_div(vh * 4, (vh + vv))
            else:
                raise ValueError(
                    f"Unsupported SAR index '{idx}' for mission '{mission_name}'. "
                    f"Allowed: {mission_cfg.get('indices')}"
                )
            out.append(da.expand_dims(band=[idx]))

        return xr.concat(out, dim="band")

    # ---------------- Optical (Sentinel-2, Landsat, etc.) ----------------
    red = green = blue = nir = None
    swir1 = swir2 = None
    rededge1 = None

    # Resolve only what’s required by requested indices
    need_red = any(i in indices for i in ["ndvi", "savi", "evi"])
    need_green = any(i in indices for i in ["ndwi", "mndwi", "ndsi"])
    need_blue = any(i in indices for i in ["evi"])
    need_nir = any(
        i in indices
        for i in ["ndvi", "ndwi", "savi", "ndmi", "ndbi", "nbr", "evi", "ndre1"]
    )
    need_swir1 = any(i in indices for i in ["ndmi", "mndwi", "ndbi", "ndsi"])
    need_swir2 = any(i in indices for i in ["nbr"])
    need_rededge1 = any(i in indices for i in ["ndre1"])

    if need_red:
        red = _require_band(stac, mission_name, "red")
    if need_green:
        green = _require_band(stac, mission_name, "green")
    if need_blue:
        blue = _require_band(stac, mission_name, "blue")

    if need_nir:
        if is_s2:
            # Sentinel-2: use 10 m NIR ("nir") and allow "nir08" only as fallback
            nir = _require_band(stac, mission_name, "nir", alternatives=["nir08"])
        else:
            nir = _require_band(stac, mission_name, "nir")

    if need_swir1:
        if is_s2:
            # Sentinel-2 SWIR1 (B11): swir16
            swir1 = _require_band(stac, mission_name, "swir16")
        elif is_ls:
            swir1 = _require_band(stac, mission_name, "swir1")
        else:
            swir1 = _require_band(stac, mission_name, "swir1", alternatives=["swir16"])

    if need_swir2:
        if is_s2:
            # Sentinel-2 SWIR2 (B12): swir22
            swir2 = _require_band(stac, mission_name, "swir22")
        elif is_ls:
            swir2 = _require_band(stac, mission_name, "swir2")
        else:
            swir2 = _require_band(stac, mission_name, "swir2", alternatives=["swir22"])

    if need_rededge1:
        rededge1 = _require_band(stac, mission_name, "rededge1")

    for idx in indices:
        if idx == "ndvi":
            da = _nd(red, nir)  # (nir - red)/(nir + red)
        elif idx == "ndwi":
            da = _nd(nir, green)  # (green - nir)/(green + nir)
        elif idx == "savi":
            da = _savi(red, nir, L=0.5)
        elif idx == "ndmi":
            da = _nd(swir1, nir)  # (nir - swir1)/(nir + swir1)
        elif idx == "nbr":
            da = _nd(swir2, nir)  # (nir - swir2)/(nir + swir2)
        elif idx == "mndwi":
            da = _nd(swir1, green)  # (green - swir1)/(green + swir1)
        elif idx == "ndbi":
            da = _nd(nir, swir1)  # (swir1 - nir)/(swir1 + nir)
        elif idx == "evi":
            da = _evi(blue=blue, red=red, nir=nir)
        elif idx == "ndre1":
            da = _nd(rededge1, nir)  # (nir - rededge1)/(nir + rededge1)
        elif idx == "ndsi":
            da = _nd(swir1, green)  # (green - swir1)/(green + swir1)
        else:
            raise ValueError(
                f"Unsupported optical index '{idx}' for mission '{mission_name}'. "
                f"Allowed: {mission_cfg.get('indices')}"
            )

        out.append(da.expand_dims(band=[idx]))

    return xr.concat(out, dim="band")


# -----------------------------------------------------------------------------
# Mission resolver: mission string/alias -> mission config dict (from missions())
# -----------------------------------------------------------------------------
def _resolve_mission_cfg(mission):
    if isinstance(mission, dict):
        return mission

    if not isinstance(mission, str):
        raise TypeError(f"mission must be str or dict, got {type(mission)}")

    # local import to avoid circular imports
    from stac2cube import missions as missions_df_fn

    df = missions_df_fn()

    m = df.loc[(df["name"] == mission) | (df["alias"] == mission)]

    if m.empty:
        known = sorted(set(df["name"]).union(set(df["alias"])))
        raise KeyError(f"Unknown mission '{mission}'. Known missions/aliases: {known}")

    return m.iloc[0].to_dict()


# -----------------------------------------------------------------------------
# Band helpers with friendly mission-specific error message format
# -----------------------------------------------------------------------------
def _available_bands_lower(stac):
    if isinstance(stac, xr.Dataset):
        return sorted([str(k).lower() for k in stac.data_vars.keys()])
    if isinstance(stac, xr.DataArray) and "band" in stac.coords:
        return sorted([str(v).lower() for v in stac.coords["band"].values])
    return []


def _band_map_lower_to_original(stac):
    if isinstance(stac, xr.Dataset):
        keys = list(stac.data_vars.keys())
        return {str(k).lower(): k for k in keys}
    if isinstance(stac, xr.DataArray) and "band" in stac.coords:
        vals = list(stac.coords["band"].values)
        return {str(v).lower(): v for v in vals}
    return {}


def _require_band(stac, mission_name, required, alternatives=None):

    required_l = str(required).lower()
    band_map = _band_map_lower_to_original(stac)
    avail_l = _available_bands_lower(stac)

    if not band_map:
        raise KeyError(
            f'Missing band. For "{mission_name}", please include "{required_l}". '
            f"Available: {avail_l}"
        )

    if required_l in band_map:
        orig = band_map[required_l]
        if isinstance(stac, xr.Dataset):
            return stac[orig]
        return stac.sel(band=orig, drop=True)

    if alternatives:
        for alt in alternatives:
            alt_l = str(alt).lower()
            if alt_l in band_map:
                orig = band_map[alt_l]
                if isinstance(stac, xr.Dataset):
                    return stac[orig]
                return stac.sel(band=orig, drop=True)

    raise KeyError(
        f'Missing band. For "{mission_name}", please include "{required_l}". '
        f"Available: {avail_l}"
    )


# -----------------------------------------------------------------------------
# Math helpers (Dask-friendly)
# -----------------------------------------------------------------------------
def _safe_div(num: xr.DataArray, den: xr.DataArray) -> xr.DataArray:
    return xr.where(den != 0, num / den, np.nan)


def _nd(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:

    return _safe_div(b - a, b + a)


def _savi(red: xr.DataArray, nir: xr.DataArray, L: float = 0.5) -> xr.DataArray:
    return _safe_div(nir - red, (nir + red + L)) * (1.0 + L)


def _evi(*, blue: xr.DataArray, red: xr.DataArray, nir: xr.DataArray) -> xr.DataArray:
    # EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
    den = nir + 6.0 * red - 7.5 * blue + 1.0
    return 2.5 * _safe_div(nir - red, den)
