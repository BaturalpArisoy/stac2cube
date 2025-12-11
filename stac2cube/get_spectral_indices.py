import xarray as xr

def calculate_spectral_index(stac, mission, indices):

    if mission == 'sentinel_1_rtc':
        if hasattr(stac, "vh"):
            vh = stac.vh
        else:
            vh = stac.sel(band="vh")
        if hasattr(stac, "vv"):
            vv = stac.vv
        else:
            vv = stac.sel(band="vv")
    else:
        if hasattr(stac, "red"):
            red = stac.red
        else:
            red = stac.sel(band="red")
        if hasattr(stac, "green"):
            green = stac.green
        else:
            green = stac.sel(band="green")
        if hasattr(stac, "blue"):
            blue = stac.blue
        else:
            blue = stac.sel(band="blue")
        if hasattr(stac, "nir"):
            nir = stac.nir
        else:
            nir = stac.sel(band="nir")

    stac_indices = []

    for index in indices:
        if index == 'ndvi':
            stac_index = _normalized_dif(index, red, nir)
        elif index == 'ndwi':
            stac_index = _normalized_dif(index, nir, green)
        elif index == 'savi':
            stac_index = _normalized_dif(index, red, nir)
        elif index == 'vh/vv':
            stac_index = vh/vv
        elif index == 'vv/vh':
            stac_index = vv/vh
        elif index == 'rvi':
            stac_index = (vh * 4) / (vh + vv)

        stac_indices.append(stac_index.assign_coords(band = index))
        
    stac_indices = xr.concat(stac_indices, dim = "band") 

    return stac_indices


def _normalized_dif(index, band1, band2):

    range = (-1, 1)

    if index == 'savi':
        factor = 0.5
        stac_index = (band2 - band1) / (band2 + band1 + factor) * (1 + factor)
    else:
        stac_index = (band2 - band1) / (band2 + band1)

#    stac_index = _normalization(stac_index, range) # THAT REQUIRES .compute(), therefore not feasible, look for an alternative!

    return stac_index


def _normalization (stac_index, range):

    min = stac_index.min().values.item()
    max = stac_index.max().values.item()
    
    zi = 2 * ((stac_index - min) / (max - min)) - 1

    return zi