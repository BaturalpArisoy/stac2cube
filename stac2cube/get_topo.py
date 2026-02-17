import os
import tempfile
import rioxarray
import xarray as xr
import whitebox

wbt = whitebox.WhiteboxTools()
wbt.set_verbose_mode(False)


def calculate_topo(dem, topographic_features):

    stac_topo_features = []

    current_dir = os.getcwd()
    tmp_dir = os.path.join(current_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        suffix=".tif", dir=tmp_dir, delete=False
    ) as tmpfile:

        dem_data = dem.squeeze()
        dem_data.rio.write_crs("EPSG:4326", inplace=True)
        dem_data.rio.to_raster(tmpfile.name)

    for feature in topographic_features:
        if feature == "slope":
            output_file = os.path.join(tmp_dir, "slope_output.tif")
            _calculate_slope(tmpfile.name, output_file)
            slope = rioxarray.open_rasterio(output_file, feature)
            stac_topo_feature = _feature_tailor(slope, dem, feature)
        elif feature == "aspect":
            output_file = os.path.join(tmp_dir, "aspect_output.tif")
            _calculate_aspect(tmpfile.name, output_file)
            aspect = rioxarray.open_rasterio(output_file, feature)
            stac_topo_feature = _feature_tailor(aspect, dem, feature)
        elif feature == "d_inf_flow_accumulation":
            output_file = os.path.join(tmp_dir, "d_inf_flow_accumulation.tif")
            _calculate_dinf_fa(tmpfile.name, output_file)
            dinf_fa = rioxarray.open_rasterio(output_file, feature)
            stac_topo_feature = _feature_tailor(dinf_fa, dem, feature)
        elif feature == "twi":
            output_file = os.path.join(tmp_dir, "twi.tif")
            _calculate_twi(output_file, tmp_dir)
            twi = rioxarray.open_rasterio(output_file, feature)
            stac_topo_feature = _feature_tailor(twi, dem, feature)

        stac_topo_features.append(stac_topo_feature)

    stac_topo_features = xr.concat(stac_topo_features, dim="band")

    _clean_tmpdir(tmp_dir)
    return stac_topo_features


def _calculate_slope(dem, output_file):
    wbt.slope(dem=dem, output=output_file, zfactor=None, units="degrees")


def _calculate_aspect(dem, output_file):
    wbt.aspect(dem=dem, output=output_file, zfactor=None)


def _calculate_dinf_fa(dem, output_file):
    wbt.d_inf_flow_accumulation(
        i=dem,
        output=output_file,
        out_type="Specific Contributing Area",
        threshold=None,
        log=False,
        clip=False,
        pntr=False,
    )


def _calculate_twi(output_file, tmp_dir):
    wbt.wetness_index(
        sca=os.path.join(tmp_dir, "d_inf_flow_accumulation.tif"),
        slope=os.path.join(tmp_dir, "slope_output.tif"),
        output=output_file,
    )


def _feature_tailor(feature, dem, feature_name):
    feature = feature.rename(
        {"y": "latitude", "x": "longitude"}
    )  # wbt generates y and x, however COP-30 images are on WGS84.
    feature = feature.assign_coords(band=[feature_name])
    feature = feature.assign_coords(latitude=dem.latitude, longitude=dem.longitude)
    return feature


def _clean_tmpdir(tmp_dir):
    for filename in os.listdir(tmp_dir):
        file_path = os.path.join(tmp_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")


"""
import os
import tempfile
import rioxarray
import xarray as xr
import whitebox
wbt = whitebox.WhiteboxTools()
wbt.set_verbose_mode(False)


def calculate_topo(dem, topographic_features):

    # Changed this from a list to a dictionary to store features by name
    stac_topo_features = {}  # Now a dictionary to store features like 'slope', 'aspect', etc.
    
    current_dir = os.getcwd()
    tmp_dir = os.path.join(current_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True) 

    with tempfile.NamedTemporaryFile(suffix='.tif', dir=tmp_dir, delete=False) as tmpfile:

        dem_data = dem.squeeze()
        dem_data.rio.set_crs("EPSG:4326", inplace=True)
        dem_data.rio.to_raster(tmpfile.name)

    
    for feature in topographic_features:
        if feature == 'slope':
            output_file = os.path.join(tmp_dir, "slope_output.tif")
            _calculate_slope(tmpfile.name, output_file)
            slope = rioxarray.open_rasterio(output_file, feature)
            stac_topo_feature = _feature_tailor(slope, dem, feature)
            
            # Store 'slope' xarray in the dictionary under its name
            stac_topo_features["slope"] = stac_topo_feature  # Changed from append to dictionary storage

        elif feature == 'aspect':
            output_file = os.path.join(tmp_dir, "aspect_output.tif")
            _calculate_aspect(tmpfile.name, output_file)
            aspect = rioxarray.open_rasterio(output_file, feature)
            stac_topo_feature = _feature_tailor(aspect, dem, feature)
            
            # Store 'aspect' xarray in the dictionary under its name
            stac_topo_features["aspect"] = stac_topo_feature  # Changed from append to dictionary storage
        

    # Concat all xarray objects stored in the dictionary by extracting their values
    stac_topo_features = xr.concat(stac_topo_features.values(), dim="band")  # Combine values from the dictionary into one xarray
    
    _clean_tmpdir(tmp_dir)
    return stac_topo_features


def _calculate_slope(dem, output_file):
    wbt.slope(
        dem=dem,  
        output=output_file, 
        zfactor=None, 
        units="degrees"
    )

def _calculate_aspect(dem, output_file):
    wbt.aspect(
        dem=dem, 
        output=output_file,
        zfactor=None
    )

def _feature_tailor(feature, dem, feature_name):
    feature = feature.rename({"y": "latitude", "x": "longitude"})  # wbt generates y and x, however COP-30 images are on WGS84.
    feature = feature.assign_coords(band=[feature_name])
    feature = feature.assign_coords(latitude=dem.latitude, longitude=dem.longitude)
    return feature

def _clean_tmpdir(tmp_dir):
    for filename in os.listdir(tmp_dir):
        file_path = os.path.join(tmp_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)  
            elif os.path.isdir(file_path):
                os.rmdir(file_path) 
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
"""
