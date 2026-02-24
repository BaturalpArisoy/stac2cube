import pandas as pd


def missions():

    columns = [
        "name",
        "alias",
        # "stac_catalog",
        "default_resolution",
        "bands",
        "indices",
        "topographic_features",
        "max_cc",
        "clip_raster",
        "cloud_masking",
        "output",
        "aggregator",
        "stats",
        "update"
    ]

    df = pd.DataFrame(columns=columns)

    sentinel_2_l2a = {
        "name": "sentinel_2_l2a",
        "alias": "s2",
        # "stac_catalog": "https://earth-search.aws.element84.com/v1/",
        "default_resolution": 10,
        "bands": [
            "coastal",
            "blue",
            "green",
            "red",
            "rededge1",
            "rededge2",
            "rededge3",
            "nir",
            "nir08",
            "nir09",
            "swir16",
            "swir22",
        ],
        "indices": [
            "ndvi",
            "ndwi",
            "savi",
            "ndmi",
            "nbr",
            "mndwi",
            "ndbi",
            "evi",
            "ndre1",
            "ndsi",
        ],
        "topographic_features": False,
        "max_cc": 100,
        "clip_raster": [True, False],
        "cloud_masking": [True, False],
        "output": "path/to/output.nc",
        "aggregator": ["mean", "median"],
        "stats": [
            "mean_timeseries",
            "mean_monthly",
            "mean_annual",
            "mean_all",
            "median_timeseries",
            "median_monthly",
            "median_annual",
            "median_all",
            "min_timeseries",
            "min_monthly",
            "min_annual",
            "min_all",
            "max_timeseries",
            "max_monthly",
            "max_annual",
            "max_all",
            "std_timeseries",
            "std_monthly",
            "std_annual",
            "std_all",
        ],
        "update": "path/to/stac.nc"
    }

    sentinel_2_l1c = {
        "name": "sentinel_2_l1c",
        "alias": "s2_l1c",
        # "stac_catalog": "https://earth-search.aws.element84.com/v1/",
        "default_resolution": 10,
        "bands": [
            "coastal",
            "blue",
            "green",
            "red",
            "rededge1",
            "rededge2",
            "rededge3",
            "nir",
            "nir08",
            "nir09",
            "cirrus",
            "swir16",
            "swir22",
        ],
        "indices": [
            "ndvi",
            "ndwi",
            "savi",
            "ndmi",
            "nbr",
            "mndwi",
            "ndbi",
            "evi",
            "ndre1",
            "ndsi",
        ],
        "topographic_features": False,
        "max_cc": 100,
        "clip_raster": [True, False],
        "cloud_masking": [True, False],
        "output": "path/to/output.nc",
        "aggregator": ["mean", "median"],
        "stats": [
            "mean_timeseries",
            "mean_monthly",
            "mean_annual",
            "mean_all",
            "median_timeseries",
            "median_monthly",
            "median_annual",
            "median_all",
            "min_timeseries",
            "min_monthly",
            "min_annual",
            "min_all",
            "max_timeseries",
            "max_monthly",
            "max_annual",
            "max_all",
            "std_timeseries",
            "std_monthly",
            "std_annual",
            "std_all",
        ],
        "update": "path/to/stac.nc",
    }

    sentinel_1_rtc = {
        "name": "sentinel_1_rtc",
        "alias": "s1",
        # "stac_catalog": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "default_resolution": 10,
        "bands": ["vh", "vv"],
        "indices": ["vh/vv", "vv/vh", "rvi"],
        "topographic_features": False,
        "max_cc": False,
        "clip_raster": [True, False],
        "cloud_masking": False,
        "output": "path/to/output.nc",
        "aggregator": ["mean", "median"],
        "stats": [
            "mean_timeseries",
            "mean_monthly",
            "mean_annual",
            "mean_all",
            "median_timeseries",
            "median_monthly",
            "median_annual",
            "median_all",
            "min_timeseries",
            "min_monthly",
            "min_annual",
            "min_all",
            "max_timeseries",
            "max_monthly",
            "max_annual",
            "max_all",
            "std_timeseries",
            "std_monthly",
            "std_annual",
            "std_all",
        ],
        "update": "path/to/stac.nc",
    }

    landsat_c2_l2 = {
        "name": "landsat_c2_l2",
        "alias": "l_oli",
        # "stac_catalog": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "default_resolution": 30,
        "bands": [
            "coastal",
            "blue",
            "green",
            "red",
            "nir",
            "swir1",
            "swir2",
            "thermal",
        ],
        "indices": [
            "ndvi",
            "ndwi",
            "savi",
            "ndmi",
            "nbr",
            "mndwi",
            "ndbi",
            "evi",
            "ndsi",
        ],
        "topographic_features": False,
        "max_cc": 100,
        "clip_raster": [True, False],
        "cloud_masking": [True, False],
        "output": "path/to/output.nc",
        "aggregator": ["mean", "median"],
        "stats": [
            "mean_timeseries",
            "mean_monthly",
            "mean_annual",
            "mean_all",
            "median_timeseries",
            "median_monthly",
            "median_annual",
            "median_all",
            "min_timeseries",
            "min_monthly",
            "min_annual",
            "min_all",
            "max_timeseries",
            "max_monthly",
            "max_annual",
            "max_all",
            "std_timeseries",
            "std_monthly",
            "std_annual",
            "std_all",
        ],
        "update": "path/to/stac.nc",
    }

    cop_dem_glo_30 = {
        "name": "cop_dem_glo_30",
        "alias": "cop_dem",
        # "stac_catalog": "https://stac.terrabyte.lrz.de/public/api/",
        "default_resolution": False,
        "bands": False,
        "indices": False,
        "topographic_features": ["slope", "aspect", "d_inf_flow_accumulation", "twi"],
        "max_cc": False,
        "clip_raster": [True, False],
        "cloud_masking": False,
        "output": "path/to/output.nc",
        "aggregator": False,
        "stats": False,
        "update": False,
    }

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    sentinel_2_l2a,
                    sentinel_2_l1c,
                    sentinel_1_rtc,
                    landsat_c2_l2,
                    cop_dem_glo_30,
                ]
            ),
        ],
        ignore_index=True,
    )
    df.style.set_properties(**{"text-align": "left"})
    pd.set_option("display.max_colwidth", None)

    return df


def missions_terrabyte():
    pass


# stac2cube/notebook_help.py

from IPython.display import display, Markdown
import ipywidgets as widgets

HELP_MD = r"""
## stac2cube parameter help

**mission**  
- Both `name` and `alias` work. Must not be `None`.  
- See `missions()` → `name` or `alias`.

**output**  
- Keep `None` to return a super-fast lazy array without computation (still good for quick visualization).  
- To export, set a NetCDF path, e.g. `./results/test.nc`.

**polygon**  
- Polygon formats: `gpkg`, `geojson`, `kml`, `kmz`, `shp`  
- Can be geographic (WGS84) or projected (e.g., UTM).  
- Can also be WGS84 bbox list: `[xmin, ymin, xmax, ymax]` (NOT projected coords). useful tool: `http://bboxfinder.com/`  
- If you don’t know the area, set `None` and use the optional leafmap selection cell.  
- If you have a polygon with multiple features, only the first feature is used. For multiple areas, run `5_Batch_Processing.ipynb`.

**resolution**  
- If `None` → uses default resolution (see `missions()` → `default_resolution`).

**daterange**  
- If `None` → every available date.  
- (1) Standard (single window): `["YYYY-MM-DD", "YYYY-MM-DD"]`
- (2) Seasonal (repeats across years): `["MM-DD", "MM-DD"]`  
  - Example vegetation season: `["04-01", "10-31"]`   
  - In seasonal mode, the window is applied to **all years** in the mission’s supported span.
- (3) Seasonal with explicit year control:  
  - `{"season": ["MM-DD", "MM-DD"], "years": "all"}`  
  - `{"season": ["MM-DD", "MM-DD"], "years": [2019, 2020, 2021]}` -> specific years
  - `{"season": ["MM-DD", "MM-DD"], "years": "2018-2024"}` -> year range

**bands**  
- If `None` → all mission bands (and SCL).  
- See `missions()` → `bands`.

**indices**  
- If `None` → no indices.  
- See `missions()` → `indices`.
- Check out for index explanation @ `https://www.indexdatabase.de/`

**clip_raster**  
- If `True`, raster is clipped to polygon area; if `False`, covers polygon bounding box.  
- Keep `False` if you plan co-registration (bbox shape works best).  
- After co-registration you can clip using `clip_stac()`.

**max_cc**  
- Maximum cloud coverage % from STAC metadata.  
- Keeping `100` is recommended for maximum availability.

**cloud_masking**  
- Scene Classification Layer masking (NOT s2cloudless threshold masking).  
- Keep `False` if you want to generate cloud mask cube and choose your own threshold (`2_Cloudmask_Data_Cube.ipynb`).  
- Set `True` for quick/rough masking (example: large areas).

**stats**  
- If `None` → no stats cubes.  
- Creates additional data variables in the output dataset with the specified statistics.  
- See `missions()` → `stats`.
- `mean_timeseries` calculates mean over the full time series; `mean_monthly` calculates mean for each month present in the time series; `mean_annual` calculates mean for each year present in the time series.
- ONE FOR ALL: `mean_all` generates `mean_timeseries` + `mean_monthly` + `mean_annual` (same for other stats).
- Disabled if `aggregator` is `NOT` `None`.

**aggregator**  
- Generates a single scene of either mean or median along the time dimension for each selected band and index.
- If `None` → no aggregation.  
- `mean` or `median`, not together.  
- See `missions()` → `aggregators`.
- Disables `stats`.

**q**  
- `True` hides print outputs except progress bar.
"""


def show_parameter_help(
    title: str = "Show parameter help",
    icon: str = "info-circle",
    open_by_default: bool = False,
):
    """
    Display a toggle button that reveals/hides the stac2cube parameter help markdown.
    Returns (toggle, output_widget) so notebooks can further customize/layout if desired.
    """
    toggle = widgets.ToggleButton(
        value=open_by_default,
        description=title if not open_by_default else "Hide parameter help",
        icon=icon,
        tooltip="Click to show/hide the parameter documentation",
    )

    out = widgets.Output()

    def _render(show: bool):
        with out:
            out.clear_output()
            if show:
                display(Markdown(HELP_MD))

    def _on_toggle(change):
        if change.get("name") != "value":
            return
        toggle.description = (
            "Hide parameter help" if change["new"] else "Show parameter help"
        )
        _render(change["new"])

    toggle.observe(_on_toggle, names="value")

    # render initial state
    _render(open_by_default)

    display(toggle, out)
    return toggle, out


COREG_HELP_MD = r"""
## Co-registration parameter help

**input_path**  
- Can be a `DataArray`, `Dataset`, or a NetCDF file path.

**grid_size**  
- The strength of the area scan. The higher, the longer it takes, but it scans more potential matching areas.  
- If the current setup still removes scenes with low cloud percentages, try increasing `grid_size`.

**max_cc**  
- Maximum cloud percentage of scenes (from cloud-masked data cube; either SCL or s2cloudless). Scenes beyond this threshold are excluded.  
- The algorithm is designed so that it detects some cloudy scenes that cannot be co-registered and automatically deletes them from the time series.  
- In this sense, the algorithm also acts as an automatic cloud-filtering system.

**time_period**  
- Selection of the time range: `["YYYY-MM-DD", "YYYY-MM-DD"]`.

**min_reliability_keep**  
- Threshold for the co-registration reliability score (percent).  
- Scenes with a score lower than this value are dropped. Very low scores often indicate highly cloudy scenes.

**min_reliability_update_ref**  
- Threshold for the co-registration reliability score (percent).  
- Scenes with a score lower than this value are kept, but the algorithm will not select them as reference for the co-registration of the next scene.

**max_cloud_update_ref**  
- Maximum cloud percentage for selecting a scene as reference.  
- Scenes above this threshold will not be selected as reference for the co-registration of the next scene.

**first_scene_mode**  
- The mode of selecting the first reference in the time series. (the first reference is crucial for the co-registration of the rest of the scenes).
- `first` selects the first scene, while `composite` creates a composite of the first `composite_window_days` days and selects the median as the first reference.
- `first` is recommended if the first scene is cloud-free, potentially a vegetation season scene. Otherwise, `composite` is recommended to create a more robust reference.

**composite_window_days**  
- Integer number of days for creating the composite if `first_scene_mode` is set to `composite`.  
- e.g. if the first scene is on `2020-01-15` and `composite_window_days` is set to `30`, the composite will calculate median of all scenes from `2020-01-15` to `2020-02-15` as the first reference.

**iteration**  
- The number of iterations to set how many time to run the co-registration process.  
- Default is `1`, however increasing the number of iterations can further improve the co-registration quality.
- `4` to `5` times is usually enough for good results.
- If the first_scene_mode is set to `composite`, the mode will be switched to `first` after the first iteration.

**output_path**  
- If `None`, the co-registered file will be exported to the same folder as the input, with the extra prefix `"_cr"`.  
- Otherwise, assign a path to a NetCDF file.
"""


def show_coregistration_parameter_help(
    title: str = "Show co-registration parameter help",
    icon: str = "info-circle",
    open_by_default: bool = False,
):
    """
    Display a toggle button that reveals/hides the co-registration parameter help markdown.
    Returns (toggle, output_widget) so notebooks can further customize/layout if desired.
    """
    toggle = widgets.ToggleButton(
        value=open_by_default,
        description=(
            title if not open_by_default else "Hide co-registration parameter help"
        ),
        icon=icon,
        tooltip="Click to show/hide the co-registration parameter documentation",
    )

    out = widgets.Output()

    def _render(show: bool):
        with out:
            out.clear_output()
            if show:
                display(Markdown(COREG_HELP_MD))

    def _on_toggle(change):
        if change.get("name") != "value":
            return
        toggle.description = (
            "Hide co-registration parameter help"
            if change["new"]
            else "Show co-registration parameter help"
        )
        _render(change["new"])

    toggle.observe(_on_toggle, names="value")

    _render(open_by_default)
    display(toggle, out)
    return toggle, out
