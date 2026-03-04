import ast
import json
import os
import re
from pathlib import Path

import xarray as xr
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output, Javascript

try:
    from ipyfilechooser import FileChooser
except Exception:
    FileChooser = None

from stac2cube import (
    missions,
    export_stac,
    export_to_cogs,
    interactive_time_view,
    save_timeseries_gif,
    calculate_statistics,
    clip_stac,
    cloud_filter,
    get_stac_layers,
    get_cloud_layers,
    coregister_cube,
    get_stac_parameters,
    mask_from_probability,
    mask_stac_clouds,
    super_resolve_cube
)


# -------------------------------------------------------------------------
# Parameter help
# -------------------------------------------------------------------------
PARAM_HELP_HTML = {
    "daterange_mode": """
    <b>Date Range Mode</b><br>
    Choose how <code>daterange</code> is interpreted:<br><br>
    <b>1) Standard (single window)</b><br>
    <code>["YYYY-MM-DD", "YYYY-MM-DD"]</code><br><br>
    <b>2) Seasonal (repeat across years)</b><br>
    <code>["MM-DD", "MM-DD"]</code><br>
    Example: vegetation season <code>["04-01", "10-31"]</code><br><br>
    <b>3) Seasonal + year control</b><br>
    <code>{"season": ["MM-DD", "MM-DD"], "years": "all"}</code><br>
    <code>{"season": ["MM-DD", "MM-DD"], "years": [2019, 2020, 2021]}</code><br>
    <code>{"season": ["MM-DD", "MM-DD"], "years": "2018-2024"}</code>
    """,
    "polygon": """
    <b>polygon</b><br>
    <b>1) Path to polygon</b><br>
    Polygon formats: <code>gpkg</code>, <code>geojson</code>, <code>kml</code>, <code>kmz</code>, <code>shp</code>.<br>
    Polygons can be geographic (WGS84) or projected (e.g., UTM).<br>
    <b>2) List of BBOX</b><br>
    Can also be a WGS84 bbox list: <code>[xmin, ymin, xmax, ymax]</code> (not projected coords). Useful tool: <code>http://bboxfinder.com/</code><br>
    <b>Note:</b> If you have multiple features, only the first feature is used.
    """,
    "clip_raster": """
    <b>clip_raster</b><br>
    <b>True</b>: clip raster to polygon area.<br>
    <b>False</b>: keep polygon bounding box extent.<br><br>
    Keep <b>False</b> if you plan co-registration (bbox shape works best).<br>
    After co-registration you can clip using <code>clip_stac()</code>.
    """,
    "max_cc": """
    <b>max_cc</b><br>
    Maximum cloud coverage (%) from STAC metadata.<br>
    Keeping <code>100</code> is recommended for maximum availability.
    """,
    "cloud_masking": """
    <b>cloud_masking</b><br>
    Uses Scene Classification Layer masking (not s2cloudless threshold masking).<br><br>
    Keep <b>False</b> if you want to generate a cloud mask cube and choose your own threshold later.<br>
    Set <b>True</b> for quick/rough masking (e.g., large areas).
    """,
    "stats": """
    <b>stats</b><br>
    If empty/None: no stats cubes.<br>
    Creates additional data variables with requested statistics.<br><br>
    Examples:
    <ul style="margin:4px 0 0 18px; padding:0;">
        <li><code>mean_timeseries</code> -> mean of all time steps</li>
        <li><code>mean_monthly</code> -> mean of each month</li>
        <li><code>mean_annual</code> -> mean of each year</li>
    </ul>
    Disabled when <code>aggregator</code> is not None.
    """,
    "aggregator": """
    <b>aggregator</b><br>
    Generates a single aggregated scene along time for each selected band/index.<br>
    Typically <code>mean</code> or <code>median</code>.<br><br>
    If <b>None</b>: no aggregation.<br>
    Setting an aggregator disables <code>stats</code>.
    """,
    "output": """
    <b>Output</b><br>
    <b>Quick Result, no Export</b> → returns lazy array, select this to check the data cube before exporting<br>
    <b>NetCDF + Output file set</b> → generates single file multispectral + multidate data cube<br>
    <b>COGs + Output directory set</b> → generates multispectral GeoTiffs per each selected date<br><br>
    <b>Tip:</b> You can generate lazily first, inspect the result, then switch export mode and export later.
    """,
    "fps": """
    <b>FPS (frames per second)</b><br>
    Controls animation playback speed.<br><br>
    <b>Higher FPS</b> → faster animation playback<br>
    <b>Lower FPS</b> → slower animation playback<br><br>
    Example: <code>fps=3</code> is a moderate speed for inspecting time series changes.
    """,
    "anim_label": """
    <b>Label</b><br>
    Shows the date of the scene on animation frames.<br><br>
    <b>True</b> → date label is visible<br>
    <b>False</b> → no date label
    """,
}


def _make_help_toggle(help_key: str):
    btn = widgets.Button(
        description="?",
        tooltip="Show help",
        layout=widgets.Layout(
            width="22px", min_width="22px", height="22px", padding="0px"
        ),
    )
    btn.add_class("stac2cube-help-btn")
    btn.style.button_color = "#2563eb"
    try:
        btn.style.text_color = "white"
    except Exception:
        pass
    try:
        btn.style.font_weight = "bold"
    except Exception:
        pass

    help_html = widgets.HTML(
        value=f"""
        <div style="
            border:1px solid #dbeafe;
            border-radius:8px;
            padding:8px 10px;
            margin:2px 0 8px 0;
            line-height:1.35;
            font-size:12.5px;
            background:#eff6ff;
        ">
            {PARAM_HELP_HTML.get(help_key, "No help available.")}
        </div>
        """,
        layout=widgets.Layout(display="none"),
    )

    def _toggle(_):
        help_html.layout.display = "" if help_html.layout.display == "none" else "none"

    btn.on_click(_toggle)
    return btn, help_html


def _with_help_left(widget, help_key: str, label_text: str = None):
    """
    Label + ? on first row, widget on next row, help box below.
    """
    btn, help_html = _make_help_toggle(help_key)

    if label_text is None and hasattr(widget, "description"):
        label_text = widget.description or ""
    label_text = (label_text or "").strip()
    if label_text and not label_text.endswith(":"):
        label_text = f"{label_text}:"

    if hasattr(widget, "description"):
        try:
            widget.description = ""
        except Exception:
            pass
    if hasattr(widget, "style"):
        try:
            widget.style.description_width = "0px"
        except Exception:
            pass

    label_html = widgets.HTML(
        value=f"""
        <div style="
            font-weight:500;
            line-height:1.2;
            white-space:nowrap;
            margin:0;
            padding:0;
        ">{label_text}</div>
        """,
        layout=widgets.Layout(width="auto"),
    )

    label_row = widgets.HBox(
        [label_html, btn],
        layout=widgets.Layout(
            width="auto",
            align_items="center",
            justify_content="flex-start",
            gap="4px",
        ),
    )

    widget_box = widgets.Box([widget], layout=widgets.Layout(width="100%"))

    return widgets.VBox(
        [label_row, widget_box, help_html], layout=widgets.Layout(width="100%")
    )


def _stacked_field(widget, label_text: str = None):
    """
    Label on first row, widget on next row (no help icon).
    """
    if label_text is None and hasattr(widget, "description"):
        label_text = widget.description or ""
    label_text = (label_text or "").strip()
    if label_text and not label_text.endswith(":"):
        label_text = f"{label_text}:"

    if hasattr(widget, "description"):
        try:
            widget.description = ""
        except Exception:
            pass
    if hasattr(widget, "style"):
        try:
            widget.style.description_width = "0px"
        except Exception:
            pass

    label_html = widgets.HTML(
        value=f"""
        <div style="
            font-weight:500;
            line-height:1.2;
            white-space:nowrap;
            margin:0;
            padding:0;
        ">{label_text}</div>
        """,
        layout=widgets.Layout(width="auto"),
    )

    widget_box = widgets.Box([widget], layout=widgets.Layout(width="100%"))
    return widgets.VBox([label_html, widget_box], layout=widgets.Layout(width="100%"))


def datacube_builder(missions_func=missions):
    
    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    xr.set_options(
        display_expand_data=False,
        display_expand_coords=True,
        display_expand_attrs=False,
        display_expand_data_vars=True,
    )
    
    def _to_list_or_empty(v):
        return v if isinstance(v, list) else []

    def _is_supported(v):
        return v is not False and v is not None

    def _pretty_mission_label(name: str):
        custom = {
            "sentinel_2_l2a": "Sentinel 2 L2A",
            "sentinel_2_l1c": "Sentinel 2 L1C",
            "sentinel_1_rtc": "Sentinel 1 RTC",
            "landsat_c2_l2": "Landsat Collection 2 Level 2",
        }
        return custom.get(name, name.replace("_", " ").title())

    def _bool_dropdown_from_metadata(value, default=False):
        if value is False:
            return {
                "options": [("Not available", None)],
                "value": None,
                "disabled": True,
            }

        options = [("False", False), ("True", True)]
        if isinstance(value, list):
            bools = [v for v in [False, True] if v in value]
            options = (
                [(str(v), v) for v in bools]
                if bools
                else [("False", False), ("True", True)]
            )

        return {
            "options": options,
            "value": (
                default if any(v == default for _, v in options) else options[0][1]
            ),
            "disabled": False,
        }

    def _band_resolution_map(mission_name: str):
        if mission_name in {"sentinel_2_l2a", "sentinel_2_l1c"}:
            return {
                "coastal": "60m",
                "blue": "10m",
                "green": "10m",
                "red": "10m",
                "rededge1": "20m",
                "rededge2": "20m",
                "rededge3": "20m",
                "nir": "10m",
                "nir08": "20m",
                "nir09": "60m",
                "cirrus": "60m",
                "swir16": "20m",
                "swir22": "20m",
            }
        elif mission_name == "sentinel_1_rtc":
            return {"vh": "10m", "vv": "10m"}
        elif mission_name == "landsat_c2_l2":
            return {
                "coastal": "30m",
                "blue": "30m",
                "green": "30m",
                "red": "30m",
                "nir": "30m",
                "swir1": "30m",
                "swir2": "30m",
                "thermal": "30m",
            }
        return {}

    def _band_options_with_resolution(mission_name: str, band_list):
        """
        Sort by native resolution when possible (10m -> 20m -> 60m etc.),
        preserving original order within same resolution.
        """
        res_map = _band_resolution_map(mission_name)
        indexed = list(enumerate(band_list))

        def _res_rank(item):
            idx, band = item
            res = str(res_map.get(band, ""))
            m = re.match(r"^(\d+)m$", res)
            if m:
                return (int(m.group(1)), idx)
            return (9999, idx)

        indexed_sorted = sorted(indexed, key=_res_rank)

        options = []
        for _, b in indexed_sorted:
            res = res_map.get(b)
            label = f"{b} ({res})" if res else str(b)
            options.append((label, b))
        return options

    def _index_fullname_map(mission_name: str):
        common = {
            "ndvi": "Normalized Difference Vegetation Index",
            "ndwi": "Normalized Difference Water Index",
            "savi": "Soil Adjusted Vegetation Index",
            "ndmi": "Normalized Difference Moisture Index",
            "nbr": "Normalized Burn Ratio",
            "mndwi": "Modified Normalized Difference Water Index",
            "ndbi": "Normalized Difference Built-up Index",
            "evi": "Enhanced Vegetation Index",
            "ndre1": "Normalized Difference Red Edge Index",
            "ndsi": "Normalized Difference Snow Index",
        }
        radar = {
            "vh/vv": "VH/VV Ratio",
            "vv/vh": "VV/VH Ratio",
            "rvi": "Radar Vegetation Index",
        }
        return radar if mission_name == "sentinel_1_rtc" else common

    def _index_options_with_fullname(mission_name: str, index_list):
        name_map = _index_fullname_map(mission_name)
        options = []
        for idx in index_list:
            full = name_map.get(idx)
            label = f"{idx} ({full})" if full else str(idx)
            options.append((label, idx))
        return options

    def _daterange_mode_placeholder(mode_value: str):
        if mode_value == "standard":
            return '["2024-04-01", "2024-04-10"]'
        elif mode_value == "seasonal":
            return '["04-01", "10-31"]'
        elif mode_value == "seasonal_years":
            return '{"season": ["04-01", "10-31"], "years": [2019, 2020, 2021]}'
        return '["2024-04-01", "2024-04-10"]'

    def _normalize_ui_path(path_str: str):
        if not path_str:
            return ""
        try:
            return os.path.normpath(str(path_str))
        except Exception:
            return str(path_str)

    # -------------------------------------------------------------------------
    # Load and prepare missions metadata
    # -------------------------------------------------------------------------
    df = missions_func().copy()

    if "name" not in df.columns:
        raise ValueError("missions() must return a DataFrame with a 'name' column.")

    # Ignore disabled DEM mission for now
    df = df[df["name"] != "cop_dem_glo_30"].reset_index(drop=True)

    if df.empty:
        raise ValueError("No missions available after filtering.")

    mission_meta = {}
    for _, row in df.iterrows():
        mission_meta[row["name"]] = row.to_dict()

    ordered_names = df["name"].tolist()
    mission_options = [(_pretty_mission_label(name), name) for name in ordered_names]

    # -------------------------------------------------------------------------
    # Widgets (Basic)
    # -------------------------------------------------------------------------
    mission_dd = widgets.Dropdown(
        options=mission_options,
        value=mission_options[0][1],
        description="Mission:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
    )

    resolution_w = widgets.IntText(
        value=10,
        description="Resolution:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
    )

    polygon_w = widgets.Text(
        value="./polygons/test.gpkg",
        description="Polygon:",
        placeholder="./polygons/test.gpkg",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
    )

    daterange_mode_w = widgets.Dropdown(
        options=[
            ("Standard (single window)", "standard"),
            ("Seasonal (repeat across years)", "seasonal"),
            ("Seasonal + year control", "seasonal_years"),
        ],
        value="standard",
        description="Date Range Mode:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
    )

    daterange_w = widgets.Text(
        value=_daterange_mode_placeholder("standard"),  # prefilled example
        description="Daterange:",
        placeholder=_daterange_mode_placeholder("standard"),
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
    )

    bands_w = widgets.SelectMultiple(
        options=[],
        value=(),
        description="Bands:",
        rows=8,
        layout=widgets.Layout(width="100%", height="220px"),
        style={"description_width": "120px"},
    )

    indices_w = widgets.SelectMultiple(
        options=[],
        value=(),
        description="Indices:",
        rows=8,
        layout=widgets.Layout(width="100%", height="220px"),
        style={"description_width": "120px"},
    )

    bands_all_btn = widgets.Button(
        description="All bands", layout=widgets.Layout(width="110px")
    )
    bands_none_btn = widgets.Button(
        description="Clear bands", layout=widgets.Layout(width="110px")
    )
    indices_all_btn = widgets.Button(
        description="All indices", layout=widgets.Layout(width="120px")
    )
    indices_none_btn = widgets.Button(
        description="Clear indices", layout=widgets.Layout(width="120px")
    )

    # -------------------------------------------------------------------------
    # Widgets (Advanced)
    # -------------------------------------------------------------------------
    clip_raster_w = widgets.Dropdown(
        options=[("False", False), ("True", True)],
        value=False,
        description="Clip raster:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
    )

    max_cc_w = widgets.IntText(
        value=100,
        description="Max CC:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
    )

    cloud_masking_w = widgets.Dropdown(
        options=[("False", False), ("True", True)],
        value=False,
        description="Cloud masking:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
    )

    stats_w = widgets.SelectMultiple(
        options=[],
        value=(),
        description="Stats:",
        rows=8,
        layout=widgets.Layout(width="100%", height="220px"),
        style={"description_width": "120px"},
    )

    stats_all_btn = widgets.Button(
        description="All stats", layout=widgets.Layout(width="110px")
    )
    stats_none_btn = widgets.Button(
        description="Clear stats", layout=widgets.Layout(width="110px")
    )

    aggregator_w = widgets.Dropdown(
        options=[("None", None)],
        value=None,
        description="Aggregator:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
    )

    # -------------------------------------------------------------------------
    # Widgets (Export)
    # -------------------------------------------------------------------------
    export_mode_w = widgets.Dropdown(
        options=[
            ("Quick Result, no Export (Lazy Array)", "lazy"),
            ("NetCDF", "netcdf"),
            ("Cloud Optimized Geotiffs (select folder)", "cogs"),
        ],
        value="lazy",
        description="Export mode:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
    )

    export_target_w = widgets.Text(
        value="",
        description="Output:",
        placeholder="Disabled (Quick Result, no Export selected)",
        disabled=True,
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
    )

    browse_polygon_btn = widgets.Button(
        description="",
        icon="folder-open",
        tooltip="Browse polygon file",
        layout=widgets.Layout(
            width="34px", min_width="34px", height="32px", padding="0px"
        ),
    )
    browse_polygon_btn.style.button_color = "#f3f4f6"

    browse_output_btn = widgets.Button(
        description="",
        icon="folder-open",
        tooltip="Browse output path",
        layout=widgets.Layout(
            width="34px", min_width="34px", height="32px", padding="0px"
        ),
        disabled=True,
    )
    browse_output_btn.style.button_color = "#f3f4f6"

    # -------------------------------------------------------------------------
    # Outputs + action buttons
    # -------------------------------------------------------------------------
    result_out = widgets.Output(layout=widgets.Layout(
        border="1px solid #e5e7eb",
        padding="10px",
        border_radius="8px",
        width="99%",
    ))

    status_out = widgets.Output(layout=widgets.Layout(
        border="1px solid #dbeafe",
        padding="10px",
        border_radius="8px",
        width="100%",
        min_height="70px",
    ))

    viz_out = widgets.Output(layout=widgets.Layout(
        border="1px solid #e5e7eb",
        padding="10px",
        border_radius="8px",
        width="99%",
        min_height="90px",
    ))

    generate_btn = widgets.Button(
        description="Build data cube",
        button_style="success",
        icon="play",
        layout=widgets.Layout(width="190px"),
    )
    export_result_btn = widgets.Button(
        description="Export current result",
        button_style="danger",
        icon="save",
        layout=widgets.Layout(width="190px"),
        disabled=True,
    )
    copy_json_btn = widgets.Button(
        description="Copy JSON",
        icon="copy",
        layout=widgets.Layout(width="140px"),  # colorless like old Generate JSON button
    )

    # -------------------------------------------------------------------------
    # Visualization widgets (disabled until cube is generated)
    # -------------------------------------------------------------------------
    viz_dropdown_btn = widgets.Button(
        description="Open interactive view (dropdown)",
        button_style="info",
        icon="image",
        layout=widgets.Layout(width="260px"),
        disabled=True,
    )

    gif_display_mode_w = widgets.Dropdown(
        options=[
            ("rgb", "rgb"),
            ("false_color", "false_color"),
            ("ndvi", "ndvi"),
            ("ndwi", "ndwi"),
        ],
        value="rgb",
        description="Display mode:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
        disabled=True,
    )

    gif_fps_w = widgets.IntText(
        value=3,
        description="FPS:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
        disabled=True,
    )

    gif_label_w = widgets.Dropdown(
        options=[("True", True), ("False", False)],
        value=True,
        description="Label:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
        disabled=True,
    )

    gif_out_path_w = widgets.Text(
        value="./animations/test_rgb.gif",
        description="Output GIF:",
        placeholder="./animations/test_rgb.gif",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
        disabled=True,
    )

    browse_gif_out_btn = widgets.Button(
        description="",
        icon="folder-open",
        tooltip="Select animation output folder",
        layout=widgets.Layout(
            width="34px", min_width="34px", height="32px", padding="0px"
        ),
        disabled=True,
    )
    browse_gif_out_btn.style.button_color = "#f3f4f6"

    viz_make_gif_btn = widgets.Button(
        description="Generate animation GIF",
        button_style="warning",
        icon="film",
        layout=widgets.Layout(width="210px"),
        disabled=True,
    )

    state = {
        "result": None,
        "last_call_params": None,
        "last_export_info": None,
        "last_auto_netcdf_suggestion": None,
        "last_auto_daterange_example": None,
        "last_auto_gif_suggestion": None,
        "last_json_syntax": None,
    }

    # -------------------------------------------------------------------------
    # File choosers (ipyfilechooser; optional)
    # -------------------------------------------------------------------------
    filechooser_available = FileChooser is not None

    polygon_fc = None
    output_fc = None
    gif_out_fc = None

    polygon_fc_box = widgets.VBox(
        [], layout=widgets.Layout(display="none", width="100%")
    )
    output_fc_box = widgets.VBox(
        [], layout=widgets.Layout(display="none", width="100%")
    )
    gif_out_fc_box = widgets.VBox(
        [], layout=widgets.Layout(display="none", width="100%")
    )

    if filechooser_available:
        try:
            polygon_fc = FileChooser(
                path=str(Path(".").resolve()),
                filename="",
                title="Select polygon file",
                show_only_dirs=False,
                select_default=False,
            )
            polygon_fc.filter_pattern = [
                "*.gpkg",
                "*.geojson",
                "*.json",
                "*.shp",
                "*.kml",
                "*.kmz",
            ]
            polygon_fc.use_dir_icons = True
            polygon_fc_box = widgets.VBox(
                [polygon_fc], layout=widgets.Layout(display="none", width="100%")
            )

            output_fc = FileChooser(
                path=str(Path(".").resolve()),
                filename="",
                title="Select output",
                show_only_dirs=False,
                select_default=False,
            )
            output_fc.use_dir_icons = True
            output_fc_box = widgets.VBox(
                [output_fc], layout=widgets.Layout(display="none", width="100%")
            )

            gif_out_fc = FileChooser(
                path=str(Path(".").resolve()),
                filename="",
                title="Select animation output folder",
                show_only_dirs=True,
                select_default=False,
            )
            gif_out_fc.use_dir_icons = True
            gif_out_fc_box = widgets.VBox(
                [gif_out_fc], layout=widgets.Layout(display="none", width="100%")
            )

        except Exception:
            filechooser_available = False
            polygon_fc = None
            output_fc = None
            gif_out_fc = None
            polygon_fc_box = widgets.VBox(
                [], layout=widgets.Layout(display="none", width="100%")
            )
            output_fc_box = widgets.VBox(
                [], layout=widgets.Layout(display="none", width="100%")
            )
            gif_out_fc_box = widgets.VBox(
                [], layout=widgets.Layout(display="none", width="100%")
            )

    # -------------------------------------------------------------------------
    # Parse / validate helpers
    # -------------------------------------------------------------------------
    def _parse_polygon_input(text: str):
        """
        Accepts:
        - empty -> None
        - path string
        - bbox list/tuple [xmin, ymin, xmax, ymax]
        """
        s = (text or "").strip()
        if s == "":
            return None

        if s.startswith("[") or s.startswith("("):
            try:
                obj = ast.literal_eval(s)
            except Exception as e:
                raise ValueError(f"Polygon bbox could not be parsed: {e}")

            if not isinstance(obj, (list, tuple)) or len(obj) != 4:
                raise ValueError(
                    "Polygon bbox must be a list/tuple of 4 values: [xmin, ymin, xmax, ymax]"
                )

            try:
                vals = [float(v) for v in obj]
            except Exception:
                raise ValueError("Polygon bbox values must be numeric")

            return vals

        return s

    def _is_str_list_len2(obj):
        return (
            isinstance(obj, (list, tuple))
            and len(obj) == 2
            and all(isinstance(x, str) for x in obj)
        )

    def _validate_date_string(s: str, pattern: str, label: str):
        if not re.match(pattern, s):
            raise ValueError(f"Invalid {label}: '{s}'")

    def _parse_daterange_input(mode: str, text: str):
        """
        Returns Python object expected by get_stac_layers:
        - None
        - ["YYYY-MM-DD", "YYYY-MM-DD"]
        - ["MM-DD", "MM-DD"]
        - {"season": [...], "years": ...}
        """
        s = (text or "").strip()
        if s == "":
            return None

        try:
            obj = ast.literal_eval(s)
        except Exception as e:
            raise ValueError(
                f"Daterange could not be parsed. Please use Python-style list/dict syntax. ({e})"
            )

        if mode == "standard":
            if not _is_str_list_len2(obj):
                raise ValueError('Standard mode expects: ["YYYY-MM-DD", "YYYY-MM-DD"]')
            for d in obj:
                _validate_date_string(d, r"^\d{4}-\d{2}-\d{2}$", "date (YYYY-MM-DD)")
            return list(obj)

        elif mode == "seasonal":
            if not _is_str_list_len2(obj):
                raise ValueError('Seasonal mode expects: ["MM-DD", "MM-DD"]')
            for d in obj:
                _validate_date_string(d, r"^\d{2}-\d{2}$", "season date (MM-DD)")
            return list(obj)

        elif mode == "seasonal_years":
            if not isinstance(obj, dict):
                raise ValueError(
                    "Seasonal + year control expects a dict, e.g. "
                    '{"season": ["04-01", "10-31"], "years": [2019, 2020]}'
                )

            if "season" not in obj or "years" not in obj:
                raise ValueError(
                    'Seasonal + year control requires keys: "season" and "years"'
                )

            season = obj["season"]
            years = obj["years"]

            if not _is_str_list_len2(season):
                raise ValueError('"season" must be ["MM-DD", "MM-DD"]')
            for d in season:
                _validate_date_string(d, r"^\d{2}-\d{2}$", "season date (MM-DD)")

            valid_years = False
            if years == "all":
                valid_years = True
            elif isinstance(years, str) and re.match(r"^\d{4}-\d{4}$", years):
                valid_years = True
            elif isinstance(years, (list, tuple)) and all(
                isinstance(y, int) for y in years
            ):
                valid_years = True

            if not valid_years:
                raise ValueError(
                    '"years" must be one of: "all", "YYYY-YYYY", or a list of years like [2019, 2020, 2021]'
                )

            return {"season": list(season), "years": years}

        else:
            raise ValueError(f"Unknown Date Range Mode: {mode}")

    # -------------------------------------------------------------------------
    # Result summary (minimal)
    # -------------------------------------------------------------------------
    def _human_readable_bytes(n):
        if n is None:
            return "unknown"
        n = float(n)
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        i = 0
        while n >= 1024 and i < len(units) - 1:
            n /= 1024.0
            i += 1
        return f"{n:.2f} {units[i]}"

    def _estimated_data_size_bytes(obj):
        """
        Estimated uncompressed data size (shape * dtype), no compute triggered.
        This is NOT final exported file size on disk.
        """
        try:
            if isinstance(obj, xr.DataArray):
                return int(getattr(obj, "nbytes", 0))
            elif isinstance(obj, xr.Dataset):
                total = 0
                for _, da in obj.data_vars.items():
                    try:
                        total += int(getattr(da, "nbytes", 0))
                    except Exception:
                        pass
                return total
            return None
        except Exception:
            return None

    def _show_result_summary(obj):
        with result_out:
            clear_output()
            est_bytes = _estimated_data_size_bytes(obj)
            print(f"Estimated data size: {_human_readable_bytes(est_bytes)}\n")
            display(obj)

    # -------------------------------------------------------------------------
    # Status helpers
    # -------------------------------------------------------------------------
    def _show_status(msg: str, clear_first=True):
        with status_out:
            if clear_first:
                clear_output()
            print(msg)

    # -------------------------------------------------------------------------
    # Export path auto-suggestion
    # -------------------------------------------------------------------------
    def _auto_netcdf_suggestion_from_polygon():
        """
        Build a default NetCDF output path from polygon input.
        - ./polygons/test.gpkg -> ./results/test.nc
        - [xmin, ymin, xmax, ymax] -> ./results/bbox.nc
        """
        raw = (polygon_w.value or "").strip()
        if raw.startswith("[") or raw.startswith("("):
            stem = "bbox"
        elif raw:
            try:
                stem = Path(raw).stem
            except Exception:
                stem = "test"
        else:
            stem = "test"

        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "test"
        return f"./results/{stem}.nc"

    def _update_netcdf_output_suggestion(force=False):
        if export_mode_w.value != "netcdf":
            return

        new_suggestion = _auto_netcdf_suggestion_from_polygon()
        current = (export_target_w.value or "").strip()
        prev_auto = state.get("last_auto_netcdf_suggestion")

        should_replace = (
            force or (current == "") or (prev_auto is not None and current == prev_auto)
        )

        if should_replace:
            export_target_w.value = new_suggestion

        state["last_auto_netcdf_suggestion"] = new_suggestion

    # -------------------------------------------------------------------------
    # Daterange auto-fill examples
    # -------------------------------------------------------------------------
    def _update_daterange_placeholder(*_, force=False):
        new_example = _daterange_mode_placeholder(daterange_mode_w.value)
        current = (daterange_w.value or "").strip()
        prev_auto = state.get("last_auto_daterange_example")

        daterange_w.placeholder = new_example

        should_replace = (
            force or (current == "") or (prev_auto is not None and current == prev_auto)
        )
        if should_replace:
            daterange_w.value = new_example

        state["last_auto_daterange_example"] = new_example

    # -------------------------------------------------------------------------
    # Visualization helpers
    # -------------------------------------------------------------------------
    def _pick_dataarray_for_visualization(result_obj):
        """
        Visualization tools should use the main time-series stack.
        Prefer Spectral_Temporal_Stack when a Dataset is returned (e.g., stats outputs).
        """
        if isinstance(result_obj, xr.DataArray):
            return result_obj

        if isinstance(result_obj, xr.Dataset):
            if "Spectral_Temporal_Stack" in result_obj.data_vars:
                return result_obj["Spectral_Temporal_Stack"]
            if len(result_obj.data_vars) == 1:
                only_name = list(result_obj.data_vars)[0]
                return result_obj[only_name]
            raise ValueError(
                "Visualization needs the main time-series stack. "
                "This result is a Dataset with multiple variables and no "
                "'Spectral_Temporal_Stack' variable was found."
            )

        raise TypeError(
            f"Unsupported result type for visualization: {type(result_obj)}"
        )

    def _set_visualization_enabled(enabled: bool):
        viz_dropdown_btn.disabled = not enabled
        gif_display_mode_w.disabled = not enabled
        gif_fps_w.disabled = not enabled
        gif_label_w.disabled = not enabled
        gif_out_path_w.disabled = not enabled
        viz_make_gif_btn.disabled = not enabled
        browse_gif_out_btn.disabled = (not enabled) or (not filechooser_available)

        if not enabled:
            with viz_out:
                clear_output()
                print("ℹ️ Build a data cube first to activate visualization tools.")

    def _auto_gif_filename_from_polygon_and_mode():
        raw = (polygon_w.value or "").strip()
        if raw.startswith("[") or raw.startswith("("):
            stem = "bbox"
        elif raw:
            try:
                stem = Path(raw).stem
            except Exception:
                stem = "test"
        else:
            stem = "test"

        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "test"
        mode = (gif_display_mode_w.value or "rgb").strip()
        return f"{stem}_{mode}.gif"

    def _auto_gif_output_suggestion():
        return f"./animations/{_auto_gif_filename_from_polygon_and_mode()}"

    def _update_gif_output_suggestion(force=False):
        new_suggestion = _auto_gif_output_suggestion()
        current = (gif_out_path_w.value or "").strip()
        prev_auto = state.get("last_auto_gif_suggestion")

        gif_out_path_w.placeholder = new_suggestion

        should_replace = (
            force or (current == "") or (prev_auto is not None and current == prev_auto)
        )
        if should_replace:
            gif_out_path_w.value = new_suggestion

        state["last_auto_gif_suggestion"] = new_suggestion

    # -------------------------------------------------------------------------
    # Core params prep + export helpers
    # -------------------------------------------------------------------------
    def _prepare_get_stac_layers_params():
        mission = mission_dd.value
        polygon = _parse_polygon_input(polygon_w.value)
        daterange = _parse_daterange_input(daterange_mode_w.value, daterange_w.value)

        resolution = None if resolution_w.disabled else int(resolution_w.value)
        max_cc = None if max_cc_w.disabled else int(max_cc_w.value)

        bands = list(bands_w.value) if len(bands_w.value) > 0 else None
        indices = list(indices_w.value) if len(indices_w.value) > 0 else None
        stats = list(stats_w.value) if len(stats_w.value) > 0 else None

        clip_raster = clip_raster_w.value
        cloud_masking = cloud_masking_w.value
        aggregator = aggregator_w.value

        export_mode = export_mode_w.value
        export_target = (export_target_w.value or "").strip() or None

        # Direct export only for NetCDF mode during generation
        output_for_get_stac = (
            export_target if (export_mode == "netcdf" and export_target) else None
        )

        params = {
            "mission": mission,
            "polygon": polygon,
            "resolution": resolution,
            "daterange": daterange,
            "bands": bands,
            "max_cc": max_cc,
            "clip_raster": clip_raster,
            "cloud_masking": cloud_masking,
            "indices": indices,
            "output": output_for_get_stac,
            "aggregator": aggregator,
            "stats": stats,
            "q": True,  # hidden in UI, keep output cleaner while progress bars still show where applicable
        }

        return params, export_mode, export_target

    def _pick_dataarray_for_cog_export(result_obj):
        """
        export_to_cogs expects a DataArray with a 'band' dimension.
        Try to extract the main stack if a Dataset is returned.
        """
        if isinstance(result_obj, xr.DataArray):
            da = result_obj
        elif isinstance(result_obj, xr.Dataset):
            if "Spectral_Temporal_Stack" in result_obj.data_vars:
                da = result_obj["Spectral_Temporal_Stack"]
            elif len(result_obj.data_vars) == 1:
                only_name = list(result_obj.data_vars)[0]
                da = result_obj[only_name]
            else:
                raise ValueError(
                    "COG export currently needs a single stack DataArray. "
                    "This result is a Dataset with multiple variables (likely stats outputs). "
                    "Please export as NetCDF or generate without stats."
                )
        else:
            raise TypeError(f"Unsupported result type for export: {type(result_obj)}")

        if "band" not in da.dims:
            raise ValueError(
                f"COG export requires a 'band' dimension. Found dims: {da.dims}"
            )

        return da

    def _export_current_result(export_mode: str, export_target: str):
        if state["result"] is None:
            raise ValueError("No generated result is available to export yet.")

        if export_mode == "lazy":
            raise ValueError(
                "Please change Export mode to NetCDF or COGs before exporting."
            )

        if not export_target:
            raise ValueError("Please provide Output file / folder before exporting.")

        obj = state["result"]

        if export_mode == "netcdf":
            target = export_target
            if not target.lower().endswith(".nc"):
                target = target + ".nc"
                export_target_w.value = target

            Path(target).parent.mkdir(parents=True, exist_ok=True)

            if isinstance(obj, xr.DataArray):
                export_stac(
                    stac=obj,
                    output=target,
                    var_name=(obj.name or "Spectral_Temporal_Stack"),
                )

            elif isinstance(obj, xr.Dataset):
                # Fix for stats datasets: Dataset may not expose .crs / .transform directly
                ref_da = None
                if "Spectral_Temporal_Stack" in obj.data_vars:
                    ref_da = obj["Spectral_Temporal_Stack"]
                elif len(obj.data_vars) > 0:
                    ref_da = obj[list(obj.data_vars)[0]]

                crs_ref = None
                transform_ref = None

                if ref_da is not None:
                    try:
                        crs_ref = getattr(ref_da, "crs", None)
                    except Exception:
                        crs_ref = None
                    try:
                        transform_ref = getattr(ref_da, "transform", None)
                    except Exception:
                        transform_ref = None

                    if crs_ref is None:
                        crs_ref = ref_da.attrs.get("crs")
                    if transform_ref is None:
                        transform_ref = ref_da.attrs.get("transform")

                export_stac(
                    stac=obj, output=target, crs=crs_ref, transform=transform_ref
                )

            else:
                raise TypeError(
                    f"Unsupported result type for NetCDF export: {type(obj)}"
                )

            return {"mode": "netcdf", "target": target}

        elif export_mode == "cogs":
            Path(export_target).mkdir(parents=True, exist_ok=True)

            # Pass the full result object:
            # - DataArray -> classic behavior
            # - Dataset (time series + stats) -> backend exports all variables
            export_to_cogs(stac=obj, output_dir=export_target, prefix="", dtype="float32")

            return {"mode": "cogs", "target": export_target}

        else:
            raise ValueError(f"Unsupported export mode: {export_mode}")

    # -------------------------------------------------------------------------
    # JSON build/copy helpers (no JSON panel shown)
    # -------------------------------------------------------------------------
    def _build_json_syntax_text():
        """
        Build JSON syntax for HPC/SLURM config usage from current UI state.
        JSON uses null/true/false (via json.dumps).
        """
        mission_name = mission_dd.value
        meta = mission_meta.get(mission_name, {})

        # Prefer alias for JSON config style; fallback to full mission name
        mission_for_json = meta.get("alias", meta.get("allias", mission_name))

        polygon = _parse_polygon_input(polygon_w.value)
        daterange = _parse_daterange_input(daterange_mode_w.value, daterange_w.value)

        resolution = None if resolution_w.disabled else int(resolution_w.value)
        max_cc = None if max_cc_w.disabled else int(max_cc_w.value)

        bands = list(bands_w.value) if len(bands_w.value) > 0 else None
        indices = list(indices_w.value) if len(indices_w.value) > 0 else None
        stats = list(stats_w.value) if len(stats_w.value) > 0 else None

        clip_raster = clip_raster_w.value
        cloud_masking = cloud_masking_w.value
        aggregator = aggregator_w.value

        export_mode = export_mode_w.value
        export_target = (
            None
            if export_target_w.disabled
            else ((export_target_w.value or "").strip() or None)
        )

        # JSON is for get_stac_layers config:
        # - lazy -> output null
        # - netcdf -> output path
        # - cogs -> output null (COG export is deferred / separate UI step)
        output_for_json = (
            export_target if (export_mode == "netcdf" and export_target) else None
        )

        json_payload = {
            "parameters": {
                "mission": mission_for_json,
                "polygon": polygon,
                "resolution": resolution,
                "daterange": daterange,
                "bands": bands,
                "indices": indices,
                "max_cc": max_cc,
                "cloud_masking": cloud_masking,
                "output": output_for_json,
                "clip_raster": clip_raster,
                "aggregator": aggregator,
                "stats": stats,
            }
        }

        json_text = json.dumps(json_payload, indent=2, ensure_ascii=False)
        state["last_json_syntax"] = json_text
        return json_text

    def _copy_json_to_clipboard(_):
        """
        Build current JSON syntax and copy it to clipboard.
        """
        try:
            text = _build_json_syntax_text()
            js_text = json.dumps(text)  # safe JS embedding

            display(
                Javascript(
                    f"""
            (async () => {{
              const text = {js_text};

              async function fallbackCopy(t) {{
                const ta = document.createElement('textarea');
                ta.value = t;
                ta.setAttribute('readonly', '');
                ta.style.position = 'fixed';
                ta.style.left = '-9999px';
                document.body.appendChild(ta);
                ta.select();
                try {{
                  document.execCommand('copy');
                }} finally {{
                  document.body.removeChild(ta);
                }}
              }}

              try {{
                if (navigator.clipboard && window.isSecureContext) {{
                  await navigator.clipboard.writeText(text);
                }} else {{
                  await fallbackCopy(text);
                }}
              }} catch (e) {{
                try {{
                  await fallbackCopy(text);
                }} catch (e2) {{
                  console.error("Clipboard copy failed", e, e2);
                }}
              }}
            }})();
            """
                )
            )

            _show_status("✅ JSON syntax copied to clipboard.")

        except Exception as e:
            _show_status(f"❌ Could not copy JSON syntax: {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------
    # File chooser helpers / callbacks (ipyfilechooser)
    # -------------------------------------------------------------------------
    def _toggle_box_display(box):
        box.layout.display = "" if box.layout.display == "none" else "none"

    def _hide_polygon_chooser():
        polygon_fc_box.layout.display = "none"

    def _hide_output_chooser():
        output_fc_box.layout.display = "none"

    def _hide_gif_out_chooser():
        gif_out_fc_box.layout.display = "none"

    def _existing_dir_or_parent(path_str: str):
        s = (path_str or "").strip()
        if not s:
            return str(Path(".").resolve())

        p = Path(s)
        if p.is_dir():
            try:
                return str(p.resolve())
            except Exception:
                return str(p)

        if p.exists():
            try:
                return str(p.parent.resolve())
            except Exception:
                return str(p.parent)

        parent = p.parent if str(p.parent) not in ("", ".") else Path(".")
        try:
            return str(parent.resolve())
        except Exception:
            return str(parent)

    def _sync_polygon_filechooser_from_text():
        if not filechooser_available or polygon_fc is None:
            return
        start_dir = _existing_dir_or_parent(polygon_w.value)
        try:
            polygon_fc.reset(path=start_dir, filename="")
        except Exception:
            try:
                polygon_fc.default_path = start_dir
                polygon_fc.default_filename = ""
            except Exception:
                pass

    def _sync_output_filechooser_from_mode_and_text():
        if not filechooser_available or output_fc is None:
            return

        mode = export_mode_w.value
        current = (export_target_w.value or "").strip()

        if mode == "lazy":
            _hide_output_chooser()
            return

        if mode == "netcdf":
            suggestion = current or _auto_netcdf_suggestion_from_polygon()
            start_dir = _existing_dir_or_parent(suggestion)
            suggested_name = Path(suggestion).name or "test.nc"
            if not suggested_name.lower().endswith(".nc"):
                suggested_name = f"{Path(suggested_name).stem}.nc"

            try:
                output_fc.reset(path=start_dir, filename=suggested_name)
            except Exception:
                try:
                    output_fc.default_path = start_dir
                    output_fc.default_filename = suggested_name
                except Exception:
                    pass

            output_fc.title = "Select NetCDF output file"
            output_fc.show_only_dirs = False
            output_fc.filter_pattern = ["*.nc"]
            output_fc.use_dir_icons = True

        elif mode == "cogs":
            start_dir = _existing_dir_or_parent(current or "./results/cogs")
            try:
                output_fc.reset(path=start_dir, filename="")
            except Exception:
                try:
                    output_fc.default_path = start_dir
                    output_fc.default_filename = ""
                except Exception:
                    pass

            output_fc.title = "Select output directory for COGs"
            output_fc.show_only_dirs = True
            try:
                output_fc.filter_pattern = None
            except Exception:
                pass
            output_fc.use_dir_icons = True

    def _sync_gif_out_filechooser_from_text():
        if not filechooser_available or gif_out_fc is None:
            return

        current = (gif_out_path_w.value or "").strip() or _auto_gif_output_suggestion()
        start_dir = _existing_dir_or_parent(current)

        try:
            gif_out_fc.reset(path=start_dir, filename="")
        except Exception:
            try:
                gif_out_fc.default_path = start_dir
                gif_out_fc.default_filename = ""
            except Exception:
                pass

        gif_out_fc.title = "Select animation output folder"
        gif_out_fc.show_only_dirs = True
        try:
            gif_out_fc.filter_pattern = None
        except Exception:
            pass
        gif_out_fc.use_dir_icons = True

    def _on_polygon_chooser_selected(chooser):
        selected = getattr(chooser, "selected", None)
        if selected:
            polygon_w.value = _normalize_ui_path(selected)
            _hide_polygon_chooser()

    def _on_output_chooser_selected(chooser):
        mode = export_mode_w.value

        if mode == "netcdf":
            selected = getattr(chooser, "selected", None)
            if selected:
                s = str(selected)
                if not s.lower().endswith(".nc"):
                    s += ".nc"
                export_target_w.value = _normalize_ui_path(s)
                _hide_output_chooser()

        elif mode == "cogs":
            selected_path = getattr(chooser, "selected_path", None) or getattr(
                chooser, "selected", None
            )
            if selected_path:
                export_target_w.value = _normalize_ui_path(selected_path)
                _hide_output_chooser()

    def _on_gif_out_chooser_selected(chooser):
        selected_dir = getattr(chooser, "selected_path", None) or getattr(
            chooser, "selected", None
        )
        if selected_dir:
            auto_name = _auto_gif_filename_from_polygon_and_mode()
            gif_out_path_w.value = _normalize_ui_path(
                str(Path(selected_dir) / auto_name)
            )
            _hide_gif_out_chooser()

    if filechooser_available and polygon_fc is not None and output_fc is not None:
        try:
            polygon_fc.register_callback(_on_polygon_chooser_selected)
            output_fc.register_callback(_on_output_chooser_selected)
            if gif_out_fc is not None:
                gif_out_fc.register_callback(_on_gif_out_chooser_selected)
        except Exception:
            filechooser_available = False

    def _on_browse_polygon_clicked(_):
        if not filechooser_available or polygon_fc is None:
            _show_status(
                "ℹ️ Optional dependency 'ipyfilechooser' is not available. Install it to use Browse buttons."
            )
            return
        _sync_polygon_filechooser_from_text()
        _toggle_box_display(polygon_fc_box)

    def _on_browse_output_clicked(_):
        if export_mode_w.value == "lazy":
            _show_status(
                "ℹ️ Output selection is disabled in 'Quick Result, no Export (Lazy Array)' mode."
            )
            return
        if not filechooser_available or output_fc is None:
            _show_status(
                "ℹ️ Optional dependency 'ipyfilechooser' is not available. Install it to use Browse buttons."
            )
            return
        _sync_output_filechooser_from_mode_and_text()
        _toggle_box_display(output_fc_box)

    def _on_browse_gif_out_clicked(_):
        if state["result"] is None:
            _show_status(
                "ℹ️ Build a data cube first to enable visualization/animation export."
            )
            return
        if not filechooser_available or gif_out_fc is None:
            _show_status(
                "ℹ️ Optional dependency 'ipyfilechooser' is not available. Install it to use Browse buttons."
            )
            return
        _sync_gif_out_filechooser_from_text()
        _toggle_box_display(gif_out_fc_box)

    # -------------------------------------------------------------------------
    # Dynamic updates
    # -------------------------------------------------------------------------
    def _apply_export_mode_defaults():
        mode = export_mode_w.value
        current = (export_target_w.value or "").strip()

        if mode == "lazy":
            export_target_w.description = "Output:"
            export_target_w.disabled = True
            browse_output_btn.disabled = True
            export_target_w.placeholder = "Disabled (Quick Result, no Export selected)"
            export_target_w.value = ""
            _hide_output_chooser()

        elif mode == "netcdf":
            export_target_w.disabled = False
            browse_output_btn.disabled = False
            export_target_w.description = "Export file:"
            export_target_w.placeholder = "./results/test.nc"

            if current in ["./results/cogs", "results/cogs", r"results\cogs"]:
                export_target_w.value = ""

            _update_netcdf_output_suggestion()
            _sync_output_filechooser_from_mode_and_text()

        elif mode == "cogs":
            export_target_w.disabled = False
            browse_output_btn.disabled = False
            export_target_w.description = "Export dir:"
            export_target_w.placeholder = "./results/cogs"
            if current == "":
                export_target_w.value = "./results/cogs"

            _sync_output_filechooser_from_mode_and_text()

    def _apply_aggregator_stats_logic(*_):
        """
        aggregator != None disables stats (per docs).
        """
        agg_selected = aggregator_w.value is not None
        meta = mission_meta[mission_dd.value]
        stats_supported = len(_to_list_or_empty(meta.get("stats"))) > 0

        stats_disabled = agg_selected or (not stats_supported)

        stats_w.disabled = stats_disabled
        stats_all_btn.disabled = stats_disabled
        stats_none_btn.disabled = stats_disabled

    def _update_from_mission(*_):
        m_name = mission_dd.value
        meta = mission_meta[m_name]

        # Resolution
        if _is_supported(meta.get("default_resolution")):
            try:
                resolution_w.value = int(meta["default_resolution"])
            except Exception:
                pass
            resolution_w.disabled = False
        else:
            resolution_w.value = 0
            resolution_w.disabled = True

        # Bands
        bands = _to_list_or_empty(meta.get("bands"))
        bands_w.options = _band_options_with_resolution(m_name, bands)
        bands_w.value = ()
        bands_w.disabled = len(bands) == 0
        bands_all_btn.disabled = len(bands) == 0
        bands_none_btn.disabled = len(bands) == 0

        # Indices
        indices = _to_list_or_empty(meta.get("indices"))
        indices_w.options = _index_options_with_fullname(m_name, indices)
        indices_w.value = ()
        indices_w.disabled = len(indices) == 0
        indices_all_btn.disabled = len(indices) == 0
        indices_none_btn.disabled = len(indices) == 0

        # Clip raster
        clip_cfg = _bool_dropdown_from_metadata(meta.get("clip_raster"), default=False)
        clip_raster_w.options = clip_cfg["options"]
        clip_raster_w.value = clip_cfg["value"]
        clip_raster_w.disabled = clip_cfg["disabled"]

        # Cloud masking
        cm_meta = meta.get("cloud_masking")
        if cm_meta is False:
            cloud_masking_w.options = [("Not available", None)]
            cloud_masking_w.value = None
            cloud_masking_w.disabled = True
        else:
            cm_cfg = _bool_dropdown_from_metadata(cm_meta, default=False)
            cloud_masking_w.options = cm_cfg["options"]
            cloud_masking_w.value = cm_cfg["value"]
            cloud_masking_w.disabled = cm_cfg["disabled"]

        # Max CC
        max_cc_meta = meta.get("max_cc")
        if max_cc_meta is False:
            max_cc_w.value = 0
            max_cc_w.disabled = True
        else:
            try:
                max_cc_w.value = int(max_cc_meta)
            except Exception:
                max_cc_w.value = 100
            max_cc_w.disabled = False

        # Stats
        stats_list = _to_list_or_empty(meta.get("stats"))

        # Hide *_all shortcuts in GUI (users can multi-select directly)
        stats_list = [
            s for s in stats_list if not (isinstance(s, str) and s.endswith("_all"))
        ]

        stats_w.options = stats_list

        stats_w.value = ()
        stats_w.disabled = len(stats_list) == 0
        stats_all_btn.disabled = len(stats_list) == 0
        stats_none_btn.disabled = len(stats_list) == 0

        # Aggregator
        agg_list = _to_list_or_empty(meta.get("aggregator"))
        agg_options = [("None", None)] + [(str(x), x) for x in agg_list]
        aggregator_w.options = agg_options
        aggregator_w.value = None
        aggregator_w.disabled = len(agg_list) == 0

        _apply_export_mode_defaults()
        _apply_aggregator_stats_logic()
        _update_daterange_placeholder()

    # -------------------------------------------------------------------------
    # Visualization callbacks
    # -------------------------------------------------------------------------
    def _on_viz_dropdown_clicked(_):
        try:
            if state["result"] is None:
                with viz_out:
                    clear_output()
                    print("ℹ️ Build a data cube first.")
                return

            da = _pick_dataarray_for_visualization(state["result"])

            with viz_out:
                clear_output()
                print("Launching interactive time viewer...")
                print("")
                print("Note: Please be patient when selecting a date, the loading speed depends on your local machine.")
                print("")
                out = interactive_time_view(stac=da, widget_type="dropdown")
                if out is not None:
                    display(out)

        except Exception as e:
            with viz_out:
                clear_output()
                print(f"❌ Visualization error: {type(e).__name__}: {e}")

    def _on_viz_make_gif_clicked(_):
        try:
            if state["result"] is None:
                with viz_out:
                    clear_output()
                    print("ℹ️ Build a data cube first.")
                return

            da = _pick_dataarray_for_visualization(state["result"])

            out_path = (gif_out_path_w.value or "").strip()
            if not out_path:
                raise ValueError("Please provide an animation output path.")
            if not out_path.lower().endswith(".gif"):
                out_path = out_path + ".gif"
                gif_out_path_w.value = out_path

            Path(out_path).parent.mkdir(parents=True, exist_ok=True)

            fps_val = int(gif_fps_w.value)
            if fps_val <= 0:
                raise ValueError("FPS must be > 0.")

            with viz_out:
                clear_output()
                print("Generating animation GIF...")
                # Animation is generated only (no preview inside GUI)
                save_timeseries_gif(
                    da=da,
                    out_path=out_path,
                    display_mode=gif_display_mode_w.value,
                    fps=fps_val,
                    label=gif_label_w.value,
                )
                print(f"✅ Animation saved: {out_path}")

        except Exception as e:
            with viz_out:
                clear_output()
                print(f"❌ Animation error: {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------
    # Main action callbacks
    # -------------------------------------------------------------------------
    def _select_all_bands(_):
        values = []
        for opt in bands_w.options:
            values.append(opt[1] if isinstance(opt, tuple) and len(opt) == 2 else opt)
        bands_w.value = tuple(values)

    def _clear_bands(_):
        bands_w.value = ()

    def _select_all_indices(_):
        values = []
        for opt in indices_w.options:
            values.append(opt[1] if isinstance(opt, tuple) and len(opt) == 2 else opt)
        indices_w.value = tuple(values)

    def _clear_indices(_):
        indices_w.value = ()

    def _select_all_stats(_):
        if not stats_w.disabled:
            stats_w.value = tuple(stats_w.options)

    def _clear_stats(_):
        stats_w.value = ()

    def _on_generate_clicked(_):
        with result_out:
            clear_output()

        try:
            params, export_mode, export_target = _prepare_get_stac_layers_params()
            state["last_call_params"] = params

            with status_out:
                clear_output()
                print("Generating data cube...")

                # Ensure parent directory exists for direct NetCDF export
                if params["output"] is not None:
                    Path(params["output"]).parent.mkdir(parents=True, exist_ok=True)

                # If get_stac_layers(output=...) internally calls export_stac(),
                # Dask ProgressBar output will print inside this status box.
                result = get_stac_layers(**params)

                state["result"] = result
                export_result_btn.disabled = False
                _set_visualization_enabled(True)
                _update_gif_output_suggestion()

                # Auto export only if COG mode + target (NetCDF direct export happens in get_stac_layers)
                if export_mode == "cogs" and export_target:
                    print("Generation finished. Exporting current result to COGs...")
                    info = _export_current_result(export_mode, export_target)
                    state["last_export_info"] = info
                    print(
                        f"✅ Data cube generation + COG export finished: {info['target']}"
                    )

                elif export_mode == "netcdf" and export_target:
                    state["last_export_info"] = {
                        "mode": "netcdf",
                        "target": export_target,
                        "via": "get_stac_layers",
                    }
                    # export_stac() already prints the file path
                    print("✅ Data cube generation finished.")

                else:
                    print("✅ Data cube generation finished. Result stored in memory.")
                    #print("")
                    print("ℹ️ Inspect it, then change Export mode if you want to export.")

            # Show preview in Result panel (not in Status)
            _show_result_summary(state["result"])

            # Auto-open Result accordion after generation
            try:
                result_acc.selected_index = 0
            except Exception:
                pass

        except Exception as e:
            _show_status(f"❌ {type(e).__name__}: {e}")

    def _on_export_result_clicked(_):
        try:
            export_mode = export_mode_w.value
            export_target = (
                None
                if export_target_w.disabled
                else ((export_target_w.value or "").strip() or None)
            )

            with status_out:
                clear_output()
                print("Exporting current result...")

                # If this calls export_stac(), Dask ProgressBar output prints here
                info = _export_current_result(export_mode, export_target)
                state["last_export_info"] = info

                # export_stac() already prints "Export is done: ..."
                if info.get("mode") != "netcdf":
                    print(f"✅ Export finished: {info['target']}")

        except Exception as e:
            _show_status(f"❌ {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------
    # Wire callbacks
    # -------------------------------------------------------------------------
    bands_all_btn.on_click(_select_all_bands)
    bands_none_btn.on_click(_clear_bands)
    indices_all_btn.on_click(_select_all_indices)
    indices_none_btn.on_click(_clear_indices)
    stats_all_btn.on_click(_select_all_stats)
    stats_none_btn.on_click(_clear_stats)

    browse_polygon_btn.on_click(_on_browse_polygon_clicked)
    browse_output_btn.on_click(_on_browse_output_clicked)
    browse_gif_out_btn.on_click(_on_browse_gif_out_clicked)

    generate_btn.on_click(_on_generate_clicked)
    export_result_btn.on_click(_on_export_result_clicked)
    copy_json_btn.on_click(_copy_json_to_clipboard)

    viz_dropdown_btn.on_click(_on_viz_dropdown_clicked)
    viz_make_gif_btn.on_click(_on_viz_make_gif_clicked)

    mission_dd.observe(_update_from_mission, names="value")
    aggregator_w.observe(_apply_aggregator_stats_logic, names="value")
    export_mode_w.observe(lambda change: _apply_export_mode_defaults(), names="value")
    daterange_mode_w.observe(
        lambda change: _update_daterange_placeholder(), names="value"
    )
    polygon_w.observe(
        lambda change: (
            _update_netcdf_output_suggestion(),
            _update_gif_output_suggestion(),
        ),
        names="value",
    )
    gif_display_mode_w.observe(
        lambda change: _update_gif_output_suggestion(), names="value"
    )

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------
    FORM_WIDTH = "96%"
    FORM_MAX_WIDTH = "950px"

    css_patch = widgets.HTML(
        """
        <style>
        .stac2cube-card {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 12px;
            background: #fafbfc;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        .stac2cube-help-btn,
        .stac2cube-help-btn button,
        .stac2cube-help-btn .widget-button {
            border-radius: 50% !important;
            width: 22px !important;
            min-width: 22px !important;
            height: 22px !important;
            min-height: 22px !important;
            padding: 0 !important;
            line-height: 20px !important;
            text-align: center !important;
        }
        .stac2cube-help-btn button,
        .stac2cube-help-btn .widget-button {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-weight: 700 !important;
        }
        </style>
        """
    )

    header = widgets.HTML(
        "<div style='margin:0 0 4px 0; font-size:28px; font-weight:700;'>Data Cube Builder</div>"
    )

    subtitle = widgets.HTML(
        "<div style='font-size:13px; color:#6b7280; margin:0 0 4px 0;'>"
        "Select Basic Parameters -> optional: select advanced parameters -> build data cube -> inspect the result -> export current result."
        "</div>"
    )

    # input rows with browse buttons on the left
    polygon_input_row = widgets.HBox(
        [browse_polygon_btn, polygon_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    polygon_input_box = widgets.VBox(
        [polygon_input_row, polygon_fc_box],
        layout=widgets.Layout(width="100%", gap="4px"),
    )

    output_input_row = widgets.HBox(
        [browse_output_btn, export_target_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    output_input_box = widgets.VBox(
        [output_input_row, output_fc_box],
        layout=widgets.Layout(width="100%", gap="4px"),
    )

    gif_output_input_row = widgets.HBox(
        [browse_gif_out_btn, gif_out_path_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    gif_output_input_box = widgets.VBox(
        [gif_output_input_row, gif_out_fc_box],
        layout=widgets.Layout(width="100%", gap="4px"),
    )

    bands_box = widgets.VBox(
        [
            _stacked_field(bands_w, "Bands"),
            widgets.HBox(
                [bands_all_btn, bands_none_btn], layout=widgets.Layout(gap="6px")
            ),
        ]
    )

    indices_box = widgets.VBox(
        [
            _stacked_field(indices_w, "Indices"),
            widgets.HBox(
                [indices_all_btn, indices_none_btn], layout=widgets.Layout(gap="6px")
            ),
        ]
    )

    stats_box = widgets.VBox(
        [
            _with_help_left(stats_w, "stats", label_text="Stats"),
            widgets.HBox(
                [stats_all_btn, stats_none_btn], layout=widgets.Layout(gap="6px")
            ),
        ]
    )

    basic_box = widgets.VBox(
        [
            #widgets.HTML("<b>Basic Parameters</b>"),
            _stacked_field(mission_dd, "Mission"),
            _stacked_field(resolution_w, "Resolution"),
            _with_help_left(polygon_input_box, "polygon", label_text="Polygon"),
            _with_help_left(
                daterange_mode_w, "daterange_mode", label_text="Date Range Mode"
            ),
            _stacked_field(daterange_w, "Daterange"),
            bands_box,
            indices_box,
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    basic_acc = widgets.Accordion(children=[basic_box], selected_index=None)
    basic_acc.set_title(0, "Basic Parameters")
    basic_acc.layout = widgets.Layout(width="100%")

    advanced_box = widgets.VBox(
        [
            #widgets.HTML("<b>Advanced Parameters</b>"),
            _with_help_left(clip_raster_w, "clip_raster", label_text="Clip raster"),
            _with_help_left(max_cc_w, "max_cc", label_text="Max CC"),
            _with_help_left(
                cloud_masking_w, "cloud_masking", label_text="Cloud masking"
            ),
            stats_box,
            _with_help_left(aggregator_w, "aggregator", label_text="Aggregator"),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    export_box = widgets.VBox(
        [
            #widgets.HTML("<b>Export Options</b>"),
            _stacked_field(export_mode_w, "Export mode"),
            _with_help_left(output_input_box, "output", label_text="Output"),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    visualization_box = widgets.VBox(
        [
            # widgets.HTML("<b>Visualization</b>"),
            # widgets.HTML("<div style='font-size:12px; color:#666;'>Available after building a data cube.</div>"),
            widgets.VBox(
                [
                    widgets.HTML("<b>1) Interactive View</b>"),
                    widgets.HTML(
                        "<div style='font-size:12px; color:#666;'>Interactive viewer will be displayed below, when clicked.</div>"
                    ),
                    viz_dropdown_btn,
                ],
                layout=widgets.Layout(width="100%", gap="6px"),
            ),
            viz_out,
            widgets.VBox(
                [
                    widgets.HTML("<b>2) Animation (export only)</b>"),
                    _stacked_field(gif_display_mode_w, "Display mode"),
                    _with_help_left(gif_fps_w, "fps", label_text="FPS"),
                    _with_help_left(gif_label_w, "anim_label", label_text="Label"),
                    _stacked_field(gif_output_input_box, "Output GIF"),
                    viz_make_gif_btn,
                ],
                layout=widgets.Layout(width="100%", gap="6px"),
            ),
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )

    # Collapsible sections
    advanced_acc = widgets.Accordion(children=[advanced_box], selected_index=None)
    advanced_acc.set_title(0, "Advanced Parameters")
    advanced_acc.layout = widgets.Layout(width="99%")

    export_acc = widgets.Accordion(children=[export_box], selected_index=None)
    export_acc.set_title(0, "Export Options")
    export_acc.layout = widgets.Layout(width="99%")

    viz_acc = widgets.Accordion(children=[visualization_box], selected_index=None)
    viz_acc.set_title(0, "Visualization")
    viz_acc.layout = widgets.Layout(width="99%")

    result_box = widgets.VBox(
        [result_out], layout=widgets.Layout(width="99%", gap="6px")
    )
    result_acc = widgets.Accordion(children=[result_box], selected_index=None)
    result_acc.set_title(0, "Result")
    result_acc.layout = widgets.Layout(width="99%")

    action_row = widgets.HBox(
        [generate_btn, export_result_btn, copy_json_btn],
        layout=widgets.Layout(gap="8px", flex_flow="row wrap"),
    )


    # --- Cards (layout only) ---
    spacer_after_export = widgets.HTML("<div style='height:6px;'></div>")
    spacer_between_cards = widgets.HTML("<div style='height:12px;'></div>")

    builder_panel = widgets.VBox(
        [basic_acc, advanced_acc, export_acc, spacer_after_export, action_row],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    builder_panel.add_class("stac2cube-card")

    result_card = widgets.VBox([result_acc], layout=widgets.Layout(width="100%"))
    result_card.add_class("stac2cube-card")

    viz_card = widgets.VBox([viz_acc], layout=widgets.Layout(width="100%"))
    viz_card.add_class("stac2cube-card")

    status_card = widgets.VBox(
        [widgets.HTML("<b>Status</b>"), status_out],
        layout=widgets.Layout(width="100%", gap="6px"),
    )
    status_card.add_class("stac2cube-card")

    ui = widgets.VBox(
        [
            css_patch,
            header,
            subtitle,

            builder_panel,

            spacer_between_cards,
            result_card,

            spacer_between_cards,
            viz_card,          # ✅ Visualization moved above Status

            spacer_between_cards,
            status_card,
        ],
        layout=widgets.Layout(
            width="50%",
            max_width=FORM_MAX_WIDTH,
            margin="0 auto",
            gap="8px",
        ),
    )

    # Initialize mission-dependent widgets and defaults
    _update_from_mission()
    _update_daterange_placeholder(force=True)
    _set_visualization_enabled(False)
    _update_gif_output_suggestion(force=True)

    with status_out:
        clear_output()
        print("ℹ️ Select at least Basic Parameters to build the data cube, with optional Advanced Parameters.")

    outer = widgets.HBox(
        [ui], layout=widgets.Layout(width="100%", justify_content="center")
    )

    display(outer)

    return {
        "ui": ui,
        "outer": outer,
        "mission_meta": mission_meta,
        "state": state,
        "widgets": {
            "mission": mission_dd,
            "resolution": resolution_w,
            "polygon": polygon_w,
            "browse_polygon_btn": browse_polygon_btn,
            "daterange_mode": daterange_mode_w,
            "daterange": daterange_w,
            "bands": bands_w,
            "indices": indices_w,
            "clip_raster": clip_raster_w,
            "max_cc": max_cc_w,
            "cloud_masking": cloud_masking_w,
            "stats": stats_w,
            "aggregator": aggregator_w,
            "export_mode": export_mode_w,
            "export_target": export_target_w,
            "browse_output_btn": browse_output_btn,
            "generate_btn": generate_btn,
            "export_result_btn": export_result_btn,
            "copy_json_btn": copy_json_btn,
            "viz_dropdown_btn": viz_dropdown_btn,
            "gif_display_mode": gif_display_mode_w,
            "gif_fps": gif_fps_w,
            "gif_label": gif_label_w,
            "gif_out_path": gif_out_path_w,
            "browse_gif_out_btn": browse_gif_out_btn,
            "viz_make_gif_btn": viz_make_gif_btn,
        },
        "outputs": {
            "result": result_out,
            "status": status_out,
            "visualization": viz_out,
        },
    }





def datacube_editor():
    """
    Data Cube Editor GUI
    --------------------
    - Load NetCDF data cube
    - Work on a current in-memory result (starts with Spectral_Temporal_Stack)
    - Slice by time and band (chained)
    - Filter by cloud coverage using existing cloud_percentage coord (chained)
    - Clip raster (vector file or bbox list; applied via Edit button)
    - Temporal composites (stats) via stac2cube.calculate_statistics (applied via Edit button)
    - Visualize (interactive dropdown + GIF generation)
    - Export current result (NetCDF / COGs)
    - Reset to loaded cube
    """

    # ---------------------------------------------------------------------
    # Help text (question-mark popups)
    # ---------------------------------------------------------------------
    HELP_HTML = {
        "cloud_filter": """
        <b>filter by cloud coverage</b><br>
        Uses the existing <code>cloud_percentage</code> coordinate stored in the data cube.<br><br>
        <b>Important:</b><br>
        This is <u>not</u> a new cloud detection / masking step and <u>not</u> STAC metadata <code>max_cc</code> filtering.<br>
        It only keeps time steps where <code>cloud_percentage &lt;= max_cloud</code>.<br><br>
        Works if your cube was already cloud-masked before (e.g. SCL masking during generation or probabilistic cloud masking workflow).<br>
        Best used before clipping and before temporal composites.<br>
        Cloud percentages are not recalculated in the editor after clipping.
        """,
        "clip_raster": """
        <b>clip raster</b><br>
        <b>1) Path to polygon</b><br>
        Polygon formats: <code>gpkg</code>, <code>geojson</code>, <code>kml</code>, <code>kmz</code>, <code>shp</code>.<br>
        Polygons can be geographic (WGS84) or projected (e.g., UTM).<br>
        <b>2) List of BBOX</b><br>
        Can also be a WGS84 bbox list: <code>[xmin, ymin, xmax, ymax]</code> (not projected coords). Useful tool: <code>http://bboxfinder.com/</code><br>
        <b>Note:</b> If you have multiple features, only the first feature is used.
        """,
        "stats": """
        <b>stats</b><br>
        If empty/None: no stats cubes.<br>
        Creates additional data variables with requested statistics.<br><br>
        Examples:
        <ul style="margin:4px 0 0 18px; padding:0;">
            <li><code>mean_timeseries</code> -> mean of all time steps</li>
            <li><code>mean_monthly</code> -> mean of each month</li>
            <li><code>mean_annual</code> -> mean of each year</li>
        </ul>
        Disabled when <code>aggregator</code> is not None.
        """,
        "fps": """
        <b>fps</b><br>
        Frames per second of the animation.<br>
        Higher values = faster animation playback.<br>
        Lower values = slower animation playback.
        """,
        "gif_label": """
        <b>label</b><br>
        If True, the date label is shown on the animation frames.
        """,
        "daterange_mode": """
        <b>date range mode</b><br>
        Choose how you want to define the requested update period.<br><br>
        <b>1) Standard (single window)</b>: <code>["YYYY-MM-DD", "YYYY-MM-DD"]</code><br>
        <b>2) Seasonal</b>: <code>["MM-DD", "MM-DD"]</code> (repeats across years)<br>
        <b>3) Seasonal + year control</b>: <code>{"season": ["MM-DD", "MM-DD"], "years": [...]}</code><br><br>
        The text box below is prefilled with an editable example for the selected mode.
        """,
        "update_cube": """
        <b>update data cube</b><br>
        Uses <code>get_stac_layers(update=...)</code> with the loaded NetCDF path to fetch only missing dates
        and return an updated <code>Spectral_Temporal_Stack</code>.<br><br>
        <b>Important:</b> This replaces the current working result with a refreshed time series.<br>
        Use it first (or by itself), then continue with other editing features (slice, clip, stats, export).
        """,
    }

    STATS_OPTIONS = [
        "mean_timeseries",
        "mean_monthly",
        "mean_annual",
        "median_timeseries",
        "median_monthly",
        "median_annual",
        "min_timeseries",
        "min_monthly",
        "min_annual",
        "max_timeseries",
        "max_monthly",
        "max_annual",
        "std_timeseries",
        "std_monthly",
        "std_annual",
    ]

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _normalize_ui_path(path_str):
        if not path_str:
            return ""
        try:
            return os.path.normpath(str(path_str))
        except Exception:
            return str(path_str)

    def _human_readable_bytes(n):
        if n is None:
            return "unknown"
        n = float(n)
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        i = 0
        while n >= 1024 and i < len(units) - 1:
            n /= 1024.0
            i += 1
        return f"{n:.2f} {units[i]}"

    def _estimated_data_size_bytes(obj):
        try:
            if isinstance(obj, xr.DataArray):
                return int(getattr(obj, "nbytes", 0))
            if isinstance(obj, xr.Dataset):
                total = 0
                for _, da in obj.data_vars.items():
                    total += int(getattr(da, "nbytes", 0))
                return total
            return None
        except Exception:
            return None

    def _show_preview(out_widget, obj, title_prefix=None):
        with out_widget:
            clear_output()
            est = _estimated_data_size_bytes(obj)
            if title_prefix:
                print(title_prefix)
            print(f"Estimated data size: {_human_readable_bytes(est)}\n")
            with xr.set_options(
                display_expand_data=False,
                display_expand_coords=True,
                display_expand_attrs=False,
                display_expand_data_vars=True,
            ):
                display(obj)

    def _show_status(msg, clear_first=True):
        with status_out:
            if clear_first:
                clear_output()
            print(msg)

    def _print_working_note():
        obj = state.get("current")
        obj_type = type(obj).__name__ if obj is not None else "None"
        #print(f"ℹ️ Updated current working result ({obj_type}).")
        print("ℹ️ Original loaded cube is preserved for 'Reset to loaded cube'.")

    def _pick_dataarray_for_visualization(obj):
        """
        Pick a DataArray from current result for visualization.
        - If DataArray: use it
        - If Dataset: prefer 'Spectral_Temporal_Stack', otherwise first data var
        """
        if isinstance(obj, xr.DataArray):
            return obj

        if isinstance(obj, xr.Dataset):
            if "Spectral_Temporal_Stack" in obj.data_vars:
                return obj["Spectral_Temporal_Stack"]
            if len(obj.data_vars) > 0:
                first_name = list(obj.data_vars)[0]
                return obj[first_name]
            raise ValueError("Dataset contains no data variables.")

        raise TypeError(f"Unsupported object type for visualization: {type(obj)}")

    def _pick_timeseries_for_stats(obj):
        """
        Return the DataArray used for temporal composites.
        Accepts:
        - DataArray (time-series cube)
        - Dataset containing 'Spectral_Temporal_Stack'
        """
        if isinstance(obj, xr.DataArray):
            return obj

        if isinstance(obj, xr.Dataset):
            if "Spectral_Temporal_Stack" in obj.data_vars:
                return obj["Spectral_Temporal_Stack"]
            raise ValueError(
                "Current result is a Dataset but does not contain 'Spectral_Temporal_Stack'."
            )

        raise TypeError(f"Unsupported object type for stats: {type(obj)}")

    def _existing_dir_or_parent(path_str):
        s = (path_str or "").strip()
        if not s:
            return str(Path(".").resolve())

        p = Path(s)
        if p.is_dir():
            try:
                return str(p.resolve())
            except Exception:
                return str(p)
        if p.exists():
            try:
                return str(p.parent.resolve())
            except Exception:
                return str(p.parent)

        parent = p.parent if str(p.parent) not in ("", ".") else Path(".")
        try:
            return str(parent.resolve())
        except Exception:
            return str(parent)

    def _loaded_stem_default():
        p = state.get("loaded_path")
        if not p:
            return "cube"
        try:
            return Path(p).stem
        except Exception:
            return "cube"

    def _auto_netcdf_export_suggestion():
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", _loaded_stem_default()).strip("._-") or "cube"
        return f"./results/{stem}_edited.nc"

    def _auto_gif_output_suggestion():
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", _loaded_stem_default()).strip("._-") or "cube"
        mode = (gif_display_mode_w.value or "rgb").strip()
        return f"./animations/{stem}_{mode}.gif"

    def _safe_copy_xarray(obj):
        try:
            return obj.copy(deep=False)
        except Exception:
            return obj

    def _normalize_transform_for_export_bool(transform):
        """
        export_stac() uses: transform = transform or stac.transform
        If transform is a numpy array, that can crash due to ambiguous truth value.
        Make it bool-safe without changing backend source code.
        """
        if transform is None:
            return None
        try:
            if isinstance(transform, np.ndarray):
                return tuple(np.asarray(transform).tolist())
        except Exception:
            pass
        return transform

    def _get_reference_crs_transform_from_loaded():
        """
        Use original loaded cube as CRS/transform reference (especially useful when
        current result became a stats Dataset).
        """
        ref = state.get("loaded_original")
        if ref is None:
            return None, None

        crs = None
        transform = None

        try:
            crs = ref.attrs.get("crs")
        except Exception:
            crs = None
        try:
            transform = ref.attrs.get("transform")
        except Exception:
            transform = None

        if crs is None:
            try:
                crs = getattr(ref, "crs", None)
            except Exception:
                crs = None

        if transform is None:
            try:
                transform = getattr(ref, "transform", None)
            except Exception:
                transform = None

        transform = _normalize_transform_for_export_bool(transform)
        return crs, transform

    def _set_export_mode_defaults():
        mode = export_mode_w.value
        current = (export_target_w.value or "").strip()

        if mode == "lazy":
            export_target_w.disabled = True
            browse_export_btn.disabled = True
            export_target_w.placeholder = "Disabled (Quick Result, no Export selected)"
            export_target_w.value = ""
            export_target_w.description = "Output:"
            if filechooser_available:
                export_fc_box.layout.display = "none"

        elif mode == "netcdf":
            export_target_w.disabled = False
            browse_export_btn.disabled = False
            export_target_w.description = "Export file:"
            export_target_w.placeholder = "./results/cube_edited.nc"

            if current in ["./results/cogs", "results/cogs", r"results\cogs"]:
                export_target_w.value = ""

            if not export_target_w.value:
                export_target_w.value = _auto_netcdf_export_suggestion()

            _sync_export_filechooser_from_mode_and_text()

        elif mode == "cogs":
            export_target_w.disabled = False
            browse_export_btn.disabled = False
            export_target_w.description = "Export dir:"
            export_target_w.placeholder = "./results/cogs"

            if not export_target_w.value:
                export_target_w.value = "./results/cogs"

            _sync_export_filechooser_from_mode_and_text()

    def _set_editor_enabled(enabled):
        # Actions
        edit_btn.disabled = not enabled
        export_current_btn.disabled = not enabled
        reset_btn.disabled = not enabled

        # Slice widgets
        _update_slice_widget_enabled_state(enabled)

        # Cloud filter
        enable_cloud_filter_w.disabled = not enabled
        cloud_max_w.disabled = not enabled

        # Clip widgets
        enable_clip_w.disabled = not enabled
        clip_geom_w.disabled = not enabled
        browse_clip_btn.disabled = (not enabled) or (not filechooser_available)

        # Stats widgets
        stats_select_w.disabled = not enabled
        stats_all_btn.disabled = not enabled
        stats_clear_btn.disabled = not enabled

        # Update widgets
        enable_update_w.disabled = not enabled
        update_daterange_mode_w.disabled = not enabled
        update_daterange_w.disabled = not enabled

        # Visualization
        viz_dropdown_btn.disabled = not enabled
        gif_display_mode_w.disabled = not enabled
        gif_fps_w.disabled = not enabled
        gif_label_w.disabled = not enabled
        gif_out_path_w.disabled = not enabled
        viz_make_gif_btn.disabled = not enabled
        browse_gif_btn.disabled = (not enabled) or (not filechooser_available)

        # Export widgets
        export_mode_w.disabled = not enabled
        if not enabled:
            export_target_w.disabled = True
            browse_export_btn.disabled = True
            export_current_btn.disabled = True
            with viz_out:
                clear_output()
                print("ℹ️ Load a cube first to activate visualization tools.")
            _set_export_mode_defaults()
        else:
            _set_export_mode_defaults()

    def _update_slice_widget_enabled_state(editor_enabled):
        obj = state.get("current")
        has_obj = editor_enabled and (obj is not None)

        has_time = False
        try:
            has_time = has_obj and ("time" in obj.dims)
        except Exception:
            has_time = False

        slice_time_w.disabled = not has_time
        slice_time_all_btn.disabled = not has_time
        slice_time_clear_btn.disabled = not has_time

        has_band = False
        try:
            has_band = has_obj and ("band" in obj.dims)
        except Exception:
            has_band = False

        slice_band_w.disabled = not has_band
        slice_band_all_btn.disabled = not has_band
        slice_band_clear_btn.disabled = not has_band

    def _populate_slice_widgets_from_current(select_all=True):
        obj = state.get("current")
        if obj is None:
            slice_time_w.options = []
            slice_time_w.value = ()
            slice_band_w.options = []
            slice_band_w.value = ()
            _update_slice_widget_enabled_state(False)
            return

        # Time options
        if "time" in obj.dims:
            try:
                tvals = obj["time"].values
                time_labels = []
                for t in tvals:
                    s = str(t)
                    if "T" in s:
                        s = s.split("T")[0]
                    time_labels.append(s)
                slice_time_w.options = time_labels
                if select_all:
                    slice_time_w.value = tuple(time_labels)
                else:
                    slice_time_w.value = tuple(time_labels[: min(1, len(time_labels))])
            except Exception:
                slice_time_w.options = []
                slice_time_w.value = ()
        else:
            slice_time_w.options = []
            slice_time_w.value = ()

        # Band options
        if "band" in obj.dims:
            try:
                bvals = [str(b) for b in obj["band"].values.tolist()]
                slice_band_w.options = bvals
                if select_all:
                    slice_band_w.value = tuple(bvals)
                else:
                    slice_band_w.value = tuple(bvals[: min(1, len(bvals))])
            except Exception:
                slice_band_w.options = []
                slice_band_w.value = ()
        else:
            slice_band_w.options = []
            slice_band_w.value = ()

        _update_slice_widget_enabled_state(True)

    def _update_gif_output_suggestion(force=False):
        new_suggestion = _auto_gif_output_suggestion()
        current = (gif_out_path_w.value or "").strip()
        prev_auto = state.get("last_auto_gif_suggestion")
        gif_out_path_w.placeholder = new_suggestion

        should_replace = force or (current == "") or (prev_auto is not None and current == prev_auto)
        if should_replace:
            gif_out_path_w.value = new_suggestion

        state["last_auto_gif_suggestion"] = new_suggestion

    def _export_current_result():
        if state["current"] is None:
            raise ValueError("No current result available. Load a cube first.")

        mode = export_mode_w.value
        target = None if export_target_w.disabled else ((export_target_w.value or "").strip() or None)

        if mode == "lazy":
            raise ValueError("Please change Export mode to NetCDF or COGs before exporting.")
        if not target:
            raise ValueError("Please provide an export file/folder path.")

        obj = state["current"]
        if not isinstance(obj, (xr.DataArray, xr.Dataset)):
            raise TypeError(f"Unsupported result type for export: {type(obj)}")

        if mode == "netcdf":
            if not target.lower().endswith(".nc"):
                target = target + ".nc"
                export_target_w.value = target

            Path(target).parent.mkdir(parents=True, exist_ok=True)

            if isinstance(obj, xr.DataArray):
                export_stac(
                    stac=obj,
                    output=target,
                    var_name=(obj.name or "Spectral_Temporal_Stack"),
                )
                return {"mode": "netcdf", "target": target}

            # Dataset export (e.g. after calculate_statistics)
            crs_ref, transform_ref = _get_reference_crs_transform_from_loaded()
            export_stac(
                stac=obj,
                output=target,
                crs=crs_ref,
                transform=transform_ref,
            )
            return {"mode": "netcdf", "target": target}

        elif mode == "cogs":
            Path(target).mkdir(parents=True, exist_ok=True)
            export_to_cogs(stac=obj, output_dir=target, prefix="", dtype="float32")
            return {"mode": "cogs", "target": target}

        else:
            raise ValueError(f"Unsupported export mode: {mode}")

    # ---------------------------------------------------------------------
    # Question mark help UI helpers
    # ---------------------------------------------------------------------
    def _help_icon_button():
        b = widgets.Button(
            description="?",
            tooltip="Show help",
            layout=widgets.Layout(width="22px", height="22px", padding="0px", min_width="22px"),
        )
        b.style.button_color = "#dbeafe"
        b.add_class("stac2cube-help-btn")
        return b

    def _stacked_field_with_help(widget, label_text, help_key):
        try:
            widget.description = ""
        except Exception:
            pass
        try:
            widget.style.description_width = "0px"
        except Exception:
            pass

        label_html = widgets.HTML(
            f"<div style='font-weight:500; line-height:1.2; margin:0; padding:0;'>{label_text}:</div>"
        )
        q_btn = _help_icon_button()

        help_box = widgets.HTML(
            value=HELP_HTML.get(help_key, ""),
            layout=widgets.Layout(
                display="none",
                border="1px solid #dbeafe",
                padding="8px",
                border_radius="8px",
                margin="2px 0 2px 0",
                width="100%",
            ),
        )

        def _toggle_help(_):
            help_box.layout.display = "" if help_box.layout.display == "none" else "none"

        q_btn.on_click(_toggle_help)

        header_row = widgets.HBox(
            [label_html, q_btn],
            layout=widgets.Layout(align_items="center", gap="6px"),
        )

        return widgets.VBox(
            [header_row, help_box, widget],
            layout=widgets.Layout(width="100%", gap="4px"),
        )

    # ---------------------------------------------------------------------
    # File chooser helpers (optional)
    # ---------------------------------------------------------------------
    filechooser_available = FileChooser is not None

    load_fc = None
    export_fc = None
    gif_fc = None
    clip_fc = None

    load_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))
    export_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))
    gif_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))
    clip_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))

    def _toggle_box_display(box):
        box.layout.display = "" if box.layout.display == "none" else "none"

    def _sync_load_filechooser_from_text():
        if not filechooser_available or load_fc is None:
            return
        current = (load_path_w.value or "").strip()
        start_dir = _existing_dir_or_parent(current)
        suggested_name = Path(current).name if current else ""
        try:
            load_fc.reset(path=start_dir, filename=suggested_name)
        except Exception:
            try:
                load_fc.default_path = start_dir
                load_fc.default_filename = suggested_name
            except Exception:
                pass

    def _sync_export_filechooser_from_mode_and_text():
        if not filechooser_available or export_fc is None:
            return

        mode = export_mode_w.value
        current = (export_target_w.value or "").strip()

        if mode == "lazy":
            export_fc_box.layout.display = "none"
            return

        if mode == "netcdf":
            suggestion = current or _auto_netcdf_export_suggestion()
            start_dir = _existing_dir_or_parent(suggestion)
            suggested_name = Path(suggestion).name or "cube_edited.nc"
            if not suggested_name.lower().endswith(".nc"):
                suggested_name = f"{Path(suggested_name).stem}.nc"

            try:
                export_fc.reset(path=start_dir, filename=suggested_name)
            except Exception:
                try:
                    export_fc.default_path = start_dir
                    export_fc.default_filename = suggested_name
                except Exception:
                    pass

            export_fc.title = "Select NetCDF export file"
            export_fc.show_only_dirs = False
            export_fc.filter_pattern = ["*.nc"]

        elif mode == "cogs":
            start_dir = _existing_dir_or_parent(current or "./results/cogs")
            try:
                export_fc.reset(path=start_dir, filename="")
            except Exception:
                try:
                    export_fc.default_path = start_dir
                    export_fc.default_filename = ""
                except Exception:
                    pass

            export_fc.title = "Select output directory for COGs"
            export_fc.show_only_dirs = True
            try:
                export_fc.filter_pattern = None
            except Exception:
                pass

    def _sync_gif_filechooser_from_text():
        if not filechooser_available or gif_fc is None:
            return
        current = (gif_out_path_w.value or "").strip() or _auto_gif_output_suggestion()
        start_dir = _existing_dir_or_parent(current)
        try:
            gif_fc.reset(path=start_dir, filename="")
        except Exception:
            try:
                gif_fc.default_path = start_dir
                gif_fc.default_filename = ""
            except Exception:
                pass
        gif_fc.title = "Select animation output folder"
        gif_fc.show_only_dirs = True
        try:
            gif_fc.filter_pattern = None
        except Exception:
            pass

    def _sync_clip_filechooser_from_text():
        if not filechooser_available or clip_fc is None:
            return
        current = (clip_geom_w.value or "").strip()

        if current.startswith("[") and current.endswith("]"):
            current = ""

        start_dir = _existing_dir_or_parent(current)
        suggested_name = Path(current).name if current else ""
        try:
            clip_fc.reset(path=start_dir, filename=suggested_name)
        except Exception:
            try:
                clip_fc.default_path = start_dir
                clip_fc.default_filename = suggested_name
            except Exception:
                pass

    if filechooser_available:
        try:
            load_fc = FileChooser(
                path=str(Path(".").resolve()),
                filename="",
                title="Select NetCDF cube",
                show_only_dirs=False,
                select_default=False,
            )
            load_fc.filter_pattern = ["*.nc"]
            load_fc.use_dir_icons = True
            load_fc_box = widgets.VBox([load_fc], layout=widgets.Layout(display="none", width="100%"))

            export_fc = FileChooser(
                path=str(Path(".").resolve()),
                filename="",
                title="Select export output",
                show_only_dirs=False,
                select_default=False,
            )
            export_fc.use_dir_icons = True
            export_fc_box = widgets.VBox([export_fc], layout=widgets.Layout(display="none", width="100%"))

            gif_fc = FileChooser(
                path=str(Path(".").resolve()),
                filename="",
                title="Select animation output folder",
                show_only_dirs=True,
                select_default=False,
            )
            gif_fc.use_dir_icons = True
            gif_fc_box = widgets.VBox([gif_fc], layout=widgets.Layout(display="none", width="100%"))

            clip_fc = FileChooser(
                path=str(Path(".").resolve()),
                filename="",
                title="Select clipping polygon file",
                show_only_dirs=False,
                select_default=False,
            )
            clip_fc.use_dir_icons = True
            try:
                clip_fc.filter_pattern = ["*.gpkg", "*.geojson", "*.kml", "*.kmz", "*.shp"]
            except Exception:
                pass
            clip_fc_box = widgets.VBox([clip_fc], layout=widgets.Layout(display="none", width="100%"))

        except Exception:
            filechooser_available = False
            load_fc = export_fc = gif_fc = clip_fc = None
            load_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))
            export_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))
            gif_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))
            clip_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))

    # ---------------------------------------------------------------------
    # Widgets
    # ---------------------------------------------------------------------
    # Loading
    load_path_w = widgets.Text(
        value="./results/test.nc",
        description="NetCDF:",
        placeholder="./results/test.nc",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "90px"},
    )

    browse_load_btn = widgets.Button(
        description="",
        icon="folder-open",
        tooltip="Browse NetCDF file",
        layout=widgets.Layout(width="34px", min_width="34px", height="32px", padding="0px"),
    )
    browse_load_btn.style.button_color = "#f3f4f6"

    load_cube_btn = widgets.Button(
        description="Load cube",
        icon="folder-open",
        button_style="info",
        layout=widgets.Layout(width="130px"),
    )

    reset_btn = widgets.Button(
        description="Reset to loaded cube",
        icon="undo",
        layout=widgets.Layout(width="180px"),
        disabled=True,
    )

    # Slice feature
    slice_time_w = widgets.SelectMultiple(
        options=[],
        value=(),
        description="Dates:",
        rows=8,
        layout=widgets.Layout(width="99%", height="200px"),
        style={"description_width": "90px"},
        disabled=True,
    )

    slice_band_w = widgets.SelectMultiple(
        options=[],
        value=(),
        description="Bands:",
        rows=8,
        layout=widgets.Layout(width="99%", height="200px"),
        style={"description_width": "90px"},
        disabled=True,
    )

    slice_time_all_btn = widgets.Button(description="All dates", layout=widgets.Layout(width="100px"), disabled=True)
    slice_time_clear_btn = widgets.Button(description="Clear dates", layout=widgets.Layout(width="110px"), disabled=True)
    slice_band_all_btn = widgets.Button(description="All bands", layout=widgets.Layout(width="100px"), disabled=True)
    slice_band_clear_btn = widgets.Button(description="Clear bands", layout=widgets.Layout(width="110px"), disabled=True)

    # Cloud filter feature (applied via Edit button)
    enable_cloud_filter_w = widgets.Checkbox(
        value=False,
        description="Enable filter",
        indent=False,
        layout=widgets.Layout(width="140px"),
        disabled=True,
    )

    cloud_max_w = widgets.IntText(
        value=100,
        description="",
        layout=widgets.Layout(width="20%"),
        disabled=True,
    )

    # Clip feature (applied via Edit button)
    enable_clip_w = widgets.Checkbox(
        value=False,
        description="Enable clip",
        indent=False,
        layout=widgets.Layout(width="140px"),
        disabled=True,
    )

    clip_geom_w = widgets.Text(
        value="",
        description="",
        placeholder="./polygons/test.gpkg  or  [xmin, ymin, xmax, ymax]",
        layout=widgets.Layout(width="80%"),
        disabled=True,
    )

    browse_clip_btn = widgets.Button(
        description="",
        icon="folder-open",
        tooltip="Browse clipping polygon file",
        layout=widgets.Layout(width="34px", min_width="34px", height="32px", padding="0px"),
        disabled=True,
    )
    browse_clip_btn.style.button_color = "#f3f4f6"

    # Temporal composites (stats) -- applied via Edit button
    stats_select_w = widgets.SelectMultiple(
        options=STATS_OPTIONS,
        value=(),
        rows=8,
        layout=widgets.Layout(width="50%", height="210px"),
        disabled=True,
    )
    stats_all_btn = widgets.Button(description="All stats", layout=widgets.Layout(width="95px"), disabled=True)
    stats_clear_btn = widgets.Button(description="Clear", layout=widgets.Layout(width="70px"), disabled=True)

    # Update Data Cube (fetch missing dates from loaded cube path)
    enable_update_w = widgets.Checkbox(
        value=False,
        description="Enable update data cube",
        indent=False,
        disabled=True,
    )

    update_daterange_mode_w = widgets.Dropdown(
        options=[
            ("Standard (single window)", "standard"),
            ("Seasonal (repeat across years)", "seasonal"),
            ("Seasonal + year control", "seasonal_years"),
        ],
        value="standard",
        description="",
        layout=widgets.Layout(width="99%"),
        disabled=True,
    )

    update_daterange_w = widgets.Text(
        value='["2024-04-01", "2024-04-10"]',
        description="",
        placeholder='["2024-04-01", "2024-04-10"]',
        layout=widgets.Layout(width="99%"),
        disabled=True,
    )

    # Export options
    export_mode_w = widgets.Dropdown(
        options=[
            ("Quick Result, no Export (Lazy Array)", "lazy"),
            ("NetCDF", "netcdf"),
            ("Cloud Optimized Geotiffs (select folder)", "cogs"),
        ],
        value="lazy",
        description="Mode:",
        layout=widgets.Layout(width="99%"),
        style={"description_width": "90px"},
        disabled=True,
    )

    export_target_w = widgets.Text(
        value="",
        description="Output:",
        placeholder="Disabled (Quick Result, no Export selected)",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "90px"},
        disabled=True,
    )

    browse_export_btn = widgets.Button(
        description="",
        icon="folder-open",
        tooltip="Browse export output",
        layout=widgets.Layout(width="34px", min_width="34px", height="32px", padding="0px"),
        disabled=True,
    )
    browse_export_btn.style.button_color = "#f3f4f6"

    # Visualization
    viz_dropdown_btn = widgets.Button(
        description="Launch interactive viewer",
        icon="image",
        button_style="info",
        layout=widgets.Layout(width="260px"),
        disabled=True,
    )

    gif_display_mode_w = widgets.Dropdown(
        options=[
            ("rgb", "rgb"),
            ("false_color", "false_color"),
            ("ndvi", "ndvi"),
            ("ndwi", "ndwi"),
        ],
        value="rgb",
        description="Mode:",
        layout=widgets.Layout(width="99%"),
        style={"description_width": "90px"},
        disabled=True,
    )

    gif_fps_w = widgets.IntText(
        value=3,
        description="FPS:",
        layout=widgets.Layout(width="99%"),
        style={"description_width": "90px"},
        disabled=True,
    )

    gif_label_w = widgets.Dropdown(
        options=[("True", True), ("False", False)],
        value=True,
        description="Label:",
        layout=widgets.Layout(width="99%"),
        style={"description_width": "90px"},
        disabled=True,
    )

    gif_out_path_w = widgets.Text(
        value="./animations/cube_rgb.gif",
        description="GIF:",
        placeholder="./animations/cube_rgb.gif",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "90px"},
        disabled=True,
    )

    browse_gif_btn = widgets.Button(
        description="",
        icon="folder-open",
        tooltip="Select GIF output folder",
        layout=widgets.Layout(width="34px", min_width="34px", height="32px", padding="0px"),
        disabled=True,
    )
    browse_gif_btn.style.button_color = "#f3f4f6"

    viz_make_gif_btn = widgets.Button(
        description="Generate animation GIF",
        icon="film",
        button_style="warning",
        layout=widgets.Layout(width="210px"),
        disabled=True,
    )

    # Actions
    edit_btn = widgets.Button(
        description="Edit data cube",
        icon="play",
        button_style="success",
        layout=widgets.Layout(width="160px"),
        disabled=True,
    )

    export_current_btn = widgets.Button(
        description="Export current result",
        icon="save",
        button_style="danger",
        layout=widgets.Layout(width="190px"),
        disabled=True,
    )

    # Outputs
    loaded_summary_out = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #e5e7eb",
            padding="10px",
            border_radius="8px",
            width="99%",
        )
    )

    result_out = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #e5e7eb",
            padding="10px",
            border_radius="8px",
            width="99%",
        )
    )

    status_out = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #dbeafe",
            padding="10px",
            border_radius="8px",
            width="100%",
            min_height="80px",
        )
    )

    viz_out = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #e5e7eb",
            padding="10px",
            border_radius="8px",
            width="99%",
            min_height="90px",
        )
    )

    # ---------------------------------------------------------------------
    # State
    # ---------------------------------------------------------------------
    state = {
        "loaded_path": None,
        "loaded_original": None,  # untouched Spectral_Temporal_Stack DataArray
        "current": None,          # working result (DataArray or Dataset after stats)
        "last_export_info": None,
        "last_auto_gif_suggestion": None,
    }

    # ---------------------------------------------------------------------
    # File chooser callbacks
    # ---------------------------------------------------------------------
    if filechooser_available and load_fc is not None:
        def _on_load_fc_selected(chooser):
            selected = getattr(chooser, "selected", None)
            if selected:
                load_path_w.value = _normalize_ui_path(selected)
                load_fc_box.layout.display = "none"

        def _on_export_fc_selected(chooser):
            mode = export_mode_w.value
            if mode == "netcdf":
                selected = getattr(chooser, "selected", None)
                if selected:
                    s = str(selected)
                    if not s.lower().endswith(".nc"):
                        s += ".nc"
                    export_target_w.value = _normalize_ui_path(s)
                    export_fc_box.layout.display = "none"
            elif mode == "cogs":
                selected_path = getattr(chooser, "selected_path", None) or getattr(chooser, "selected", None)
                if selected_path:
                    export_target_w.value = _normalize_ui_path(selected_path)
                    export_fc_box.layout.display = "none"

        def _on_gif_fc_selected(chooser):
            selected_dir = getattr(chooser, "selected_path", None) or getattr(chooser, "selected", None)
            if selected_dir:
                auto_name = Path(_auto_gif_output_suggestion()).name
                gif_out_path_w.value = _normalize_ui_path(str(Path(selected_dir) / auto_name))
                gif_fc_box.layout.display = "none"

        def _on_clip_fc_selected(chooser):
            selected = getattr(chooser, "selected", None)
            if selected:
                clip_geom_w.value = _normalize_ui_path(selected)
                clip_fc_box.layout.display = "none"

        try:
            load_fc.register_callback(_on_load_fc_selected)
            export_fc.register_callback(_on_export_fc_selected)
            gif_fc.register_callback(_on_gif_fc_selected)
            if clip_fc is not None:
                clip_fc.register_callback(_on_clip_fc_selected)
        except Exception:
            filechooser_available = False

    def _on_browse_load_clicked(_):
        if not filechooser_available or load_fc is None:
            _show_status("ℹ️ Optional dependency 'ipyfilechooser' is not available. Install it to use Browse buttons.")
            return
        _sync_load_filechooser_from_text()
        _toggle_box_display(load_fc_box)

    def _on_browse_export_clicked(_):
        if export_mode_w.value == "lazy":
            _show_status("ℹ️ Output selection is disabled in 'Quick Result, no Export (Lazy Array)' mode.")
            return
        if not filechooser_available or export_fc is None:
            _show_status("ℹ️ Optional dependency 'ipyfilechooser' is not available. Install it to use Browse buttons.")
            return
        _sync_export_filechooser_from_mode_and_text()
        _toggle_box_display(export_fc_box)

    def _on_browse_gif_clicked(_):
        if state["current"] is None:
            _show_status("ℹ️ Load a cube first to enable visualization tools.")
            return
        if not filechooser_available or gif_fc is None:
            _show_status("ℹ️ Optional dependency 'ipyfilechooser' is not available. Install it to use Browse buttons.")
            return
        _sync_gif_filechooser_from_text()
        _toggle_box_display(gif_fc_box)

    def _on_browse_clip_clicked(_):
        if state["current"] is None:
            _show_status("ℹ️ Load a cube first to enable editing features.")
            return
        if not filechooser_available or clip_fc is None:
            _show_status("ℹ️ Optional dependency 'ipyfilechooser' is not available. Install it to use Browse buttons.")
            return
        _sync_clip_filechooser_from_text()
        _toggle_box_display(clip_fc_box)

    # ---------------------------------------------------------------------
    # Feature helpers
    # ---------------------------------------------------------------------
    def _daterange_mode_example(mode_value: str):
        if mode_value == "standard":
            return '["2024-04-01", "2024-04-10"]'
        elif mode_value == "seasonal":
            return '["04-01", "10-31"]'
        elif mode_value == "seasonal_years":
            return '{"season": ["04-01", "10-31"], "years": [2019, 2020, 2021]}'
        return '["2024-04-01", "2024-04-10"]'

    def _update_update_daterange_example(force=False):
        new_example = _daterange_mode_example(update_daterange_mode_w.value)
        current = (update_daterange_w.value or "").strip()
        prev_auto = state.get("last_auto_update_daterange_example")

        update_daterange_w.placeholder = new_example
        should_replace = force or (current == "") or (prev_auto is not None and current == prev_auto)
        if should_replace:
            update_daterange_w.value = new_example

        state["last_auto_update_daterange_example"] = new_example

    def _is_str_list_len2(obj):
        return (
            isinstance(obj, (list, tuple))
            and len(obj) == 2
            and all(isinstance(x, str) for x in obj)
        )

    def _validate_date_string(s: str, pattern: str, label: str):
        if not re.match(pattern, s):
            raise ValueError(f"Invalid {label}: '{s}'")

    def _parse_daterange_input(mode: str, text: str):
        s = (text or "").strip()
        if s == "":
            return None

        try:
            obj = ast.literal_eval(s)
        except Exception as e:
            raise ValueError(
                f"Daterange could not be parsed. Please use Python-style list/dict syntax. ({e})"
            )

        if mode == "standard":
            if not _is_str_list_len2(obj):
                raise ValueError('Standard mode expects: ["YYYY-MM-DD", "YYYY-MM-DD"]')
            for d in obj:
                _validate_date_string(d, r"^\d{4}-\d{2}-\d{2}$", "date (YYYY-MM-DD)")
            return list(obj)

        if mode == "seasonal":
            if not _is_str_list_len2(obj):
                raise ValueError('Seasonal mode expects: ["MM-DD", "MM-DD"]')
            for d in obj:
                _validate_date_string(d, r"^\d{2}-\d{2}$", "season date (MM-DD)")
            return list(obj)

        if mode == "seasonal_years":
            if not isinstance(obj, dict):
                raise ValueError(
                    'Seasonal + year control expects a dict, e.g. {"season": ["04-01", "10-31"], "years": [2019, 2020]}'
                )
            if "season" not in obj or "years" not in obj:
                raise ValueError('Seasonal + year control requires keys: "season" and "years"')

            season = obj["season"]
            years = obj["years"]

            if not _is_str_list_len2(season):
                raise ValueError('"season" must be ["MM-DD", "MM-DD"]')
            for d in season:
                _validate_date_string(d, r"^\d{2}-\d{2}$", "season date (MM-DD)")

            valid_years = False
            if years == "all":
                valid_years = True
            elif isinstance(years, str) and re.match(r"^\d{4}-\d{4}$", years):
                valid_years = True
            elif isinstance(years, (list, tuple)) and all(isinstance(y, int) for y in years):
                valid_years = True

            if not valid_years:
                raise ValueError(
                    '"years" must be one of: "all", "YYYY-YYYY", or a list like [2019, 2020, 2021]'
                )

            return {"season": list(season), "years": years}

        raise ValueError(f"Unknown Date Range Mode: {mode}")
    
    
    def _parse_clip_geometry_input(raw_text):
        """
        Returns either:
        - bbox list [xmin, ymin, xmax, ymax] (floats)
        - path string
        - None (if empty)
        """
        s = (raw_text or "").strip()
        if not s:
            return None

        if s.startswith("[") and s.endswith("]"):
            try:
                obj = ast.literal_eval(s)
            except Exception as e:
                raise ValueError(f"Invalid bbox list syntax: {s}") from e

            if not (isinstance(obj, (list, tuple)) and len(obj) == 4):
                raise ValueError("BBox must be a list/tuple with 4 values: [xmin, ymin, xmax, ymax]")

            try:
                vals = [float(v) for v in obj]
            except Exception as e:
                raise ValueError("BBox values must be numeric.") from e

            return vals

        return s

    def _apply_slice_feature(obj):
        """
        Apply time/band slicing to current working result.
        Empty selection means 'keep all' for that dimension.
        Works for DataArray and Dataset if dims exist.
        """
        if obj is None:
            raise ValueError("No current result available.")

        out = obj
        changed = False
        changes = []

        if "time" in out.dims:
            selected_dates = list(slice_time_w.value)
            all_dates = list(slice_time_w.options)
            if len(selected_dates) > 0 and len(selected_dates) < len(all_dates):
                out = out.sel(time=selected_dates)
                changed = True
                changes.append(f"time={len(selected_dates)} scene(s)")
            elif len(selected_dates) == 0:
                changes.append("time=all (empty selection interpreted as no filter)")

        if "band" in out.dims:
            selected_bands = list(slice_band_w.value)
            all_bands = list(slice_band_w.options)
            if len(selected_bands) > 0 and len(selected_bands) < len(all_bands):
                out = out.sel(band=selected_bands)
                changed = True
                changes.append(f"band={len(selected_bands)} band(s)")
            elif len(selected_bands) == 0:
                changes.append("band=all (empty selection interpreted as no filter)")

        return out, changed, changes

    def _apply_cloud_filter_feature(obj):
        """
        Apply cloud coverage filtering using stac2cube.cloud_filter() and existing cloud_percentage coord.
        If current result is Dataset (e.g. after stats), filter Spectral_Temporal_Stack and drop stale stats.
        """
        if not enable_cloud_filter_w.value:
            return obj, False, []

        max_cloud = int(cloud_max_w.value)
        if max_cloud < 0 or max_cloud > 100:
            raise ValueError("Max cloud % must be between 0 and 100.")

        # Dataset case -> filter time series and drop stats
        if isinstance(obj, xr.Dataset):
            if "Spectral_Temporal_Stack" not in obj.data_vars:
                raise ValueError(
                    "Current Dataset does not contain 'Spectral_Temporal_Stack' for cloud filtering."
                )
            da = obj["Spectral_Temporal_Stack"]
            if "time" not in da.dims:
                raise ValueError("Cloud filtering requires a 'time' dimension.")
            if "cloud_percentage" not in da.coords:
                raise ValueError(
                    "Current cube has no 'cloud_percentage' coordinate. "
                    "This feature works only if the cube was already cloud-masked "
                    "(e.g. SCL during generation or probabilistic cloud masking workflow)."
                )

            before_n = int(da.sizes.get("time", 0))
            attrs_ref = dict(getattr(da, "attrs", {}) or {})
            filtered = cloud_filter(da, max_cloud=max_cloud)
            try:
                filtered.attrs.update(attrs_ref)
            except Exception:
                pass
            after_n = int(filtered.sizes.get("time", 0))

            msgs = [
                "cloud_filter applied ",
                f"max_cloud={max_cloud}%",
                f"Scenes kept: {after_n} / {before_n}",
                f"Removed scenes: {max(0, before_n - after_n)}",
                "Previous stats were removed because cloud filtering changes selected time steps.",
            ]
            if after_n == 0:
                msgs.append("Warning: no scenes remain after filtering.")
            return filtered, True, msgs

        # DataArray case (normal)
        if isinstance(obj, xr.DataArray):
            da = obj
            if "time" not in da.dims:
                raise ValueError("Cloud filtering requires a 'time' dimension.")
            if "cloud_percentage" not in da.coords:
                raise ValueError(
                    "Current cube has no 'cloud_percentage' coordinate. "
                    "This feature works only if the cube was already cloud-masked "
                    "(e.g. SCL during generation or probabilistic cloud masking workflow)."
                )

            before_n = int(da.sizes.get("time", 0))
            attrs_ref = dict(getattr(da, "attrs", {}) or {})
            filtered = cloud_filter(da, max_cloud=max_cloud)
            try:
                filtered.attrs.update(attrs_ref)
            except Exception:
                pass
            after_n = int(filtered.sizes.get("time", 0))

            msgs = [
                "cloud_filter applied ",
                f"max_cloud={max_cloud}%",
                f"Scenes kept: {after_n} / {before_n}",
                f"Removed scenes: {max(0, before_n - after_n)}",
            ]
            if after_n == 0:
                msgs.append("Warning: no scenes remain after filtering.")
            return filtered, True, msgs

        raise TypeError(f"Unsupported object type for cloud filtering: {type(obj)}")

    
    def _apply_update_feature(obj):
        """
        Update the loaded cube by requesting a new daterange via:
        - get_stac_layers(update=...) for Spectral_Temporal_Stack cubes
        - get_cloud_layers(update=..., threshold=None) for Cloud_Stack cubes (probability only)
        """
        if not enable_update_w.value:
            return obj, False, []

        loaded_path = state.get("loaded_path")
        if not loaded_path:
            raise ValueError("No loaded cube path available for update.")

        daterange = _parse_daterange_input(update_daterange_mode_w.value, update_daterange_w.value)
        if not daterange:
            raise ValueError("Please provide a daterange for Update Data Cube.")

        # Which kind of cube was loaded?
        loaded_var = state.get("loaded_var")
        if not loaded_var:
            try:
                loaded_var = state.get("loaded_original").name
            except Exception:
                loaded_var = None

        # ------------------------------------------------------------------
        # Cloud cube update (Cloud_Stack) -> cloud probability only
        # ------------------------------------------------------------------
        if loaded_var == "Cloud_Stack":
            # threshold is intentionally None: return probability only
            import inspect

            sig = inspect.signature(get_cloud_layers)
            kwargs = {
                "update": loaded_path,
                "daterange": daterange,
                "threshold": None,
            }
            if "output" in sig.parameters:
                kwargs["output"] = None  # in-memory
            if "q" in sig.parameters:
                kwargs["q"] = True       # silent for GUI

            updated = get_cloud_layers(**kwargs)

            # Normalize to DataArray
            if isinstance(updated, xr.Dataset):
                if "Cloud_Stack" in updated.data_vars:
                    updated = updated["Cloud_Stack"]
                elif len(updated.data_vars) > 0:
                    updated = updated[list(updated.data_vars)[0]]
                else:
                    raise ValueError("Cloud update returned a Dataset with no data variables.")

            if not isinstance(updated, xr.DataArray):
                raise TypeError(f"Cloud update returned unsupported object type: {type(updated)}")

            msgs = [
                "update_data_cube applied (get_cloud_layers(update=...))",
                f"daterange={daterange}",
                "threshold=None (cloud probability only)",
                "Current working result was replaced with the updated Cloud_Stack.",
                "Tip: disable Update now if you want to continue with slicing/clipping/stats on the updated cube.",
            ]
            return updated, True, msgs

        # ------------------------------------------------------------------
        # Spectral cube update (existing behavior)
        # ------------------------------------------------------------------
        loaded_ref = state.get("loaded_original")
        cloud_masking_flag = False
        try:
            cloud_masking_flag = bool(
                loaded_ref is not None and ("cloud_percentage" in loaded_ref.coords)
            )
        except Exception:
            cloud_masking_flag = False

        updated = get_stac_layers(
            update=loaded_path,
            daterange=daterange,
            max_cc=100,
            clip_raster=False,
            cloud_masking=cloud_masking_flag,
            stats=None,
            aggregator=None,
            output=None,  # return in memory
            q=True,       # silent for GUI
        )

        if isinstance(updated, xr.Dataset):
            if "Spectral_Temporal_Stack" not in updated.data_vars:
                raise ValueError("Update returned a Dataset without 'Spectral_Temporal_Stack'.")
            updated = updated["Spectral_Temporal_Stack"]

        if not isinstance(updated, xr.DataArray):
            raise TypeError(f"Update returned unsupported object type: {type(updated)}")

        msgs = [
            "update_data_cube applied (get_stac_layers(update=...))",
            f"daterange={daterange}",
            f"cloud_masking auto-detected from loaded cube: {cloud_masking_flag}",
            "Current working result was replaced with the updated Spectral_Temporal_Stack.",
            "Tip: disable Update now if you want to continue with slicing/clipping/stats on the updated cube.",
        ]
        return updated, True, msgs
    
    
    
    
    def _apply_clip_feature(obj):
        """
        Apply clipping using stac2cube.clip_stac().

        Behavior:
        - If clip checkbox is disabled -> no change
        - If enabled but no clip input -> raises clear error
        - If current is DataArray -> clip directly
        - If current is Dataset with Spectral_Temporal_Stack -> clip time series and
          drop old stats (they become invalid after spatial clip)
        """
        if not enable_clip_w.value:
            return obj, False, []

        geom = _parse_clip_geometry_input(clip_geom_w.value)
        if geom is None:
            raise ValueError("Clip is enabled, but no polygon/bbox was provided.")

        if isinstance(obj, xr.Dataset):
            if "Spectral_Temporal_Stack" not in obj.data_vars:
                raise ValueError(
                    "Current Dataset does not contain 'Spectral_Temporal_Stack' for clipping."
                )
            da = obj["Spectral_Temporal_Stack"]
            clipped = clip_stac(da, polygon=geom)

            msgs = ["clip_raster applied"]
            if isinstance(geom, (list, tuple)):
                msgs.append("Clip input type: bbox list")
            else:
                msgs.append(f"Clip input type: vector file ({Path(str(geom)).name})")
            msgs.append("Previous stats were removed because clipping changes the raster extent.")
            return clipped, True, msgs

        if isinstance(obj, xr.DataArray):
            clipped = clip_stac(obj, polygon=geom)
            msgs = ["clip_raster applied"]
            if isinstance(geom, (list, tuple)):
                msgs.append("Clip input type: bbox list")
            else:
                msgs.append(f"Clip input type: vector file ({Path(str(geom)).name})")
            return clipped, True, msgs

        raise TypeError(f"Unsupported object type for clipping: {type(obj)}")

    def _apply_stats_feature(obj):
        """
        Apply temporal composites using stac2cube.calculate_statistics().
        Returns (new_obj, changed, messages).
        """
        selected = list(stats_select_w.value)
        if not selected:
            return obj, False, []

        da = _pick_timeseries_for_stats(obj)
        if "time" not in da.dims:
            raise ValueError(
                "Temporal composites require a 'time' dimension. "
                "Use 'Reset to loaded cube' if you are currently on a non-temporal result."
            )

        ds_stats = calculate_statistics(da, selected)
        msgs = [
            f"stats={len(selected)} selection(s)",
            "Temporal composites generated (time series + stats).",
            "This should usually be the LAST step before exporting.",
        ]
        return ds_stats, True, msgs

    # ---------------------------------------------------------------------
    # Core callbacks
    # ---------------------------------------------------------------------
    def _on_load_cube_clicked(_):
        path = (load_path_w.value or "").strip()
        if not path:
            _show_status("❌ Please provide a NetCDF file path.")
            return
        if not path.lower().endswith(".nc"):
            _show_status("❌ Please select a NetCDF file (.nc).")
            return
        if not Path(path).exists():
            _show_status(f"❌ File not found: {path}")
            return

        try:
            with status_out:
                clear_output()
                print("Loading data cube from NetCDF...")

                with xr.open_dataset(path) as ds:
                    ds_loaded = ds.load()

                # Accept either cube name
                if "Spectral_Temporal_Stack" in ds_loaded.data_vars:
                    var_name = "Spectral_Temporal_Stack"
                elif "Cloud_Stack" in ds_loaded.data_vars:
                    var_name = "Cloud_Stack"
                else:
                    raise ValueError(
                        "NetCDF does not contain 'Spectral_Temporal_Stack' or 'Cloud_Stack'. "
                        f"Found data_vars: {list(ds_loaded.data_vars)}"
                    )

                loaded = ds_loaded[var_name]

                state["loaded_path"] = path
                state["loaded_var"] = var_name
                state["loaded_original"] = loaded
                state["current"] = _safe_copy_xarray(loaded)

                _show_preview(loaded_summary_out, state["loaded_original"])
                _show_preview(result_out, state["current"])

                _populate_slice_widgets_from_current(select_all=True)
                _set_editor_enabled(True)
                _update_gif_output_suggestion(force=True)
                _update_update_daterange_example(force=True)

                if export_mode_w.value == "netcdf" and not export_target_w.value:
                    export_target_w.value = _auto_netcdf_export_suggestion()

                print(f"✅ Loaded cube: {path}")
                #print("✅ Working object initialized from: Spectral_Temporal_Stack (DataArray)")
                _print_working_note()

            try:
                loaded_summary_acc.selected_index = 0
            except Exception:
                pass
            try:
                result_acc.selected_index = 0
            except Exception:
                pass

        except Exception as e:
            _show_status(f"❌ {type(e).__name__}: {e}")

    def _on_reset_clicked(_):
        if state["loaded_original"] is None:
            _show_status("ℹ️ No loaded cube to reset to yet.")
            return

        state["current"] = _safe_copy_xarray(state["loaded_original"])
        _populate_slice_widgets_from_current(select_all=True)
        _show_preview(result_out, state["current"])

        with status_out:
            clear_output()
            print("✅ Working result reset to original loaded cube.")
            _print_working_note()

        try:
            result_acc.selected_index = 0
        except Exception:
            pass

    def _reset_feature_checkboxes_after_edit():
        # Uncheck feature toggles to prevent accidental re-application
        enable_cloud_filter_w.value = False
        enable_clip_w.value = False
        enable_update_w.value = False

    def _on_edit_clicked(_):
        if state["current"] is None:
            _show_status("❌ Load a cube first.")
            return

        try:
            with status_out:
                clear_output()
                print("Applying editing features to current working result...")

                current_obj = state["current"]
                changed_any = False
                messages = []

                if enable_update_w.value:
                    #print("ℹ️ Update Data Cube is enabled.")
                    #print("ℹ️ In this run, other feature selections are ignored and the working result will be replaced from the loaded cube path.")
                    current_obj, changed_update, update_msgs = _apply_update_feature(current_obj)
                    changed_any = changed_any or changed_update
                    messages.extend(update_msgs)
                else:
                    # 1) Slice
                    current_obj, changed_slice, slice_msgs = _apply_slice_feature(current_obj)
                    changed_any = changed_any or changed_slice
                    messages.extend(slice_msgs)

                    # 2) Filter by Cloud Coverage
                    current_obj, changed_cloud, cloud_msgs = _apply_cloud_filter_feature(current_obj)
                    changed_any = changed_any or changed_cloud
                    messages.extend(cloud_msgs)

                    # 3) Clip Raster
                    current_obj, changed_clip, clip_msgs = _apply_clip_feature(current_obj)
                    changed_any = changed_any or changed_clip
                    messages.extend(clip_msgs)

                    # 4) Temporal composites (stats)
                    current_obj, changed_stats, stats_msgs = _apply_stats_feature(current_obj)
                    changed_any = changed_any or changed_stats
                    messages.extend(stats_msgs)

                state["current"] = current_obj

                # Refresh UI from updated current cube
                _populate_slice_widgets_from_current(select_all=True)
                _show_preview(result_out, state["current"])

                if changed_any:
                    print("✅ Edit finished.")
                    if messages:
                        print("Applied:")
                        for m in messages:
                            print(f"- {m}")
                else:
                    print("✅ Edit finished (no changes applied).")
                    print(
                        "Tip: select a subset of dates/bands, enable cloud filter, enable clip with a geometry, "
                        "and/or choose statistics before clicking 'Edit data cube'."
                    )
                
                _reset_feature_checkboxes_after_edit()

                _print_working_note()

                # Optional auto-export if export mode+target already selected
                mode = export_mode_w.value
                target = None if export_target_w.disabled else ((export_target_w.value or "").strip() or None)
                if mode != "lazy" and target:
                    print("")
                    print("Export mode is set and output path is provided.")
                    print("Exporting current result...")
                    info = _export_current_result()
                    state["last_export_info"] = info
                    if info.get("mode") != "netcdf":
                        print(f"✅ Export finished: {info['target']}")
                    else:
                        print("✅ Export finished.")

            try:
                result_acc.selected_index = 0
            except Exception:
                pass

        except Exception as e:
            _show_status(f"❌ {type(e).__name__}: {e}")

    def _on_export_current_clicked(_):
        if state["current"] is None:
            _show_status("❌ No current result available. Load and/or edit a cube first.")
            return

        try:
            with status_out:
                clear_output()
                print("Exporting current result...")
                info = _export_current_result()
                state["last_export_info"] = info

                # export_stac() already prints "Export is done: ..."
                if info.get("mode") != "netcdf":
                    print(f"✅ Export finished: {info['target']}")

                _print_working_note()

        except Exception as e:
            _show_status(f"❌ {type(e).__name__}: {e}")

    def _on_viz_dropdown_clicked(_):
        if state["current"] is None:
            with viz_out:
                clear_output()
                print("ℹ️ Load a cube first.")
            return

        try:
            da = _pick_dataarray_for_visualization(state["current"])
            with viz_out:
                clear_output()
                if isinstance(state["current"], xr.Dataset) and da.name != "Spectral_Temporal_Stack":
                    print(f"ℹ️ Visualizing dataset variable: {da.name}")
                print("Launching interactive viewer...")
                out = interactive_time_view(stac=da, widget_type="dropdown")
                if out is not None:
                    display(out)
        except Exception as e:
            with viz_out:
                clear_output()
                print(f"❌ Visualization error: {type(e).__name__}: {e}")

    def _on_make_gif_clicked(_):
        if state["current"] is None:
            with viz_out:
                clear_output()
                print("ℹ️ Load a cube first.")
            return

        try:
            da = _pick_dataarray_for_visualization(state["current"])

            if "time" not in da.dims:
                raise ValueError(
                    f"Animation generation requires a 'time' dimension. Found dims: {da.dims}. "
                    "If you are viewing a stats-only variable, use 'Reset to loaded cube' "
                    "or visualize 'Spectral_Temporal_Stack'."
                )

            out_path = (gif_out_path_w.value or "").strip()
            if not out_path:
                raise ValueError("Please provide a GIF output path.")
            if not out_path.lower().endswith(".gif"):
                out_path += ".gif"
                gif_out_path_w.value = out_path

            fps_val = int(gif_fps_w.value)
            if fps_val <= 0:
                raise ValueError("FPS must be > 0.")

            Path(out_path).parent.mkdir(parents=True, exist_ok=True)

            with viz_out:
                clear_output()
                print("Generating animation GIF...")
                save_timeseries_gif(
                    da=da,
                    out_path=out_path,
                    display_mode=gif_display_mode_w.value,
                    fps=fps_val,
                    label=gif_label_w.value,
                )
                print(f"✅ Animation saved: {out_path}")

        except Exception as e:
            with viz_out:
                clear_output()
                print(f"❌ Animation error: {type(e).__name__}: {e}")

    # ---------------------------------------------------------------------
    # Small selection helper callbacks
    # ---------------------------------------------------------------------
    def _select_all_dates(_):
        slice_time_w.value = tuple(slice_time_w.options)

    def _clear_dates(_):
        slice_time_w.value = ()

    def _select_all_bands(_):
        slice_band_w.value = tuple(slice_band_w.options)

    def _clear_bands(_):
        slice_band_w.value = ()

    def _select_all_stats(_):
        stats_select_w.value = tuple(stats_select_w.options)

    def _clear_stats(_):
        stats_select_w.value = ()

    # ---------------------------------------------------------------------
    # Observe / wire
    # ---------------------------------------------------------------------
    browse_load_btn.on_click(_on_browse_load_clicked)
    browse_export_btn.on_click(_on_browse_export_clicked)
    browse_gif_btn.on_click(_on_browse_gif_clicked)
    browse_clip_btn.on_click(_on_browse_clip_clicked)

    load_cube_btn.on_click(_on_load_cube_clicked)
    reset_btn.on_click(_on_reset_clicked)
    edit_btn.on_click(_on_edit_clicked)
    export_current_btn.on_click(_on_export_current_clicked)

    slice_time_all_btn.on_click(_select_all_dates)
    slice_time_clear_btn.on_click(_clear_dates)
    slice_band_all_btn.on_click(_select_all_bands)
    slice_band_clear_btn.on_click(_clear_bands)

    stats_all_btn.on_click(_select_all_stats)
    stats_clear_btn.on_click(_clear_stats)

    viz_dropdown_btn.on_click(_on_viz_dropdown_clicked)
    viz_make_gif_btn.on_click(_on_make_gif_clicked)

    export_mode_w.observe(lambda change: _set_export_mode_defaults(), names="value")
    gif_display_mode_w.observe(lambda change: _update_gif_output_suggestion(), names="value")
    update_daterange_mode_w.observe(lambda change: _update_update_daterange_example(), names="value")

    # ---------------------------------------------------------------------
    # Layout helpers
    # ---------------------------------------------------------------------
    def _stacked_field(widget, label_text):
        try:
            widget.description = ""
        except Exception:
            pass
        try:
            widget.style.description_width = "0px"
        except Exception:
            pass

        label = widgets.HTML(
            f"<div style='font-weight:500; line-height:1.2; margin:0; padding:0;'>{label_text}:</div>"
        )
        return widgets.VBox([label, widget], layout=widgets.Layout(width="100%", gap="4px"))

    # ---------------------------------------------------------------------
    # Build layout
    # ---------------------------------------------------------------------
    header = widgets.HTML(
        "<div style='margin:0 0 4px 0; font-size:28px; font-weight:700;'>Data Cube Editor</div>"
    )

    subtitle = widgets.HTML(
        "<div style='font-size:13px; color:#6b7280; margin:0 0 4px 0;'>"
        "Load a data cube -> select editing feature(s) -> edit data cube -> inspect the result -> export current result."
        "</div>"
    )

    # Loading section
    load_input_row = widgets.HBox(
        [browse_load_btn, load_path_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    load_input_box = widgets.VBox(
        [load_input_row, load_fc_box],
        layout=widgets.Layout(width="100%", gap="4px"),
    )

    loading_box = widgets.VBox(
        [
            widgets.HTML("<b>Loading</b>"),
            widgets.HTML("<div style='font-size:12px; color:#666;'>NetCDF only (Geotiffs are not supported as editor input).</div>"),
            _stacked_field(load_input_box, "Data cube path"),
            widgets.HBox([load_cube_btn, reset_btn], layout=widgets.Layout(gap="8px", flex_flow="row wrap")),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    loaded_summary_box = widgets.VBox([loaded_summary_out], layout=widgets.Layout(width="100%"))
    loaded_summary_acc = widgets.Accordion(children=[loaded_summary_box], selected_index=None)
    loaded_summary_acc.set_title(0, "Loaded data cube")
    loaded_summary_acc.layout = widgets.Layout(width="100%")

    # Slice feature
    slice_time_box = widgets.VBox(
        [
            _stacked_field(slice_time_w, "Dates"),
            widgets.HBox([slice_time_all_btn, slice_time_clear_btn], layout=widgets.Layout(gap="6px")),
        ],
        layout=widgets.Layout(width="50%", gap="6px"),
    )

    slice_band_box = widgets.VBox(
        [
            _stacked_field(slice_band_w, "Bands"),
            widgets.HBox([slice_band_all_btn, slice_band_clear_btn], layout=widgets.Layout(gap="6px")),
        ],
        layout=widgets.Layout(width="50%", gap="6px"),
    )

    slice_feature_box = widgets.VBox(
        [
            #widgets.HTML("<b>Slice Data Cube</b>"),
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Select a subset of dates and/or bands.</b>. "
                "</div>"
            ),
            slice_time_box,
            slice_band_box,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )

    slice_acc = widgets.Accordion(children=[slice_feature_box], selected_index=None)
    slice_acc.set_title(0, "Slice Data Cube")
    slice_acc.layout = widgets.Layout(width="99%")

    # Cloud filter feature (NEW)
    cloud_filter_controls = widgets.VBox(
        [
            enable_cloud_filter_w,
            _stacked_field_with_help(cloud_max_w, "Max cloud %", "cloud_filter"),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    cloud_filter_feature_box = widgets.VBox(
        [
            #widgets.HTML("<b>Filter by Cloud Coverage</b>"),
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "This is only possible if the data cube is already cloud-masked. "
                "(either with SCL during data cube generation or masked by a cloud data cube)"
                "</div>"
            ),
            cloud_filter_controls,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )

    cloud_filter_acc = widgets.Accordion(children=[cloud_filter_feature_box], selected_index=None)
    cloud_filter_acc.set_title(0, "Filter by Cloud Coverage")
    cloud_filter_acc.layout = widgets.Layout(width="99%")

    # Clip feature
    clip_input_row = widgets.HBox(
        [browse_clip_btn, clip_geom_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )

    clip_input_box = widgets.VBox(
        [clip_input_row, clip_fc_box],
        layout=widgets.Layout(width="100%", gap="4px"),
    )

    clip_controls_box = widgets.VBox(
        [
            enable_clip_w,
            _stacked_field_with_help(clip_input_box, "Polygon / BBOX", "clip_raster"),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    clip_feature_box = widgets.VBox(
        [
            #widgets.HTML("<b>Clip Raster</b>"),
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Clip data cube (clip raster) by providing a vector file path or a WGS84 bbox list."
                "</div>"
            ),
            clip_controls_box,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )

    clip_acc = widgets.Accordion(children=[clip_feature_box], selected_index=None)
    clip_acc.set_title(0, "Clip Raster")
    clip_acc.layout = widgets.Layout(width="99%")

    # Temporal Composites (stats) feature
    stats_inner_widget = widgets.VBox(
        [
            stats_select_w,
            widgets.HBox([stats_all_btn, stats_clear_btn], layout=widgets.Layout(gap="6px")),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    stats_feature_box = widgets.VBox(
        [
            #widgets.HTML("<b>Temporal Composites</b>"),
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Select statistics to build temporal composites. <br>"
                "<b>This should be the last step before exporting, because it creates extra composite layers besides the original data cube.</b>"
                "</div>"
            ),
            _stacked_field_with_help(stats_inner_widget, "Stats", "stats"),
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )

    stats_acc = widgets.Accordion(children=[stats_feature_box], selected_index=None)
    stats_acc.set_title(0, "Temporal Composites")
    stats_acc.layout = widgets.Layout(width="99%")



    # Update Data Cube feature
    update_feature_box = widgets.VBox(
        [
            #widgets.HTML("<b>Update Data Cube</b>"),
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Fetch missing dates for the loaded date cube in the given date range. "
                "Instead of re-building the entire data cube, it only computes the new scenes.<br>"
                "<b>This feature is recommended to be used alone without in sequence with other features.</b>"
                "</div>"
            ),
            widgets.VBox(
                [
                    enable_update_w,
                    _stacked_field_with_help(update_daterange_mode_w, "Date Range Mode", "daterange_mode"),
                    _stacked_field(update_daterange_w, "Daterange"),
                ],
                layout=widgets.Layout(width="100%", gap="6px"),
            ),
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )

    update_acc = widgets.Accordion(children=[update_feature_box], selected_index=None)
    update_acc.set_title(0, "Update Data Cube")
    update_acc.layout = widgets.Layout(width="99%")




    # Export Options
    export_input_row = widgets.HBox(
        [browse_export_btn, export_target_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    export_input_box = widgets.VBox(
        [export_input_row, export_fc_box],
        layout=widgets.Layout(width="100%", gap="4px"),
    )

    export_box = widgets.VBox(
        [
            #widgets.HTML("<b>Export Options</b>"),
            widgets.HTML("<div style='font-size:12px; color:#666;'>Exports the current result in the desired format.</div>"),
            _stacked_field(export_mode_w, "Export mode"),
            _stacked_field(export_input_box, "Output"),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    export_acc = widgets.Accordion(children=[export_box], selected_index=None)
    export_acc.set_title(0, "Export Options")
    export_acc.layout = widgets.Layout(width="99%")

    # Features group
    features_box = widgets.VBox(
        [
            widgets.HTML("<b>Features</b>"),
            widgets.HTML("<div style='font-size:12px; color:#666;'>Multiple features can be selected before editing the data cube.</div>"),
            #widgets.HTML("<div style='font-size:12px; color:#666;'>- Do not forget to uncheck boxes after editing a data cube to prevent.</div>"),
            slice_acc,
            cloud_filter_acc,
            clip_acc,
            stats_acc,
            update_acc,
            export_acc,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )

    # Actions
    action_row = widgets.HBox(
        [edit_btn, export_current_btn],
        layout=widgets.Layout(gap="8px", flex_flow="row wrap"),
    )

    # Result accordion
    result_box = widgets.VBox([result_out], layout=widgets.Layout(width="100%"))
    result_acc = widgets.Accordion(children=[result_box], selected_index=None)
    result_acc.set_title(0, "Result")
    result_acc.layout = widgets.Layout(width="100%")

    # Visualization accordion (at the end)
    gif_input_row = widgets.HBox(
        [browse_gif_btn, gif_out_path_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    gif_input_box = widgets.VBox(
        [gif_input_row, gif_fc_box],
        layout=widgets.Layout(width="100%", gap="4px"),
    )

    visualization_box = widgets.VBox(
        [
            #widgets.HTML("<b>Visualization</b>"),
            #widgets.HTML("<div style='font-size:12px; color:#666;'>Available after loading a cube. For GIFs, a time dimension is required.</div>"),
            widgets.VBox(
                [
                    widgets.HTML("<b>1) Interactive View</b>"),
                    #widgets.HTML("<div style='font-size:12px; color:#666;'>Dropdown mode.</div>"),
                    viz_dropdown_btn,
                ],
                layout=widgets.Layout(width="100%", gap="6px"),
            ),
            viz_out,
            widgets.VBox(
                [
                    widgets.HTML("<b>2) Animation (export only)</b>"),
                    _stacked_field(gif_display_mode_w, "Display mode"),
                    _stacked_field_with_help(gif_fps_w, "FPS", "fps"),
                    _stacked_field_with_help(gif_label_w, "Label", "gif_label"),
                    _stacked_field(gif_input_box, "Output GIF"),
                    viz_make_gif_btn,
                ],
                layout=widgets.Layout(width="100%", gap="6px"),
            ),
            
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )

    viz_acc = widgets.Accordion(children=[visualization_box], selected_index=None)
    viz_acc.set_title(0, "Visualization")
    viz_acc.layout = widgets.Layout(width="100%")

    # Spacers
    spacer_after_loaded = widgets.HTML("<div style='height:10px;'></div>")
    spacer_after_buttons = widgets.HTML("<div style='height:8px;'></div>")

    # --- NEW: wrap sections into cards (layout only) ---
    loading_card = widgets.VBox([loading_box], layout=widgets.Layout(width="100%"))
    loading_card.add_class("stac2cube-card")

    loaded_summary_card = widgets.VBox([loaded_summary_acc], layout=widgets.Layout(width="100%"))
    loaded_summary_card.add_class("stac2cube-card")

    features_card = widgets.VBox(
        [features_box, widgets.HTML("<div style='height:6px;'></div>"), action_row],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    features_card.add_class("stac2cube-card")

    result_card = widgets.VBox([result_acc], layout=widgets.Layout(width="100%"))
    result_card.add_class("stac2cube-card")

    viz_card = widgets.VBox([viz_acc], layout=widgets.Layout(width="100%"))
    viz_card.add_class("stac2cube-card")

    status_card = widgets.VBox(
        [widgets.HTML("<b>Status</b>"), status_out],
        layout=widgets.Layout(width="100%", gap="6px"),
    )
    status_card.add_class("stac2cube-card")


    # Main UI
    # Spacers (keep them simple)
    spacer_small = widgets.HTML("<div style='height:6px;'></div>")
    spacer_med = widgets.HTML("<div style='height:12px;'></div>")

    ui = widgets.VBox(
        [
            header,
            subtitle,

            loading_card,
            spacer_small,
            loaded_summary_card,

            spacer_med,
            features_card,

            spacer_med,
            result_card,

            spacer_med,
            viz_card,

            spacer_med,
            status_card,
        ],
        layout=widgets.Layout(
            width="50%",
            max_width="980px",
            margin="0 auto",
            gap="0px",
        ),
    )

    ui.add_class("stac2cube-root")

    # ---------------------------------------------------------------------
    # Initialize + styling  (inject CSS BEFORE display)
    # ---------------------------------------------------------------------
    display(
        widgets.HTML(
            """
            <style>
            .stac2cube-help-btn button {
                border-radius: 999px !important;
                font-weight: 700 !important;
                line-height: 1 !important;
                border: 1px solid #93c5fd !important;
            }

            /* Root: kill tiny horizontal overflow */
            .stac2cube-root {
                overflow-x: hidden !important;
            }

            /* Cards */
            .stac2cube-card {
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                padding: 12px;
                background: #fafbfc;
                box-shadow: 0 1px 2px rgba(0,0,0,0.04);
                box-sizing: border-box;
                overflow-x: hidden;
                min-width: 0;
            }

            /* Prevent accordion panels and nested widget boxes from overflowing */
            .stac2cube-card .p-Accordion-child,
            .stac2cube-card .p-Accordion-child > .p-Widget,
            .stac2cube-card .widget-vbox,
            .stac2cube-card .widget-hbox {
                min-width: 0 !important;
                max-width: 100% !important;
                box-sizing: border-box !important;
            }

            /* Constrain text inputs, dropdowns, and select widgets inside cards */
            .stac2cube-card .widget-text input,
            .stac2cube-card .widget-dropdown select,
            .stac2cube-card .widget-select-multiple select {
                min-width: 0 !important;
                max-width: 100% !important;
                box-sizing: border-box !important;
                overflow-x: hidden !important;
            }

            /* Outputs: avoid xarray repr forcing horizontal scrollbars */
            .stac2cube-root .widget-output {
                overflow-x: hidden !important;
            }
            .stac2cube-root .widget-output pre {
                white-space: pre-wrap !important;
                overflow-wrap: anywhere !important;
            }
            </style>
            """
        )
    )

    outer = widgets.HBox([ui], layout=widgets.Layout(width="100%", justify_content="center"))

    _set_editor_enabled(False)
    _show_status("ℹ️ Load a NetCDF cube to start editing.")
    _update_gif_output_suggestion(force=True)
    _update_update_daterange_example(force=True)

    display(outer)

    return {
        "ui": ui,
        "outer": outer,
        "state": state,
        "widgets": {
            "load_path": load_path_w,
            "load_cube_btn": load_cube_btn,
            "reset_btn": reset_btn,
            "slice_time": slice_time_w,
            "slice_band": slice_band_w,
            "enable_cloud_filter": enable_cloud_filter_w,
            "cloud_max": cloud_max_w,
            "enable_clip": enable_clip_w,
            "clip_geom": clip_geom_w,
            "stats_select": stats_select_w,
            "edit_btn": edit_btn,
            "enable_update": enable_update_w,
            "update_daterange_mode": update_daterange_mode_w,
            "update_daterange": update_daterange_w,
            "export_mode": export_mode_w,
            "export_target": export_target_w,
            "export_current_btn": export_current_btn,
            "viz_dropdown_btn": viz_dropdown_btn,
            "gif_display_mode": gif_display_mode_w,
            "gif_fps": gif_fps_w,
            "gif_label": gif_label_w,
            "gif_out_path": gif_out_path_w,
            "viz_make_gif_btn": viz_make_gif_btn,
        },
        "outputs": {
            "loaded_summary": loaded_summary_out,
            "result": result_out,
            "status": status_out,
            "visualization": viz_out,
        },
    }


def ard_cube_tools():
    
    xr.set_options(
        display_expand_data=False,
        display_expand_coords=True,
        display_expand_attrs=False,
        display_expand_data_vars=True,
    )

    # -----------------------------------------
    # CSS (card design)
    # -----------------------------------------
    css_patch = widgets.HTML(
        """
        <style>
        .stac2cube-card {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 12px;
            background: #fafbfc;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        button.stac2cube-help-btn,
        button.stac2cube-help-btn:hover,
        button.stac2cube-help-btn:active,
        button.stac2cube-help-btn:focus {
        border-radius: 9999px !important;
        width: 18px !important;
        min-width: 18px !important;
        height: 18px !important;
        min-height: 18px !important;
        padding: 0 !important;
        background: #2D7FF9 !important;
        color: #ffffff !important;
        border: 0 !important;
        outline: none !important;
        box-shadow: none !important;

        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        line-height: 18px !important;
        font-weight: 700 !important;
        }
        </style>
        """
    )

    # -----------------------------------------
    # State
    # -----------------------------------------
    state = {
        "loaded_path": None,
        "loaded_obj": None,          
        "current_result_path": None, 
    }

    # -----------------------------------------
    # Outputs
    # -----------------------------------------
    loaded_summary_out = widgets.Output(layout=widgets.Layout(width="99%", max_height="420px", overflow="auto"))
    result_out = widgets.Output(layout=widgets.Layout(width="99%", max_height="420px", overflow="auto"))
    status_out = widgets.Output(layout=widgets.Layout(width="99%", max_height="260px", overflow="auto"))

    def _status(*lines, append=False):
        with status_out:
            if not append:
                clear_output()
            for ln in lines:
                print(ln)

    def _show_loaded_summary(obj):
        with loaded_summary_out:
            clear_output()
            display(obj)

    def _show_result_from_path(nc_path: str):
        with result_out:
            clear_output()
            if not nc_path:
                print("No exported result yet.")
                return

            p = Path(nc_path)
            if not p.exists():
                print(f"Exported file not found: {p.as_posix()}")
                return

            print(f"Exported file: {p.as_posix()}")
            with xr.open_dataset(p) as ds:
                if "Spectral_Temporal_Stack" in ds.data_vars:
                    display(ds["Spectral_Temporal_Stack"])
                elif "Cloud_Stack" in ds.data_vars:
                    display(ds["Cloud_Stack"])
                else:
                    display(ds)

        try:
            result_acc.selected_index = 0
        except Exception:
            pass

    # -----------------------------------------
    # File chooser helper (optional)
    # -----------------------------------------
    def _guess_dir_from_text(text_value: str) -> str:
        s = (text_value or "").strip()
        if not s:
            return os.getcwd()
        p = Path(s).expanduser()
        if p.is_dir():
            return str(p)
        if p.parent.exists():
            return str(p.parent)
        return os.getcwd()

    def _attach_filechooser(
        browse_btn: widgets.Button,
        text_widget: widgets.Text,
        title: str,
        pattern=None,
        select_dirs: bool = False,
    ):
        """
        Folder icon toggles a FileChooser under the textbox.
        Selecting a file/folder immediately writes into the textbox and closes the chooser.
        (No 'Use selected' / 'Close' buttons.)
        """
        fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))

        if FileChooser is None:
            browse_btn.disabled = True
            browse_btn.tooltip = "Install ipyfilechooser (pip install ipyfilechooser) or type the path manually."
            return fc_box

        def _toggle(_):
            if fc_box.layout.display == "none":
                start_dir = _guess_dir_from_text(text_widget.value)
                fc = FileChooser(start_dir)
                fc.title = f"<b>{title}</b>"
                fc.use_dir_icons = True
                fc.show_only_dirs = bool(select_dirs)
                if pattern is not None:
                    fc.filter_pattern = pattern

                def _on_select(_chooser):
                    chosen = fc.selected_path if select_dirs else fc.selected
                    if chosen:
                        text_widget.value = str(Path(chosen)).replace("\\", "/")
                        fc_box.layout.display = "none"

                # Auto-apply selection
                try:
                    fc.register_callback(_on_select)
                except Exception:
                    pass

                fc_box.children = [fc]
                fc_box.layout.display = ""
            else:
                fc_box.layout.display = "none"

        browse_btn.on_click(_toggle)
        return fc_box

    # -----------------------------------------
    # Naming suggestions
    # -----------------------------------------
    def _stem_from_loaded():
        if state["loaded_path"]:
            return Path(state["loaded_path"]).stem
        return "cube"

    def _dir_from_loaded():
        if state["loaded_path"]:
            return Path(state["loaded_path"]).parent
        return Path("./results")

    def _suggest_masked_path(threshold: int):
        base = _stem_from_loaded()
        outdir = _dir_from_loaded()
        return (outdir / f"{base}_masked_{int(threshold)}.nc").as_posix()

    def _suggest_cr_path():
        base = _stem_from_loaded()
        outdir = _dir_from_loaded()
        return (outdir / f"{base}_cr.nc").as_posix()

    def _suggest_sr_path():
        base = _stem_from_loaded()
        outdir = _dir_from_loaded()
        return (outdir / f"{base}_sr.nc").as_posix()
    
    def _suggest_clouds_path_from_loaded():
        p = Path(state["loaded_path"])
        return (p.parent / f"{p.stem}_cloud.nc").as_posix()

    # -----------------------------------------
    # Header
    # -----------------------------------------
    header = widgets.HTML("<div style='margin:0 0 4px 0; font-size:28px; font-weight:700;'>Analysis Ready Data Cube Tools</div>")
    subtitle = widgets.HTML(
        "<div style='font-size:13px; color:#6b7280; margin:0 0 4px 0;'>"
        "• Cloud masking • Co-registration • Super-resolution"
        "</div>"
    )

    # -----------------------------------------
    # Loading card (same pattern as editor)
    # -----------------------------------------
    load_path_w = widgets.Text(value="./results/test.nc", layout=widgets.Layout(width="100%"))
    browse_load_btn = widgets.Button(icon="folder-open", description="", layout=widgets.Layout(width="36px"))
    load_cube_btn = widgets.Button(description="Load cube", button_style="primary", icon="upload", layout=widgets.Layout(width="140px"))
    #reset_btn = widgets.Button(description="Reset to loaded cube", icon="undo", layout=widgets.Layout(width="180px"), disabled=True)

    load_fc_box = _attach_filechooser(browse_load_btn, load_path_w, title="Select NetCDF file", pattern=["*.nc", "*"], select_dirs=False)




    COREG_HELP = {
        "grid_size": """
    <b>grid_size</b><br>
    The strength of the area scan. The higher, the longer it takes, but it scans more potential matching areas.<br>
    If the current setup still removes scenes with low cloud percentages, try increasing it.<br>
    Increasing grid size does not guarantee that more scenes will be kept, but it can help in some cases.
    """,
        "max_cc": """
    <b>max_cc</b><br>
    Maximum cloud percentage of scenes (from cloud-masked data cube; either SCL or s2cloudless). Scenes beyond this threshold are excluded.<br>
    The algorithm already detects some cloudy scenes that cannot be co-registered and automatically deletes them from the time series.<br>
    However, filtering cloudy scenes can sometimes improve the performance of the co-registration, especially in highly cloud covered regions.
    """,
        "time_period": """
    <b>time_period</b><br>
    Selection of the time range: <code>["YYYY-MM-DD", "YYYY-MM-DD"]</code>.<br>
    The co-registration is performed on the selected time range.<br> 
    It can be useful to exclude problematic surfaces for co-registration algorithm (e.g. snow & ice).
    """,
        "min_reliability_keep": """
    <b>min_reliability_keep</b><br>
    Threshold for the co-registration reliability score (%). Scenes with a score lower than this value are dropped.<br>
    Very low scores often indicate highly cloudy scenes.
    """,
        "min_reliability_update_ref": """
    <b>min_reliability_update_ref</b><br>
    Threshold for the co-registration reliability score (%). Scenes with a score lower than this value are kept,<br>
    but the algorithm will not select them as reference for the co-registration of the next scene.
    """,
        "max_cloud_update_ref": """
    <b>max_cloud_update_ref</b><br>
    Maximum cloud percentage for selecting a scene as reference. Scenes above this threshold will not be selected as reference<br>
    for the co-registration of the following scene.
    """,
        "first_scene_mode": """
    <b>first_scene_mode</b><br>
    Mode for selecting the first reference in the time series (crucial for the rest).<br>
    <code>first</code> selects the first scene; <code>composite</code> creates a composite of the first <code>composite_window_days</code> days and selects the median.<br>
    Use <code>first</code> if the first scene is cloud-free; otherwise <code>composite</code> is more robust.
    """,
        "composite_window_days": """
    <b>composite_window_days</b><br>
    Days used for composite if <code>first_scene_mode="composite"</code>.<br>
    Example: first scene 2020-01-15 and <code>composite_window_days=30</code> → median of scenes from 2020-01-15 to 2020-02-15.
    """,
        "iteration": """
    <b>iteration</b><br>
    Number of iterations to run co-registration. Default 1; 4–5 is usually enough for good results.<br>
    If <code>first_scene_mode="composite"</code>, it switches to <code>first</code> after the first iteration.
    """,
    }

    def _make_help_btn(help_html: str):
        btn = widgets.Button(description="?", layout=widgets.Layout(width="18px", height="18px", padding="0px"))
        btn.add_class("stac2cube-help-btn")

        # Optional: keep (doesn't affect circle, but reinforces colors)
        btn.style.button_color = "#2D7FF9"
        btn.style.text_color = "white"
        btn.style.font_weight = "700"

        help_box = widgets.HTML(
            f"<div style='font-size:12px; color:#444; margin-top:4px; display:none;'>{help_html}</div>"
        )

        def _toggle(_):
            cur = help_box.value
            if "display:none" in cur:
                help_box.value = cur.replace("display:none", "display:block")
            else:
                help_box.value = cur.replace("display:block", "display:none")

        btn.on_click(_toggle)
        return btn, help_box

    def _stacked_field_with_help(widget, label_text: str, help_key: str):
        try:
            widget.description = ""
        except Exception:
            pass
        try:
            widget.style.description_width = "0px"
        except Exception:
            pass

        help_btn, help_box = _make_help_btn(COREG_HELP[help_key])
        label_row = widgets.HBox(
            [
                widgets.HTML(f"<div style='font-weight:500; line-height:1.2; margin:0; padding:0;'>{label_text}:</div>"),
                help_btn,
            ],
            layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
        )
        return widgets.VBox([label_row, widget, help_box], layout=widgets.Layout(width="100%", gap="4px"))






    def _stacked_field(widget, label_text):
        label = widgets.HTML(f"<div style='font-weight:500; line-height:1.2; margin:0; padding:0;'>{label_text}:</div>")
        return widgets.VBox([label, widget], layout=widgets.Layout(width="100%", gap="4px"))

    load_input_row = widgets.HBox([browse_load_btn, load_path_w], layout=widgets.Layout(width="100%", gap="6px", align_items="center"))
    load_input_box = widgets.VBox([load_input_row, load_fc_box], layout=widgets.Layout(width="100%", gap="4px"))

    loading_box = widgets.VBox(
        [
            widgets.HTML("<b>Loading</b>"),
            widgets.HTML("<div style='font-size:12px; color:#666;'>NetCDF only (COGs are not supported as input).</div>"),
            _stacked_field(load_input_box, "Data cube path"),
            widgets.HBox([load_cube_btn], layout=widgets.Layout(gap="8px", flex_flow="row wrap")),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )
    loading_card = widgets.VBox([loading_box], layout=widgets.Layout(width="99%"))
    loading_card.add_class("stac2cube-card")

    # Loaded cube accordion (same as editor)
    loaded_summary_box = widgets.VBox([loaded_summary_out], layout=widgets.Layout(width="100%"))
    loaded_summary_acc = widgets.Accordion(children=[loaded_summary_box], selected_index=None)
    loaded_summary_acc.set_title(0, "Loaded data cube")
    loaded_summary_acc.layout = widgets.Layout(width="99%")
    loaded_summary_card = widgets.VBox([loaded_summary_acc], layout=widgets.Layout(width="100%"))
    loaded_summary_card.add_class("stac2cube-card")

    # -----------------------------------------
    # Tools card (3 tool accordions + separate buttons)
    # -----------------------------------------

    # --- Tool 1: Cloud Masking Data Cube (a) Fully Automated Workflow) ---
    # NOTE: threshold + outputs live inside sub-accordion (a)

    # Widgets used in sub-accordion (a)
    mask_threshold_w = widgets.BoundedIntText(value=70, min=0, max=100, step=1, layout=widgets.Layout(width="120px"))

    export_clouds_w = widgets.Checkbox(
        value=False,
        description="",
        indent=False,
        layout=widgets.Layout(width="22px", min_width="22px"),
    )
    try:
        export_clouds_w.style.description_width = "0px"
    except Exception:
        pass

    export_clouds_label = widgets.HTML(
        "<div style='font-size:13px; line-height:1.2; white-space:normal;'>"
        "Also export cloud probability layers (recommended)"
        "</div>"
    )

    export_clouds_row = widgets.HBox(
        [export_clouds_w, export_clouds_label],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )

    clouds_out_w = widgets.Text(value="", layout=widgets.Layout(width="100%"), disabled=True)
    browse_clouds_out_btn = widgets.Button(icon="folder-open", description="", layout=widgets.Layout(width="36px"))
    browse_clouds_out_btn.disabled = True
    clouds_out_fc_box = _attach_filechooser(
        browse_clouds_out_btn,
        clouds_out_w,
        title="Select output NetCDF for cloud probability layers",
        pattern=["*.nc", "*"],
        select_dirs=False,
    )
    clouds_out_row = widgets.HBox(
        [browse_clouds_out_btn, clouds_out_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    clouds_out_box = widgets.VBox([clouds_out_row, clouds_out_fc_box], layout=widgets.Layout(width="100%", gap="4px"))

    masked_out_w = widgets.Text(value="", layout=widgets.Layout(width="100%"))
    browse_masked_out_btn = widgets.Button(icon="folder-open", description="", layout=widgets.Layout(width="36px"))
    masked_out_fc_box = _attach_filechooser(
        browse_masked_out_btn,
        masked_out_w,
        title="Select output NetCDF for masked cube",
        pattern=["*.nc", "*"],
        select_dirs=False,
    )
    masked_out_row = widgets.HBox(
        [browse_masked_out_btn, masked_out_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    masked_out_box = widgets.VBox([masked_out_row, masked_out_fc_box], layout=widgets.Layout(width="100%", gap="4px"))

    mask_and_export_btn = widgets.Button(
        description="Mask and Export",
        button_style="success",
        icon="play",
        layout=widgets.Layout(width="170px"),
    )

    def _suggest_clouds_path():
        base = _stem_from_loaded()
        outdir = _dir_from_loaded()
        return (outdir / f"{base}_cloud.nc").as_posix()

    def _refresh_mask_outputs(force=False):
        # Always suggest masked output based on threshold (unless user already typed a custom one and force=False)
        suggested_masked = _suggest_masked_path(int(mask_threshold_w.value))
        if force or (not masked_out_w.value.strip()):
            masked_out_w.value = suggested_masked

        # Suggest clouds output only when enabled
        if export_clouds_w.value:
            suggested_clouds = _suggest_clouds_path()
            if force or (not clouds_out_w.value.strip()):
                clouds_out_w.value = suggested_clouds

    def _on_export_clouds_toggle(change):
        if change.get("name") != "value":
            return
        enabled = bool(export_clouds_w.value)
        clouds_out_w.disabled = not enabled
        browse_clouds_out_btn.disabled = not enabled
        if enabled:
            _refresh_mask_outputs(force=True)

    export_clouds_w.observe(_on_export_clouds_toggle, names="value")

    def _on_threshold_change(change):
        if change.get("name") == "value" and state["loaded_path"]:
            # Always keep masked output synced to threshold unless user overwrote it manually (simple approach: always update)
            masked_out_w.value = _suggest_masked_path(int(mask_threshold_w.value))

    mask_threshold_w.observe(_on_threshold_change, names="value")

    def _ensure_nc_suffix(path_str: str) -> str:
        p = Path(path_str)
        if p.suffix.lower() != ".nc":
            p = p.with_suffix(".nc")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p.as_posix()

    def _on_mask_and_export_clicked(_):
        if state["loaded_obj"] is None or not state["loaded_path"]:
            _status("❌ Load a cube first.")
            return
        if get_cloud_layers is None:
            _status("❌ get_cloud_layers is not available. Check your stac2cube installation/imports.")
            return

        threshold = int(mask_threshold_w.value)

        out_masked = (masked_out_w.value or "").strip()
        if not out_masked:
            out_masked = _suggest_masked_path(threshold)
            masked_out_w.value = out_masked
        out_masked = _ensure_nc_suffix(out_masked)

        out_clouds = None
        if export_clouds_w.value:
            tmp = (clouds_out_w.value or "").strip()
            if not tmp:
                tmp = _suggest_clouds_path()
                clouds_out_w.value = tmp
            out_clouds = _ensure_nc_suffix(tmp)

        _status(
            "Masking and exporting...",
            f"masking (input) = {state['loaded_path']}",
            f"threshold = {threshold}",
            f"output_masked = {out_masked}",
            f"output_clouds = {out_clouds if out_clouds else 'None'}",
        )

        try:
            # Run your tool (exports inside)
            with status_out:
                # keep the lines you've already printed, then run the tool so its prints show here
                get_cloud_layers(
                    masking=state["loaded_path"],
                    output_clouds=out_clouds,
                    output_masked=out_masked,
                    threshold=threshold,
                )

            state["current_result_path"] = out_masked
            _show_result_from_path(out_masked)

            lines = [f"✅ Mask and export finished: {out_masked}"]
            if out_clouds:
                lines.append(f"✅ Cloud layers exported: {out_clouds}")
            _status(*lines)

        except Exception as e:
            _status(f"❌ {type(e).__name__}: {e}")

    mask_and_export_btn.on_click(_on_mask_and_export_clicked)

    # Sub-accordions inside Tool 1
    # --- Pretty layout for Tool 1a ---
    threshold_row = widgets.HBox(
        [
            widgets.HTML("<div style='font-weight:500;'>Threshold (%):</div>"),
            mask_threshold_w,
        ],
        layout=widgets.Layout(width="100%", gap="8px", align_items="center"),
    )

    exports_header = widgets.HTML("<div style='font-weight:700; margin-top:4px;'>Exporting Setup:</div>")

    
    mask_a_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Masks out the loaded data cube with a single known threshold value.<br> Optionally, exports time series of 'cloud probability + selected threshold binary maps', <br> Cloud probability time series can be used at in step (ii) of the manual workflow to experiment with different thresholds without re-computing probabilities."
                "</div>"
            ),
            threshold_row,
            exports_header,
            _stacked_field(masked_out_box, "Output masked cube (NetCDF)"),
            export_clouds_row,
            _stacked_field(clouds_out_box, "Output cloud layers (NetCDF)"),
            mask_and_export_btn,
        ],
        layout=widgets.Layout(width="100%", gap="10px"),
    )



    mask_a_acc = widgets.Accordion(children=[mask_a_box], selected_index=None)
    mask_a_acc.set_title(0, "a) Fully Automated Workflow")
    mask_a_acc.layout = widgets.Layout(width="99%")

    

    # -----------------------------
    # Tool 1b: Manually Build Cloud Masking Data Cube (UI skeleton)
    # -----------------------------

    # i) Build Cloud Mask Data Cube
    b1_cloud_out_w = widgets.Text(value="", layout=widgets.Layout(width="100%"))
    browse_b1_cloud_out_btn = widgets.Button(icon="folder-open", description="", layout=widgets.Layout(width="36px"))
    b1_cloud_out_fc_box = _attach_filechooser(
        browse_b1_cloud_out_btn,
        b1_cloud_out_w,
        title="Select output NetCDF for cloud probability cube",
        pattern=["*.nc", "*"],
        select_dirs=False,
    )
    b1_cloud_out_row = widgets.HBox(
        [browse_b1_cloud_out_btn, b1_cloud_out_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    b1_cloud_out_box = widgets.VBox([b1_cloud_out_row, b1_cloud_out_fc_box], layout=widgets.Layout(width="100%", gap="4px"))
    b1_build_btn = widgets.Button(description="Build and Export", button_style="success", icon="play", layout=widgets.Layout(width="170px"))

    b1_thresholds_w = widgets.Text(
        value="",  # ✅ empty means None
        placeholder="70  or  [50, 70, 90]  or (leave empty for probability only)",
        layout=widgets.Layout(width="320px"),
    )


    def _ensure_nc_suffix(path_str: str) -> str:
        p = Path(path_str)
        if p.suffix.lower() != ".nc":
            p = p.with_suffix(".nc")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p.as_posix()

    def _on_b1_build_clicked(_):
        if state["loaded_path"] is None:
            _status("❌ Load a cube first.")
            return
        if get_cloud_layers is None:
            _status("❌ get_cloud_layers is not available.")
            return
        if get_stac_parameters is None:
            _status("❌ get_stac_parameters is not available.")
            return

        out_cloud = (b1_cloud_out_w.value or "").strip()
        if not out_cloud:
            # default: same folder, *_cloud.nc
            out_cloud = (Path(state["loaded_path"]).with_name(f"{Path(state['loaded_path']).stem}_cloud.nc")).as_posix()
            b1_cloud_out_w.value = out_cloud
        out_cloud = _ensure_nc_suffix(out_cloud)
        b2_prob_in_w.value = out_cloud

        raw_thr = (b1_thresholds_w.value or "").strip()
        if raw_thr == "":
            thresholds = None
        else:
            try:
                parsed = ast.literal_eval(raw_thr)
            except Exception:
                # allow simple "70" without brackets
                if raw_thr.isdigit():
                    parsed = int(raw_thr)
                else:
                    raise ValueError("Thresholds must be empty, an int (e.g. 70), or a list like [50, 70, 90].")

            if isinstance(parsed, (int, np.integer)):
                thresholds = int(parsed)
            elif isinstance(parsed, (list, tuple)) and all(isinstance(x, (int, np.integer)) for x in parsed):
                thresholds = [int(x) for x in parsed]
            else:
                raise ValueError("Thresholds must be empty, an int (e.g. 70), or a list like [50, 70, 90].")
            
        _status(
            "Building cloud probability data cube...",
            f"loaded cube = {state['loaded_path']}",
            f"output_clouds = {out_cloud}",
            f"threshold = {thresholds}",
        )

        try:
            # Extract polygon + daterange from the loaded cube (same dates, but no masking)
            params = get_stac_parameters(state["loaded_path"])
            polygon = params["polygon"]
            daterange = params["daterange"]

            with status_out:
                # Capture progress prints from get_cloud_layers
                cloud_da = get_cloud_layers(
                    polygon=polygon,
                    daterange=daterange,
                    output_clouds=out_cloud,
                    output_masked=None,
                    threshold=thresholds,          # probability only
                    clip_raster=False,
                    masking=None,            # IMPORTANT: do not trigger masking branch
                    update=None,
                )

            state["current_result_path"] = out_cloud
            _show_result_from_path(out_cloud)
            _status(f"✅ Cloud probability cube exported: {out_cloud}")

        except Exception as e:
            _status(f"❌ {type(e).__name__}: {e}")

    b1_build_btn.on_click(_on_b1_build_clicked)





    b1_box = widgets.VBox(
        [
            widgets.HTML("<div style='font-size:12px; color:#666;'>"
                        "Builds cloud data cube.<br> If threshold is not given, only cloud probability cube will be built. "
                        "In that case, binary mask(s) with threshold(s) can be generated in step (ii)."
                        "</div>"),
            _stacked_field(b1_thresholds_w, "Threshold(s)"),
            _stacked_field(b1_cloud_out_box, "Output cloud probability cube (NetCDF)"),
            b1_build_btn,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    b1_acc = widgets.Accordion(children=[b1_box], selected_index=None)
    b1_acc.set_title(0, "i) Build Cloud Mask Data Cube")
    b1_acc.layout = widgets.Layout(width="100%")

    # ii) (Optional) Generate Masks from Probability Map
    b2_prob_in_w = widgets.Text(value="", layout=widgets.Layout(width="100%"))
    browse_b2_prob_in_btn = widgets.Button(icon="folder-open", description="", layout=widgets.Layout(width="36px"))
    b2_prob_in_fc_box = _attach_filechooser(
        browse_b2_prob_in_btn,
        b2_prob_in_w,
        title="Select cloud probability cube (NetCDF)",
        pattern=["*.nc", "*"],
        select_dirs=False,
    )
    b2_prob_in_row = widgets.HBox([browse_b2_prob_in_btn, b2_prob_in_w],
                                layout=widgets.Layout(width="100%", gap="6px", align_items="center"))
    b2_prob_in_box = widgets.VBox([b2_prob_in_row, b2_prob_in_fc_box], layout=widgets.Layout(width="100%", gap="4px"))

    b2_thresholds_w = widgets.Text(
        value="",
        placeholder="70  or  [50, 70, 90]",
        layout=widgets.Layout(width="420px"),
    )
    
    b2_generate_btn = widgets.Button(description="Generate and Overwrite", button_style="success", icon="play",
                                    layout=widgets.Layout(width="210px"))









    def _parse_thresholds_text(raw: str):
        s = (raw or "").strip()
        if s == "":
            return None
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            if s.isdigit():
                parsed = int(s)
            else:
                raise ValueError("Thresholds must be empty, an int (e.g. 70), or a list like [50, 70, 90].")

        if isinstance(parsed, (int, np.integer)):
            return int(parsed)
        if isinstance(parsed, (list, tuple)) and all(isinstance(x, (int, np.integer)) for x in parsed):
            return [int(x) for x in parsed]
        raise ValueError("Thresholds must be empty, an int (e.g. 70), or a list like [50, 70, 90].")


    def _on_b2_generate_overwrite_clicked(_):
        if state["loaded_path"] is None:
            _status("❌ Load a cube first.")
            return
        if mask_from_probability is None or export_stac is None:
            _status("❌ mask_from_probability/export_stac not available. Check stac2cube imports.")
            return

        prob_path = (b2_prob_in_w.value or "").strip()
        if not prob_path:
            _status("❌ Please provide an input probability cube path.")
            return

        thresholds = _parse_thresholds_text(b2_thresholds_w.value)
        if thresholds is None:
            _status("ℹ️ No thresholds provided. Nothing to do (leaving probability cube unchanged).")
            return

        p = Path(prob_path)
        if not p.exists():
            _status(f"❌ File not found: {p.as_posix()}")
            return

        _status(
            "Generating masks from probability map...",
            f"input/overwrite file = {p.as_posix()}",
            f"thresholds = {thresholds}",
        )

        try:
            # Load existing cloud cube
            with xr.open_dataset(p) as ds:
                if "Cloud_Stack" not in ds.data_vars:
                    raise ValueError("NetCDF does not contain 'Cloud_Stack'.")
                cloud = ds["Cloud_Stack"].load()

            # --- Select probability band (this is the only input to mask_from_probability) ---
            cloud_prob = cloud.sel(band="cloud_prob")

            # --- Generate new masks (we keep average_over/dilation_size hidden in UI) ---
            new_masks = mask_from_probability(
                cloud_probability=cloud_prob,
                threshold=thresholds,
                average_over=4,
                dilation_size=2,
            )  # -> bands: cloud_mask_XX

            # --- Keep existing Cloud_Stack, but drop any mask bands we're about to regenerate ---
            new_band_names = set(map(str, new_masks["band"].values))
            base_bands = [b for b in map(str, cloud["band"].values) if b not in new_band_names]
            base = cloud.sel(band=base_bands)

            # --- Append the new masks (probability stays, old non-conflicting masks stay) ---
            combined = xr.concat([base, new_masks], dim="band").transpose("time", "band", "y", "x")
            combined.name = "Cloud_Stack"

            # --- Overwrite the same file (no need to pass crs/transform) ---
            with status_out:
                export_stac(combined, p.as_posix(), overwrite=True, var_name="Cloud_Stack")

            state["current_result_path"] = p.as_posix()
            _show_result_from_path(p.as_posix())
            _status(f"✅ Masks generated and file overwritten: {p.as_posix()}")

        except Exception as e:
            _status(f"❌ {type(e).__name__}: {e}")


    b2_generate_btn.on_click(_on_b2_generate_overwrite_clicked)









    b2_box = widgets.VBox(
        [
            widgets.HTML("<div style='font-size:12px; color:#666;'>"
                        "Adds one or more <b>binary mask layers</b> from the probability map, "
                        "so you can apply different thresholds later without recomputing probabilities.<br>"
                        "<b>Warning:</b> This overwrites the input NetCDF (keeps cloud_prob, adds/updates mask bands)."
                        "</div>"),
            _stacked_field(b2_prob_in_box, "Input probability cube (NetCDF)"),
            _stacked_field(b2_thresholds_w, "Thresholds"),
            b2_generate_btn,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    b2_acc = widgets.Accordion(children=[b2_box], selected_index=None)
    b2_acc.set_title(0, "ii) (Optional) Generate Masks from Probability Map")
    b2_acc.layout = widgets.Layout(width="100%")

    



    # iii) Mask out Data Cube (by single threshold value) — NEW design

    # Cloud cube selector (separate from main loaded cube)
    b3_cloud_path_w = widgets.Text(value="", layout=widgets.Layout(width="100%"))
    browse_b3_cloud_btn = widgets.Button(icon="folder-open", description="", layout=widgets.Layout(width="36px"))
    b3_cloud_fc_box = _attach_filechooser(
        browse_b3_cloud_btn,
        b3_cloud_path_w,
        title="Select cloud cube (NetCDF with Cloud_Stack)",
        pattern=["*.nc", "*"],
        select_dirs=False,
    )
    b3_cloud_row = widgets.HBox(
        [browse_b3_cloud_btn, b3_cloud_path_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    b3_cloud_box = widgets.VBox([b3_cloud_row, b3_cloud_fc_box], layout=widgets.Layout(width="100%", gap="4px"))

    load_cloud_btn = widgets.Button(
        description="Load cloud cube",
        button_style="primary",
        icon="upload",
        layout=widgets.Layout(width="160px"),
    )

    # Mask band dropdown (populated after loading cloud cube)
    b3_mask_band_w = widgets.Dropdown(
        options=[],
        value=None,
        description="",
        layout=widgets.Layout(width="60%"),
        disabled=True,
    )

    # Output masked cube path
    b3_masked_out_w = widgets.Text(value="", layout=widgets.Layout(width="100%"))
    browse_b3_masked_out_btn = widgets.Button(icon="folder-open", description="", layout=widgets.Layout(width="36px"))
    b3_masked_out_fc_box = _attach_filechooser(
        browse_b3_masked_out_btn,
        b3_masked_out_w,
        title="Select output masked cube (NetCDF)",
        pattern=["*.nc", "*"],
        select_dirs=False,
    )
    b3_masked_out_row = widgets.HBox(
        [browse_b3_masked_out_btn, b3_masked_out_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    b3_masked_out_box = widgets.VBox([b3_masked_out_row, b3_masked_out_fc_box], layout=widgets.Layout(width="100%", gap="4px"))

    b3_mask_btn = widgets.Button(
        description="Mask and Export",
        button_style="success",
        icon="play",
        layout=widgets.Layout(width="170px"),
    )

    # Keep separate state for cloud cube used in iii)
    state["cloud_path"] = None
    state["cloud_mask_bands"] = []

    def _extract_thr_suffix(mask_band: str):
        """
        For band like 'cloud_mask_70' -> returns '70'
        Otherwise returns the raw band string.
        """
        s = str(mask_band)
        m = re.search(r"(\d+)$", s)
        return m.group(1) if m else s

    def _suggest_masked_output_from_selection():
        if not state.get("loaded_path"):
            return ""
        if not b3_mask_band_w.value:
            return ""

        thr = _extract_thr_suffix(b3_mask_band_w.value)
        p = Path(state["loaded_path"])
        out = p.parent / f"{p.stem}_masked_{thr}.nc"
        return out.as_posix()

    def _on_load_cloud_clicked(_):
        if state.get("loaded_path") is None:
            _status("❌ Load the main data cube first.")
            return

        cloud_path = (b3_cloud_path_w.value or "").strip()
        if not cloud_path:
            _status("❌ Please select a cloud cube NetCDF path.")
            return

        p = Path(cloud_path)
        if not p.exists():
            _status(f"❌ File not found: {p.as_posix()}")
            return

        try:
            _status("Loading cloud cube (for masks)...", f"path = {p.as_posix()}")

            with xr.open_dataset(p) as ds:
                if "Cloud_Stack" not in ds.data_vars:
                    raise ValueError("NetCDF does not contain 'Cloud_Stack'.")
                cloud = ds["Cloud_Stack"]
                if "band" not in cloud.dims:
                    raise ValueError("Cloud_Stack has no 'band' dimension.")
                bands = [str(b) for b in cloud["band"].values]

            # Exclude probability band
            mask_bands = [b for b in bands if b != "cloud_prob"]
            if not mask_bands:
                raise ValueError("No mask bands found. Cloud_Stack only contains 'cloud_prob'?")

            state["cloud_path"] = p.as_posix()
            state["cloud_mask_bands"] = mask_bands

            # Populate dropdown
            b3_mask_band_w.options = mask_bands
            b3_mask_band_w.value = mask_bands[0]
            b3_mask_band_w.disabled = False

            # Auto-suggest output based on selected band
            b3_masked_out_w.value = _suggest_masked_output_from_selection()

            _status("✅ Cloud cube loaded for masking.", f"Available masks: {mask_bands}")

        except Exception as e:
            _status(f"❌ {type(e).__name__}: {e}")

    load_cloud_btn.on_click(_on_load_cloud_clicked)

    def _on_mask_band_change(change):
        if change.get("name") != "value":
            return
        if not b3_mask_band_w.value:
            return
        # Always refresh suggestion on band change
        b3_masked_out_w.value = _suggest_masked_output_from_selection()

    b3_mask_band_w.observe(_on_mask_band_change, names="value")

    def _on_b3_mask_export_clicked(_):
        if state.get("loaded_path") is None:
            _status("❌ Load the main data cube first.")
            return
        if mask_stac_clouds is None:
            _status("❌ mask_stac_clouds is not available. Check stac2cube imports.")
            return
        if state.get("cloud_path") is None:
            _status("❌ Load a cloud cube first (blue button).")
            return
        if not b3_mask_band_w.value:
            _status("❌ Please select a mask band (e.g., cloud_mask_70).")
            return

        out_path = (b3_masked_out_w.value or "").strip()
        if not out_path:
            out_path = _suggest_masked_output_from_selection()
            b3_masked_out_w.value = out_path

        out_path = _ensure_nc_suffix(out_path)
        mask_layer = str(b3_mask_band_w.value)

        _status(
            "Masking and exporting...",
            f"input cube = {state['loaded_path']}",
            f"cloud cube = {state['cloud_path']}",
            f"mask_layer = {mask_layer}",
            f"output = {out_path}",
        )

        try:
            with status_out:
                res = mask_stac_clouds(
                    stac=state["loaded_path"],
                    cloud=state["cloud_path"],
                    mask_layer=mask_layer,
                    output=out_path,
                )

            # Show result from exported masked cube
            state["current_result_path"] = out_path
            _show_result_from_path(out_path)
            _status(f"✅ Masked cube exported: {out_path}")

        except Exception as e:
            _status(f"❌ {type(e).__name__}: {e}")

    b3_mask_btn.on_click(_on_b3_mask_export_clicked)

    b3_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Masks the <b>loaded data cube</b> using a selected binary mask band from a <b>Cloud_Stack</b> cube.<br>"
                "Load a cloud cube, pick a mask band (not cloud_prob), then export a masked cube."
                "</div>"
            ),
            _stacked_field(b3_cloud_box, "Cloud data cube (NetCDF)"),
            load_cloud_btn,
            _stacked_field(b3_mask_band_w, "Mask band"),
            _stacked_field(b3_masked_out_box, "Output masked cube (NetCDF)"),
            b3_mask_btn,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )

    b3_acc = widgets.Accordion(children=[b3_box], selected_index=None)
    b3_acc.set_title(0, "iii) Mask out Data Cube (by single threshold value)")
    b3_acc.layout = widgets.Layout(width="100%")









    # Tool 1b container
    mask_b_box = widgets.VBox(
        [
            widgets.HTML("<div style='font-size:12px; color:#666;'>"
                        "Manual workflow: build probability cube → (optional) generate binary masks → apply one threshold to mask the cube."
                        "</div>"),
            b1_acc,
            b2_acc,
            b3_acc,
        ],
        layout=widgets.Layout(width="100%", gap="10px"),
    )

    mask_b_acc = widgets.Accordion(children=[mask_b_box], selected_index=None)
    mask_b_acc.set_title(0, "b) Manually Build Cloud Masking Data Cube")
    mask_b_acc.layout = widgets.Layout(width="99%")


    mask_tool_box = widgets.VBox(
        [
            #widgets.HTML("<b>1) Cloud Masking Data Cube</b>"),
            widgets.HTML("<div style='font-size:12px; color:#666;'>If you already know the threshold value, proceed with Fully Automated Workflow. <br>If not, build your cloud data cube manually and inspect the result. <br>Cloud masking data cube can be also loaded and exported as Geotiffs with Data Cube Editor. </div>"),
            mask_a_acc,
            mask_b_acc,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    mask_tool_acc = widgets.Accordion(children=[mask_tool_box], selected_index=None)
    mask_tool_acc.set_title(0, "1) Cloud Masking Data Cube")
    mask_tool_acc.layout = widgets.Layout(width="99%")

    # --- Tool 2: Co-register Data Cube ---

    # Compact, consistent widget widths (no percent widths)
    cr_grid_size_w = widgets.BoundedIntText(value=7, min=1, max=50, step=1, layout=widgets.Layout(width="200px"))
    cr_max_cc_w = widgets.BoundedIntText(value=100, min=0, max=100, step=1, layout=widgets.Layout(width="200px"))

    cr_time_period_w = widgets.Text(
        value="",
        placeholder='["2023-04-01", "2023-12-31"]',
        layout=widgets.Layout(width="200px"),
    )

    cr_min_rel_keep_w = widgets.BoundedFloatText(value=10.0, min=0.0, max=100.0, step=1.0, layout=widgets.Layout(width="200px"))
    cr_min_rel_update_ref_w = widgets.BoundedFloatText(value=70.0, min=0.0, max=100.0, step=1.0, layout=widgets.Layout(width="200px"))
    cr_max_cloud_update_ref_w = widgets.BoundedFloatText(value=20.0, min=0.0, max=100.0, step=1.0, layout=widgets.Layout(width="200px"))

    cr_first_scene_mode_w = widgets.Dropdown(
        options=[("first", "first"), ("composite", "composite")],
        value="first",
        layout=widgets.Layout(width="200px"),
    )

    cr_composite_window_days_w = widgets.BoundedIntText(value=30, min=1, max=365, step=1, layout=widgets.Layout(width="200px"))
    cr_composite_window_days_w.disabled = True

    cr_iteration_w = widgets.BoundedIntText(value=5, min=1, max=10, step=1, layout=widgets.Layout(width="200px"))

    # output path
    cr_out_w = widgets.Text(value="", layout=widgets.Layout(width="100%"))
    browse_cr_out_btn = widgets.Button(icon="folder-open", description="", layout=widgets.Layout(width="36px"))
    cr_out_fc_box = _attach_filechooser(
        browse_cr_out_btn,
        cr_out_w,
        title="Select output NetCDF for co-registered cube",
        pattern=["*.nc", "*"],
        select_dirs=False,
    )
    cr_out_row = widgets.HBox(
        [browse_cr_out_btn, cr_out_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    cr_out_box = widgets.VBox([cr_out_row, cr_out_fc_box], layout=widgets.Layout(width="100%", gap="4px"))

    cr_run_btn = widgets.Button(
        description="Co-register and Export",
        button_style="success",
        icon="play",
        layout=widgets.Layout(width="210px"),
    )

    def _on_first_scene_mode_change(change):
        if change.get("name") != "value":
            return
        cr_composite_window_days_w.disabled = (cr_first_scene_mode_w.value != "composite")

    cr_first_scene_mode_w.observe(_on_first_scene_mode_change, names="value")

    def _parse_time_period(txt: str):
        s = (txt or "").strip()
        if s == "":
            return None
        import ast
        obj = ast.literal_eval(s)
        if not (isinstance(obj, (list, tuple)) and len(obj) == 2 and all(isinstance(x, str) for x in obj)):
            raise ValueError('time_period must be ["YYYY-MM-DD","YYYY-MM-DD"] or empty.')
        return list(obj)

    def _on_coregister_clicked(_):
        if state.get("loaded_path") is None:
            _status("❌ Load a cube first.")
            return
        if coregister_cube is None:
            _status("❌ coregister_cube is not available. Check stac2cube imports.")
            return

        out_path = (cr_out_w.value or "").strip()
        if not out_path:
            out_path = _suggest_cr_path()
            cr_out_w.value = out_path
        out_path = _ensure_nc_suffix(out_path)

        try:
            time_period = _parse_time_period(cr_time_period_w.value)
        except Exception as e:
            _status(f"❌ ValueError: {e}")
            return

        _status(
            "Co-registering and exporting...",
            f"input_path = {state['loaded_path']}",
            f"output_path = {out_path}",
        )

        try:
            with status_out:
                coregister_cube(
                    input_path=state["loaded_path"],
                    grid_size=int(cr_grid_size_w.value),
                    max_cc=int(cr_max_cc_w.value),
                    time_period=time_period,
                    min_reliability_keep=float(cr_min_rel_keep_w.value),
                    min_reliability_update_ref=float(cr_min_rel_update_ref_w.value),
                    max_cloud_update_ref=float(cr_max_cloud_update_ref_w.value),
                    first_scene_mode=str(cr_first_scene_mode_w.value),
                    composite_window_days=int(cr_composite_window_days_w.value),
                    iteration=int(cr_iteration_w.value),
                    output_path=out_path,
                )

            state["current_result_path"] = out_path
            _show_result_from_path(out_path)
            _status(f"✅ Co-registration finished and exported: {out_path}")

        except Exception as e:
            _status(f"❌ {type(e).__name__}: {e}")

    cr_run_btn.on_click(_on_coregister_clicked)


    # --- Pretty layout (compact rows) ---

    def _auto(box):
        # Make field blocks not stretch to full width
        box.layout.width = "auto"
        box.layout.flex = "0 0 auto"
        return box

    def _row(fields, gap_px=16):
        # gap_px controls the distance between boxes reliably
        items = []
        for i, f in enumerate(fields):
            f.layout.width = "auto"
            f.layout.flex = "0 0 auto"
            items.append(f)
            if i < len(fields) - 1:
                items.append(widgets.HTML(f"<div style='width:{gap_px}px;'></div>"))

        return widgets.HBox(
            items,
            layout=widgets.Layout(
                width="100%",
                justify_content="flex-start",
                align_items="flex-start",
                flex_flow="row wrap",
            ),
        )

    row1 = widgets.VBox(
        [
            widgets.HTML("<b>Filter time-series (optional)</b>"),
            _row(
                [
                    _stacked_field_with_help(cr_max_cc_w, "Max Cloud Coverage", "max_cc"),
                    _stacked_field_with_help(cr_time_period_w, "Time Period", "time_period"),
                ],
                gap_px=20,
            ),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    row2 = widgets.VBox(
        [
            widgets.HTML("<b>Primary Parameters</b>"),
            _row(
                [
                    _stacked_field_with_help(cr_grid_size_w, "Grid Size", "grid_size"),
                    _stacked_field_with_help(cr_iteration_w, "Iteration", "iteration"),
                ],
                gap_px=20,
            ),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    row3 = widgets.VBox(
        [
            widgets.HTML("<b>Secondary Parameters</b>"),
            _row(
                [
                    _stacked_field_with_help(cr_min_rel_keep_w, "Min Reliability to Keep Scenes", "min_reliability_keep"),
                    _stacked_field_with_help(cr_min_rel_update_ref_w, "Min Reliability to Update Reference", "min_reliability_update_ref"),
                    _stacked_field_with_help(cr_max_cloud_update_ref_w, "Max Cloud Coverage to Update Reference", "max_cloud_update_ref"),
                ],
                gap_px=20,
            ),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    row4 = widgets.VBox(
        [
            widgets.HTML("<b>First Scene Behavior</b>"),
            _row(
                [
                    _stacked_field_with_help(cr_first_scene_mode_w, "First Reference Scene", "first_scene_mode"),
                    _stacked_field_with_help(cr_composite_window_days_w, "Composite Window (days)", "composite_window_days"),
                ],
                gap_px=20,
            ),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    section_spacer = widgets.HTML("<div style='height:10px;'></div>")  # adjust 6/8/10/12

    cr_tool_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Co-registers the data cube.<br>"
                "Performs the best in relatively larger areas with heterogeneous land cover."
                "</div>"
            ),
            row1,
            section_spacer,
            row2,
            section_spacer,
            row3,
            section_spacer,
            row4,
            section_spacer,
            _stacked_field(cr_out_box, "Output NetCDF"),
            section_spacer,
            cr_run_btn,
        ],
        layout=widgets.Layout(width="100%", gap="10px"),
    )

    cr_tool_acc = widgets.Accordion(children=[cr_tool_box], selected_index=None)
    cr_tool_acc.set_title(0, "2) Co-register Data Cube")
    cr_tool_acc.layout = widgets.Layout(width="99%")

   



    # --- Tool 3: Super-resolve Data Cube (single mode dropdown) ---

    def _suggest_sr_path_from_loaded():
        if state.get("loaded_path"):
            p = Path(state["loaded_path"])
            return (p.parent / f"{p.stem}_sr.nc").as_posix()
        return "./results/cube_sr.nc"

    sr_mode_w = widgets.Dropdown(
        options=[
            ("10-m RGBN to 2.5-m", "rgbn"),
            ("10-m Full Spectral to 2.5-m", "full_spectral"),
            ("20-m Bands to 10-m", "20to10"),
        ],
        value="rgbn",
        layout=widgets.Layout(width="320px"),
    )

    SR_DESC = {
        "rgbn": (
            "- Required band setup -> <code>blue, green, red, nir</code><br>"
            "- If exist, indices must be only 10-meter resolution ones, e.g., ndvi, ndwi<br>"
            "- Use this model if you don't have 20-m bands. Much faster model!"
        ),
        "full_spectral": (
            "- Required band setup -> <code>blue, green, red, nir, nir08, rededge1, rededge2, rededge3, swir16, swir22</code><br>"
            "- If exist, indices can be both 10 and 20-meter resolution ones, e.g., ndvi, ndwi, ndmi<br>"
            "- Use this model only if you need to super resolve 20-meter bands.<br>"
            "- Even if you need to super-resolve one of the 20-meter bands, still need to include all of the required ones."
        ),
        "20to10": "Under development :)",
    }

    sr_desc_html = widgets.HTML(
        f"<div style='font-size:12px; color:#666;'>{SR_DESC[sr_mode_w.value]}</div>"
    )

    # Output NetCDF
    sr_out_w = widgets.Text(value="", layout=widgets.Layout(width="100%"))
    browse_sr_out_btn = widgets.Button(icon="folder-open", description="", layout=widgets.Layout(width="36px"))
    sr_out_fc_box = _attach_filechooser(
        browse_sr_out_btn,
        sr_out_w,
        title="Select output NetCDF for super-resolved cube",
        pattern=["*.nc", "*"],
        select_dirs=False,
    )
    sr_out_row = widgets.HBox([browse_sr_out_btn, sr_out_w], layout=widgets.Layout(width="100%", gap="6px", align_items="center"))
    sr_out_box = widgets.VBox([sr_out_row, sr_out_fc_box], layout=widgets.Layout(width="100%", gap="4px"))

    sr_run_btn = widgets.Button(
        description="Super-resolve and Export",
        button_style="success",
        icon="play",
        layout=widgets.Layout(width="220px"),
    )

    def _on_sr_mode_change(change):
        if change.get("name") != "value":
            return
        mode = sr_mode_w.value
        sr_desc_html.value = f"<div style='font-size:12px; color:#666;'>{SR_DESC[mode]}</div>"

        # Disable run for under-development mode
        if mode == "20to10":
            sr_run_btn.disabled = True
        else:
            # enabled state will also depend on load status via _set_enabled_after_load
            sr_run_btn.disabled = (state.get("loaded_path") is None)

    sr_mode_w.observe(_on_sr_mode_change, names="value")

    
    def _on_sr_run_clicked(_):
        if state.get("loaded_path") is None:
            _status("❌ Load a cube first.")
            return
        if super_resolve_cube is None:
            _status("❌ super_resolve_cube is not available. Check stac2cube imports.")
            return

        mode = sr_mode_w.value
        if mode == "20to10":
            _status("ℹ️ 20-m Bands to 10-m is under development :)")
            return

        out_path = (sr_out_w.value or "").strip()
        if not out_path:
            out_path = _suggest_sr_path_from_loaded()
            sr_out_w.value = out_path
        out_path = _ensure_nc_suffix(out_path)

        p_out = Path(out_path)
        existed_before = p_out.exists()
        old_mtime = p_out.stat().st_mtime if existed_before else None
        old_size = p_out.stat().st_size if existed_before else None

        _status(
            "Super-resolving and exporting...",
            f"input_path = {state['loaded_path']}",
            f"mode = {mode}",
            f"output_path = {out_path}",
        )

        try:
            # Capture progress prints from the tool
            with status_out:
                super_resolve_cube(
                    input_path=state["loaded_path"],
                    output_path=out_path,
                    model_type=("rgbn" if mode == "rgbn" else "full_spectral"),
                )

            # --- Verify export actually happened (prevents false ✅) ---
            if not p_out.exists():
                with status_out:
                    print("❌ Super-resolution failed: output file was not created.")
                return

            new_mtime = p_out.stat().st_mtime
            new_size = p_out.stat().st_size

            if existed_before and (new_mtime == old_mtime) and (new_size == old_size):
                with status_out:
                    print(
                        "❌ Super-resolution failed: output file was not updated "
                        "(likely missing required bands for the selected mode)."
                    )
                return

            # Ensure file is readable
            try:
                with xr.open_dataset(p_out) as _:
                    pass
            except Exception as e:
                with status_out:
                    print(f"❌ Super-resolution failed: output file is not readable ({type(e).__name__}: {e})")
                return

            # Success
            state["current_result_path"] = out_path
            _show_result_from_path(out_path)
            _status(f"✅ Super-resolution finished and exported: {out_path}")

        except Exception as e:
            _status(f"❌ {type(e).__name__}: {e}")


    sr_run_btn.on_click(_on_sr_run_clicked)

    # Initial disable (until load)
    sr_run_btn.disabled = True

    sr_tool_box = widgets.VBox(
        [
            #widgets.HTML("<b>3) Super-resolve Data Cube</b>"),
            widgets.HTML("<div style='font-size:12px; color:#666;'>Super resolves the loaded data cube. Select one of the three modes below.</div>"),
            _stacked_field(sr_mode_w, "Mode"),
            sr_desc_html,
            _stacked_field(sr_out_box, "Output NetCDF"),
            sr_run_btn,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )

    sr_tool_acc = widgets.Accordion(children=[sr_tool_box], selected_index=None)
    sr_tool_acc.set_title(0, "3) Super-resolve Data Cube")
    sr_tool_acc.layout = widgets.Layout(width="100%")










    sr_tool_acc = widgets.Accordion(children=[sr_tool_box], selected_index=None)
    sr_tool_acc.set_title(0, "3) Super-resolve Data Cube")
    sr_tool_acc.layout = widgets.Layout(width="99%")

    tools_box = widgets.VBox(
        [
            widgets.HTML("<b>Tools</b>"),
            widgets.HTML("<div style='font-size:12px; color:#666;'>Each tool exports its result to NetCDF (no COG export here).</div>"),
            mask_tool_acc,
            cr_tool_acc,
            sr_tool_acc,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    tools_card = widgets.VBox([tools_box], layout=widgets.Layout(width="100%"))
    tools_card.add_class("stac2cube-card")

    # -----------------------------------------
    # Result card (shows exported file result)
    # -----------------------------------------
    # Result accordion
    result_box = widgets.VBox(
        [
            widgets.HTML("<div style='font-size:12px; color:#666;'>Shows the exported NetCDF result after a tool run.</div>"),
            result_out,
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    result_acc = widgets.Accordion(children=[result_box], selected_index=None)
    result_acc.set_title(0, "Result")
    result_acc.layout = widgets.Layout(width="99%")

    result_card = widgets.VBox([result_acc], layout=widgets.Layout(width="100%"))
    result_card.add_class("stac2cube-card")

    # -----------------------------------------
    # Status card
    # -----------------------------------------
    status_card = widgets.VBox(
        [
            widgets.HTML("<b>Status</b>"),
            status_out,
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )
    status_card.add_class("stac2cube-card")

    # -----------------------------------------
    # Initialize tool output suggestions (after load)
    # -----------------------------------------
    def _refresh_output_suggestions():
        _refresh_mask_outputs(force=True)
        b1_cloud_out_w.value = _suggest_clouds_path()
        b2_prob_in_w.value = b1_cloud_out_w.value
        b3_cloud_path_w.value = _suggest_clouds_path()
        cr_out_w.value = _suggest_cr_path()
        sr_path = _suggest_sr_path_from_loaded()
        sr_out_w.value = _suggest_sr_path_from_loaded()

    def _set_enabled_after_load(enabled: bool):
        #reset_btn.disabled = not enabled

        # Tool 1a controls
        mask_threshold_w.disabled = not enabled
        masked_out_w.disabled = not enabled
        browse_masked_out_btn.disabled = not enabled

        b1_cloud_out_w.disabled = not enabled
        browse_b1_cloud_out_btn.disabled = not enabled
        b1_build_btn.disabled = not enabled

        b2_prob_in_w.disabled = not enabled
        browse_b2_prob_in_btn.disabled = not enabled
        b2_thresholds_w.disabled = not enabled
        b2_generate_btn.disabled = not enabled

        export_clouds_w.disabled = not enabled
        # clouds_out depends on checkbox
        clouds_out_w.disabled = (not enabled) or (not export_clouds_w.value)
        browse_clouds_out_btn.disabled = (not enabled) or (not export_clouds_w.value)

        mask_and_export_btn.disabled = not enabled

        b3_cloud_path_w.disabled = not enabled
        browse_b3_cloud_btn.disabled = not enabled
        load_cloud_btn.disabled = not enabled

        # dropdown remains disabled until cloud cube is loaded
        if not enabled:
            b3_mask_band_w.disabled = True

        b3_masked_out_w.disabled = not enabled
        browse_b3_masked_out_btn.disabled = not enabled
        b3_mask_btn.disabled = not enabled

        # Tool 2/3 Co-registration
        cr_grid_size_w.disabled = not enabled
        cr_max_cc_w.disabled = not enabled
        cr_time_period_w.disabled = not enabled
        cr_min_rel_keep_w.disabled = not enabled
        cr_min_rel_update_ref_w.disabled = not enabled
        cr_max_cloud_update_ref_w.disabled = not enabled
        cr_first_scene_mode_w.disabled = not enabled
        cr_composite_window_days_w.disabled = (not enabled) or (cr_first_scene_mode_w.value != "composite")
        cr_iteration_w.disabled = not enabled
        cr_out_w.disabled = not enabled
        browse_cr_out_btn.disabled = not enabled
        cr_run_btn.disabled = not enabled

        # Tool 3/3 Super-resolution
        sr_mode_w.disabled = not enabled
        sr_out_w.disabled = not enabled
        browse_sr_out_btn.disabled = not enabled

        # run enabled only if loaded AND mode not under development
        if not enabled:
            sr_run_btn.disabled = True
        else:
            sr_run_btn.disabled = (sr_mode_w.value == "20to10")


    _set_enabled_after_load(False)

    # -----------------------------------------
    # Events
    # -----------------------------------------
    def _on_load_clicked(_):
        path = (load_path_w.value or "").strip()
        if not path:
            _status("❌ Please select a NetCDF path.")
            return
        p = Path(path)
        if not p.exists():
            _status(f"❌ File not found: {p.as_posix()}")
            return

        try:
            _status("Loading cube...")
            with xr.open_dataset(p) as ds:
                if "Spectral_Temporal_Stack" in ds.data_vars:
                    obj = ds["Spectral_Temporal_Stack"].load()
                else:
                    obj = ds.load()

            state["loaded_path"] = p.as_posix()
            state["loaded_obj"] = obj
            #reset_btn.disabled = False
            _set_enabled_after_load(True)
            _refresh_output_suggestions()

            _show_loaded_summary(obj)
            loaded_summary_acc.selected_index = 0
            _status("✅ Cube loaded.", f"Loaded path: {state['loaded_path']}", "Select one of the listed tools to proceed.")
        except Exception as e:
            _status(f"❌ {type(e).__name__}: {e}")

    def _on_reset_clicked(_):
        if state["loaded_obj"] is None:
            _status("❌ No loaded cube to reset to.")
            return
        # For this UI, reset just clears the result display pointer
        state["current_result_path"] = None
        with result_out:
            clear_output()
            print("No exported result yet.")
        _status("✅ Reset done. (No exported result selected.)")


    load_cube_btn.on_click(_on_load_clicked)
    #reset_btn.on_click(_on_reset_clicked)

    # Tool buttons: skeleton only (no logic wired yet)
    def _run_tool_stub(tool_name: str, out_path: str):
        if state["loaded_obj"] is None or not state["loaded_path"]:
            _status("❌ Load a cube first.")
            return
        if not out_path.strip():
            _status("❌ Please set an output NetCDF path.")
            return
        _status(
            f"🚧 {tool_name} is not wired yet (skeleton).",
            f"Would export to: {Path(out_path).as_posix()}",
            "Next: we will plug in your real tool functions + parameters step by step.",
        )
        # no result to show yet

    

    def _on_run_cr(_):
        _run_tool_stub("Tool 2: Co-register Data Cube", cr_out_w.value)

    #def _on_run_sr(_):
     #   _run_tool_stub("Tool 3: Super-resolve Data Cube", sr_out_w.value)


    
    

    # -----------------------------------------
    # Compose UI (cards + spacing)
    # -----------------------------------------
    spacer_small = widgets.HTML("<div style='height:6px;'></div>")
    spacer_med = widgets.HTML("<div style='height:12px;'></div>")

    ui = widgets.VBox(
        [
            css_patch,
            header,
            subtitle,

            loading_card,
            spacer_small,
            loaded_summary_card,

            spacer_med,
            tools_card,

            spacer_med,
            result_card,

            spacer_med,
            status_card,
        ],
        layout=widgets.Layout(width="96%", max_width="980px", margin="0 auto", gap="8px"),
    )

    outer = widgets.HBox([ui], layout=widgets.Layout(width="100%", justify_content="center"))
    display(outer)

    _status("ℹ️ Load a data cube to start. Then select one of the tools.")

    return {
        "ui": ui,
        "outer": outer,
        "state": state,
        "widgets": {
            "load_path": load_path_w,
            "load_cube_btn": load_cube_btn,
            #"reset_btn": reset_btn,
            "mask_threshold": mask_threshold_w,
            "cr_out": cr_out_w,
            #"sr_out": sr_out_w,
            #"run_sr_btn": run_sr_btn,
            "loaded_summary_acc": loaded_summary_acc,
            "mask_tool_acc": mask_tool_acc,
            "cr_tool_acc": cr_tool_acc,
            "sr_tool_acc": sr_tool_acc,
        },
        "outputs": {
            "loaded_summary": loaded_summary_out,
            "result": result_out,
            "status": status_out,
        },
    }
