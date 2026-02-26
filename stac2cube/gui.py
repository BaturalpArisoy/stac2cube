import ast
import json
import os
import re
from pathlib import Path

import xarray as xr
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output, Javascript

try:
    from ipyfilechooser import FileChooser
except Exception:
    FileChooser = None

from stac2cube import (
    missions,
    get_stac_layers,
    export_stac,
    export_to_cogs,
    interactive_time_view,
    save_timeseries_gif,
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


def datacube_generator_GUI(missions_func=missions):
    
    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
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
        description="Generate data cube",
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
        "<div style='margin:0 0 4px 0; font-size:28px; font-weight:700;'>Data Cube Generator</div>"
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
            widgets.HTML("<b>Basic Parameters</b>"),
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

    advanced_box = widgets.VBox(
        [
            widgets.HTML("<b>Advanced Parameters</b>"),
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
            widgets.HTML("<b>Export Options</b>"),
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
    advanced_acc.set_title(0, "Advanced parameters")
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

    ui = widgets.VBox(
        [
            css_patch,
            header,
            basic_box,
            advanced_acc,
            export_acc,
            action_row,
            result_acc,
            widgets.HTML("<b>Status</b>"),
            status_out,
            viz_acc,
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
