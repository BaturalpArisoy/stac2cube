import ast
import fnmatch
import io
import json
import os
import re
import tempfile
from contextlib import redirect_stdout
from datetime import date as _date, datetime as _datetime
from pathlib import Path

import xarray as xr
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output, Javascript, HTML

try:
    from ipyfilechooser import FileChooser
except Exception:
    FileChooser = None

from stac2cube import (
    missions,
    export_stac,
    export_to_cogs,
    open_cube,
    is_zarr_path,
    resolve_cube_path,
    interactive_time_view,
    interactive_cloud_overlay_view,
    save_timeseries_gif,
    calculate_statistics,
    calculate_spectral_index,
    clip_stac,
    reproject_stac,
    cloud_filter,
    get_stac_layers,
    get_cloud_layers,
    build_cloud_mask_cube,
    update_cloud_mask_cube,
    get_shadow_layers,
    add_shadow_masks_to_cloud_stack,
    coregister_cube,
    spectral_profiler,
    get_stac_parameters,
    mask_from_probability,
    mask_stac_clouds,
    super_resolve_cube,
    check_scene_availability,
    preview_scene_footprints,
)
from .auxiliary import GRAZE_THRESHOLD as _GRAZE_THRESHOLD
from .auxiliary import export_cube_statistics
# The CSV lands next to the cube, so both GUIs derive its path with the same
# rule export_cube_statistics itself uses for a bare path.
from .auxiliary import _default_csv_path as _statistics_csv_path
from .main import settings_sidecar_path
from .mosaic import mosaic_cubes, mosaic_layers

from .get_data import (
    SCENE_METADATA_AVAILABILITY,
    probe_native_crs,
    validate_target_crs,
)
from .clip import compute_scene_coverage

# The custom-composite rows validate through the SAME parser calculate_statistics
# uses, so the interface can never accept a composite a headless run rejects.
from .get_statistics import _parse_custom as _parse_custom_composite
from .get_statistics import _VALID_OPS as _COMPOSITE_OPS
from .stac_processing import is_iso_date, is_mmdd

from .gui_common import (
    human_readable_bytes as _human_readable_bytes,
    estimated_data_size_bytes as _estimated_data_size_bytes,
    normalize_ui_path as _normalize_ui_path,
    existing_dir_or_parent as _existing_dir_or_parent,
    is_str_list_len2 as _is_str_list_len2,
    validate_date_string as _validate_date_string,
    parse_daterange_input as _parse_daterange_input,
    friendly_error as _friendly_error,
    gui_css_widget as _gui_css_widget,
    stacked_field as _stacked_field,
    stacked_field_with_help as _field_with_help,
    field_group as _field_group,
    boxed_control as _boxed,
    help_button as _help_button,
    make_viz_renderer_control as _make_viz_renderer_control,
)


if FileChooser is None:
    _CubeFileChooser = None
else:

    class _CubeFileChooser(FileChooser):
        """FileChooser that picks a ``.zarr`` store in one click.

        A Zarr cube is a directory, so the stock chooser walks INTO it and the
        user has to select some file inside (``zarr.json``, a coordinate
        folder, ...). Here a ``*.zarr`` folder behaves like a file: clicking it
        fills the filename box instead of opening it, so ``selected`` is the
        store root itself.

        Only applies when the store name matches the chooser's file filter, so
        a NetCDF-only or polygon chooser keeps browsing ``.zarr`` folders as
        plain directories.
        """

        def _is_pickable_store(self, path, name):
            if self._show_only_dirs or not name:
                return False
            if not is_zarr_path(name):
                return False
            if self._filter_pattern and not _fnmatch_any(name, self._filter_pattern):
                return False
            return os.path.isdir(os.path.join(path, name))

        def _on_dircontent_select(self, change):
            new = change.get("new")
            if new is None:
                return
            base = self._expand_path(self._pathlist.value)
            name = self._map_disp_to_name[new]
            if self._is_pickable_store(base, name):
                # Stay in the parent folder, put the store in the filename box.
                self._set_form_values(base, name)
                return
            super()._on_dircontent_select(change)

        def _set_form_values(self, path, filename):
            super()._set_form_values(path, filename)
            # The base class clears the highlight and disables Select for every
            # directory entry; undo both when that directory is a Zarr store.
            if not self._is_pickable_store(path, filename):
                return
            disp = getattr(self, "_map_name_to_disp", {}).get(filename)
            if disp is not None and disp in (self._dircontent.options or ()):
                self._dircontent.unobserve(self._on_dircontent_select, names="value")
                self._dircontent.value = disp
                self._dircontent.observe(self._on_dircontent_select, names="value")
            if self._gb.layout.display is None:
                already_selected = (
                    self._selected_path is not None
                    and self._selected_filename is not None
                    and os.path.join(path, filename)
                    == os.path.join(self._selected_path, self._selected_filename)
                )
                self._select.disabled = already_selected


def _fnmatch_any(name, patterns):
    """True when ``name`` matches any of the fnmatch ``patterns`` (case-insensitive)."""
    if isinstance(patterns, str):
        patterns = [patterns]
    low = str(name).lower()
    return any(fnmatch.fnmatch(low, str(p).lower()) for p in patterns)


# Shown when the user tries to export while Export mode is still "Quick Result"
# (lazy). Used both as the early-return guard message in the export buttons and
# as the ValueError raised deeper in the export helpers.
_EXPORT_MODE_REMINDER = (
    "ʕ•ᴥ•ʔ Mr. Bear would like to remind you to change Export mode to NetCDF, "
    "Zarr or COGs before exporting, thanks."
)


# Tempo of the busy indicator, in seconds.
_BEAR_HOP = 0.55            # one bear hop
_BEAR_FALL = 0.35           # crumb falling from the cookie into the row
_BEAR_BOUNCE = 0.225        # one crumb's up-down in the wave (stays constant)
_BEAR_COOKIE_GAP = 0.5      # beat while a fresh cookie arrives
_BEAR_PAUSE = 1.2           # final beat before the whole loop restarts
_BEAR_COOKIES = 20          # cookies on the plate (~76s loop, repeats on longer builds)
_BEAR_BITES = 4             # quarters per cookie

_BEAR_BITE_AT = _BEAR_HOP * 0.45   # bite lands at the top of the hop
_BEAR_FALL_FROM = -16.0            # px above the row where a crumb spawns
_BEAR_CRUMBS = _BEAR_COOKIES * _BEAR_BITES

# The bear eats at a constant pace: every round takes the same time, whatever
# the crumb count. The wave has to fit inside that fixed round, so as crumbs
# pile up the crest simply travels faster (the gap between neighbouring crumbs
# shrinks) while each crumb's own bounce stays the same length.
_BEAR_ROUND = _BEAR_BITE_AT + _BEAR_FALL + _BEAR_BOUNCE
_BEAR_WAVE_WINDOW = _BEAR_ROUND - _BEAR_BITE_AT   # bite -> end of the round


def _bear_schedule():
    """Timeline of one indicator cycle.

    Each hop is a bite: the bear takes a quarter of the current cookie and the
    bite drops a crumb into the row underneath it. Once a cookie is finished a
    fresh one pops in, but the crumbs are never cleared - so the wave has more
    ground to cover with every bite of every cookie.

    The bear's pace never changes: a round is always _BEAR_ROUND long. The wave
    is what adapts - the bite kicks it off, and the gap between neighbouring
    crumbs shrinks just enough for the crest to cross the whole row within the
    round. That makes the crest reach the newest crumb exactly as it lands, for
    any number of crumbs, so the wave always keeps pace with the eating.

    Returns (cycle_length, hop_starts, bite_times, crumb_land_times,
    {crumb_index: [wave_start_times]}, fresh_cookie_times).
    """
    t = 0.0
    hops, bites, lands, fresh = [], [], [], []
    waves = {i: [] for i in range(1, _BEAR_CRUMBS + 1)}
    for n in range(1, _BEAR_CRUMBS + 1):
        hops.append(t)
        bite = t + _BEAR_BITE_AT
        land = bite + _BEAR_FALL
        bites.append(bite)
        lands.append(land)
        if n == 1:
            # Nothing to travel through yet, so the lone crumb bounces on
            # landing rather than on the bite.
            waves[1].append(land)
        else:
            # Crest leaves at the bite and has (window - bounce) to reach the
            # last crumb; that lands it there exactly at _BEAR_FALL, which is
            # the moment the newest crumb touches down.
            stagger = (_BEAR_WAVE_WINDOW - _BEAR_BOUNCE) / (n - 1)
            for i in range(1, n + 1):      # wave across every crumb so far
                waves[i].append(bite + stagger * (i - 1))
        t += _BEAR_ROUND
        if n % _BEAR_BITES == 0 and n < _BEAR_CRUMBS:
            fresh.append(t + _BEAR_COOKIE_GAP * 0.4)
            t += _BEAR_COOKIE_GAP
    return t + _BEAR_PAUSE, hops, bites, lands, waves, fresh


def _bear_pct(t, total):
    return round(100.0 * t / total, 3)


def _bear_snap(total):
    """Epsilon for a change that should be instant rather than interpolated.

    It has to be a fraction of the cycle, not a fixed number of seconds: keyframe
    positions are percentages rounded to 3 decimals, so a fixed epsilon collapses
    onto the same percentage once the cycle gets long, and CSS then smoothly
    interpolates the step instead of snapping it.
    """
    return total * 0.0005


def _bear_hop_keyframes(total, hops):
    frames = [(0.0, "translateY(0)")]
    for s in hops:
        frames += [
            (s, "translateY(0)"),
            (s + _BEAR_HOP * 0.45, "translateY(-8px)"),
            (s + _BEAR_HOP * 0.9, "translateY(0)"),
        ]
    frames.append((total, "translateY(0)"))
    body = "".join(f"{_bear_pct(t, total)}%{{transform:{v};}}" for t, v in frames)
    return "@keyframes s2cBearHop{" + body + "}"


def _bear_cookie_keyframes(total, bites, fresh):
    """A quarter vanishes at each bite, with a small recoil; when a cookie is
    finished the next one pops in at full size."""
    eps = _bear_snap(total)
    frames = [(0.0, 0, "translateX(0) scale(1)")]
    for k, b in enumerate(bites):
        j = k % _BEAR_BITES
        frames += [
            (max(b - eps, 0.0), 25 * j, "translateX(0) scale(1)"),
            (b, 25 * (j + 1), "translateX(2px) scale(1)"),
            (min(b + 0.12, total), 25 * (j + 1), "translateX(0) scale(1)"),
        ]
    for c in fresh:
        frames += [
            (max(c - eps, 0.0), 100, "translateX(0) scale(1)"),
            (c, 0, "translateX(0) scale(0.5)"),
            (min(c + 0.22, total), 0, "translateX(0) scale(1)"),
        ]
    frames.append((total, 100, "translateX(0) scale(1)"))
    frames.sort(key=lambda f: f[0])
    # Clipped from the LEFT: the bear stands to the left of the cookie, so the
    # edge nearest him is the one that disappears.
    body = "".join(
        "%s%%{clip-path:inset(0 0 0 %s%%);-webkit-clip-path:inset(0 0 0 %s%%);"
        "transform:%s;}" % (_bear_pct(t, total), c, c, tr)
        for t, c, tr in frames
    )
    return "@keyframes s2cCookie{" + body + "}"


def _bear_crumb_keyframes(total, bites, lands, waves, i):
    """Crumb i: hidden -> spawns at the cookie on bite i -> falls into the row
    below -> then bounces once per wave that follows."""
    eps = _bear_snap(total)
    b, land = bites[i - 1], lands[i - 1]
    frames = [
        (0.0, 0.0, _BEAR_FALL_FROM),
        (max(b - eps, 0.0), 0.0, _BEAR_FALL_FROM),   # invisible until bitten
        (b, 1.0, _BEAR_FALL_FROM),                   # pops in at the cookie
        (land, 0.55, 0.0),                           # lands in the row
    ]
    for s in waves[i]:
        frames += [
            (s, 0.55, 0.0),
            (s + _BEAR_BOUNCE * 0.5, 1.0, -4.0),
            (s + _BEAR_BOUNCE, 0.55, 0.0),
        ]
    frames.append((total, 0.55, 0.0))
    frames.sort(key=lambda f: f[0])
    body = "".join(
        f"{_bear_pct(t, total)}%{{opacity:{o};transform:translateY({y}px);}}"
        for t, o, y in frames
    )
    return "@keyframes s2cCrumb%d{%s}" % (i, body)


def _busy_bear_html(lead, tail=""):
    """Mr.-Bear-eating-cookies progress indicator.

    Browser-side CSS, so it keeps moving even while the kernel is blocked on a
    slow STAC call - shows the user it isn't stuck. `lead` is the main message;
    optional `tail` is shown after it (e.g. a parenthetical note).
    """
    total, hops, bites, lands, waves, fresh = _bear_schedule()
    dur = round(total, 3)
    css = [
        _bear_hop_keyframes(total, hops),
        _bear_cookie_keyframes(total, bites, fresh),
    ]
    css += [
        _bear_crumb_keyframes(total, bites, lands, waves, i)
        for i in range(1, _BEAR_CRUMBS + 1)
    ]
    crumb_rules = "".join(
        ".s2c-crumbs span:nth-child(%d){animation:s2cCrumb%d %ss infinite linear;}"
        % (i, i, dur)
        for i in range(1, _BEAR_CRUMBS + 1)
    )
    tail_span = f"<span style='font-size:13px;'> {tail}</span>" if tail else ""
    return (
        "<style>"
        + "".join(css)
        + ".s2c-bunny{display:inline-block;"
        "animation:s2cBearHop %ss infinite linear;}" % dur
        + ".s2c-cookie{display:inline-block;"
        "animation:s2cCookie %ss infinite linear;}" % dur
        + ".s2c-crumbs span{display:inline-block;font-weight:700;opacity:0;}"
        + crumb_rules
        + "</style>"
        "<span class='s2c-bunny' style='font-size:13px;'>\u0295\u2022\u1d25\u2022\u0294</span>"
        # The cookie gets its own positioning context so the crumb row can sit
        # directly underneath it - the crumbs fall off the cookie, so that is
        # where they should land.
        "<span class='s2c-cookie-stack' style='position:relative;"
        "display:inline-block;margin-left:4px;padding-bottom:16px;'>"
        "<span class='s2c-cookie' style='font-size:15px;'>\U0001f36a</span>"
        "<span class='s2c-crumbs' style='position:absolute;left:0;top:15px;"
        "font-size:16px;letter-spacing:2px;white-space:nowrap;'>"
        + "".join("<span>.</span>" for _ in range(_BEAR_CRUMBS))
        + "</span></span>"
        + f"<span style='font-size:13px;'> {lead}</span>"
        + tail_span
    )


def _raster_layer_names(ds):
    """Names of data variables that look like raster layers (have y/x dims).

    Filters out helper variables such as 'spatial_ref', which xarray decodes
    as a dimensionless data variable in exported cubes.
    """
    return [
        str(v) for v in ds.data_vars
        if ("y" in ds[v].dims and "x" in ds[v].dims)
    ]


def _layer_display_name(name):
    """User-facing label for a layer variable. 'Time_Series' is the
    internal name of the full time series, so show it as 'Time Series'; other
    layers (temporal composites like median_timeseries) keep their own name."""
    return "Time Series" if str(name) == "Time_Series" else str(name)


def _layer_dropdown_options(ds, names):
    """(label, value) dropdown options showing each layer's dims and sizes.

    The value stays the real variable name; only the label is friendly.
    """
    options = []
    for name in names:
        dims = ", ".join(f"{d}: {ds[name].sizes[d]}" for d in ds[name].dims)
        options.append((f"{_layer_display_name(name)}  ({dims})", name))
    return options


# -------------------------------------------------------------------------
# Parameter help
# -------------------------------------------------------------------------
PARAM_HELP_HTML = {
    "daterange_mode": """
    <b>Season mode</b><br>
    <b>1) Seasonal (all available years)</b><br>
    <code>{"season": ["MM-DD", "MM-DD"], "years": "all"}</code><br>
    The season for every year the mission has data.<br><br>
    <b>2) Seasonal (year range)</b><br>
    <code>{"season": ["MM-DD", "MM-DD"], "years": "2019-2024"}</code><br>
    Every year from first to last (inclusive).<br><br>
    <b>3) Seasonal (selected years only)</b><br>
    <code>{"season": ["MM-DD", "MM-DD"], "years": [2019, 2021, 2023]}</code><br>
    Only the years you list.<br><br>
    <i>Season is MM-DD (no year), e.g. a vegetation season ["04-01", "10-31"]. A season that
    crosses the new year (e.g. ["11-01", "03-31"]) is handled automatically.</i>
    """,
    "polygon": """
    <b>polygon</b><br>
    <b>1) Path to polygon</b><br>
    Polygon formats: <code>gpkg</code>, <code>geojson</code>, <code>kml</code>, <code>kmz</code>, <code>shp</code>.<br>
    Polygons can be geographic (WGS84) or projected (e.g., UTM).<br>
    <b>2) List of BBOX</b><br>
    Can also be a WGS84 bbox list: <code>[xmin, ymin, xmax, ymax]</code> (not projected coords). Useful tool: <code>http://bboxfinder.com/</code><br>
    <b>3) Draw on a map</b><br>
    Tick <b>"Draw the area on a map instead"</b> below to open an interactive map. Use the draw tools at the top-left of the map to draw a rectangle or polygon over your area, then click <b>"Use drawn area"</b>. Your exact outline is saved as a GeoJSON file in the <code>polygons</code> folder and the box above is pointed at it (overwriting any path you typed). To change the background map, use the toolbar at the top-right of the map.<br>
    """,
    "footprint_prefilter": """
    <b>Skip scenes that barely touch your area</b><br>
    A satellite passes over your area on several different orbits. Some of them
    only clip a corner of it. Those acquisitions still cost a <b>full-size time
    step</b> in the cube - the whole rectangle is stored for every date, even
    where there is no data - and they come back every revisit cycle, so on a
    sprawling area they can easily be <b>half of the total size and build
    time</b>.<br><br>
    This box removes them <b>before anything is downloaded</b>, which is what
    makes it fast: a skipped date is never loaded, never cloud-checked and never
    given memory. Every other filter in stac2cube runs on the finished cube, so
    it only trims the result.<br><br>
    <b>How to choose a number:</b> click <b>Check area coverage</b> above. The
    orbits at the bottom of that table are the ones this box removes. Type a
    value just above them - it will suggest <code>10</code> when it finds dates
    that barely reach your area. <code>0</code> (default) keeps everything.<br><br>
    <b>Keep it low.</b> This is measured from the scene outlines the catalogue
    publishes, not from the pixels, so it is an estimate: on one test area
    Planetary Computer and terrabyte agreed exactly while Element84 read about
    3 points lower on partial scenes. It is meant for removing near-misses, not
    for judging whether a scene is complete.<br><br>
    <b>Not the same as "Remove partially missing scenes"</b> (Advanced &rarr;
    Overlapping Tile Handling). That one measures real pixels after downloading,
    and it is the only one that can catch a scene whose outline looks complete
    but whose data is missing or broken. Use this box to avoid pointless
    downloads, and that one to guarantee complete scenes.
    """,
    "clip_raster": """
    <b>Clip data cube to polygon boundaries</b><br>
    <b>Off (default)</b>: the cube covers the polygon's <b>bounding box</b> — a clean
    rectangle. This is intentional: a rectangle gives the best results for
    <b>co-registration</b> and <b>super-resolution</b>.<br>
    <b>On</b>: the cube is cut to the <b>exact polygon outline</b>; pixels outside the
    shape become empty (NaN).<br><br>
    If you prefer to clip later instead, you can use <code>clip_stac()</code>.
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
    "keep_clouds": """
    <b>keep_clouds</b> (depends on <b>cloud_masking</b>)<br>
    Only selectable when <b>cloud_masking</b> is True (it needs the Scene
    Classification Layer).<br><br>
    <b>False</b> (default): cloudy pixels are removed (set to no-data).<br>
    <b>True</b>: the imagery is left untouched - clouds stay visible - but each
    scene is still tagged with a <code>cloud_percentage</code> (computed from the
    SCL, exactly like the masking would). So you keep the visual integrity of the
    scenes and can still filter out the fully-clouded dates by cloud %.<br><br>
    Currently SCL-only (Sentinel-2 L2A).
    """,
    "nir_dark_threshold": """
    <b>NIR Dark Threshold</b><br>
    A cloud shadow candidate is a pixel darker than this value in the
    near-infrared band (reflectance, 0-1). Water and no-data pixels are
    excluded automatically.<br><br>
    <b>Lower</b> (e.g. 0.15): stricter - only very dark shadows are masked,
    fewer false positives.<br>
    <b>Higher</b> (e.g. 0.25): catches lighter shadows, but dark surfaces
    (asphalt, dense city, dark soil) start being masked too.<br><br>
    Default: <code>0.18</code>.
    """,
    "shadow_proj_distance": """
    <b>Projection Distance (km)</b><br>
    How far each detected cloud is projected in the direction opposite the sun
    to search for its shadow. The shadow distance depends on cloud height:
    higher clouds cast shadows further away.<br><br>
    <b>1 km</b> (default): good for low/mid clouds, fewest false
    positives.<br>
    <b>Larger</b> (2-3 km): catches shadows of high clouds, but flags more dark
    surfaces along the way.<br><br>
    Shadows whose parent cloud lies <b>outside the data cube</b> cannot be
    projected at any distance.
    """,
    "resampling_method": """
    <b>Resampling Method</b><br>
    How pixels are interpolated when the source imagery is resampled onto your
    cube grid (e.g. 20 m bands to a 10 m cube).<br><br>
    <b>nearest</b> (default): keeps original pixel values, blockier look, recommended for analysis.<br>
    <b>bilinear</b>: smooth, recommended for better visual quality.<br>
    <b>bicubic</b>: smoothest visual, slightly slower.<br><br>
    The Scene Classification Layer (<code>scl</code>) is <b>always</b> loaded
    with nearest resampling regardless of this choice - interpolating between
    class codes would produce meaningless values.
    """,
    "crs": """
    <b>Output Projection (CRS)</b><br>
    The coordinate reference system your cube's grid is drawn in. Every scene is
    loaded onto this one grid, so scenes delivered in a different projection are
    re-drawn (resampled) into it. No scene is ever dropped because of this.<br><br>
    <b>Automatic</b> (default): uses the projection that natively covers most of
    your area, so the least data has to be re-drawn.<br><br>
    <b>Detected projections</b>: the projections your scenes actually come in.
    Sentinel-2 is delivered in UTM zones, e.g. <code>EPSG:32632</code> is
    <b>UTM zone 32N</b>. Note this is about which <b>tile</b> your data comes from,
    not which zone your polygon sits in on a map: tiles are 110 km squares that
    extend past the zone boundary, so an area straddling a zone line is very often
    still delivered entirely in one projection.<br><br>
    <b>Custom</b>: any projected, metre-based CRS. Typical reasons: an equal-area
    CRS when you report areas (e.g. <code>EPSG:3035</code>, ETRS89-LAEA Europe),
    a national grid, or matching a partner dataset.<br>
    Enter an EPSG code such as <code>EPSG:3035</code> or <code>EPSG:32632</code>.
    Geographic (degree-based) CRSs like <code>EPSG:4326</code> and non-metre CRSs
    like <code>EPSG:2263</code> (US survey foot) are rejected, because Resolution
    is expressed in CRS units and would silently mean degrees or feet.
    """,
    "stats": """
    <b>stats</b><br>
    Examples (for a date range of 2020-01-01 to 2021-12-31):
    <ul style="margin:4px 0 0 18px; padding:0;">
        <li><code>mean_timeseries</code> -> one image: the mean of every scene across 2020-2021</li>
        <li><code>mean_monthly</code> -> one image per month present: mean of Jan 2020, Feb 2020, ... Dec 2021</li>
        <li><code>mean_annual</code> -> one image per year: mean of 2020, and mean of 2021</li>
    </ul>
    """,
    "custom_composites": """
    <b>Custom Composites</b><br>
    A period you define yourself, instead of a whole month or year.<br><br>
    <b>Every year</b> repeats the period in each year the cube covers. A spring
    of <code>04-01</code> to <code>06-21</code> named <code>spring_mean</code>
    gives <code>spring_mean_2024</code>, <code>spring_mean_2025</code>, and so
    on - one image per year.<br><br>
    <b>Single window</b> uses full dates and gives one image, named exactly as
    you type it.<br><br>
    Both dates are included. A period that starts later than it ends
    (<code>12-01</code> to <code>02-28</code>) runs over New Year and is named
    after the year it starts in.<br><br>
    Composites are calculated from the dates left in the <b>Result</b> section
    above, after the cloud and coverage filters - no new scenes are downloaded.
    A year with no scene in the period is skipped.<br><br>
    Cloudy pixels that were masked out are left out of the statistic, so a pixel
    can be built from fewer dates than the period contains, and a pixel masked
    on every date stays empty.
    """,
    # Kept for the legacy `aggregator` parameter of get_stac_layers; the GUI's
    # own control is now the Temporal Composites section.
    "aggregator": """
    <b>Temporal Composite</b> (legacy parameter)<br>
    Collapses the time axis into a single scene (<code>mean</code> or
    <code>median</code>) per band/index, replacing the time series.<br><br>
    The interface no longer uses it: tick <b>Mean/Median of the time series</b>
    in <b>Temporal Composites</b> and untick <b>Keep the full time series</b>
    for the same result, with the composite named after the statistic.<br><br>
    """,
    "export_mode": """
    <b>Which format should I pick?</b><br>
    <b>NetCDF (.nc)</b> - one single file. Works with every Analysis Ready Data
    (ARD) cube tool here and is easy to copy or share. A solid all-round default.<br><br>
    <b>Zarr (.zarr)</b> - a chunked folder. Also works with every ARD cube tool,
    and is the quickest to read and write for very large cubes. To share it, zip
    the folder first.<br><br>
    <b>Cloud Optimized GeoTIFFs</b> - a folder with one GeoTIFF per date. The
    bands are already mapped, so you can drag them straight into QGIS (or another
    GIS) and view them right away. NOT accepted by the ARD cube tools. Best when
    your goal is viewing or sharing individual scenes.
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


def _with_help_left(widget, help_key, label_text=None):
    """Builder field with a help (?) toggle. Resolves the help text from the
    builder's PARAM_HELP_HTML and delegates layout to the shared helper."""
    return _field_with_help(widget, label_text, PARAM_HELP_HTML.get(help_key, ""))


def _polygon_coreg_size_hint(polygon, resolution=None):
    """Warn (string) when the AOI is likely too small for good
    co-registration; None when it is fine or cannot be judged.

    Co-registration accuracy is texture-limited on small areas (measured
    ~0.3 px residual on a 2.9 x 1.2 km AOI); the matching window is
    256 px, so the suggested minimum bounding-box edge is 256 * resolution
    (2.6 km at 10 m). Users who need a small cube can build a larger one,
    co-register it, and clip afterwards.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import box, shape

        if isinstance(polygon, (list, tuple)) and len(polygon) == 4:
            gdf = gpd.GeoDataFrame(
                geometry=[box(*map(float, polygon))], crs="EPSG:4326"
            )
        elif isinstance(polygon, dict) and "coordinates" in polygon:
            gdf = gpd.GeoDataFrame(geometry=[shape(polygon)], crs="EPSG:4326")
        elif isinstance(polygon, str):
            gdf = gpd.read_file(polygon)
            if gdf.empty:
                return None
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
        else:
            return None

        utm = gdf.estimate_utm_crs()
        minx, miny, maxx, maxy = gdf.to_crs(utm).total_bounds
        w_km = (maxx - minx) / 1000.0
        h_km = (maxy - miny) / 1000.0
        res = float(resolution) if resolution else 10.0
        min_edge_km = 256.0 * res / 1000.0
        if min(w_km, h_km) < min_edge_km:
            return (
                f"The area is small for good quality co-registration: the bounding "
                f"box is {w_km:.1f} x {h_km:.1f} km, while the suggested minimum "
                f"edge length is {min_edge_km:.1f} km at {res:.0f} m resolution. "
                "You can still build and co-register this cube, but shift accuracy "
                "will be texture-limited. Tip: build a larger cube, co-register it, "
                "then clip to your area of interest."
            )
    except Exception:
        return None
    return None


def _enlarge_polygon_for_coreg(polygon, resolution=None):
    """Enlarge the AOI so its bounding box meets the suggested minimum
    co-registration edge (256 px * resolution, i.e. 2.6 km at 10 m). Only
    edges below the minimum are expanded, symmetrically, so the original
    area stays centered (e.g. 5 x 2 km -> 5 x 2.6 km).

    Input / output:
    - bbox [xmin, ymin, xmax, ymax] (EPSG:4326) -> new bbox (list of 4 floats)
    - polygon file path -> path (str) of a NEW file written next to the
      original with an '_enlarged' stem suffix. Every feature whose bounding
      box is below the minimum edge is replaced by its enlarged bounding
      RECTANGLE; features already large enough keep their exact geometry.

    Note: enlargement is rectangular (bounding box based), so an irregular
    drawn polygon becomes a rectangle in the enlarged file - 'Clip to exact
    polygon outline' will then clip to that rectangle.
    """
    import geopandas as gpd
    from shapely.geometry import box

    res = float(resolution) if resolution else 10.0
    min_edge_m = 256.0 * res

    def _expand(minx, miny, maxx, maxy):
        if (maxx - minx) < min_edge_m:
            pad = (min_edge_m - (maxx - minx)) / 2.0
            minx, maxx = minx - pad, maxx + pad
        if (maxy - miny) < min_edge_m:
            pad = (min_edge_m - (maxy - miny)) / 2.0
            miny, maxy = miny - pad, maxy + pad
        return minx, miny, maxx, maxy

    # Bbox input: expand in the metric UTM CRS, then return the EPSG:4326
    # bounds of the expanded rectangle (the tightest axis-aligned 4326 bbox
    # that still contains it).
    if isinstance(polygon, (list, tuple)) and len(polygon) == 4:
        gdf = gpd.GeoDataFrame(
            geometry=[box(*map(float, polygon))], crs="EPSG:4326"
        )
        utm = gdf.estimate_utm_crs()
        expanded = box(*_expand(*gdf.to_crs(utm).total_bounds))
        back = gpd.GeoDataFrame(geometry=[expanded], crs=utm).to_crs("EPSG:4326")
        return [float(v) for v in back.total_bounds]

    if not isinstance(polygon, str):
        raise ValueError(
            "Only a polygon file path or a [xmin, ymin, xmax, ymax] bbox "
            "can be resized."
        )

    src = Path(polygon)
    if not src.exists():
        raise ValueError(f"Polygon file not found: {src}")
    gdf = gpd.read_file(src)
    if gdf.empty:
        raise ValueError(f"Polygon file has no features: {src}")
    orig_crs = gdf.crs
    if orig_crs is None:
        gdf = gdf.set_crs("EPSG:4326")
        orig_crs = gdf.crs

    utm = gdf.estimate_utm_crs()
    utm_gdf = gdf.to_crs(utm)
    new_geoms = []
    for geom in utm_gdf.geometry:
        minx, miny, maxx, maxy = geom.bounds
        if (maxx - minx) < min_edge_m or (maxy - miny) < min_edge_m:
            new_geoms.append(box(*_expand(minx, miny, maxx, maxy)))
        else:
            new_geoms.append(geom)
    out = utm_gdf.set_geometry(new_geoms).to_crs(orig_crs)

    # Re-clicking on an already-enlarged file overwrites it instead of
    # stacking suffixes (foo_enlarged_enlarged...).
    stem = src.stem
    if not stem.endswith("_enlarged"):
        stem += "_enlarged"
    out_path = src.with_name(stem + src.suffix)
    out.to_file(out_path)
    return out_path.as_posix()


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
            "scl": "20m",
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
        # scl is a classification layer, not a spectral band - keep it at
        # the very end of the list regardless of its native resolution.
        if str(band).lower() == "scl":
            return (99999, idx)
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

    # Required bands per index, mirroring calculate_spectral_index() in
    # get_spectral_indices.py. Values are conceptual band slots translated to
    # mission-specific band names (e.g. swir1 -> swir16 on Sentinel-2) by
    # _index_required_band_slots() below. Keep in sync with that module.
    _OPTICAL_INDEX_BANDS = {
        "ndvi": ["red", "nir"],
        "ndwi": ["green", "nir"],
        "savi": ["red", "nir"],
        "ndmi": ["nir", "swir1"],
        "nbr": ["nir", "swir2"],
        "mndwi": ["green", "swir1"],
        "ndbi": ["nir", "swir1"],
        "evi": ["blue", "red", "nir"],
        "ndre1": ["nir", "rededge1"],
        "ndsi": ["green", "swir1"],
    }
    _SAR_INDEX_BANDS = {
        "vh/vv": ["vh", "vv"],
        "vv/vh": ["vv", "vh"],
        "rvi": ["vh", "vv"],
    }

    def _index_required_band_slots(mission_name: str, idx):
        """Mission-specific band requirements for one index.

        Returns a list of "slots"; each slot is a tuple of band names of which
        ANY ONE satisfies the requirement (e.g. Sentinel-2 NIR accepts "nir"
        or the "nir08" fallback). Returns None for indices whose requirements
        are not tracked here, so they are never greyed out by mistake. Mirrors
        the _require_band() calls in calculate_spectral_index().
        """
        idx = str(idx).lower()
        is_s2 = mission_name.startswith("sentinel_2")
        is_ls = mission_name.startswith("landsat")
        req = (
            _SAR_INDEX_BANDS if mission_name == "sentinel_1_rtc"
            else _OPTICAL_INDEX_BANDS
        ).get(idx)
        if req is None:
            return None
        slots = []
        for concept in req:
            if concept == "nir" and is_s2:
                slots.append(("nir", "nir08"))
            elif concept == "swir1":
                slots.append(
                    ("swir16",) if is_s2
                    else ("swir1",) if is_ls
                    else ("swir1", "swir16")
                )
            elif concept == "swir2":
                slots.append(
                    ("swir22",) if is_s2
                    else ("swir2",) if is_ls
                    else ("swir2", "swir22")
                )
            else:
                slots.append((concept,))
        return slots

    def _index_requirements_text(mission_name: str, idx):
        """Human-readable required-bands list, e.g. 'red, nir/nir08'."""
        slots = _index_required_band_slots(mission_name, idx)
        if not slots:
            return ""
        return ", ".join("/".join(s) for s in slots)

    def _daterange_mode_placeholder(mode_value: str):
        if mode_value == "seasonal_all":
            return '{"season": ["04-01", "10-31"], "years": "all"}'
        elif mode_value == "seasonal_range":
            return '{"season": ["04-01", "10-31"], "years": "2019-2024"}'
        elif mode_value == "seasonal_selected":
            return '{"season": ["04-01", "10-31"], "years": [2019, 2021, 2023]}'
        return '{"season": ["04-01", "10-31"], "years": "all"}'

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

    # Only the two Sentinel-2 collections are wired end-to-end today. The others
    # stay listed - so it is visible which missions the table knows about - but
    # they cannot be picked: ipywidgets offers no way to grey out a single
    # Dropdown option, so the selection is snapped back by the guard below (same
    # limitation that turned the index list into checkboxes).
    _SELECTABLE_MISSIONS = ("sentinel_2_l2a", "sentinel_2_l1c")

    def _mission_option_label(name: str):
        label = _pretty_mission_label(name)
        return label if name in _SELECTABLE_MISSIONS else f"{label} (not available yet)"

    mission_options = [(_mission_option_label(name), name) for name in ordered_names]

    # -------------------------------------------------------------------------
    # Widgets (Basic)
    # -------------------------------------------------------------------------
    _default_mission = next(
        (n for n in ordered_names if n in _SELECTABLE_MISSIONS), ordered_names[0]
    )

    mission_dd = widgets.Dropdown(
        options=mission_options,
        value=_default_mission,
        description="Mission:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
    )

    # Reason shown when a listed-but-unavailable mission is clicked.
    mission_note = widgets.HTML("")

    _mission_reverting = {"busy": False}

    def _guard_mission_choice(change):
        """Snap the dropdown back when the picked mission is not available yet.

        Registered before every other mission observer, and the revert is a
        nested value change, so the rest of the form is re-synced against the
        mission that is actually kept.
        """
        if _mission_reverting["busy"]:
            return
        if change["new"] in _SELECTABLE_MISSIONS:
            mission_note.value = ""
            return

        keep = change["old"] if change["old"] in _SELECTABLE_MISSIONS else _default_mission
        _mission_reverting["busy"] = True
        try:
            mission_dd.value = keep
        finally:
            _mission_reverting["busy"] = False

        mission_note.value = (
            "<div style='font-size:12px; color:#92400e; background:#fffbeb; "
            "border:1px solid #fde68a; border-radius:6px; padding:6px 8px;'>"
            f"<b>{_pretty_mission_label(change['new'])}</b> is not available yet - "
            f"kept {_pretty_mission_label(keep)}.</div>"
        )

    mission_dd.observe(_guard_mission_choice, names="value")

    source_w = widgets.Dropdown(
        options=[
            ("Planetary Computer (Microsoft)", "planetary_computer"),
            ("Element84 (Earth Search)", "element84"),
            ("terrabyte (DLR)", "terrabyte"),
            ("Copernicus Data Space Ecosystem (Copernicus)", "cdse"),
        ],
        value="planetary_computer",
        description="Data Source:",
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
        description="",
        placeholder="./polygons/test.gpkg",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "0px"},
    )

    daterange_mode_w = widgets.Dropdown(
        options=[
            ("Seasonal (all available years)", "seasonal_all"),
            ("Seasonal (year range)", "seasonal_range"),
            ("Seasonal (selected years only)", "seasonal_selected"),
        ],
        value="seasonal_all",
        description="Season mode:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
    )

    daterange_w = widgets.Text(
        value=_daterange_mode_placeholder("seasonal_all"),  # prefilled example
        description="Daterange:",
        placeholder=_daterange_mode_placeholder("seasonal_all"),
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
    )

    # Friendly date inputs for the common single-window case. The Python-style
    # text box above (daterange_w) is kept for the advanced seasonal / multi-year
    # modes, shown only when "Advanced date options" is ticked.
    date_from_w = widgets.DatePicker(
        value=_date(2024, 4, 1),
        layout=widgets.Layout(width="100%"),
    )
    date_to_w = widgets.DatePicker(
        value=_date(2024, 4, 10),
        layout=widgets.Layout(width="100%"),
    )
    advanced_dates_w = widgets.Checkbox(
        value=False,
        description="Use a seasonal date range (repeating across years)",
        indent=False,
        # Full row width: without it the checkbox keeps ipywidgets' default
        # ~300px inline width and ellipsizes the label ("..."). 99% not 100%:
        # at 100% the widget's internal margins overflow the Time Period group
        # by a sliver and draw a useless horizontal scrollbar.
        layout=widgets.Layout(width="99%"),
    )

    bands_w = widgets.SelectMultiple(
        options=[],
        value=(),
        description="Bands:",
        rows=8,
        layout=widgets.Layout(width="100%", height="220px"),
        style={"description_width": "120px"},
    )

    # Indices are individual checkbox rows (not a SelectMultiple like the
    # bands) so that each index can be greyed out on its own when the bands it
    # needs are not selected above - ipywidgets cannot disable single options
    # inside a SelectMultiple.
    _index_rows = {}  # index name -> its Checkbox row

    # overflow "hidden auto" = x hidden, y auto: scroll vertically when the
    # list outgrows max_height, but clip the classic 1px horizontal sliver
    # that would otherwise draw a useless horizontal scrollbar across the box.
    # (ipywidgets 8 has no Layout.overflow_y trait, only the shorthand.)
    indices_w = widgets.VBox(
        [],
        layout=widgets.Layout(width="100%", max_height="260px", overflow="hidden auto"),
    )

    def _index_row_html(mission_name, idx, missing_slots=None):
        full = _index_fullname_map(mission_name).get(str(idx))
        label = f"<b>{idx}</b>" + (f" ({full})" if full else "")
        req = _index_requirements_text(mission_name, idx)
        if req:
            label += f" - {req}"
        if missing_slots:
            miss = ", ".join("/".join(s) for s in missing_slots)
            return (
                f"<span style='color:#9ca3af;'>{label} "
                f"<i>(missing: {miss})</i></span>"
            )
        return label

    def _refresh_index_availability(*_):
        """Grey out (and untick) indices whose required bands are not selected."""
        m_name = mission_dd.value
        selected = {str(b).lower() for b in bands_w.value}
        for idx, cb in _index_rows.items():
            slots = _index_required_band_slots(m_name, idx)
            if slots is None:
                # Requirements not tracked for this index: keep it selectable.
                missing = []
            else:
                missing = [s for s in slots if not any(b in selected for b in s)]
            if missing and cb.value:
                cb.value = False
            cb.disabled = bool(missing)
            cb.description = _index_row_html(m_name, idx, missing or None)

    def _set_index_options(mission_name, index_list):
        _index_rows.clear()
        if not index_list:
            indices_w.children = (
                widgets.HTML(
                    "<i style='color:#6b7280;'>No indices available for this "
                    "mission.</i>"
                ),
            )
            return
        for idx in index_list:
            cb = widgets.Checkbox(
                value=False,
                indent=False,
                description="",
                layout=widgets.Layout(width="100%"),
            )
            # The label is the checkbox's own description (HTML enabled), so
            # clicking the text toggles the box, and a wide label column lets
            # long index names wrap instead of being clipped.
            cb.description_allow_html = True
            cb.style.description_width = "330px"
            _index_rows[str(idx)] = cb
        indices_w.children = tuple(_index_rows.values())
        _refresh_index_availability()

    def _selected_index_values():
        return [
            idx for idx, cb in _index_rows.items() if cb.value and not cb.disabled
        ]

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
    clip_raster_w = widgets.Checkbox(
        value=False,
        description="Clip to exact polygon outline",
        indent=False,
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

    # Action selector once Cloud masking is on: either mask out the cloudy pixels
    # (default) OR keep them visible but still tag each scene with a cloud % (for
    # artists / filtering). Framing it as "Mask or Keep" avoids the confusing
    # "enable masking in order to keep clouds" of a separate Keep-Clouds toggle.
    # Only meaningful when Cloud masking is on; greyed to "Mask clouds" otherwise.
    keep_clouds_w = widgets.Dropdown(
        options=[("Mask clouds", "mask"), ("Keep clouds", "keep")],
        value="mask",
        description="Mask or keep:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
        disabled=True,
    )

    # Export the binary SCL cloud mask (cloud_mask_output) as its own NetCDF, so a
    # kept-clouds cube can still be masked / filtered / co-registered later. Only
    # meaningful when cloud detection is on. When True, a path field appears
    # (auto-filled <polygon>_mask_binary.nc); when False the path box is hidden.
    export_mask_w = widgets.Dropdown(
        options=[("False", False), ("True", True)],
        value=False,
        description="Export mask:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
        disabled=True,
    )
    cloud_mask_output_w = widgets.Text(
        value="",
        placeholder="./results/<polygon>_mask_binary.nc",
        layout=widgets.Layout(width="100%"),
        disabled=True,
    )
    export_mask_path_box = widgets.VBox(
        [_stacked_field(cloud_mask_output_w, "Binary mask output (.nc)")],
        layout=widgets.Layout(width="100%", display="none"),  # shown only when True
    )

    def _sync_export_mask_visibility(change=None):
        """Path box shows only when Export mask is True."""
        export_mask_path_box.layout.display = (
            "" if export_mask_w.value is True else "none"
        )

    def _export_mask_user_controlled():
        """Export mask is the user's to set under 'No masking' (preset 1) and
        'Free settings' (preset 3); the 'Mask clouds' preset keeps it greyed."""
        return _cloud_preset_state["n"] in (1, 3)

    def _sync_export_mask_path_enabled():
        """Path field is editable when the user controls Export mask (No masking /
        Free settings) and it is on; otherwise greyed. Auto-fill when relevant."""
        on = (export_mask_w.value is True) and (cloud_masking_w.value is True)
        if on and not (cloud_mask_output_w.value or "").strip():
            cloud_mask_output_w.value = _auto_mask_binary_suggestion()
        cloud_mask_output_w.disabled = not (_export_mask_user_controlled() and on)

    def _sync_keep_clouds_enabled(change=None):
        """Mask-or-Keep AND Export-mask both need cloud detection on (they use the
        SCL layer). When it's off, force them back to their defaults and grey them
        out. Governs the manual (Free Settings) interactions; presets set states
        directly and override afterwards."""
        on = (cloud_masking_w.value is True)
        if not on and keep_clouds_w.value != "mask":
            keep_clouds_w.value = "mask"
        keep_clouds_w.disabled = not on
        if not on and export_mask_w.value is not False:
            export_mask_w.value = False
        export_mask_w.disabled = not on
        _sync_export_mask_path_enabled()

    cloud_masking_w.observe(_sync_keep_clouds_enabled, names="value")
    export_mask_w.observe(
        lambda c: (_sync_export_mask_visibility(), _sync_export_mask_path_enabled()),
        names="value",
    )

    # -------------------------------------------------------------------------
    # Cloud Shadow Masking (GEE s2cloudless approach on the SCL cloud mask).
    # Usable only when clouds are actually being MASKED (shadow is projected
    # from the detected clouds - Rule 1) and the nir band is selected (the
    # dark-pixel test needs it). Its two parameters stay greyed until shadow
    # masking is switched on (Rule 2).
    # -------------------------------------------------------------------------
    shadow_masking_w = widgets.Checkbox(
        value=False,
        description="Mask cloud shadows",
        indent=False,
        disabled=True,
    )
    shadow_nir_dark_w = widgets.FloatText(
        value=0.18,
        step=0.01,
        description="NIR dark threshold:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "160px"},
        disabled=True,
    )
    shadow_proj_dist_w = widgets.FloatText(
        value=1.0,
        step=0.5,
        description="Projection distance (km):",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "160px"},
        disabled=True,
    )
    # One-line reason why the checkbox is greyed, updated by the sync below.
    shadow_gate_note = widgets.HTML("")

    def _shadow_masking_allowed():
        """Rule 1 + band requirement + mission capability, with the reason."""
        if mission_dd.value != "sentinel_2_l2a":
            return False, "Available for Sentinel-2 L2A only."
        if not (cloud_masking_w.value is True and keep_clouds_w.value == "mask"):
            return False, (
                "Requires <b>Mask Clouds</b> (select the Mask Clouds preset, or "
                "set Cloud Detection True + Mask or Keep = Mask clouds)."
            )
        if "nir" not in [str(b).lower() for b in bands_w.value]:
            return False, "Requires the <b>nir</b> band to be selected above."
        return True, ""

    def _sync_shadow_masking_enabled(change=None):
        allowed, reason = _shadow_masking_allowed()
        if not allowed and shadow_masking_w.value:
            shadow_masking_w.value = False
        shadow_masking_w.disabled = not allowed
        shadow_gate_note.value = (
            ""
            if allowed
            else f"<div style='font-size:12px; color:#9a3412;'>{reason}</div>"
        )
        # Rule 2: the two parameters open up only while shadow masking is on.
        params_on = allowed and (shadow_masking_w.value is True)
        shadow_nir_dark_w.disabled = not params_on
        shadow_proj_dist_w.disabled = not params_on

    shadow_masking_w.observe(_sync_shadow_masking_enabled, names="value")
    bands_w.observe(_sync_shadow_masking_enabled, names="value")
    cloud_masking_w.observe(_sync_shadow_masking_enabled, names="value")
    keep_clouds_w.observe(_sync_shadow_masking_enabled, names="value")
    mission_dd.observe(_sync_shadow_masking_enabled, names="value")
    _sync_shadow_masking_enabled()  # initial state (greyed with reason)

    # -------------------------------------------------------------------------
    # Resampling method (Advanced): spectral bands are resampled onto the cube
    # grid with this method; categorical layers (scl) are pinned to nearest
    # inside get_stac regardless of the choice.
    # -------------------------------------------------------------------------
    resampling_w = widgets.Dropdown(
        options=[("nearest (default)", "nearest"), ("bilinear", "bilinear"),
                 ("bicubic", "bicubic")],
        value="nearest",
        description="Resampling:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
    )

    # -------------------------------------------------------------------------
    # Tile handling (Advanced): how AOIs that straddle two adjacent Sentinel-2
    # MGRS tiles are handled. "mosaic" (default) merges the tiles of each solar
    # day into one timestep (current behaviour); "separate" keeps each tile's
    # acquisition as its own timestep with a `tile` coordinate. Sentinel-2 L2A
    # only (get_stac_layers rejects it elsewhere); greyed otherwise.
    # -------------------------------------------------------------------------
    tile_handling_w = widgets.Dropdown(
        options=[("Mosaic tiles", "mosaic"),
                 ("Separate tiles", "separate")],
        value="mosaic",
        description="Tiles:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
    )
    tile_handling_note = widgets.HTML("")

    def _sync_tile_handling(change=None):
        """Enable only for Sentinel-2 L2A; show a one-line reason otherwise, and
        an explainer when 'Separate tiles' is chosen."""
        s2l2a = mission_dd.value == "sentinel_2_l2a"
        if not s2l2a:
            if tile_handling_w.value != "mosaic":
                tile_handling_w.value = "mosaic"
            tile_handling_w.disabled = True
            tile_handling_note.value = (
                "<div style='font-size:12px; color:#9a3412;'>"
                "Separating tiles is available for Sentinel-2 L2A only.</div>"
            )
            return
        tile_handling_w.disabled = False
        if tile_handling_w.value == "separate":
            tile_handling_note.value = (
                "<div style='font-size:12px; color:#1e40af; background:#eff6ff; "
                "border:1px solid #bfdbfe; border-radius:6px; padding:6px 8px;'>"
                "Each scene that straddles two adjacent Sentinel-2 tiles is kept "
                "as its own timestep (with a <b>tile</b> coordinate) instead of "
                "being merged. Cannot be combined with a Temporal Composite.</div>"
            )
        else:
            tile_handling_note.value = ""

    tile_handling_w.observe(_sync_tile_handling, names="value")
    mission_dd.observe(_sync_tile_handling, names="value")

    # -------------------------------------------------------------------------
    # Across-track (East-West): partial-scene handling. Scenes near a swath /
    # orbit edge image only part of the AOI (the rest loads as NaN). This lets
    # the user drop those partial scenes. Optical time-series missions only.
    # -------------------------------------------------------------------------
    _PARTIAL_OK_MISSIONS = ("sentinel_2_l2a", "sentinel_2_l1c", "landsat_c2_l2")
    partial_scene_w = widgets.Dropdown(
        options=[("Keep all scenes", "keep"),
                 ("Remove partially missing scenes", "remove")],
        value="keep",
        description="Coverage:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
    )
    # Minimum share of the area a scene must image to be kept (percent). Maps to
    # min_scene_coverage = value / 100. Shown only in "Remove" mode.
    min_coverage_w = widgets.BoundedIntText(
        value=90, min=0, max=100, step=1,
        description="Min coverage %:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
    )
    # _boxed() strips the widget's own description, so the threshold field would
    # show as a bare "90". Use a stacked label instead to keep it self-explaining.
    min_coverage_box = widgets.VBox(
        [_stacked_field(min_coverage_w, "Min coverage %")],
        layout=widgets.Layout(width="100%", display="none"),  # shown in Remove mode
    )
    partial_scene_note = widgets.HTML("")

    def _sync_partial_scene(change=None):
        """Enable for optical time-series missions; reveal the threshold and
        explain when 'Remove' is on."""
        ok = mission_dd.value in _PARTIAL_OK_MISSIONS
        if not ok:
            if partial_scene_w.value != "keep":
                partial_scene_w.value = "keep"
            partial_scene_w.disabled = True
            min_coverage_box.layout.display = "none"
            partial_scene_note.value = (
                "<div style='font-size:12px; color:#9a3412;'>"
                "Available for optical missions (Sentinel-2, Landsat).</div>"
            )
            return
        partial_scene_w.disabled = False
        _remove = partial_scene_w.value == "remove"
        min_coverage_box.layout.display = "" if _remove else "none"
        if _remove:
            partial_scene_note.value = (
                "<div style='font-size:12px; color:#1e40af; background:#eff6ff; "
                "border:1px solid #bfdbfe; border-radius:6px; padding:6px 8px;'>"
                "Scenes imaging less than the <b>Min coverage %</b> of the area "
                "(swath / orbit edge) are dropped. Clouds do not count as "
                "missing - a fully-imaged but cloudy scene is kept. "
                "Note: building the preview can take longer.</div>"
            )
        else:
            partial_scene_note.value = ""

    partial_scene_w.observe(_sync_partial_scene, names="value")
    mission_dd.observe(_sync_partial_scene, names="value")

    # -------------------------------------------------------------------------
    # Scene Specific Metadata (Advanced): optional per-scene STAC item
    # properties attached to the cube as (time,) coordinates, so scenes can be
    # queried/filtered by them later (e.g. keep only relative orbit 65). The
    # option list follows the selected Data Source: fields a source does not
    # publish are hidden entirely, so nothing all-NaN can be selected
    # (availability verified per source in get_data.SCENE_METADATA_AVAILABILITY).
    # -------------------------------------------------------------------------
    _SCENE_METADATA_LABELS = [
        ("acq_datetime - full acquisition timestamp", "acq_datetime"),
        ("sun_azimuth - mean solar azimuth [deg]", "sun_azimuth"),
        ("sun_elevation - mean solar elevation [deg]", "sun_elevation"),
        ("view_azimuth - mean viewing azimuth [deg]", "view_azimuth"),
        ("incidence_angle - mean viewing incidence angle [deg]", "incidence_angle"),
        ("relative_orbit - relative orbit number", "relative_orbit"),
        ("processing_baseline - processing baseline version", "processing_baseline"),
    ]
    scene_metadata_w = widgets.SelectMultiple(
        options=[],
        value=(),
        description="Metadata:",
        rows=7,
        layout=widgets.Layout(width="100%", height="190px"),
        style={"description_width": "120px"},
    )
    scene_meta_all_btn = widgets.Button(
        description="All fields", layout=widgets.Layout(width="110px")
    )
    scene_meta_none_btn = widgets.Button(
        description="Clear fields", layout=widgets.Layout(width="110px")
    )
    # One-line availability note under the list (why some fields are hidden).
    scene_meta_note = widgets.HTML("")

    # Download the granule metadata XML (MTD_TL.xml) of every scene at export
    # time, into <export target>_granule_metadata/. Off by default. Greyed on
    # terrabyte: its STAC publishes cluster-local file:// asset paths that are
    # not downloadable from outside the cluster. 99% width (not 100%): at 100%
    # the checkbox's internal margins overflow the group by a sliver and draw
    # a useless horizontal scrollbar (same fix as the seasonal-dates checkbox).
    export_granule_meta_w = widgets.Checkbox(
        value=False,
        description="Export Granule Metadata",
        indent=False,
        disabled=True,
        layout=widgets.Layout(width="99%"),
    )
    export_granule_meta_note = widgets.HTML("")

    def _sync_scene_metadata_options(change=None):
        """Rebuild the option list from the mission + Data Source. Fields the
        source does not publish are hidden (not greyed), and any selection of
        a now-hidden field is dropped."""
        m_name = mission_dd.value
        src = source_w.value
        s2 = m_name in ("sentinel_2_l2a", "sentinel_2_l1c")
        if s2 and src:
            avail = SCENE_METADATA_AVAILABILITY.get(src, [])
            keep = tuple(v for v in scene_metadata_w.value if v in avail)
            scene_metadata_w.options = [
                (lbl, v) for lbl, v in _SCENE_METADATA_LABELS if v in avail
            ]
            scene_metadata_w.value = keep
            scene_metadata_w.disabled = False
            scene_meta_all_btn.disabled = False
            scene_meta_none_btn.disabled = False
            hidden = [v for _, v in _SCENE_METADATA_LABELS if v not in avail]
            scene_meta_note.value = (
                ""
                if not hidden
                else (
                    "<div style='font-size:12px; color:#9a3412;'>"
                    f"Not published by this data source (hidden): "
                    f"<b>{', '.join(hidden)}</b>. Available with "
                    "<b>terrabyte</b> and <b>cdse</b>.</div>"
                )
            )
            # Granule metadata XML download: terrabyte's asset hrefs are
            # cluster-local file:// paths - not downloadable from here.
            if src == "terrabyte":
                if export_granule_meta_w.value:
                    export_granule_meta_w.value = False
                export_granule_meta_w.disabled = True
                export_granule_meta_note.value = (
                    "<div style='font-size:12px; color:#9a3412;'>"
                    "Not downloadable from the terrabyte catalogue (it stores "
                    "cluster-local file paths).</div>"
                )
            else:
                export_granule_meta_w.disabled = False
                export_granule_meta_note.value = ""
        else:
            scene_metadata_w.options = []
            scene_metadata_w.value = ()
            scene_metadata_w.disabled = True
            scene_meta_all_btn.disabled = True
            scene_meta_none_btn.disabled = True
            scene_meta_note.value = (
                "<div style='font-size:12px; color:#9a3412;'>"
                "Available for Sentinel-2 missions only.</div>"
            )
            if export_granule_meta_w.value:
                export_granule_meta_w.value = False
            export_granule_meta_w.disabled = True
            export_granule_meta_note.value = ""

    scene_meta_all_btn.on_click(
        lambda _b: setattr(
            scene_metadata_w, "value",
            tuple(v for _, v in scene_metadata_w.options),
        )
    )
    scene_meta_none_btn.on_click(
        lambda _b: setattr(scene_metadata_w, "value", ())
    )
    # Follow BOTH: a mission switch rewrites the source options (see
    # _update_from_mission, which also calls this sync directly so the widget
    # can never lag one event behind), and a source change refilters the list.
    source_w.observe(_sync_scene_metadata_options, names="value")
    mission_dd.observe(_sync_scene_metadata_options, names="value")

    # -------------------------------------------------------------------------
    # Guided cloud presets: four plain-language choices that drive the three raw
    # cloud parameters for the user, so most people never have to touch the raw
    # widgets. Selecting 1-3 sets the parameters AND greys them out (so it feels
    # "taken care of"); option 4 unlocks them for manual editing. Mutually
    # exclusive - one is always selected - rendered as checkboxes per design.
    # -------------------------------------------------------------------------
    def _make_preset_row(title_html, desc_html=None):
        # Only the option TITLE is the checkbox's description (HTML enabled),
        # so clicking the title toggles the box - not just the little square.
        # The explanation is a separate, non-clickable HTML line underneath,
        # indented to sit flush under the title and wrapping freely at any
        # panel width. (The old single wide label forced a fixed 330px column
        # that overflowed narrow panels with a horizontal scrollbar.)
        cb = widgets.Checkbox(
            value=False,
            indent=False,
            description=title_html,
            layout=widgets.Layout(width="100%"),
        )
        cb.description_allow_html = True
        cb.style.description_width = "initial"
        children = [cb]
        if desc_html:
            children.append(
                widgets.HTML(
                    "<div style='font-size:12px; color:#6b7280; line-height:1.4; "
                    f"margin:-4px 0 0 26px;'>{desc_html}</div>"
                )
            )
        row = widgets.VBox(
            children,
            # x hidden: clip the 1px flex sliver instead of drawing a useless
            # horizontal scrollbar across the row.
            layout=widgets.Layout(width="100%", overflow="hidden"),
        )
        return cb, row

    cloud_preset1_cb, _cp1_row = _make_preset_row(
        "<b>No Masking</b>",
        "Keep all pixels. Select this if you want use s2cloudless masking "
        "later. Each scene is tagged with a cloud % so you can filter cloudy "
        "dates, saving you to apply s2cloudless on obviously fully cloudy "
        "scenes. Optionally export a binary mask to apply later.",
    )
    cloud_preset2_cb, _cp2_row = _make_preset_row(
        "<b>Mask Clouds</b>",
        "Remove cloudy pixels automatically (SCL). The quick way to a clean "
        "cube.",
    )
    cloud_preset3_cb, _cp3_row = _make_preset_row(
        "<b>Free Settings</b>",
        "Dude, I know what I am doing...",
    )

    _cloud_preset_cbs = [
        cloud_preset1_cb, cloud_preset2_cb, cloud_preset3_cb,
    ]
    _cloud_preset_label = widgets.HTML(
        "<div style='font-weight:600; font-size:13px; color:#374151;'>"
        "Choose how to handle clouds:</div>"
    )
    _cloud_preset_box = widgets.VBox(
        [_cloud_preset_label, _cp1_row, _cp2_row, _cp3_row],
        layout=widgets.Layout(width="100%", gap="6px", overflow="hidden"),
    )

    _cloud_preset_guard = {"busy": False}
    # The active preset (1-4) and what the current mission actually supports. The
    # mission gate is the hard limit; the preset's greying applies on top, so a
    # mission that can't do SCL masking stays disabled even under the manual
    # preset, and re-selecting a mission re-applies the active preset.
    _cloud_preset_state = {"n": 1}
    _cloud_caps = {"masking": True, "max_cc": True}

    def _set_cloud_params_disabled(disabled):
        """Grey (or free) the raw cloud widgets together. Mission capability is a
        hard gate: an unsupported control stays disabled even when a preset would
        otherwise free it."""
        hard_no_mask = not _cloud_caps["masking"]
        cloud_masking_w.disabled = disabled or hard_no_mask
        max_cc_w.disabled = disabled or (not _cloud_caps["max_cc"])
        export_mask_w.disabled = disabled or hard_no_mask
        if disabled or hard_no_mask:
            keep_clouds_w.disabled = True
            cloud_mask_output_w.disabled = True
        else:
            # Manual mode: Mask-or-Keep and Export-mask follow their dependency on
            # Cloud Detection (this also handles the path field + auto-fill).
            _sync_keep_clouds_enabled()
        _sync_export_mask_visibility()

    def _apply_cloud_preset(n):
        if n == 3:
            _set_cloud_params_disabled(False)  # Free settings: unlock, keep values
            return
        # Both remaining presets turn SCL cloud detection ON: preset 1 to compute
        # the cloud % while KEEPING every pixel, preset 2 to actually mask. Clamp
        # to what the mission offers (only None when masking is unavailable -> the
        # preset then simply can't detect / mask).
        valid = [v for _, v in cloud_masking_w.options]
        if True in valid:
            cloud_masking_w.value = True
        elif False in valid:
            cloud_masking_w.value = False
        keep_clouds_w.value = "keep" if n == 1 else "mask"
        max_cc_w.value = 100
        # Binary mask export defaults OFF for both; under "No masking" the user may
        # switch it on (it stays interactive), so they can produce a mask to apply
        # in the Data Cube Editor later.
        export_mask_w.value = False
        _set_cloud_params_disabled(True)
        if n == 1:
            # "No masking": free the Export-mask toggle (only when the mission can
            # actually detect clouds), so it is the one raw control the user keeps.
            export_mask_w.disabled = not (cloud_masking_w.value is True)
            _sync_export_mask_visibility()
            _sync_export_mask_path_enabled()

    def _select_cloud_preset(n, apply=True):
        _cloud_preset_state["n"] = n
        _cloud_preset_guard["busy"] = True
        try:
            for i, cb in enumerate(_cloud_preset_cbs, start=1):
                cb.value = (i == n)
        finally:
            _cloud_preset_guard["busy"] = False
        if apply:
            _apply_cloud_preset(n)

    def _on_cloud_preset_toggle(n):
        def _handler(change):
            if _cloud_preset_guard["busy"]:
                return
            if change["new"]:
                _select_cloud_preset(n)            # uncheck the others + apply
            else:
                # one option must always stay selected: re-check the active one
                _cloud_preset_guard["busy"] = True
                change["owner"].value = True
                _cloud_preset_guard["busy"] = False
        return _handler

    for _i, _cb in enumerate(_cloud_preset_cbs, start=1):
        _cb.observe(_on_cloud_preset_toggle(_i), names="value")

    # Default: option 1 (no masking, parameters greyed) - same baseline as before.
    _select_cloud_preset(1)

    # Result-panel filter: keep only scenes with cloud_percentage <= this value.
    # Lives under the Result date/cloud table (placed into result_box later) and
    # drives the table, visualization, and export at once. Pure coord selection
    # on the existing cloud_percentage coord - no recompute, instantly reversible
    # by raising the value back up. Enabled only after a build that carries a
    # cloud_percentage coord (i.e. cloud masking was on); greyed out otherwise.
    result_cloud_max_w = widgets.BoundedIntText(
        value=100,
        min=0,
        max=100,
        step=1,
        description="Max cloud %:",
        # ~25% narrower than the original 190px.
        layout=widgets.Layout(width="143px"),
        style={"description_width": "78px"},
        disabled=True,
    )

    # Result-panel filter #2, directly below Max cloud %: keep only scenes that
    # image at least this share of the area (the scene_coverage coord, 0..100%).
    # Drops across-track / swath-edge and faulty partial acquisitions WITHOUT a
    # rebuild - the same job the Advanced "Overlapping Tile Handling ->
    # Across-track" dropdown does at build time, but reversible and composable
    # with the cloud filter (a scene must pass BOTH).
    #
    # 0 = keep everything, so the filter is off by default. Enabled like the
    # cloud box: only when the build actually carries usable coverage numbers -
    # i.e. a READY (eager) scene_coverage coord, which is what cloud detection
    # produces from the SCL "imaged" boolean. With cloud detection off the coord
    # is left lazy on purpose (reading it would download a band), and the whole
    # Result panel stays read-free, so the box is greyed out rather than
    # triggering a hidden read.
    result_coverage_min_w = widgets.BoundedIntText(
        value=0,
        min=0,
        max=100,
        step=1,
        description="Min coverage %:",
        layout=widgets.Layout(width="167px"),
        style={"description_width": "102px"},
        disabled=True,
    )

    # Result-panel date picker: tick/untick individual acquisition dates to keep
    # or drop them. Like the Max cloud % filter, this is a reversible VIEW on the
    # built cube - state["result"] is never mutated, and unticked dates stay in
    # the list so a misclick is one click to undo. It composes with the cloud
    # filter: a date survives only if it is ticked AND under the cloud threshold.
    # Cloud % (when present) is shown per date. Populated + enabled after a
    # single-cube build with a time dimension; hidden/greyed for multi-feature
    # batches or cubes without a time axis. No pixels are read - times and
    # cloud_percentage are metadata, so this stays fully lazy.
    # Back to the original fixed 240x88 - a relative width overflows its
    # container here (the select's border/padding are added on top of 100%,
    # which is what produced the horizontal scrollbar). The entries are kept
    # SHORT ("2024-04-01 · 0% · 100%") so both percentages still fit at this
    # size, with the meaning carried by the legend above the box.
    result_date_w = widgets.SelectMultiple(
        options=[],
        value=(),
        description="",
        rows=4,
        layout=widgets.Layout(width="240px", height="88px"),
        style={"description_width": "0px"},
        disabled=True,
    )
    result_date_legend = widgets.HTML(
        "<div style='font-size:11px; color:#6b7280; line-height:1.3;'>"
        "date &nbsp;·&nbsp; cloud % &nbsp;·&nbsp; coverage %</div>"
    )
    result_date_all_btn = widgets.Button(
        description="All dates", layout=widgets.Layout(width="100px"), disabled=True
    )
    result_date_clear_btn = widgets.Button(
        description="Clear dates", layout=widgets.Layout(width="110px"), disabled=True
    )

    # The "these are filters" note lives BELOW the Max cloud box (placed into
    # result_box later), so it reads as the last step after the user has
    # optionally filtered. Filled in only when a ready cube is shown; cleared to
    # empty (renders nothing) otherwise.
    _RESULT_VIZ_NOTE_HTML = (
        "<div style='font-size:12px; color:#1e3a8a; background:#eff6ff; "
        "border:1px solid #bfdbfe; border-radius:6px; padding:6px 8px; "
        "margin:0;'>"
        "<ul style='margin:0; padding:0 0 0 18px;'>"
        "<li><b>Max cloud %</b> (estimated per scene, even if not masked) and "
        "<b>Date Selection</b> are filters: they decide which dates stay. "
        "<b>Temporal Composites</b> (changes Result), <b>Visualization</b> and "
        "<b>Export</b> below all use only those filtered dates.</li>"
        "<li><b>Export Current Result</b> exports exactly what is seen in the "
        "Result section.</li>"
        "<li><b>Strategy for s2cloudless:</b> filter out highly cloudy scenes, "
        "so the computationally heavy s2cloudless algorithm is not wasted on "
        "them - it saves you time and RAM by not exporting these useless "
        "scenes, and saves storage space too.</li>"
        "</ul>"
        "</div>"
    )
    # Collapsed by default, same one-line toggle as the notes strip above: it is
    # guidance to open when wanted, not something to re-read after every filter
    # change. Open/closed survives re-renders (_show_result_summary runs on each
    # filter change), so a strip that snapped shut each time would be unusable.
    result_viz_note_w = widgets.HTML(value="")
    result_viz_note_toggle = widgets.Button(
        description="",
        layout=widgets.Layout(width="100%", height="auto", display="none"),
    )
    result_viz_note_toggle.add_class("stac2cube-notes-toggle")
    result_viz_note_row = widgets.VBox(
        [result_viz_note_toggle, result_viz_note_w],
        layout=widgets.Layout(width="100%", gap="4px", margin="10px 0 0 0"),
    )
    result_viz_note_row.add_class("stac2cube-notes-row")

    # Co-registration size warning: when the AOI is small for good co-registration
    # this is shown BELOW the visualize/export note in the Result section (not in
    # Status, which is now taken by the "preview ready" message). Filled in by
    # _show_result_summary from state["coreg_size_hint"]; cleared otherwise.
    result_coreg_warn_w = widgets.HTML(
        value="", layout=widgets.Layout(flex="1 1 auto")
    )

    # Notes strip: ONE collapsed line at the top of the Result section holding
    # every blue ℹ️ notice the build produced (multi-swath coverage, projection).
    # Collapsed by default and never yellow - these describe what happened, they
    # are not a to-do list, and three stacked warning boxes made people feel they
    # had to clear them before the cube was usable. The genuinely actionable
    # notice (Area Size, which carries a Resize and Re-build button) deliberately
    # stays OUTSIDE this strip, in its own yellow box.
    #
    # The body's children are rebuilt on each render, so the strip needs no
    # per-notice widget and notices cannot go stale.
    result_notes_body = widgets.VBox(
        [], layout=widgets.Layout(width="100%", gap="6px", display="none")
    )
    # width="auto": the header fills the row by stretching, not by a percentage -
    # a rounded-up 100% loses its right border to the row's clip edge (see the
    # .stac2cube-notes-toggle rule in gui_common).
    result_notes_toggle = widgets.Button(
        description="",
        layout=widgets.Layout(width="auto", height="auto"),
    )
    result_notes_toggle.add_class("stac2cube-notes-toggle")
    # 99%, not 100%: at full width the strip ran visibly past the text it holds
    # and read as the widest thing in the Result panel. The trim applies to the
    # ROW, so the header button and every notice box below it shrink together and
    # their edges stay flush.
    result_notes_row = widgets.VBox(
        [result_notes_toggle, result_notes_body],
        layout=widgets.Layout(width="99%", gap="4px", display="none"),
    )
    result_notes_row.add_class("stac2cube-notes-row")
    # One-click fix for the warning above: writes an enlarged copy of the
    # polygon (short bbox edges expanded to the minimum co-registration edge,
    # original area kept centered) and re-runs the build with it - every other
    # parameter stays as selected. Shown only while the warning is shown
    # (see _set_coreg_warning).
    coreg_resize_btn = widgets.Button(
        description="Resize and Re-build Data Cube",
        button_style="warning",
        icon="expand",
        layout=widgets.Layout(width="auto", display="none", flex="0 0 auto"),
    )

    # -------------------------------------------------------------------------
    # Temporal Composites: statistics reduced over the dates kept in the Result
    # panel. The two most used ones (mean / median of the whole series) get
    # their own highlighted checkboxes; everything else - min/max/std and the
    # monthly / annual variants - lives in the "More composites" list below.
    #
    # Composites are computed AFTER the date and cloud filters, from the
    # surviving scenes, and they need no rebuild: the build only fetches the
    # time series, every composite is derived from it (see _apply_composites).
    # -------------------------------------------------------------------------
    _COMMON_COMPOSITES = ("mean_timeseries", "median_timeseries")

    comp_mean_w = widgets.Checkbox(
        value=False,
        description="Mean of the time series",
        indent=False,
        layout=widgets.Layout(width="99%"),
    )
    comp_median_w = widgets.Checkbox(
        value=False,
        description="Median of the time series",
        indent=False,
        layout=widgets.Layout(width="99%"),
    )

    # "More composites": the same SelectMultiple as before, minus the two
    # promoted above so no composite can be picked twice.
    stats_w = widgets.SelectMultiple(
        options=[],
        value=(),
        description="",
        rows=8,
        layout=widgets.Layout(width="100%", height="220px"),
        style={"description_width": "0px"},
    )

    stats_all_btn = widgets.Button(
        description="All", layout=widgets.Layout(width="70px")
    )
    stats_none_btn = widgets.Button(
        description="Clear", layout=widgets.Layout(width="70px")
    )

    # Off -> the time series is dropped from the export and only the composites
    # are written ("I just want the median"). Force-ticked and greyed while no
    # composite is selected, since dropping it then would leave nothing.
    keep_ts_w = widgets.Checkbox(
        value=True,
        description="Keep the full time series",
        indent=False,
        disabled=True,
        layout=widgets.Layout(width="99%"),
    )
    keep_ts_note = widgets.HTML("")

    # -- Custom Composites: user-defined periods, one row each ------------------
    # A row is either a season ("Every year", MM-DD, expanded to name_YYYY per
    # year the cube covers) or a single window (full dates, one variable). Rows
    # are added and removed at runtime, so the container's children are rebuilt
    # rather than declared here (see _custom_add_row / _custom_remove_row).
    custom_rows_box = widgets.VBox(layout=widgets.Layout(width="100%", gap="4px"))
    custom_add_btn = widgets.Button(
        description="Add composite",
        icon="plus",
        layout=widgets.Layout(width="150px"),
    )
    # Red list of the rows that are incomplete or wrong, and therefore ignored.
    custom_error_note = widgets.HTML("")

    # Legacy: the mean/median Temporal Composite dropdown this section replaces.
    # Kept as a hidden widget (never shown, never emitted) so the mission-meta
    # wiring and the paste-settings path that still reference it keep working.
    aggregator_w = widgets.Dropdown(
        options=[("None", None)],
        value=None,
        description="Temporal Composite:",
        layout=widgets.Layout(width="100%", display="none"),
        style={"description_width": "150px"},
    )
    # ipywidgets shows a BLANK label for value=None even when a ("None", None)
    # option exists, so set the label explicitly to display "None".
    aggregator_w.label = "None"

    # -------------------------------------------------------------------------
    # Widgets (Export)
    # -------------------------------------------------------------------------
    export_mode_w = widgets.Dropdown(
        options=[
            ("NetCDF (accepted for ARD Cube Tools)", "netcdf"),
            ("Zarr (accepted for ARD Cube Tools)", "zarr"),
            ("Geotiffs, Cloud Optimized (select a folder, NOT accepted for ARD Cube Tools)", "cogs"),
        ],
        value="netcdf",
        description="Export mode:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "120px"},
    )

    export_target_w = widgets.Text(
        value="",
        description="",
        placeholder="",
        disabled=False,
        layout=widgets.Layout(width="100%"),
        style={"description_width": "0px"},
    )

    # Lossless zlib compression for the exported cube. NetCDF only: COGs and
    # Zarr are already compressed by their own codecs, so the checkbox is
    # hidden for those modes (see _apply_compress_visibility).
    export_compress_w = widgets.Checkbox(
        value=False,
        description="Lossless compression (zlib)",
        indent=False,
        layout=widgets.Layout(width="auto"),
    )
    export_compress_warn_html = widgets.HTML(
        "<div style='font-size:12px; color:#b00020;'>"
        "⚠️ <b>Warning:</b> compression shrinks the output file a further "
        "~20-40% (scene-dependent), but the export step takes roughly "
        "<b>10x longer</b>. Enable it only for archiving, when disk space "
        "matters more than your time.</div>"
    )
    export_compress_warn_html.layout.display = "none"

    # Writes a small .vrt beside the NetCDF that labels every band with its date
    # and band name, so a GIS shows "2024-04-01 red" instead of "Band 3". NetCDF
    # only: COGs already carry band names, and a VRT over a Zarr store cannot
    # read its pixels back (see write_qgis_vrt).
    export_vrt_w = widgets.Checkbox(
        value=False,
        description="Export Band Mapping for GIS Tools (.vrt)",
        indent=False,
        layout=widgets.Layout(width="auto"),
    )
    export_vrt_note_html = widgets.HTML(
        "<div style='font-size:12px; color:#555; margin-left:2px;'>"
        "&#8505;&#65039; Open the <b>.vrt</b> file in QGIS, not the .nc, "
        "and keep both files in the same folder.</div>"
    )
    export_vrt_note_html.layout.display = "none"

    # Writes the current build settings as a config JSON beside the export -
    # same name, same folder, '.json' instead of the cube extension (for a COG
    # folder: inside it, named after the folder). Same content as Copy
    # Settings, so the file re-runs headless on SLURM and pastes back into this
    # form. Applies to all three export modes.
    export_settings_w = widgets.Checkbox(
        value=False,
        description="Export Settings (.json)",
        indent=False,
        layout=widgets.Layout(width="auto"),
    )
    export_settings_note_html = widgets.HTML(
        "<div style='font-size:12px; color:#555; margin-left:2px;'>"
        "&#8505;&#65039; Saved next to the export with the same name "
        "(<b>.json</b>). Same content as <b>Copy Settings</b>: re-run it on "
        "SLURM, or load it back with <b>Paste Settings</b>.</div>"
    )
    export_settings_note_html.layout.display = "none"

    # Statistics report: written AFTER the cube, read back from the file just
    # exported. That is what makes it free of extra scene reads - export_stac
    # streams and keeps nothing, so computing the statistics from the lazy
    # result instead would fetch every scene from the archive a second time.
    # NetCDF / Zarr only: a COG folder has no cube file to read back.
    export_csv_w = widgets.Checkbox(
        value=False,
        description="Export Statistics Report (.csv)",
        indent=False,
        layout=widgets.Layout(width="auto"),
    )
    export_csv_note_html = widgets.HTML(
        "<div style='font-size:12px; color:#555; margin-left:2px;'>"
        "&#8505;&#65039; Per-band mean, median, min, max and standard "
        "deviation for every date, year and month of the exported cube. "
        "Saved next to it as <b>&lt;cube&gt;_statistics.csv</b>.</div>"
    )
    export_csv_note_html.layout.display = "none"

    def _gate_export_option(box, note, available, reason):
        """Grey out an export option the current export mode cannot honour.

        Greyed and still visible, never removed: a box that disappears when the
        mode changes reads as a bug, and the user loses the fact that the
        option exists at all. A ticked box is unticked as it is disabled, so
        the export can never carry an intent the mode cannot honour, and the
        tooltip says why it is unavailable.
        """
        if not available and box.value:
            box.value = False
        box.disabled = not available
        box.tooltip = "" if available else reason
        note.layout.display = "" if (available and box.value) else "none"

    def _apply_compress_visibility(*_):
        netcdf_only = export_mode_w.value == "netcdf"
        # zlib compresses the cube file itself; Zarr and COGs already carry
        # their own codecs, so there is nothing for it to do there.
        _gate_export_option(
            export_compress_w, export_compress_warn_html, netcdf_only,
            "Available for NetCDF exports only: Zarr stores and GeoTIFFs are "
            "already compressed by their own codecs.",
        )
        # A VRT cannot read a Zarr store's pixels back, and COGs already carry
        # their band names, so the band mapping is NetCDF-only.
        _gate_export_option(
            export_vrt_w, export_vrt_note_html, netcdf_only,
            "Available for NetCDF exports only: a VRT cannot read a Zarr "
            "store's pixels, and GeoTIFFs already carry their band names.",
        )
        # The settings JSON works for every mode, so only its note toggles.
        export_settings_note_html.layout.display = (
            "" if export_settings_w.value else "none"
        )
        # The statistics CSV is read back from the exported cube, so it needs a
        # cube FILE - which a folder of GeoTIFFs is not.
        _gate_export_option(
            export_csv_w, export_csv_note_html,
            export_mode_w.value in ("netcdf", "zarr"),
            "Available for NetCDF and Zarr exports only: the table is read "
            "back from the exported cube, and a GeoTIFF folder is not one.",
        )

    export_compress_w.observe(_apply_compress_visibility, names="value")
    export_vrt_w.observe(_apply_compress_visibility, names="value")
    export_settings_w.observe(_apply_compress_visibility, names="value")
    export_csv_w.observe(_apply_compress_visibility, names="value")

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
        max_height="260px",
        overflow="auto",
    ))

    viz_out = widgets.Output(layout=widgets.Layout(
        border="1px solid #e5e7eb",
        padding="10px",
        border_radius="8px",
        width="99%",
        min_height="90px",
    ))

    # Animation status has its own box: the viewer and the animation maker are
    # different tools, so GIF prompts/errors must never land in the interactive
    # view's output.
    anim_out = widgets.Output(layout=widgets.Layout(
        border="1px solid #e5e7eb",
        padding="10px",
        border_radius="8px",
        width="99%",
        min_height="40px",
    ))

    generate_btn = widgets.Button(
        description="Build Data Cube Preview",
        button_style="success",
        icon="play",
        layout=widgets.Layout(width="220px"),
    )
    export_result_btn = widgets.Button(
        description="Export Current Result",
        icon="download",
        layout=widgets.Layout(width="220px"),
        disabled=True,
    )
    # Warm, energetic orange - a confident "go" call-to-action that stands out
    # from the green Build button without the alarm of the old red (danger).
    export_result_btn.style.button_color = "#f97316"
    copy_json_btn = widgets.Button(
        description="Copy Settings",
        icon="copy",
        layout=widgets.Layout(width="150px"),  # colorless like old Generate JSON button
    )
    # Counterpart of Copy Settings: take a settings JSON (from this GUI or from a
    # SLURM config file) and push it back into the widgets. The browser clipboard
    # cannot be read from the kernel, so the button reveals a paste box instead;
    # the settings are applied as soon as valid JSON lands in it.
    paste_json_btn = widgets.Button(
        description="Paste Settings",
        icon="paste",
        layout=widgets.Layout(width="150px"),
    )
    paste_json_area_w = widgets.Textarea(
        value="",
        placeholder='Paste the copied settings here (Ctrl+V) - {"parameters": {...}}',
        layout=widgets.Layout(width="99%", height="120px"),
        continuous_update=True,   # paste syncs immediately, no blur needed
    )
    paste_json_hint = widgets.HTML(
        "<div style='font-size:12px; color:#6b7280;'>"
        "Paste a settings JSON here (Ctrl+V). It is applied to the form "
        "automatically as soon as it parses.</div>"
    )
    paste_json_box = widgets.VBox(
        [paste_json_hint, paste_json_area_w],
        layout=widgets.Layout(width="100%", display="none"),  # toggled by the button
    )

    # -------------------------------------------------------------------------
    # Visualization widgets (disabled until cube is generated)
    # -------------------------------------------------------------------------
    viz_dropdown_btn = widgets.Button(
        description="Open Interactive View",
        button_style="info",
        icon="image",
        layout=widgets.Layout(width="260px"),
        disabled=True,
    )

    viz_renderer_w, viz_renderer_box = _make_viz_renderer_control()

    # Multi-feature picker: choose which cube (polygon feature) the viz tools act
    # on. Only shown when a build returns several features.
    viz_feature_w = widgets.Dropdown(
        options=[("Feature 1", 0)],
        value=0,
        description="Visualize feature:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="auto"),
        disabled=True,
    )

    # Which layer of the result to view: the time series, or any composite the
    # Temporal Composites section produced. Repopulated whenever the result or
    # the composite selection changes (_refresh_viz_layers). A composite has no
    # time axis, so picking one hides the viewer's date control and greys the
    # GIF maker, which needs per-date frames.
    viz_layer_w = widgets.Dropdown(
        options=[("Time Series", "Time_Series")],
        value="Time_Series",
        description="Layer:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="auto"),
        disabled=True,
    )
    viz_layer_note = widgets.HTML("")

    # Viewing resolution. A frame is drawn at most a couple of thousand pixels
    # wide, so on a large AOI the full-detail read fetches far more pixels than
    # can ever be shown. Picking a coarser value re-reads the scenes from the
    # archive's own reduced-resolution copies, which is a genuinely smaller
    # download - decimating the built cube is not, because the pixels have to
    # be fetched before they can be thrown away. Default None = as built, so
    # nothing changes unless the user asks for it.
    viz_resolution_w = widgets.FloatText(
        value=float(resolution_w.value or 10),
        description="View resolution (m):",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="auto"),
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

    # Animation rendering sections mirror the interactive viewer: presets,
    # single band (grey levels) and custom RGB (free band mapping). Band
    # dropdowns are populated from the built cube.
    gif_section_w = widgets.ToggleButtons(
        options=[
            ("Presets", "preset"),
            ("Single band", "band"),
            ("Custom RGB", "custom"),
        ],
        value="preset",
        style={"button_width": "110px"},
        disabled=True,
    )

    gif_band_dd = widgets.Dropdown(
        options=[],
        description="Band:",
        layout=widgets.Layout(width="260px"),
        disabled=True,
    )

    _gif_chan_layout = widgets.Layout(width="180px")
    _gif_chan_style = {"description_width": "24px"}
    gif_r_dd = widgets.Dropdown(options=[], description="R:", layout=_gif_chan_layout,
                                style=_gif_chan_style, disabled=True)
    gif_g_dd = widgets.Dropdown(options=[], description="G:", layout=_gif_chan_layout,
                                style=_gif_chan_style, disabled=True)
    gif_b_dd = widgets.Dropdown(options=[], description="B:", layout=_gif_chan_layout,
                                style=_gif_chan_style, disabled=True)

    gif_stretch_w = widgets.FloatRangeSlider(
        value=(2.0, 98.0),
        min=0.0,
        max=100.0,
        step=0.5,
        description="Stretch (%):",
        continuous_update=False,
        readout_format=".1f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="380px"),
        disabled=True,
    )

    # Contextual rows: only the active section's controls are visible.
    gif_preset_box = widgets.VBox(
        [_stacked_field(gif_display_mode_w, "Display mode")],
        layout=widgets.Layout(width="100%"),
    )
    gif_band_box = widgets.VBox(
        [gif_band_dd], layout=widgets.Layout(width="100%", display="none")
    )
    gif_custom_box = widgets.VBox(
        [widgets.HBox([gif_r_dd, gif_g_dd, gif_b_dd],
                      layout=widgets.Layout(gap="8px"))],
        layout=widgets.Layout(width="100%", display="none"),
    )
    gif_stretch_box = widgets.VBox(
        [gif_stretch_w],
        layout=widgets.Layout(width="100%", gap="0px", display="none"),
    )

    def _sync_gif_section_visibility():
        sec = gif_section_w.value
        gif_preset_box.layout.display = "" if sec == "preset" else "none"
        gif_band_box.layout.display = "" if sec == "band" else "none"
        gif_custom_box.layout.display = "" if sec == "custom" else "none"
        # Presets keep their fixed scaling; the stretch applies to band/custom.
        gif_stretch_box.layout.display = "" if sec in ("band", "custom") else "none"

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
        "last_auto_mask_binary_suggestion": None,
        "cloud_mask_result": None,   # in-memory binary mask held from the build
        # Coarse re-read of the current build for fast browsing, kept so the
        # same resolution is not fetched twice: {"res", "stamp", "cube"}.
        "viz_preview": None,
        "last_auto_daterange_example": None,
        "last_auto_gif_suggestion": None,
        "last_json_syntax": None,
        "draw_map": None,        # leafmap map, created lazily on first use
        "draw_box_built": False,
    }

    # Re-entrancy guard so programmatic resets of result_cloud_max_w.value
    # (e.g. snapping back to 100 on a fresh build) don't trigger the filter
    # change handler and re-render mid-build.
    _cloud_filter_guard = {"busy": False}

    # Same guard for the Result panel's Min coverage % box (reset to 0 on a
    # fresh build), kept separate so one filter's reset never swallows the
    # other's change event.
    _cov_filter_guard = {"busy": False}

    # Same idea for the Result date picker: populating it (and its select-all)
    # on a fresh build sets result_date_w.value programmatically, which must not
    # fire the date-change re-render mid-build.
    _date_filter_guard = {"busy": False}

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
            polygon_fc = _CubeFileChooser(
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

            output_fc = _CubeFileChooser(
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

            gif_out_fc = _CubeFileChooser(
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

            if vals[0] >= vals[2] or vals[1] >= vals[3]:
                raise ValueError(
                    "Polygon bbox must cover a real area: it needs xmin < xmax and "
                    f"ymin < ymax. Got [xmin, ymin, xmax, ymax] = {vals}."
                )

            return vals

        return s

    # -------------------------------------------------------------------------
    # Result summary (minimal)
    # -------------------------------------------------------------------------
    def _friendly_cube_summary_html(obj):
        """Plain-language overview of a single result cube, for users who don't
        read xarray reprs. Built ONLY from metadata the cube actually carries
        (attrs + materialized coords) — nothing is computed and nothing is
        invented: missing fields show '-'."""
        da = obj
        if isinstance(obj, xr.Dataset):
            da = obj.get("Time_Series")
            if da is None and len(obj.data_vars):
                da = obj[list(obj.data_vars)[0]]

        attrs = dict(getattr(da, "attrs", {}) or {})

        mission = attrs.get("mission")
        try:
            mission_label = _pretty_mission_label(mission) if mission else "-"
        except Exception:
            mission_label = mission or "-"

        indices = [str(i) for i in (attrs.get("indices") or [])]
        bands = [str(b) for b in (attrs.get("spectral_bands") or [])]
        if not bands and da is not None and "band" in getattr(da, "coords", {}):
            # Fallback: band coordinate minus the indices.
            bands = [str(b) for b in da.coords["band"].values if str(b) not in indices]

        crs = attrs.get("crs") or "-"
        est = _human_readable_bytes(_estimated_data_size_bytes(obj))

        # Extra layers when stats were added (Dataset with more variables).
        extra_layers = []
        if isinstance(obj, xr.Dataset):
            extra_layers = [v for v in obj.data_vars if v != "Time_Series"]

        rcell = "padding:4px 12px; text-align:right; color:#374151;"
        lcell = "padding:4px 12px; text-align:left;"
        label = (
            "padding:5px 14px 5px 0; color:#6b7280; white-space:nowrap; "
            "vertical-align:top; width:1%; border-bottom:1px solid #f3f4f6;"
        )
        value = "padding:5px 0; color:#111827; border-bottom:1px solid #f3f4f6;"

        def _row(k, v):
            return f"<tr><td style='{label}'>{k}</td><td style='{value}'>{v}</td></tr>"

        info_rows = [
            _row("Mission", mission_label),
            _row("Bands", ", ".join(bands) if bands else "-"),
            _row("Indices", ", ".join(indices) if indices else "-"),
            _row("CRS", crs),
            _row("Estimated data size", f"<b>{est}</b>"),
        ]
        if extra_layers:
            info_rows.append(_row("Additional layers", ", ".join(extra_layers)))

        # Sentinel-2 scene metadata coords + the multi-granule merge note.
        # np.asarray().ravel(): the attr is a list in memory / Zarr but an
        # ndarray (or scalar string for one field) after a NetCDF roundtrip.
        sm_attr = attrs.get("scene_metadata")
        sm_fields = (
            [str(s) for s in np.asarray(sm_attr).ravel()] if sm_attr is not None else []
        )
        if sm_fields:
            info_rows.append(_row("Scene metadata", ", ".join(sm_fields)))
        try:
            _sm_multi = int(attrs.get("scene_metadata_multiday", 0) or 0)
        except Exception:
            _sm_multi = 0
        scene_meta_info_html = ""
        if sm_fields and _sm_multi > 0:
            _tiles_attr = attrs.get("tile_id")
            _tiles = (
                [str(t) for t in np.asarray(_tiles_attr).ravel()]
                if _tiles_attr is not None
                else []
            )
            _tiles_txt = f" (tiles: {', '.join(_tiles)})" if len(_tiles) > 1 else ""
            _n_dates = (
                int(da.sizes.get("time", 0))
                if da is not None and hasattr(da, "sizes")
                else 0
            )
            scene_meta_info_html = (
                "<div style='font-size:12px; color:#1e40af; line-height:1.5; "
                "background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; "
                "padding:8px 10px; margin-top:10px;'>"
                f"ℹ️ On <b>{_sm_multi}</b> of {_n_dates} dates your polygon is "
                f"covered by more than one Sentinel-2 granule{_tiles_txt}. The "
                "scene metadata of those dates merges the granules: angle "
                "values are the <b>per-date mean</b> and acq_datetime is the "
                "<b>earliest</b> acquisition."
                "</div>"
            )

        # Per-date table with cloud % when the cube carries it; '-' otherwise.
        dates_html = ""
        has_time = da is not None and "time" in getattr(da, "coords", {})
        if has_time:
            tv = da.coords["time"].values
            cloud = None
            if "cloud_percentage" in da.coords:
                try:
                    cloud = list(da.coords["cloud_percentage"].values)
                except Exception:
                    cloud = None
            # Separate-tile cubes carry a per-timestep `tile` coordinate; show
            # it as a middle column so the two same-date tiles are legible.
            tiles_per_step = None
            if "tile" in da.coords:
                try:
                    tiles_per_step = [str(t) for t in da.coords["tile"].values]
                except Exception:
                    tiles_per_step = None
            # Per-scene AOI coverage (0..1), shown as a standard column right
            # after cloud % - it is a normal property of every cube now, not tied
            # to the warning or to the Across-track setting. The one exclusion is
            # a still-LAZY coord (cloud detection was off): reading it would force
            # a band read, so the column is simply omitted rather than slow the
            # preview down.
            coverage = None
            _cov_c = da.coords.get("scene_coverage")
            if _cov_c is not None and (
                getattr(getattr(_cov_c, "data", None), "chunks", None) is None
            ):
                try:
                    coverage = list(_cov_c.values)
                except Exception:
                    coverage = None
            th = "padding:4px 12px; border-bottom:1px solid #d1d5db; color:#374151;"
            rows = []
            for i, t in enumerate(tv):
                cp = "-"
                if cloud is not None and i < len(cloud):
                    try:
                        cp = f"{int(cloud[i])}%"
                    except Exception:
                        cp = "-"
                # Separate mode keeps sub-day timestamps, so show the time too
                # (two tiles of one day would otherwise read as the same row).
                date_txt = str(t)[:16] if tiles_per_step is not None else str(t)[:10]
                tile_cell = ""
                if tiles_per_step is not None:
                    tv_i = tiles_per_step[i] if i < len(tiles_per_step) else "-"
                    tile_cell = f"<td style='{lcell}'>{tv_i}</td>"
                cov_cell = ""
                if coverage is not None:
                    try:
                        cov_cell = f"<td style='{rcell}'>{int(round(float(coverage[i]) * 100))}%</td>"
                    except Exception:
                        cov_cell = f"<td style='{rcell}'>-</td>"
                rows.append(
                    f"<tr><td style='{lcell}'>{date_txt}</td>"
                    f"{tile_cell}"
                    f"<td style='{rcell}'>{cp}</td>"
                    f"{cov_cell}</tr>"
                )
            # Show at most 4 dates up front (matches the height of the info block
            # on the left); the rest folds behind a native <details> expander so a
            # long time series can't stretch the Result panel forever.
            table_open = (
                "<table style='border-collapse:collapse; margin-top:4px; width:100%; "
                "font-size:12.5px;'>"
            )
            _tile_th = (
                f"<th style='{th} text-align:left;'>tile</th>"
                if tiles_per_step is not None
                else ""
            )
            _cov_th = (
                f"<th style='{th} text-align:right;'>coverage</th>"
                if coverage is not None
                else ""
            )
            header_row = (
                f"<tr style='background:#f3f4f6;'><th style='{th} text-align:left;'>date</th>"
                f"{_tile_th}"
                f"<th style='{th} text-align:right;'>cloud</th>"
                f"{_cov_th}</tr>"
            )
            dates_html = (
                f"<div style='font-weight:600;'>Dates ({len(tv)})</div>"
                + table_open + header_row + "".join(rows[:4]) + "</table>"
            )
            if len(rows) > 4:
                dates_html += (
                    "<details style='margin-top:2px;'>"
                    "<summary style='cursor:pointer; color:#2563eb; font-size:12px;'>"
                    f"Show all {len(tv)} dates</summary>"
                    + table_open + "".join(rows[4:]) + "</table>"
                    + "</details>"
                )
        else:
            info_rows.append(
                _row("Dates", "- (time dimension collapsed by Temporal Composite)")
            )

        # Two flexible columns (key facts | dates table) so the summary fills the
        # Result panel's width instead of clustering top-left; wraps when narrow.
        info_table = (
            "<table style='border-collapse:collapse; width:100%; font-size:13px;'>"
            + "".join(info_rows)
            + "</table>"
        )
        dates_col = (
            f"<div style='flex:1 1 280px; min-width:240px;'>{dates_html}</div>"
            if dates_html
            else ""
        )
        # The visualize/export note is rendered as a separate widget below the
        # Max cloud box (result_viz_note_w), not inline here.
        return (
            "<div style='font-size:13px; width:100%;'>"
            "<b>(っ◕‿◕)っ Your data cube is ready!</b>"
            "<div style='display:flex; flex-wrap:wrap; gap:10px 36px; margin-top:8px; "
            "align-items:flex-start;'>"
            f"<div style='flex:1 1 340px; min-width:280px;'>{info_table}</div>"
            + dates_col
            + "</div>"
            + scene_meta_info_html
            + "</div>"
        )

    def _cube_is_empty(c):
        """True if a cube carries no actual data (None, no dims, or a zero-length
        dimension such as time=0). Structural only - no compute is triggered."""
        if not isinstance(c, (xr.DataArray, xr.Dataset)):
            return True
        da = c
        if isinstance(c, xr.Dataset):
            da = c.get("Time_Series")
            if da is None and len(c.data_vars):
                da = c[list(c.data_vars)[0]]
        if da is None:
            return True
        sizes = getattr(da, "sizes", {}) or {}
        if not sizes:
            return True
        return any(int(v) == 0 for v in sizes.values())

    def _result_is_empty(obj):
        """True when there is nothing worth showing as a 'ready' cube."""
        if obj is None:
            return True
        if isinstance(obj, list):
            cubes = [c for c in obj if isinstance(c, (xr.DataArray, xr.Dataset))]
            if not cubes:
                return True
            return all(_cube_is_empty(c) for c in cubes)
        return _cube_is_empty(obj)

    # Shared style for the blue Result-panel notices.
    #
    # Blue ℹ️ = "here is what happened", yellow ⚠️ = "you may want to act". The
    # split is deliberate and yellow is deserved by exactly ONE notice - the Area
    # Size one, which carries a Resize and Re-build button (it styles itself in
    # _coreg_warn_html, hence no shared yellow constant here). Everything else is
    # blue: three identical yellow boxes made people feel they had to clear a
    # to-do list before their cube was usable, when nothing was actually wrong.
    _INFO_BOX = (
        "<div style='font-size:12px; color:#1e3a8a; background:#eff6ff; "
        "border:1px solid #bfdbfe; border-radius:6px; padding:6px 8px; "
        "margin:0; box-sizing:border-box;'>"
    )

    def _compute_projection_hint(obj):
        """Were any scenes re-drawn into a different projection?

        Returns a ``(title, html)`` note for the Result panel's notes strip, or
        None when the cube is entirely native. Two wordings, never both - the
        situations are mutually exclusive:

        * "Projection Information" - the user asked for a CRS none of the scenes
          come in, so this confirms what they requested.
        * "Multiple Projections Detected" - the area itself spans projections,
          so the build had to choose one.

        Reads NOTHING: everything comes from attrs the build already recorded
        (`crs`, `native_crs`, `native_crs_share`), which are only present when a
        reprojection actually happened. Never raises - a hint must not break a
        build.
        """
        try:
            da = obj
            if isinstance(obj, xr.Dataset):
                da = obj.get("Time_Series")
                if da is None and len(obj.data_vars):
                    da = obj[list(obj.data_vars)[0]]
            if da is None:
                return None
            attrs = getattr(da, "attrs", {}) or {}
            natives = attrs.get("native_crs")
            if natives is None:
                return None
            natives = [str(c) for c in np.asarray(natives).ravel()]
            if not natives:
                return None
            target = str(attrs.get("crs", "") or "")

            shares = attrs.get("native_crs_share")
            share_by_crs = {}
            if shares is not None:
                vals = np.asarray(shares).ravel().tolist()
                if len(vals) == len(natives):
                    share_by_crs = dict(zip(natives, vals))

            def _fmt(c):
                s = share_by_crs.get(c)
                pct = f" (covers {s * 100:.0f}% of your area)" if s is not None else ""
                mark = " <b>&larr; used</b>" if c == target else ""
                return f"<code>{c}</code>{pct}{mark}"

            listing = ", ".join(_fmt(c) for c in natives)

            if target and target not in natives:
                # Deliberate non-native choice, e.g. an equal-area CRS. The user
                # asked for this, so it is information, not a warning.
                return "Projection Information", (
                    f"{_INFO_BOX}ℹ️ <b>Projection Information:</b> Your cube uses "
                    f"<code>{target}</code>, remember that provided native "
                    f"projections are: ({listing}). All of them were stretched to "
                    "fit the user-selected projection.</div>"
                )
            return "Multiple Projections Detected", (
                f"{_INFO_BOX}ℹ️ <b>Multiple Projections Detected:</b> This area's "
                f"scenes come in more than one projection: {listing}. <b>All "
                "scenes were kept "
                "and mosaicked</b>; those not native to "
                f"<code>{target}</code> were re-drawn into it (pixel size and "
                "area are slightly distorted far from that zone's central "
                "meridian). You can change CRS under "
                "<b>Advanced Parameters → Output Projection (CRS)</b>.</div>"
            )
        except Exception:
            return None

    def _compute_multiswath_hint(obj):
        """GUI-only detection: does the built AOI sit across multiple swaths, so
        some scenes image only part of it? Returns a ``(title, html)`` note for
        the Result panel's notes strip, pointing at the Across-track
        partial-scene removal, or None when there's nothing to flag. Never
        raises (a hint must never break a build).

        Two modes, chosen by the FRACTION of partial scenes, so the message names
        the actual cause instead of always blaming swath overlap:
          * >= 20% partial -> "Overlapping Tiles Detected" (systematic: an AOI
            straddling two swaths is partial on roughly every other orbit).
          * < 20% partial  -> "Incomplete Scenes Detected" (sporadic: a faulty or
            partially-missing acquisition), naming the offending dates when few
            so they can be unticked in Dates before export - no rebuild needed.

        Reads NOTHING: it uses the eager scene_coverage coord that get_stac_layers
        already computed from the SCL "imaged" boolean (sharing the cloud-% read).
        That coord is the true swath coverage - cloud-aware by construction (a
        cloudy-but-complete scene reads ~1.0). When cloud detection was OFF there
        is no SCL, so scene_coverage is left lazy (band-0) and unread; the warning
        deliberately stays silent there rather than force a read.
        """
        try:
            da = obj
            if isinstance(obj, xr.Dataset):
                da = obj.get("Time_Series")
                if da is None and len(obj.data_vars):
                    da = obj[list(obj.data_vars)[0]]
            if da is None or "time" not in getattr(da, "dims", ()):
                return None
            if int(da.sizes.get("time", 0)) < 2:
                return None
            # Already handled by the user -> nothing to suggest.
            if str(da.attrs.get("partial_scene_handling", "keep")) == "remove":
                return None
            # Only suggest the tool where it actually applies (optical missions).
            if str(da.attrs.get("mission", "")) not in _PARTIAL_OK_MISSIONS:
                return None

            # Use the ready coverage coord only. If it is absent (old cube) or
            # still lazy (cloud detection was off), stay silent - do not trigger
            # a read just to show a warning.
            sc = da.coords.get("scene_coverage")
            if sc is None:
                return None
            if getattr(getattr(sc, "data", None), "chunks", None) is not None:
                return None  # lazy -> no SCL was read -> no warning by design
            swath = np.asarray(sc.values, dtype=float)

            valid = ~np.isnan(swath)
            if int(valid.sum()) < 2:
                return None
            partial = valid & (swath < 0.9)
            n_partial = int(partial.sum())
            if n_partial == 0:
                return None
            n_total = int(valid.sum())
            frac = n_partial / float(n_total)
            pct_txt = f"{frac * 100:.1f}%" if frac < 0.01 else f"{round(frac * 100)}%"
            _verb = "images" if n_partial == 1 else "image"

            # The Result panel's own Min coverage % box does this without a
            # rebuild, and it is enabled exactly when this warning can appear
            # (both need the ready scene_coverage coord), so point there.
            _filter_hint = "set <b>Min coverage %</b> below"
            # Which cause? An AOI straddling two swaths is partial on roughly
            # every other orbit (systematic, tens of percent), whereas a faulty /
            # incomplete acquisition is a handful of scenes. The 20% cut sits
            # between the two regimes; the sporadic wording says "most likely"
            # because an AOI barely clipping a swath edge can also land low.
            if frac >= 0.20:
                return "Overlapping Tiles Detected", (
                    f"{_INFO_BOX}"
                    "ℹ️ <b>Overlapping Tiles Detected:</b> This area is "
                    "potentially covered by "
                    f"multiple swaths: <b>{n_partial}</b> of {n_total} scenes "
                    f"({pct_txt}) {_verb} less than 90% of it. To drop these "
                    f"partially-missing scenes, {_filter_hint}.</div>"
                )

            # Sporadic: name the offending dates (when few) so they can simply be
            # unticked in the Dates list before exporting - no rebuild needed.
            dates_txt = ""
            if n_partial <= 5:
                try:
                    tv = np.asarray(da["time"].values)
                    items = [
                        f"{str(tv[i])[:10]} ({round(float(swath[i]) * 100)}%)"
                        for i in np.flatnonzero(partial)
                    ]
                    label = "Affected date" + ("s" if len(items) > 1 else "")
                    dates_txt = f" <b>{label}:</b> {', '.join(items)}."
                except Exception:
                    dates_txt = ""
            _them = "them" if n_partial > 1 else "it"
            return "Incomplete Scenes Detected", (
                f"{_INFO_BOX}"
                "ℹ️ <b>Incomplete Scenes Detected:</b> "
                f"<b>{n_partial}</b> of {n_total} scenes ({pct_txt}) {_verb} less "
                # No trailing space: dates_txt already leads with one, and is ""
                # when there are too many dates to list.
                "than 90% of this area."
                f"{dates_txt} You can untick "
                f"{_them} in <b>Dates</b> below before exporting, or {_filter_hint}."
                "</div>"
            )
        except Exception:
            return None

    # Expanded/collapsed survives re-renders: _show_result_summary runs on every
    # cloud-filter and date-filter change, and a strip that snapped shut each
    # time would be unusable while reading it.
    _notes_open = {"open": False}

    def _render_notes_toggle(n):
        arrow = "▾" if _notes_open["open"] else "▸"
        word = "note" if n == 1 else "notes"
        # "ℹ️" with the emoji variation selector, matching the notice boxes. The
        # bare "ℹ" (U+2139) renders as a tiny serif letter, not an icon.
        result_notes_toggle.description = (
            f"{arrow}  ℹ️  {n} {word} about this data cube"
        )
        result_notes_body.layout.display = "" if _notes_open["open"] else "none"

    def _on_notes_toggle(_btn):
        _notes_open["open"] = not _notes_open["open"]
        _render_notes_toggle(len(result_notes_body.children))

    result_notes_toggle.on_click(_on_notes_toggle)

    # Same collapse mechanics for the "What is next?" guidance strip.
    _next_open = {"open": False}

    def _render_next_toggle():
        arrow = "▾" if _next_open["open"] else "▸"
        result_viz_note_toggle.description = f"{arrow}  ℹ️  What is next?"
        result_viz_note_w.layout.display = "" if _next_open["open"] else "none"

    def _on_next_toggle(_btn):
        _next_open["open"] = not _next_open["open"]
        _render_next_toggle()

    result_viz_note_toggle.on_click(_on_next_toggle)

    def _set_result_viz_note(show):
        """Show or hide the collapsed "What is next?" strip."""
        if not show:
            result_viz_note_w.value = ""
            result_viz_note_toggle.layout.display = "none"
            result_viz_note_w.layout.display = "none"
            return
        result_viz_note_w.value = _RESULT_VIZ_NOTE_HTML
        result_viz_note_toggle.layout.display = ""
        _render_next_toggle()

    def _set_result_notes(notes):
        """Fill the notes strip from a list of ``(title, html)`` pairs (None
        entries dropped). An empty list hides the strip entirely, so a cube with
        nothing to report shows no extra line at all."""
        notes = [n for n in (notes or []) if n]
        if not notes:
            result_notes_body.children = ()
            result_notes_row.layout.display = "none"
            return
        # width="100%", matching the header button exactly - see the
        # .stac2cube-notes-row rules in gui_common for why "auto" drifts.
        result_notes_body.children = tuple(
            widgets.HTML(html_, layout=widgets.Layout(width="100%"))
            for _title, html_ in notes
        )
        _render_notes_toggle(len(notes))
        result_notes_row.layout.display = ""

    def _coreg_warn_html():
        """Yellow co-registration size warning for the Result panel, or empty
        string when the last build carried no such hint. Deliberately NOT part of
        the blue notes strip: it is the one notice with something to decide, and
        its Resize and Re-build button must not be collapsed out of view."""
        hint = state.get("coreg_size_hint")
        if not hint:
            return ""
        return (
            "<div style='font-size:12px; color:#92400e; "
            "background:#fef3c7; border:1px solid #fcd34d; "
            "border-radius:8px; padding:8px 10px; margin:0;'>"
            f"⚠️ <b>Area Size Warning:</b> {hint}</div>"
        )

    def _set_coreg_warning(html):
        """Show/clear the co-registration size warning together with its
        'Resize and Re-build' button - the button (and the row itself) only
        exist while the warning is visible, so an empty warning adds no gap at
        the top of the Result section."""
        html = html or ""
        result_coreg_warn_w.value = html
        coreg_resize_btn.layout.display = "" if html else "none"
        result_coreg_warn_row.layout.display = "" if html else "none"

    def _show_result_summary(obj):
        # The visualize/export note (a sibling widget below the Max cloud box) only
        # makes sense next to a ready cube - hide it for empty/failed results.
        empty = _result_is_empty(obj)
        _set_result_viz_note(not empty)
        # Notices live at the TOP of the Result section: the yellow Area Size
        # warning first (the only one with something to decide), then the blue
        # notes strip. Both collapse away entirely when they have nothing to say.
        _set_coreg_warning("" if empty else _coreg_warn_html())
        _set_result_notes([] if empty else [
            state.get("multiswath_hint"),
            state.get("projection_hint"),
        ])
        with result_out:
            clear_output()

            # Never show a 'ready' cube when there is no data: a failed/empty build
            # must read as a failure here, not as success sitting next to a Status
            # error.
            if _result_is_empty(obj):
                display(HTML(
                    "<div style='font-size:13px; color:#991b1b; background:#fef2f2; "
                    "border:1px solid #fecaca; border-radius:6px; padding:8px 10px;'>"
                    "(╥﹏╥) No data cube to show - the build did not produce any data. "
                    "See the <b>Status</b> panel for the reason.</div>"
                ))
                return

            # A polygon FILE with several features returns a LIST of cubes (one
            # per feature). Dumping every xarray repr is unreadable, so show a
            # compact table instead.
            if isinstance(obj, list):
                _show_multi_feature_summary(obj)
                return

            # (The multi-swath note is rendered above result_out, inside the
            # notes strip - see the top of this function.)

            # Single cube: friendly summary by default; the bold toggle swaps in
            # the raw xarray repr for power users.
            nerd_w = widgets.Checkbox(
                value=False,
                description="Details for Xarray-Nerds",
                indent=False,
                # Shrink to content and push to the far right of the row.
                # justify_content alone won't move a Checkbox reliably, so we
                # constrain the width and use margin-left:auto.
                layout=widgets.Layout(width="max-content", margin="0 0 0 auto"),
            )
            nerd_w.add_class("stac2cube-nerd-toggle")
            body = widgets.Output()

            def _render(*_):
                with body:
                    clear_output()
                    if nerd_w.value:
                        est_bytes = _estimated_data_size_bytes(obj)
                        print(
                            f"Estimated data size: {_human_readable_bytes(est_bytes)}\n"
                        )
                        display(obj)
                    else:
                        display(HTML(_friendly_cube_summary_html(obj)))

            nerd_w.observe(_render, names="value")
            _render()
            toggle_row = widgets.HBox(
                [nerd_w],
                layout=widgets.Layout(width="100%", justify_content="flex-end"),
            )
            display(widgets.VBox(
                [toggle_row, body],
                layout=widgets.Layout(width="100%", gap="4px"),
            ))

    def _show_multi_feature_summary(cubes):
        """Compact, readable summary for a multi-feature batch result (a list of
        cubes, one per polygon feature). Time/dates are shown PER feature because
        separate areas can return different acquisition dates; bands/CRS go in the
        header only when identical across EVERY feature, else flagged as varying.
        No compute is triggered (.nbytes/.sizes/coords are materialized metadata)."""
        n = len(cubes)
        if n == 0:
            print("No data cubes were produced.")
            return

        def _main_da(c):
            if isinstance(c, xr.Dataset):
                if "Time_Series" in c.data_vars:
                    return c["Time_Series"]
                return next(iter(c.data_vars.values()), None)
            return c

        def _cube_bytes(c):
            try:
                return int(getattr(c, "nbytes", 0))
            except Exception:
                return 0

        def _band_key(c):
            da = _main_da(c)
            try:
                if da is not None and "band" in da.coords:
                    return tuple(str(b) for b in da.coords["band"].values)
            except Exception:
                pass
            return None

        def _time_key(c):
            da = _main_da(c)
            try:
                if da is not None and "time" in da.coords:
                    return tuple(str(t) for t in da.coords["time"].values)
            except Exception:
                pass
            return None

        def _crs_of(c):
            da = _main_da(c)
            try:
                return da.attrs.get("crs") if da is not None else None
            except Exception:
                return None

        def _is_cube(c):
            return isinstance(c, (xr.DataArray, xr.Dataset))

        # Defensive: if the result list ever contains a non-cube entry (a failed
        # feature), split it out so failures are reported honestly instead of
        # rendered as blank "ready" rows.
        valid = [c for c in cubes if _is_cube(c)]
        failed = [c for c in cubes if not _is_cube(c)]
        n_ok = len(valid)

        total = _human_readable_bytes(sum(_cube_bytes(c) for c in valid))

        # Bands are requested identically for every feature, so they're the one
        # truly shared field shown in the header. CRS and time/dates can differ
        # per area, so those are shown per feature in the table below.
        time_uniform = len({_time_key(c) for c in valid}) <= 1

        common = []
        _bands = _band_key(valid[0]) if valid else None
        if _bands:
            common.append("Bands: " + ", ".join(_bands))

        cell = "padding:3px 10px; text-align:right;"
        lcell = "padding:3px 10px; text-align:left;"
        th = "padding:3px 10px; border-bottom:1px solid #d1d5db; color:#374151;"
        rows = []
        for i, c in enumerate(cubes, 1):
            if not _is_cube(c):
                reason = getattr(c, "error", "could not be generated")
                rows.append(
                    f"<tr><td style='{cell} color:#b91c1c;'>{i}</td>"
                    f"<td colspan='4' style='{lcell} color:#b91c1c;'>"
                    f"failed - {reason}</td></tr>"
                )
                continue
            da = _main_da(c)
            s = getattr(da, "sizes", {}) if da is not None else {}
            rows.append(
                f"<tr><td style='{cell}'>{i}</td>"
                f"<td style='{cell}'>{s.get('time', '–')}</td>"
                f"<td style='{lcell}'>{_crs_of(c) or '–'}</td>"
                f"<td style='{cell}'>{s.get('y', '–')} × {s.get('x', '–')}</td>"
                f"<td style='{cell}'>{_human_readable_bytes(_cube_bytes(c))}</td></tr>"
            )

        warn = ""
        if not time_uniform:
            warn = (
                "<div style='font-size:12px; color:#92400e; background:#fffbeb; "
                "border:1px solid #fde68a; border-radius:6px; padding:6px 8px; "
                "margin-top:6px;'>⚠️ Time steps differ between features - each cube "
                "keeps only the dates with scenes over its own area (see the "
                "<b>time</b> column; counts can match while the actual dates differ).</div>"
            )

        # Batch mode always ends with SEPARATE cubes, which is exactly the input
        # the mosaic tool exists for - so point at it here, where the user is
        # looking at the list of pieces. Shown only with at least two successful
        # cubes: with one there is nothing to join and the note would be noise.
        mosaic_hint = ""
        if n_ok >= 2:
            _red = "color:#b91c1c; font-weight:600;"
            mosaic_hint = (
                f"{_INFO_BOX.replace('margin:0;', 'margin-top:6px;')}"
                "ℹ️ You can mosaic these cubes into one with "
                f"<span style='{_red}'>Data Cube Editor</span> &rarr; "
                f"<span style='{_red}'>Mosaic Data Cubes</span>."
                "</div>"
            )

        fail_note = ""
        if failed:
            failed_nums = ", ".join(f"#{getattr(m, 'feature', '?')}" for m in failed)
            fail_note = (
                "<div style='font-size:12px; color:#991b1b; background:#fef2f2; "
                "border:1px solid #fecaca; border-radius:6px; padding:6px 8px; "
                f"margin-top:6px;'>(╥﹏╥) {len(failed)} of {n} feature(s) could not be "
                f"generated ({failed_nums}). They are listed as <b>failed</b> below "
                "and are skipped on export/visualization. See the table for the reason.</div>"
            )

        html = (
            "<div style='font-size:13px;'>"
            f"<b>✅ {n_ok} data cube(s)</b> generated from {n} polygon feature(s). "
            f"Total estimated size: <b>{total}</b>."
            + (
                "<div style='font-size:12px; color:#6b7280; margin-top:2px;'>"
                + " · ".join(common) + "</div>"
                if common else ""
            )
            + warn
            + mosaic_hint
            + fail_note
            + "<table style='border-collapse:collapse; margin-top:8px; font-size:12px;'>"
            + "<tr style='background:#f3f4f6;'>"
            + f"<th style='{th} text-align:right;'>#</th>"
            + f"<th style='{th} text-align:right;'>time</th>"
            + f"<th style='{th} text-align:left;'>CRS</th>"
            + f"<th style='{th} text-align:right;'>y × x</th>"
            + f"<th style='{th} text-align:right;'>est. size</th></tr>"
            + "".join(rows)
            + "</table>"
            + "<div style='font-size:12px; color:#6b7280; margin-top:6px;'>"
            + "Each feature is a separate cube. On export, <b>NetCDF</b> writes one "
            + "file per feature (<code>name_01.nc</code>, <code>name_02.nc</code>, …); "
            + "<b>COGs</b> write one subfolder per feature (<code>01/</code>, "
            + "<code>02/</code>, … each holding that feature's dated tiles)."
            + "</div></div>"
        )
        display(HTML(html))

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

        When the polygon is an absolute path (e.g. one picked via the file
        chooser), the suggestion is returned as an absolute path too. SLURM
        jobs run from a different working directory, so a relative
        "./results/..." would not resolve there - a full path can be copied
        straight into the job script.
        """
        raw = (polygon_w.value or "").strip()
        is_bbox = raw.startswith("[") or raw.startswith("(")
        if is_bbox:
            stem = "bbox"
        elif raw:
            try:
                stem = Path(raw).stem
            except Exception:
                stem = "test"
        else:
            stem = "test"

        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "test"

        # If a polygon file was selected, polygon_w holds an absolute path.
        # Mirror it to an absolute "<base>/results/<stem>.nc" so the
        # recommendation is a hard-coded path, not a relative one.
        if not is_bbox and raw:
            try:
                poly_path = Path(raw)
                if poly_path.is_absolute():
                    out_dir = poly_path.parent.parent / "results"
                    return str(out_dir / f"{stem}.nc")
            except Exception:
                pass

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

    def _auto_mask_binary_suggestion():
        """Binary-mask output path: the main NetCDF suggestion with a
        '_mask_binary' stem suffix (<polygon>_mask_binary.nc), so it sits right
        next to the cube."""
        base = _auto_netcdf_suggestion_from_polygon()  # ./results/<stem>.nc
        stem, ext = os.path.splitext(base)
        return f"{stem}_mask_binary{ext or '.nc'}"

    def _update_mask_binary_suggestion(force=False):
        """Keep the binary-mask path following the polygon, exactly like the main
        output field. Only acts while Export mask is on; leaves a user-typed path
        alone (only replaces empty / previously-auto values)."""
        if export_mask_w.value is not True:
            return
        new_suggestion = _auto_mask_binary_suggestion()
        current = (cloud_mask_output_w.value or "").strip()
        prev_auto = state.get("last_auto_mask_binary_suggestion")
        should_replace = (
            force or (current == "") or (prev_auto is not None and current == prev_auto)
        )
        if should_replace:
            cloud_mask_output_w.value = new_suggestion
        state["last_auto_mask_binary_suggestion"] = new_suggestion

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

    def _resolve_daterange():
        """Build the daterange object get_stac_layers expects, from whichever
        date input is active: the simple From/To pickers, or (when 'Advanced date
        options' is on) the Python-style text field. Returns the same shapes as
        before -- e.g. ["YYYY-MM-DD", "YYYY-MM-DD"] -- so the backend is unchanged.
        """
        if advanced_dates_w.value:
            return _parse_daterange_input(daterange_mode_w.value, daterange_w.value)

        d_from = date_from_w.value
        d_to = date_to_w.value
        if d_from is None or d_to is None:
            raise ValueError("Please choose both a 'From' and a 'To' date.")
        if d_from > d_to:
            raise ValueError("The 'From' date is after the 'To' date — please swap them.")
        return [d_from.isoformat(), d_to.isoformat()]

    # -------------------------------------------------------------------------
    # Visualization helpers
    # -------------------------------------------------------------------------
    def _pick_dataarray_for_visualization(result_obj):
        """The layer to display: whichever the Layer dropdown selects.

        Falls back to the time series (or the single variable of a one-layer
        cube) when the dropdown's choice is not in this result - e.g. right
        after the composite selection changed but before the dropdown was
        repopulated.
        """
        if isinstance(result_obj, xr.DataArray):
            return result_obj

        if isinstance(result_obj, xr.Dataset):
            wanted = viz_layer_w.value
            if wanted and wanted in result_obj.data_vars:
                return result_obj[wanted]
            if "Time_Series" in result_obj.data_vars:
                return result_obj["Time_Series"]
            layers = _raster_layer_names(result_obj)
            if layers:
                return result_obj[layers[0]]
            raise ValueError(
                "This result has no raster layer to visualize "
                f"(variables: {list(result_obj.data_vars)})."
            )

        raise TypeError(
            f"Unsupported result type for visualization: {type(result_obj)}"
        )

    def _active_result_cube(composite=True):
        """The cube the viz tools act on: for a multi-feature list, the one
        chosen in the feature dropdown; for a single cube, that cube; otherwise
        None.

        composite=True (default) includes the Temporal Composites, so the Layer
        dropdown can show them. The GIF maker asks for composite=False: it
        animates per-date frames, which only the time series has."""
        obj = state["result"]
        if obj is None:
            return None
        if isinstance(obj, list):
            cubes = [c for c in obj if isinstance(c, (xr.DataArray, xr.Dataset))]
            if not cubes:
                return None
            idx = viz_feature_w.value if isinstance(viz_feature_w.value, int) else None
            if (
                isinstance(idx, int)
                and 0 <= idx < len(obj)
                and isinstance(obj[idx], (xr.DataArray, xr.Dataset))
            ):
                return _effective_result(obj[idx], composite=composite)
            return _effective_result(cubes[0], composite=composite)
        return _effective_result(obj, composite=composite)

    def _refresh_viz_layers():
        """Repopulate the Layer dropdown from the current effective result, and
        grey the GIF maker when the chosen layer has no time axis."""
        try:
            cube = _active_result_cube()
        except Exception:
            cube = None

        if cube is None:
            viz_layer_w.options = [("Time Series", "Time_Series")]
            viz_layer_w.value = "Time_Series"
            viz_layer_w.disabled = True
            viz_layer_note.value = ""
            return

        if isinstance(cube, xr.DataArray):
            names = [cube.name or "Time_Series"]
            opts = [(_layer_display_name(names[0]), names[0])]
        else:
            names = _raster_layer_names(cube)
            opts = [(_layer_display_name(n), n) for n in names]
        if not opts:
            opts = [("Time Series", "Time_Series")]
            names = ["Time_Series"]

        previous = viz_layer_w.value
        viz_layer_w.options = opts
        viz_layer_w.value = previous if previous in names else names[0]
        # Only worth choosing when there is more than one layer.
        viz_layer_w.disabled = len(opts) < 2

        # A composite has no per-date frames to animate.
        has_time = False
        try:
            sel = viz_layer_w.value
            da = cube[sel] if isinstance(cube, xr.Dataset) else cube
            has_time = "time" in getattr(da, "dims", ())
        except Exception:
            has_time = False
        viz_make_gif_btn.disabled = not has_time
        viz_layer_note.value = (
            ""
            if has_time
            else (
                "<div style='font-size:12px; color:#9a3412;'>"
                "This layer is a single composite image - it has no dates to "
                "scrub through or animate, so the GIF maker is off.</div>"
            )
        )

    # -------------------------------------------------------------------------
    # Result-panel cloud filter (drives table + visualization + export at once)
    # -------------------------------------------------------------------------
    def _result_cloud_pct(obj):
        """The cloud_percentage coordinate of a single cube, or None if it has
        none. Structural lookup only - nothing is computed."""
        da = obj
        if isinstance(obj, xr.Dataset):
            da = obj.get("Time_Series")
            if da is None and len(obj.data_vars):
                da = obj[list(obj.data_vars)[0]]
        if da is None:
            return None
        if "cloud_percentage" not in getattr(da, "coords", {}):
            return None
        return da.coords["cloud_percentage"]

    def _result_has_cloud_pct(obj):
        """True when obj (cube or list of cubes) carries a cloud_percentage
        coord, i.e. it was built with cloud masking and is filterable."""
        if obj is None:
            return False
        if isinstance(obj, list):
            return any(
                _result_has_cloud_pct(c)
                for c in obj
                if isinstance(c, (xr.DataArray, xr.Dataset))
            )
        return _result_cloud_pct(obj) is not None

    def _apply_cloud_threshold(obj, thr=None):
        """Return a view of obj keeping only timesteps with
        cloud_percentage <= thr. Pure coord selection on the existing
        cloud_percentage coord - no recompute, fully reversible by raising thr.

        thr defaults to the Result panel's Max cloud % box. thr >= 100 (or a
        cube with no cloud_percentage / no time dim) passes through unchanged,
        so non-masked builds are never altered."""
        if obj is None:
            return None
        if thr is None:
            thr = int(result_cloud_max_w.value)
        if thr >= 100:
            return obj
        if isinstance(obj, list):
            return [
                _apply_cloud_threshold(c, thr)
                if isinstance(c, (xr.DataArray, xr.Dataset))
                else c
                for c in obj
            ]
        cp = _result_cloud_pct(obj)
        if cp is None or "time" not in getattr(obj, "dims", {}):
            return obj
        keep = np.asarray((cp <= int(thr)).values)
        if keep.all():
            return obj
        return obj.isel(time=np.flatnonzero(keep))

    def _result_scene_cov(da):
        """The READY (already computed) scene_coverage coord of a cube, or None.

        A still-lazy coord is treated as absent: materializing it means reading a
        band, and no Result-panel control is allowed to trigger that (same rule
        the Dates table, the date picker and the swath warning follow)."""
        if not isinstance(da, (xr.DataArray, xr.Dataset)):
            return None
        sc = getattr(da, "coords", {}).get("scene_coverage")
        if sc is None:
            return None
        if getattr(getattr(sc, "data", None), "chunks", None) is not None:
            return None
        return sc

    def _result_has_scene_cov(obj):
        """True when obj (cube or list of cubes) carries a ready scene_coverage
        coord, i.e. it can be filtered by how much of the area each scene
        images."""
        if obj is None:
            return False
        if isinstance(obj, list):
            return any(_result_has_scene_cov(c) for c in obj)
        return _result_scene_cov(obj) is not None

    def _apply_coverage_threshold(obj, thr=None):
        """Return a view of obj keeping only timesteps whose scene_coverage is
        >= thr percent. Pure coord selection on the existing scene_coverage
        coord - no recompute, fully reversible by lowering thr.

        thr defaults to the Result panel's Min coverage % box. thr <= 0 (or a
        cube with no ready scene_coverage / no time dim) passes through
        unchanged, so the filter is a no-op until the user asks for it. NaN
        coverage counts as failing, mirroring the cloud filter's `cp <= thr`."""
        if obj is None:
            return None
        if thr is None:
            thr = int(result_coverage_min_w.value)
        if thr <= 0:
            return obj
        if isinstance(obj, list):
            return [
                _apply_coverage_threshold(c, thr)
                if isinstance(c, (xr.DataArray, xr.Dataset))
                else c
                for c in obj
            ]
        sc = _result_scene_cov(obj)
        if sc is None or "time" not in getattr(obj, "dims", {}):
            return obj
        try:
            cov = np.asarray(sc.values, dtype=float)
        except Exception:
            return obj
        # The coord is a fraction (0..1); the box is a percentage.
        keep = ~np.isnan(cov) & (cov * 100.0 >= float(thr))
        if keep.all():
            return obj
        return obj.isel(time=np.flatnonzero(keep))

    def _dates_passing_filters():
        """The ISO timestamps that pass the current Max cloud % AND Min
        coverage % boxes, or None when neither filters (or the cube carries
        neither coord). A NaN percentage counts as failing, matching
        _apply_cloud_threshold and _apply_coverage_threshold."""
        thr_cloud = int(result_cloud_max_w.value)
        thr_cov = int(result_coverage_min_w.value)
        if thr_cloud >= 100 and thr_cov <= 0:
            return None
        obj = state["result"]
        if obj is None or isinstance(obj, list):
            return None
        if "time" not in getattr(obj, "dims", {}):
            return None
        try:
            tvals = np.asarray(obj["time"].values)
        except Exception:
            return None

        def _vals(coord, scale):
            if coord is None:
                return None
            try:
                v = np.asarray(coord.values, dtype=float) * scale
            except Exception:
                return None
            return v if v.shape[0] == tvals.shape[0] else None

        pct = _vals(_result_cloud_pct(obj), 1.0) if thr_cloud < 100 else None
        cov = _vals(_result_scene_cov(obj), 100.0) if thr_cov > 0 else None
        if pct is None and cov is None:
            return None

        passing = set()
        for i, t in enumerate(tvals):
            if pct is not None and not (
                not np.isnan(pct[i]) and pct[i] <= thr_cloud
            ):
                continue
            if cov is not None and not (
                not np.isnan(cov[i]) and cov[i] >= thr_cov
            ):
                continue
            passing.add(str(t))
        return passing

    def _sync_date_picker_to_cloud():
        """Untick the dates the Result scene filters removed, so the picker shows
        what actually survives instead of leaving dropped dates highlighted.

        The user's own ticks are remembered separately in
        state["date_user_selection"], so raising the threshold brings their
        dates straight back - the filter stays as reversible as before, it just
        no longer lies about which dates are in the cube.
        """
        if result_date_w.disabled or not result_date_w.options:
            return
        all_vals = _result_date_all_values()
        chosen = state.get("date_user_selection")
        if chosen is None:
            chosen = set(all_vals)
        passing = _dates_passing_filters()
        new_val = tuple(
            v
            for v in all_vals
            if v in chosen and (passing is None or v in passing)
        )
        if new_val == tuple(result_date_w.value):
            return
        # Guarded: this is a display sync, not a user choice - it must neither
        # re-render (the caller does that once) nor overwrite what the user
        # picked.
        _date_filter_guard["busy"] = True
        try:
            result_date_w.value = new_val
        finally:
            _date_filter_guard["busy"] = False

    def _sync_cloud_filter_enabled(change=None):
        """The Result filter only makes sense when cloud masking is on AND the
        current build actually carries a cloud_percentage coord. Grey it out
        otherwise (incl. before any build)."""
        masking_on = bool(cloud_masking_w.value)
        has_pct = _result_has_cloud_pct(state["result"])
        result_cloud_max_w.disabled = not (masking_on and has_pct)

    def _sync_coverage_filter_enabled(change=None):
        """The Min coverage % filter needs per-scene coverage numbers that are
        already computed. Grey it out otherwise (before any build, for a cube
        whose scene_coverage is still lazy because cloud detection was off, or
        for missions that carry no coverage at all)."""
        result_coverage_min_w.disabled = not _result_has_scene_cov(state["result"])

    # -------------------------------------------------------------------------
    # Result-panel date picker (third reversible view; composes with the cloud
    # and coverage filters). All are pure time-axis selections on the built cube;
    # the panel,
    # visualization and export all read the combined view via _effective_result,
    # so nothing is stored twice and state["result"] is never mutated.
    # -------------------------------------------------------------------------
    def _result_date_all_values():
        """The value part (unique ISO timestamp string) of every option currently
        in the date picker, in order."""
        vals = []
        for o in result_date_w.options:
            vals.append(o[1] if isinstance(o, tuple) else o)
        return vals

    def _apply_date_selection(obj):
        """Return a view of obj keeping only the acquisition dates ticked in the
        Result date picker. Pure coord selection on the existing time axis - no
        recompute, fully reversible by re-ticking. Passes obj through unchanged
        when the picker is empty/disabled, when nothing is deselected, for
        multi-feature lists, or for cubes without a time dim. Matching is on the
        full ISO timestamp string, so it survives the cloud filter's isel and is
        unambiguous even when two scenes share a calendar day."""
        if obj is None:
            return None
        if isinstance(obj, list):
            return obj
        opts = _result_date_all_values()
        if not opts:
            return obj
        sel = set(result_date_w.value)
        if len(sel) >= len(opts):  # everything ticked -> nothing to drop
            return obj
        if "time" not in getattr(obj, "dims", {}):
            return obj
        tvals = obj["time"].values
        keep = np.array([str(t) in sel for t in tvals])
        if keep.all():
            return obj
        return obj.isel(time=np.flatnonzero(keep))

    # -- Custom Composites rows ------------------------------------------------
    # Each row keeps its own widgets in a dict; custom_rows_box.children holds
    # the rendered HBoxes in the same order, so a row is removed by rebuilding
    # both lists.
    _custom_rows = []

    def _custom_row_is_blank(row):
        """A freshly added row the user has not touched yet - ignored quietly
        instead of being reported as an error."""
        return not any(
            (row[k].value or "").strip() for k in ("start", "end", "name")
        )

    def _custom_row_spec(row):
        """The dict this row stands for, without validating it."""
        return {
            "op": row["op"].value,
            ("season" if row["mode"].value == "season" else "window"): [
                (row["start"].value or "").strip(),
                (row["end"].value or "").strip(),
            ],
            "name": (row["name"].value or "").strip(),
        }

    def _custom_row_error(row, names_seen):
        """A short, plain problem description for this row, or None when it is
        fine. Short messages for the everyday mistakes, then the shared parser
        as the final authority so nothing the headless run rejects gets through
        here."""
        seasonal = row["mode"].value == "season"
        start = (row["start"].value or "").strip()
        end = (row["end"].value or "").strip()
        name = (row["name"].value or "").strip()
        fmt = "MM-DD" if seasonal else "YYYY-MM-DD"

        if not start or not end:
            return f"fill both dates as {fmt}."
        if not name:
            return "give the composite a name."
        if name in names_seen:
            return f"the name '{name}' is already used by another row."

        try:
            _parse_custom_composite(_custom_row_spec(row))
        except ValueError:
            # Re-say the parser's complaint in the interface's own words.
            checker = is_mmdd if seasonal else is_iso_date
            if not checker(start) or not checker(end):
                return f"dates must be written as {fmt}."
            if not seasonal and start > end:
                return "the start date is after the end date."
            return (
                "the name can only use letters, digits and _ , and cannot start "
                "with a digit."
            )
        return None

    def _custom_composite_specs():
        """The valid custom composites, in row order. Blank and broken rows are
        left out - _custom_validate() is what tells the user about them.

        Emits the INPUT form ({"op", "season"|"window", "name"}), not the
        parser's normal form: this list goes straight into Copy Settings, and a
        SLURM run has to be able to feed it back to calculate_statistics.
        """
        specs = []
        names_seen = set()
        for row in _custom_rows:
            if _custom_row_is_blank(row) or row["op"].disabled:
                continue
            if _custom_row_error(row, names_seen) is not None:
                continue
            names_seen.add((row["name"].value or "").strip())
            spec = _custom_row_spec(row)
            _parse_custom_composite(spec)  # validated, but keep the input form
            specs.append(spec)
        return specs

    def _custom_validate():
        """Refresh the red note under the rows. Rows listed here are ignored,
        never silently applied."""
        problems = []
        names_seen = set()
        for i, row in enumerate(_custom_rows):
            if _custom_row_is_blank(row) or row["op"].disabled:
                continue
            err = _custom_row_error(row, names_seen)
            if err:
                problems.append(f"Row {i + 1}: {err}")
            else:
                names_seen.add((row["name"].value or "").strip())
        if problems:
            custom_error_note.value = (
                "<div style='font-size:12px; color:#b91c1c; background:#fef2f2; "
                "border:1px solid #fecaca; border-radius:6px; padding:6px 8px;'>"
                "Ignored until fixed:<br>" + "<br>".join(problems) + "</div>"
            )
        else:
            custom_error_note.value = ""

    def _custom_row_widgets():
        """One editable row: period type, the two dates, the statistic, the
        name, and the button that removes it."""
        mode = widgets.Dropdown(
            options=[("Every year", "season"), ("Single window", "window")],
            value="season",
            layout=widgets.Layout(width="130px"),
        )
        # continuous_update=False: the value only changes when the field is left
        # (or Enter is pressed), so a half-typed date is never checked and the
        # Result panel is not re-rendered on every keystroke.
        start = widgets.Text(
            value="", placeholder="MM-DD", continuous_update=False,
            layout=widgets.Layout(width="105px"),
        )
        end = widgets.Text(
            value="", placeholder="MM-DD", continuous_update=False,
            layout=widgets.Layout(width="105px"),
        )
        op = widgets.Dropdown(
            options=sorted(_COMPOSITE_OPS),
            value="mean",
            layout=widgets.Layout(width="105px"),
        )
        name = widgets.Text(
            value="", placeholder="name", continuous_update=False,
            layout=widgets.Layout(width="150px"),
        )
        remove = widgets.Button(
            icon="times",
            tooltip="Remove this composite",
            layout=widgets.Layout(width="38px"),
        )
        return {
            "mode": mode, "start": start, "end": end,
            "op": op, "name": name, "remove": remove,
        }

    def _custom_sync_placeholders(row):
        """Season rows take MM-DD, single windows take full dates."""
        hint = "MM-DD" if row["mode"].value == "season" else "YYYY-MM-DD"
        row["start"].placeholder = hint
        row["end"].placeholder = hint

    def _custom_render():
        custom_rows_box.children = tuple(row["box"] for row in _custom_rows)

    def _custom_add_row(_=None, values=None):
        """Add a row, optionally pre-filled from a pasted setting."""
        row = _custom_row_widgets()
        row["box"] = widgets.HBox(
            [row["mode"], row["start"], row["end"], row["op"], row["name"],
             row["remove"]],
            layout=widgets.Layout(
                width="100%", gap="4px", flex_flow="row wrap", align_items="center"
            ),
        )
        if values:
            row["mode"].value = values.get("mode", "season")
            row["start"].value = values.get("start", "")
            row["end"].value = values.get("end", "")
            row["op"].value = values.get("op", "mean")
            row["name"].value = values.get("name", "")
        _custom_sync_placeholders(row)

        def _changed(*_a):
            _custom_sync_placeholders(row)
            _on_composites_change()

        for key in ("mode", "start", "end", "op", "name"):
            row[key].observe(_changed, names="value")
        row["remove"].on_click(lambda _b: _custom_remove_row(row))

        # Inherit the enabled/disabled state of the section (mission gating).
        for key in ("mode", "start", "end", "op", "name", "remove"):
            row[key].disabled = stats_w.disabled

        _custom_rows.append(row)
        _custom_render()
        if values is None:
            # A blank row changes nothing yet; just refresh the note.
            _custom_validate()
        else:
            _on_composites_change()
        return row

    def _custom_remove_row(row):
        if row in _custom_rows:
            _custom_rows.remove(row)
        _custom_render()
        _on_composites_change()

    def _custom_clear_rows():
        _custom_rows.clear()
        _custom_render()
        custom_error_note.value = ""

    custom_add_btn.on_click(_custom_add_row)

    def _selected_composites():
        """The composites chosen in the Temporal Composites section, in a stable
        order: the two promoted ones, then the "More composites" list, then the
        Custom Composites rows (dicts, understood by calculate_statistics).
        Empty list = time series only."""
        tokens = []
        if comp_mean_w.value and not comp_mean_w.disabled:
            tokens.append("mean_timeseries")
        if comp_median_w.value and not comp_median_w.disabled:
            tokens.append("median_timeseries")
        if not stats_w.disabled:
            tokens.extend(str(s) for s in stats_w.value)
        tokens.extend(_custom_composite_specs())
        return tokens

    def _stats_tokens_used():
        """The composite tokens the Result/export should carry. Read from the
        WIDGET, not from the stored build params: composites are derived from
        the built time series, so changing them never needs a rebuild."""
        return _selected_composites() or None

    def _stack_of(obj):
        """The time-series DataArray of a cube, or None when it has none."""
        if isinstance(obj, xr.DataArray):
            return obj
        if isinstance(obj, xr.Dataset):
            return obj.get("Time_Series")
        return None

    def _apply_composites_single(filtered_cube, tokens, keep_ts):
        """Add the requested composites to ONE filtered cube, then optionally
        drop the time series.

        Reduced from the ALREADY-FILTERED stack, so every composite describes
        exactly the dates kept in the Result panel. Uses the same
        calculate_statistics() the headless path uses, so a SLURM run of the
        copied settings produces identical layers. Lazy: the reductions are dask
        graphs and nothing computes until preview or export.
        """
        stack = _stack_of(filtered_cube)
        if stack is None:
            return filtered_cube
        if "time" not in stack.dims or stack.sizes.get("time", 0) == 0:
            return filtered_cube

        base_attrs = dict(getattr(filtered_cube, "attrs", {}) or {})
        out = calculate_statistics(stack, tokens)
        out.attrs.update(base_attrs)

        if not keep_ts and "Time_Series" in out.data_vars:
            remaining = [
                v for v in out.data_vars if v != "Time_Series"
            ]
            if remaining:
                out = out.drop_vars("Time_Series")
                # No variable uses the time axis once the series is gone; drop
                # the orphaned time coords so the cube does not advertise dates
                # it no longer holds (mirrors main._drop_timeseries).
                if not any("time" in out[v].dims for v in out.data_vars):
                    orphans = [
                        n for n, c in out.coords.items() if "time" in c.dims
                    ]
                    if orphans:
                        out = out.drop_vars(orphans)
        return out

    def _apply_composites(obj):
        """Apply the Temporal Composites selection to the filtered result.

        No-op when no composite is ticked - the cube stays the plain time
        series. Never mutates state["result"]; a derived copy is returned.

        A custom composite can still be rejected here for something the rows
        cannot check on their own - a name already taken by a band or by a
        preset composite, or a period holding no scene at all. Report it in the
        same red note and hand back the uncomposited cube, so the Result panel
        keeps working instead of raising.
        """
        tokens = _selected_composites()
        if not tokens or obj is None:
            return obj
        keep_ts = bool(keep_ts_w.value) or keep_ts_w.disabled
        try:
            return _apply_composites_inner(obj, tokens, keep_ts)
        except ValueError as exc:
            custom_error_note.value = (
                "<div style='font-size:12px; color:#b91c1c; background:#fef2f2; "
                "border:1px solid #fecaca; border-radius:6px; padding:6px 8px;'>"
                f"Composites not applied: {exc}</div>"
            )
            return obj

    def _apply_composites_inner(obj, tokens, keep_ts):
        if isinstance(obj, list):
            return [
                _apply_composites_single(o, tokens, keep_ts)
                if isinstance(o, (xr.DataArray, xr.Dataset))
                else o
                for o in obj
            ]
        return _apply_composites_single(obj, tokens, keep_ts)

    def _effective_result(obj, composite=True):
        """The cube as the Result panel / export should see it: the built cube
        with all three reversible views applied - the Max cloud % threshold, the
        Min coverage % threshold and the date picker - and (when composite=True)
        the Temporal Composites computed over the surviving dates.
        state["result"] itself is never mutated.

        The two scene filters are AND-ed (a date must be clear enough AND
        complete enough), which is also the order get_stac_layers applies them
        in, so a copied config reproduces this view.

        composite=False returns the filtered time series without composites;
        visualization uses it for the time slider and the GIF, which need the
        per-date axis. The viewer picks a composite layer through its own Layer
        dropdown instead."""
        filtered = _apply_date_selection(
            _apply_coverage_threshold(_apply_cloud_threshold(obj))
        )
        if composite:
            return _apply_composites(filtered)
        return filtered

    def _populate_result_dates(obj):
        """Fill the Result date picker from a single cube's time axis (one entry
        per acquisition date, with cloud % when available) and select all. Multi-
        feature lists and time-less cubes hide + disable the picker. Guarded so
        the programmatic select-all does not fire the change handler mid-build."""
        def _disable():
            _date_filter_guard["busy"] = True
            try:
                result_date_w.options = []
                result_date_w.value = ()
            finally:
                _date_filter_guard["busy"] = False
            state["date_user_selection"] = None
            result_date_w.disabled = True
            result_date_all_btn.disabled = True
            result_date_clear_btn.disabled = True
            result_date_acc.selected_index = None
            result_date_row.layout.display = "none"

        # Only single cubes with a time dim are date-selectable in this version.
        if obj is None or isinstance(obj, list):
            _disable()
            return
        if "time" not in getattr(obj, "dims", {}):
            _disable()
            return
        try:
            tvals = obj["time"].values
        except Exception:
            _disable()
            return
        if len(tvals) == 0:
            _disable()
            return

        # Cloud % per timestep is optional (only present on cloud-masked builds).
        cp = _result_cloud_pct(obj)
        cvals = None
        if cp is not None:
            try:
                cvals = np.asarray(cp.values)
                if cvals.shape[0] != len(tvals):
                    cvals = None
            except Exception:
                cvals = None

        # Scene coverage % per timestep (a default coord on every new cube).
        # Skipped while it is still LAZY (cloud detection off) so filling the
        # picker never forces a band read.
        covvals = None
        try:
            _sc = obj.coords.get("scene_coverage")
            if _sc is not None and (
                getattr(getattr(_sc, "data", None), "chunks", None) is None
            ):
                _cv = np.asarray(_sc.values, dtype=float)
                if _cv.shape[0] == len(tvals):
                    covvals = _cv
        except Exception:
            covvals = None

        options = []
        for i, t in enumerate(tvals):
            iso = str(t)  # unique per timestamp -> stable option value
            date_str = iso.split("T")[0] if "T" in iso else iso
            # Same order as the Dates table: cloud % first, then coverage %.
            # Bare numbers keep the row narrow enough to avoid a horizontal
            # scrollbar; result_date_legend above the box names the columns.
            parts = []
            if cvals is not None:
                try:
                    parts.append(f"{float(cvals[i]):.0f}%")
                except Exception:
                    pass
            if covvals is not None and not np.isnan(covvals[i]):
                parts.append(f"{covvals[i] * 100:.0f}%")
            label = f"{date_str} · " + " · ".join(parts) if parts else date_str
            options.append((label, iso))

        _date_filter_guard["busy"] = True
        try:
            result_date_w.options = options
            result_date_w.value = tuple(v for _, v in options)  # all selected
        finally:
            _date_filter_guard["busy"] = False
        # A fresh build resets the remembered user choice to "all dates"; the
        # cloud filter is then applied on top by the caller's re-render.
        state["date_user_selection"] = {v for _, v in options}
        result_date_w.disabled = False
        result_date_all_btn.disabled = False
        result_date_clear_btn.disabled = False
        result_date_row.layout.display = ""

    def _rerender_for_scene_filter(empty_msg):
        """Shared re-render for the two Result scene filters (Max cloud % and
        Min coverage %). Both are threshold views on the same time axis, so they
        need exactly the same follow-up: sync the date picker, then re-render the
        panel, the viewer and the GIF suggestion from the combined view.

        ``empty_msg`` is the HTML shown when THIS filter's threshold leaves no
        scene at all - a build that succeeded must not be reported as 'no data'.
        """
        # Reflect the new threshold in the date picker first (guarded, so it
        # re-renders only once, below).
        _sync_date_picker_to_cloud()
        # Emptiness is checked on the filtered TIME SERIES (composite=False): a
        # composite of zero dates would drop the time axis and read as a valid
        # (all-NaN) scene, hiding the "change the threshold" hint.
        filtered = _effective_result(state["result"], composite=False)
        if _result_is_empty(filtered) and not _result_is_empty(state["result"]):
            _set_result_viz_note(False)
            _set_result_notes([])
            _set_coreg_warning("")
            with result_out:
                clear_output()
                display(HTML(
                    "<div style='font-size:13px; color:#92400e; background:#fffbeb; "
                    "border:1px solid #fde68a; border-radius:6px; padding:10px 12px;'>"
                    f"{empty_msg}</div>"
                ))
            return
        _show_result_summary(_apply_composites(filtered))
        _refresh_viz_layers()
        _update_gif_output_suggestion()

    def _on_result_cloud_max_change(change=None):
        """Re-render the Result panel for the new Max cloud %. Visualization and
        export read the same view (via _effective_result), so the three stay in
        sync without storing a second copy of the cube."""
        if _cloud_filter_guard["busy"]:
            return
        if state["result"] is None:
            return
        _rerender_for_scene_filter(
            f"No scenes at or below <b>{int(result_cloud_max_w.value)}%</b> "
            "cloud cover. Raise <b>Max cloud %</b> to bring dates back."
        )

    def _on_result_coverage_min_change(change=None):
        """Re-render the Result panel for the new Min coverage %. Same mechanics
        as the cloud filter - a reversible time selection on the built cube -
        and the two compose: a date is kept only if it passes both."""
        if _cov_filter_guard["busy"]:
            return
        if state["result"] is None:
            return
        _rerender_for_scene_filter(
            "No scenes image at least "
            f"<b>{int(result_coverage_min_w.value)}%</b> of the area. Lower "
            "<b>Min coverage %</b> to bring dates back."
        )

    def _on_result_date_change(change=None):
        """Re-render the Result panel when the ticked dates change. Viz + export
        read the same selection via _effective_result, so all three stay in sync.
        An empty selection (or the cloud filter having removed the rest) shows a
        friendly hint rather than the generic 'no data' failure card."""
        if _date_filter_guard["busy"]:
            return
        if state["result"] is None:
            return
        # Record the user's own choice. Dates the cloud / coverage filters are
        # currently hiding are kept in the remembered set: the user cannot see
        # them to untick, so unticking a visible one must not silently discard
        # them - relaxing a threshold still brings them back.
        _passing = _dates_passing_filters()
        _prev = state.get("date_user_selection")
        _hidden = (
            set()
            if (_passing is None or _prev is None)
            else {v for v in _prev if v not in _passing}
        )
        state["date_user_selection"] = set(result_date_w.value) | _hidden
        # See _on_result_cloud_max_change: check emptiness on the time series, not
        # the composite (which would drop the time axis and mask an empty view).
        filtered = _effective_result(state["result"], composite=False)
        if _result_is_empty(filtered) and not _result_is_empty(state["result"]):
            _set_result_viz_note(False)
            _set_result_notes([])
            _set_coreg_warning("")
            with result_out:
                clear_output()
                display(HTML(
                    "<div style='font-size:13px; color:#92400e; background:#fffbeb; "
                    "border:1px solid #fde68a; border-radius:6px; padding:10px 12px;'>"
                    "No dates selected (or the cloud / coverage filters removed "
                    "the rest). "
                    "Tick at least one date - or press <b>All dates</b> - to bring "
                    "the cube back."
                    "</div>"
                ))
            return
        _show_result_summary(_apply_composites(filtered))
        _refresh_viz_layers()
        _update_gif_output_suggestion()

    def _on_result_dates_all(_):
        # "All dates" restores the user's full choice; the cloud and coverage
        # filters are then re-applied on top, so filtered-out dates stay
        # unticked (and come back by relaxing the threshold).
        state["date_user_selection"] = set(_result_date_all_values())
        _sync_date_picker_to_cloud()
        _on_result_date_change()

    def _on_result_dates_clear(_):
        state["date_user_selection"] = set()
        result_date_w.value = ()

    result_cloud_max_w.observe(_on_result_cloud_max_change, names="value")
    result_coverage_min_w.observe(_on_result_coverage_min_change, names="value")
    cloud_masking_w.observe(_sync_cloud_filter_enabled, names="value")
    result_date_w.observe(_on_result_date_change, names="value")
    result_date_all_btn.on_click(_on_result_dates_all)
    result_date_clear_btn.on_click(_on_result_dates_clear)

    # Coarser viewing resolutions, as multiples of the build resolution. Kept to
    # round factors so the label reads as a plain metre value (10 m -> 20/30/60/
    def _build_resolution():
        """Metres per pixel the current result was built at, or None."""
        params = state.get("last_call_params") or {}
        base = params.get("resolution")
        return float(base) if base else None

    def _refresh_viz_resolution():
        """Reset the viewing resolution to the build resolution.

        Free-form: any value is allowed, the build resolution is only the
        starting point. Enabled only when the params that produced the cube are
        still known - a coarser view is a re-query with those same params, so
        without them there is nothing to re-query."""
        base = _build_resolution()
        if base is None or state["result"] is None:
            viz_resolution_w.disabled = True
            return
        viz_resolution_w.value = base
        viz_resolution_w.disabled = False

    def _preview_cube_at(res):
        """The cube re-read at ``res`` metres, reusing the build's own settings.

        Cached per (resolution, build), so re-opening the viewer at a resolution
        already fetched costs nothing. Every output path is stripped: a preview
        must never write a file or replace the held cloud mask.
        """
        # The feature index is part of the key: a multi-feature build keeps one
        # state["result"], so without it switching feature would keep serving
        # the previous feature's preview.
        feat = viz_feature_w.value if isinstance(viz_feature_w.value, int) else 0
        cache = state.get("viz_preview") or {}
        stamp = (id(state["result"]), feat)
        if cache.get("res") == res and cache.get("stamp") == stamp:
            return cache["cube"]

        params = dict(state["last_call_params"])
        params.update(
            resolution=res,
            output=None,            # never write from a preview
            return_cloud_mask=False,  # the held mask stays the built one
            q=True,
        )
        built = get_stac_layers(**params)
        if isinstance(built, tuple):   # defensive: return_cloud_mask is off
            built = built[0]

        # Multi-feature builds come back as a list; take the feature the viz
        # tools are pointed at, so the preview shows the same polygon.
        if isinstance(built, list):
            cubes = [c for c in built if isinstance(c, (xr.DataArray, xr.Dataset))]
            if not cubes:
                raise ValueError("The preview build produced no cube.")
            built = built[feat] if (0 <= feat < len(built) and isinstance(
                built[feat], (xr.DataArray, xr.Dataset))) else cubes[0]

        state["viz_preview"] = {"res": res, "stamp": stamp, "cube": built}
        return built

    def _viz_cube_for_display():
        """The DataArray the viewer should show, honouring the resolution pick.

        At full detail this is exactly what it always was. At a coarser setting
        the preview cube is restricted to the dates the Result panel currently
        shows and then run through the same composite step, so the preview
        matches the filtered result date-for-date. The dates are COPIED from
        the full-resolution view rather than re-filtered: cloud percentages are
        measured per pixel grid, so re-applying the threshold on a coarser grid
        could quietly select a different set of scenes.
        """
        full = _active_result_cube()
        base = _build_resolution()
        try:
            res = float(viz_resolution_w.value)
        except (TypeError, ValueError):
            res = None
        # Blank, nonsensical, or simply the build resolution -> the cube in
        # hand already IS that view, so re-reading it would only cost a second
        # download of the same pixels.
        if full is None or not res or res <= 0 or (base and res == base):
            return _pick_dataarray_for_visualization(full)

        preview = _preview_cube_at(res)

        # Match the Result panel's surviving dates.
        kept = _active_result_cube(composite=False)
        times = None
        if kept is not None:
            _k = kept if isinstance(kept, xr.DataArray) else (
                kept["Time_Series"] if "Time_Series" in getattr(kept, "data_vars", {})
                else None
            )
            if _k is not None and "time" in _k.dims:
                times = np.asarray(_k["time"].values)
        if times is not None and "time" in getattr(preview, "dims", {}):
            preview = preview.sel(time=times, method="nearest")
            preview = preview.assign_coords(time=times)
        elif times is not None and isinstance(preview, xr.Dataset):
            if "Time_Series" in preview.data_vars:
                _p = preview["Time_Series"].sel(time=times, method="nearest")
                preview = preview.assign(
                    Time_Series=_p.assign_coords(time=times)
                )

        preview = _apply_composites(preview)
        return _pick_dataarray_for_visualization(preview)

    def _refresh_viz_feature_options():
        """Show + populate the feature picker only when the result is a list of
        several cubes; hide it for a single cube."""
        obj = state["result"]
        # Only offer features that actually produced a cube; any non-cube entry
        # (a failed feature) is not selectable for visualization.
        if isinstance(obj, list):
            opts = [
                (f"Feature {i + 1}", i)
                for i, c in enumerate(obj)
                if isinstance(c, (xr.DataArray, xr.Dataset))
            ]
        else:
            opts = []
        if len(opts) > 1:
            viz_feature_w.options = opts
            viz_feature_w.value = opts[0][1]
            viz_feature_w.disabled = False
            viz_feature_box.layout.display = ""
        else:
            viz_feature_box.layout.display = "none"

    def _set_visualization_enabled(enabled: bool):
        viz_dropdown_btn.disabled = not enabled
        if not enabled:
            viz_resolution_w.disabled = True
        gif_section_w.disabled = not enabled
        gif_display_mode_w.disabled = not enabled
        gif_band_dd.disabled = not enabled
        gif_r_dd.disabled = not enabled
        gif_g_dd.disabled = not enabled
        gif_b_dd.disabled = not enabled
        gif_stretch_w.disabled = not enabled
        gif_fps_w.disabled = not enabled
        gif_label_w.disabled = not enabled
        gif_out_path_w.disabled = not enabled
        viz_make_gif_btn.disabled = not enabled
        browse_gif_out_btn.disabled = (not enabled) or (not filechooser_available)

        if not enabled:
            with viz_out:
                clear_output()
                print("ℹ️ Build a data cube first to activate visualization tools.")
            with anim_out:
                clear_output()

    def _refresh_gif_band_options():
        """Populate the animation band selectors from the active cube's bands."""
        try:
            cube = _active_result_cube(composite=False)
            if cube is None:
                return
            da = _pick_dataarray_for_visualization(cube)
        except Exception:
            return

        if "band" in da.dims:
            bands = [str(b) for b in da.coords["band"].values]
        else:
            bands = [str(da.name) if da.name is not None else "layer"]
        lower = [b.lower() for b in bands]

        def _default(name, idx):
            if name in lower:
                return bands[lower.index(name)]
            return bands[min(idx, len(bands) - 1)]

        for dd, default in (
            (gif_band_dd, bands[0]),
            (gif_r_dd, _default("red", 0)),
            (gif_g_dd, _default("green", 1)),
            (gif_b_dd, _default("blue", 2)),
        ):
            old = dd.value
            dd.options = bands
            dd.value = old if old in bands else default

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
        mode = _gif_mode_token()
        return f"{stem}_{mode}.gif"

    def _gif_mode_token():
        """Filename token describing the current animation rendering choice."""
        sec = gif_section_w.value
        if sec == "band":
            b = str(gif_band_dd.value or "band")
            return re.sub(r"[^A-Za-z0-9._-]+", "_", f"band_{b}")
        if sec == "custom":
            return "customRGB"
        return (gif_display_mode_w.value or "rgb").strip()

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
        daterange = _resolve_daterange()

        resolution = None if resolution_w.disabled else int(resolution_w.value)
        max_cc = None if max_cc_w.disabled else int(max_cc_w.value)

        bands = list(bands_w.value) if len(bands_w.value) > 0 else None
        # Catch an empty band selection here with a friendly message instead of
        # letting None crash deep inside get_stac() ("'NoneType' object is not
        # iterable"). Only required when bands are actually offered for the
        # mission (widget enabled and has options).
        if bands is None and not bands_w.disabled and len(bands_w.options) > 0:
            raise ValueError(
                "ʕ•ᴥ•ʔ Mr. Bear noticed that you haven't selected any band and "
                "strongly asking you to select at least one band to continue."
            )
        indices = _selected_index_values() or None

        clip_raster = bool(clip_raster_w.value)
        cloud_masking = cloud_masking_w.value
        # Keep clouds only applies when cloud masking is on (needs the SCL layer).
        keep_clouds = (cloud_masking is True) and (keep_clouds_w.value == "keep")
        # Shadow masking: the widget can only be True when its gate allows it
        # (Mask Clouds + nir band + S2 L2A), but re-assert the conditions here
        # so a stale widget state can never produce an invalid call.
        shadow_masking = (
            (shadow_masking_w.value is True)
            and (cloud_masking is True)
            and not keep_clouds
        )
        source = source_w.value  # None for non-S2 missions

        export_mode = export_mode_w.value
        export_target = (export_target_w.value or "").strip() or None

        # "Build Data Cube Preview" always produces a lazy in-memory preview and
        # writes nothing to disk; the actual file is written only when the user
        # clicks "Export Current Result". So the build never passes an output path.
        output_for_get_stac = None

        # The GUI never writes the mask *during* the build. Instead it asks
        # get_stac_layers to hand back the mask in memory (return_cloud_mask) and
        # writes it itself at export time. That way a lazy/preview build writes
        # nothing, but a later "Export current results" (or a NetCDF build) still
        # produces the mask file. SLURM keeps using cloud_mask_output directly.
        return_cloud_mask = (cloud_masking is True) and (export_mask_w.value is True)

        params = {
            "mission": mission,
            "polygon": polygon,
            "resolution": resolution,
            "daterange": daterange,
            "bands": bands,
            "max_cc": max_cc,
            "clip_raster": clip_raster,
            "cloud_masking": cloud_masking,
            "keep_clouds": keep_clouds,
            "shadow_masking": shadow_masking,
            "nir_dark_threshold": float(shadow_nir_dark_w.value),
            "shadow_proj_distance": float(shadow_proj_dist_w.value),
            "return_cloud_mask": return_cloud_mask,
            "indices": indices,
            "output": output_for_get_stac,
            # Temporal Composites are intentionally NOT sent to the build. The
            # GUI builds the plain time series so the user can filter by
            # date/cloud first, then derives every composite from the filtered
            # result client-side (see _apply_composites / _effective_result) -
            # which also means changing a composite needs no rebuild. Copy
            # Settings still writes stats + keep_timeseries for headless/SLURM
            # runs, where interactive filtering isn't available.
            "aggregator": None,
            "stats": None,
            "source": source,
            "resampling_method": resampling_w.value,
            # None = Automatic: get_stac_layers picks the projection natively
            # covering most of the area. A typed CRS wins over the dropdown.
            "crs": _effective_crs(),
            # Sentinel-2 only; the widget is cleared + disabled for other
            # missions, so this is None there.
            "scene_metadata": (
                list(scene_metadata_w.value)
                if len(scene_metadata_w.value) > 0
                else None
            ),
            # Mosaic (default) vs separate N-S tiles. Greyed to "mosaic" for
            # non-S2-L2A missions, so this is always "mosaic" there.
            "tile_handling": tile_handling_w.value,
            # Across-track: keep (default) vs remove partial swath/orbit-edge
            # scenes. Greyed to "keep" for non-optical missions.
            "partial_scene_handling": partial_scene_w.value,
            # Min share of the AOI a scene must image to be kept (percent box
            # -> fraction). Only used in remove mode.
            "min_scene_coverage": float(min_coverage_w.value) / 100.0,
            # Pre-load footprint prefilter (percent box -> fraction). None when
            # 0, so the build skips the whole step rather than running it as a
            # no-op that would still reproject every footprint.
            "min_footprint_coverage": (
                (float(skip_footprint_w.value) / 100.0)
                if float(skip_footprint_w.value) > 0
                else None
            ),
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
            if "Time_Series" in result_obj.data_vars:
                da = result_obj["Time_Series"]
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

    def _export_feature_list(cubes, export_mode, export_target):
        """Export a multi-feature batch (list of cubes), one output per feature.
        Each cube is written then released, so RAM stays ~one feature:
          - NetCDF: <stem>_<i>.nc   (polygons_01.nc, polygons_02.nc, ...)
          - COGs:   <folder>/<i>/<date>.tif   (one subfolder per feature)

        The index is zero-padded to the batch size, so the files sort in build
        order; unpadded they run _1, _10, _11, ... _9.
        """
        cubes = [c for c in cubes if isinstance(c, (xr.DataArray, xr.Dataset))]
        n = len(cubes)
        if n == 0:
            raise ValueError("No exportable cubes in the result.")
        pad = max(2, len(str(n)))

        def _ref(c):
            da = c
            if isinstance(c, xr.Dataset):
                da = c.get("Time_Series")
                if da is None and len(c.data_vars):
                    da = c[list(c.data_vars)[0]]
            crs_ref = da.attrs.get("crs") if da is not None else None
            transform_ref = da.attrs.get("transform") if da is not None else None
            return crs_ref, transform_ref

        written = []
        if export_mode in ("netcdf", "zarr"):
            want_ext = ".zarr" if export_mode == "zarr" else ".nc"
            stem, ext = os.path.splitext(export_target)
            if ext.lower() not in (".nc", ".zarr"):
                stem = export_target
            ext = want_ext
            compress = bool(export_compress_w.value)
            want_vrt = bool(export_vrt_w.value) and export_mode == "netcdf"
            for i, c in enumerate(cubes, 1):
                out_i = f"{stem}_{i:0{pad}d}{ext}"
                Path(out_i).parent.mkdir(parents=True, exist_ok=True)
                if isinstance(c, xr.DataArray):
                    export_stac(c, out_i, var_name=(c.name or "Time_Series"),
                                compress=compress, vrt=want_vrt)
                else:
                    crs_ref, transform_ref = _ref(c)
                    export_stac(c, out_i, crs=crs_ref, transform=transform_ref,
                                compress=compress, vrt=want_vrt)
                written.append(out_i)
            return {"mode": export_mode, "target": export_target, "files": written, "count": n}

        if export_mode == "cogs":
            base = Path(export_target)
            for i, c in enumerate(cubes, 1):
                sub = base / f"{i:0{pad}d}"
                sub.mkdir(parents=True, exist_ok=True)
                export_to_cogs(stac=c, output_dir=str(sub), prefix="", dtype="float32")
                written.append(str(sub))
            return {"mode": "cogs", "target": str(base), "folders": written, "count": n}

        raise ValueError(f"Unsupported export mode: {export_mode}")

    def _write_held_cloud_mask():
        """Write the in-memory binary mask held from the build to its path, if the
        mask export is enabled. Mirrors the main NetCDF write; skipped for COG (by
        design). Returns the path(s) written, or None. Cheap no-op when disabled.

        A batch build holds a list of masks -> one file per feature
        (<stem>_<i>.nc), matching how the cubes are split.
        """
        if export_mask_w.value is not True:
            return None
        if export_mode_w.value == "cogs":
            return None  # COG builds don't get a mask (by design)
        mask = state.get("cloud_mask_result")
        if mask is None:
            return None
        path = (cloud_mask_output_w.value or "").strip()
        if not path:
            return None
        if not path.lower().endswith(".nc"):
            path = path + ".nc"

        if isinstance(mask, list):
            stem, ext = os.path.splitext(path)
            written = []
            i = 0
            pad = max(2, len(str(len(mask))))
            for m in mask:
                if not isinstance(m, (xr.DataArray, xr.Dataset)):
                    continue
                i += 1
                out_i = f"{stem}_{i:0{pad}d}{ext}"
                Path(out_i).parent.mkdir(parents=True, exist_ok=True)
                export_stac(m, out_i, var_name="Cloud_Stack",
                            compress=bool(export_compress_w.value))
                written.append(out_i)
            return written or None

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        export_stac(mask, path, var_name="Cloud_Stack",
                    compress=bool(export_compress_w.value))
        return path

    def _write_granule_metadata_xmls():
        """Download the granule metadata XMLs (MTD_TL.xml) for the scenes of
        the just-exported result, into <export target>_granule_metadata/
        (for COGs: <folder>/granule_metadata/). Runs only when the Export
        Granule Metadata box is ticked; a metadata-only STAC re-query, no
        pixels. Uses the PRE-composite filtered view so the XMLs match the
        scenes that actually fed the export even when a Temporal Composite
        collapsed the time axis. Returns the folder(s) written, or None.
        """
        if export_granule_meta_w.value is not True or export_granule_meta_w.disabled:
            return None
        target = (export_target_w.value or "").strip()
        if not target:
            return None
        from stac2cube.get_data import export_granule_metadata

        if export_mode_w.value == "cogs":
            base_dir = str(Path(target) / "granule_metadata")
        else:
            stem, ext = os.path.splitext(target)
            base_dir = (
                stem if ext.lower() in (".nc", ".zarr") else target
            ) + "_granule_metadata"

        # composite=False: keep the per-scene time axis for date matching.
        obj = _effective_result(state["result"], composite=False)

        def _main_da(c):
            if isinstance(c, xr.Dataset):
                da = c.get("Time_Series")
                if da is None and len(c.data_vars):
                    da = c[list(c.data_vars)[0]]
                return da
            return c

        if isinstance(obj, list):
            written = []
            i = 0
            pad = max(2, len(str(len(obj))))
            for c in obj:
                da = _main_da(c)
                if da is None:
                    continue
                i += 1
                export_granule_metadata(da, f"{base_dir}_{i:0{pad}d}", q=True)
                written.append(f"{base_dir}_{i:0{pad}d}")
            return written or None

        da = _main_da(obj)
        if da is None:
            return None
        export_granule_metadata(da, base_dir, q=True)
        return base_dir

    def _write_settings_sidecar():
        """Write the current settings as a config JSON next to the exported
        cube: same folder, same name, '.json' instead of the cube extension
        (for a COG folder: inside it, named after the folder). Content is
        exactly what Copy Settings produces, so the file runs headless on
        SLURM and pastes back into this form. Runs only when the Export
        Settings box is ticked. Returns the path written, or None.

        A multi-feature batch gets ONE file at the un-suffixed export path -
        the same config rebuilds every feature (it carries the multi-feature
        polygon file itself).
        """
        if export_settings_w.value is not True:
            return None
        target = (export_target_w.value or "").strip()
        if not target:
            return None
        path = settings_sidecar_path(target, export_mode_w.value)
        if not path:
            return None
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_build_json_syntax_text())
        return path

    def _write_statistics_csv(info):
        """Statistics CSV next to the exported cube, read back FROM the file
        that was just written.

        Reading the file rather than the in-memory result is what keeps this
        free: export_stac streams and retains nothing, so reducing the lazy
        result here would fetch every scene from the archive a second time.
        The written file is local, and the numbers are bit-identical (verified
        on a NetCDF, a zlib NetCDF and a Zarr round trip).

        A multi-feature batch gets one CSV per cube, like the .vrt: statistics
        describe one cube, so a single file for N features would describe none
        of them. Returns the list of paths written, or None.
        """
        if export_csv_w.value is not True:
            return None
        # COGs are a folder of GeoTIFFs with no cube file to read back.
        if info.get("mode") not in ("netcdf", "zarr"):
            return None
        targets = info.get("files") or (
            [info.get("target")] if info.get("target") else []
        )
        written = []
        for target in targets:
            if not target or not os.path.exists(target):
                continue
            csv_path = _statistics_csv_path(target)
            export_cube_statistics(target, csv_path=csv_path, q=True)
            written.append(csv_path)
        return written or None

    def _export_current_result(export_mode: str, export_target: str):
        if state["result"] is None:
            raise ValueError("No generated result is available to export yet.")

        if export_mode == "lazy":
            raise ValueError(_EXPORT_MODE_REMINDER)

        if not export_target:
            raise ValueError("Please provide Output file / folder before exporting.")

        # Guard against exporting a fully-filtered-out selection: with a Temporal
        # Composite active this would silently write an all-NaN scene (no time
        # axis left to flag as empty), so check the pre-composite series first.
        if _result_is_empty(_effective_result(state["result"], composite=False)) \
                and not _result_is_empty(state["result"]):
            raise ValueError(
                "The current date / max cloud % selection keeps no scenes. "
                "Relax the filters in the Result panel before exporting."
            )

        # Export exactly what the Result panel shows: if the Max cloud % filter
        # and/or the date picker are active, write that filtered view, not the
        # full build. When a Temporal Composite is set, this is the collapsed
        # single scene (composite defaults to True in _effective_result).
        obj = _effective_result(state["result"])

        # Multi-feature batch result: a list of cubes -> one output per feature.
        if isinstance(obj, list):
            return _export_feature_list(obj, export_mode, export_target)

        if export_mode in ("netcdf", "zarr"):
            # NetCDF -> single .nc file; Zarr -> a .zarr store. Same code path;
            # export_stac dispatches on the extension. Swap any existing cube
            # extension so switching mode never yields e.g. "cube.nc.zarr".
            ext = ".zarr" if export_mode == "zarr" else ".nc"
            target = export_target
            low = target.lower()
            if low.endswith(".nc") or low.endswith(".zarr"):
                target = target[: target.rfind(".")]
            if not target.lower().endswith(ext):
                target = target + ext
            export_target_w.value = target

            Path(target).parent.mkdir(parents=True, exist_ok=True)
            compress = bool(export_compress_w.value)
            # NetCDF-only sidecar; export_stac ignores it on the Zarr path, but
            # gate it here too so the intent is visible at the call site.
            want_vrt = bool(export_vrt_w.value) and export_mode == "netcdf"

            if isinstance(obj, xr.DataArray):
                export_stac(
                    stac=obj,
                    output=target,
                    var_name=(obj.name or "Time_Series"),
                    compress=compress,
                    vrt=want_vrt,
                )

            elif isinstance(obj, xr.Dataset):
                # Fix for stats datasets: Dataset may not expose .crs / .transform directly
                ref_da = None
                if "Time_Series" in obj.data_vars:
                    ref_da = obj["Time_Series"]
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
                    stac=obj, output=target, crs=crs_ref, transform=transform_ref,
                    compress=compress, vrt=want_vrt,
                )

            else:
                raise TypeError(
                    f"Unsupported result type for {export_mode} export: {type(obj)}"
                )

            return {"mode": export_mode, "target": target}

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
        daterange = _resolve_daterange()

        resolution = None if resolution_w.disabled else int(resolution_w.value)
        max_cc = None if max_cc_w.disabled else int(max_cc_w.value)

        bands = list(bands_w.value) if len(bands_w.value) > 0 else None
        indices = _selected_index_values() or None
        # Temporal Composites section -> stats + keep_timeseries. Headless
        # applies them after the same date/cloud filters this JSON carries, so
        # a SLURM run reduces over exactly the scenes the Result panel shows.
        stats = _selected_composites() or None
        keep_timeseries = bool(keep_ts_w.value) or keep_ts_w.disabled

        clip_raster = bool(clip_raster_w.value)
        cloud_masking = cloud_masking_w.value
        keep_clouds = (cloud_masking is True) and (keep_clouds_w.value == "keep")
        shadow_masking = (
            (shadow_masking_w.value is True)
            and (cloud_masking is True)
            and not keep_clouds
        )
        cloud_mask_output = None
        if (cloud_masking is True) and (export_mask_w.value is True):
            cloud_mask_output = (cloud_mask_output_w.value or "").strip() or None
        # The mean/median dropdown retired into the Temporal Composites section;
        # its job is now stats + keep_timeseries, so no aggregator is emitted.
        aggregator = None

        export_mode = export_mode_w.value
        export_target = (
            None
            if export_target_w.disabled
            else ((export_target_w.value or "").strip() or None)
        )

        # JSON is for get_stac_layers config, and a SLURM run always writes, so
        # every export mode carries its target: a FILE for netcdf / zarr, a
        # FOLDER for cogs. export_format says which - get_stac_layers dispatches
        # on it (cogs -> export_to_cogs) instead of guessing from the path.
        output_for_json = export_target or None
        export_format_for_json = export_mode if output_for_json else None

        # Result panel's "Max cloud %" -> scene_cloud_coverage. Only meaningful
        # with cloud detection on (the cloud_percentage coord it filters does
        # not exist otherwise), and 100 means "keep everything" - emit null for
        # both so the JSON only carries the parameter when it actually filters.
        scene_cloud_coverage = None
        if cloud_masking is True and int(result_cloud_max_w.value) < 100:
            scene_cloud_coverage = int(result_cloud_max_w.value)

        # Result panel's "Date Selection" -> dates. Emitted only when the user
        # actually deselected something: with every date ticked the list would
        # just pin the build to what this preview happened to find, so null
        # (keep everything) is the honest config. The picker is empty for
        # multi-feature batches, so a batch config never carries dates -
        # get_stac_layers rejects that combination anyway. An emptied selection
        # is emitted as [] on purpose: it fails loudly there instead of
        # silently exporting every date.
        # Coverage filter -> partial_scene_handling / min_scene_coverage. TWO
        # widgets can ask for it: the Advanced "Overlapping Tile Handling ->
        # Across-track" dropdown (drops partial scenes during the build) and the
        # Result panel's "Min coverage %" (the same threshold, applied to the
        # built cube without a rebuild). get_stac_layers has ONE coverage filter,
        # so emit the STRICTER of the two - that reproduces exactly the scenes
        # the Result panel is currently showing.
        partial_handling_for_json = partial_scene_w.value
        min_cov_for_json = float(min_coverage_w.value) / 100.0
        _result_min_cov = float(result_coverage_min_w.value) / 100.0
        if _result_min_cov > 0:
            if partial_handling_for_json == "remove":
                min_cov_for_json = max(min_cov_for_json, _result_min_cov)
            else:
                partial_handling_for_json = "remove"
                min_cov_for_json = _result_min_cov

        dates_for_json = None
        _all_result_dates = _result_date_all_values()
        if _all_result_dates:
            _sel_dates = [
                d for d in _all_result_dates if d in set(result_date_w.value)
            ]
            if len(_sel_dates) < len(_all_result_dates):
                dates_for_json = _sel_dates

        json_payload = {
            "parameters": {
                "mission": mission_for_json,
                "source": source_w.value,
                "polygon": polygon,
                "resolution": resolution,
                "daterange": daterange,
                "bands": bands,
                "indices": indices,
                "max_cc": max_cc,
                "scene_cloud_coverage": scene_cloud_coverage,
                "cloud_masking": cloud_masking,
                "keep_clouds": keep_clouds,
                # Shadow masking rides on the SCL cloud mask, so it is only
                # meaningful when clouds are actually masked (same gate the
                # build applies in _collect_params).
                "shadow_masking": shadow_masking,
                "nir_dark_threshold": float(shadow_nir_dark_w.value),
                "shadow_proj_distance": float(shadow_proj_dist_w.value),
                "cloud_mask_output": cloud_mask_output,
                "dates": dates_for_json,
                "output": output_for_json,
                "export_format": export_format_for_json,
                "clip_raster": clip_raster,
                "resampling_method": resampling_w.value,
                # null = Automatic (chosen from the scenes at build time).
                "crs": _effective_crs(),
                # A temporal composite collapses the time dimension the
                # per-scene coords live on - get_stac_layers rejects the
                # combination, so emit null when an aggregator is set.
                "scene_metadata": (
                    list(scene_metadata_w.value)
                    if (len(scene_metadata_w.value) > 0 and not aggregator)
                    else None
                ),
                # Granule metadata XML folder, derived from the output path.
                # Needs a concrete output (lazy/COG JSON runs have none) and
                # no aggregator (get_stac_layers rejects that combination).
                "metadata_output": (
                    os.path.splitext(output_for_json)[0] + "_granule_metadata"
                    if (
                        export_granule_meta_w.value is True
                        and not export_granule_meta_w.disabled
                        and output_for_json
                        and not aggregator
                    )
                    else None
                ),
                # Separate tiles are incompatible with a temporal composite
                # (get_stac_layers rejects the combination), so fall back to
                # mosaic when an aggregator is set.
                "tile_handling": (
                    "mosaic" if aggregator else tile_handling_w.value
                ),
                # Coverage filter (compatible with a composite: partials are
                # dropped before the collapse). Merged from the Advanced
                # Across-track dropdown and the Result panel's Min coverage % -
                # see partial_handling_for_json above.
                "partial_scene_handling": partial_handling_for_json,
                "min_scene_coverage": min_cov_for_json,
                # Pre-load footprint prefilter. Emitted straight from its own
                # widget with no merging: unlike the two coverage filters above,
                # nothing else in the GUI can ask for it, and it acts BEFORE the
                # load - so on SLURM it is the one setting that makes the job
                # cheaper rather than only making its output smaller.
                "min_footprint_coverage": (
                    (float(skip_footprint_w.value) / 100.0)
                    if float(skip_footprint_w.value) > 0
                    else None
                ),
                "aggregator": aggregator,
                "stats": stats,
                "keep_timeseries": keep_timeseries,
                # zlib compression is a NetCDF-only option (Zarr and COGs bring
                # their own codecs), and the checkbox is hidden - not reset -
                # for the other modes, so pin it to the mode it applies to.
                "compress": bool(export_compress_w.value) and export_mode == "netcdf",
                # Same NetCDF-only pinning for the QGIS band-mapping sidecar.
                "vrt": bool(export_vrt_w.value) and export_mode == "netcdf",
                # The settings file describes itself: a config that asks for
                # the JSON sidecar writes one again on the next (SLURM) run.
                # Needs an export target to sit next to, so it is pinned to
                # false for a lazy config with no output.
                "export_settings": (
                    bool(export_settings_w.value) and bool(output_for_json)
                ),
                # Read back from the exported cube, so it needs one: pinned to
                # the two modes that write a cube file, like 'vrt' is pinned to
                # NetCDF. Headless honours it in get_stac_layers.
                "statistics_csv": (
                    bool(export_csv_w.value)
                    and bool(output_for_json)
                    and export_mode in ("netcdf", "zarr")
                ),
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

            _show_status("✅ Settings copied to clipboard.")

        except Exception as e:
            _show_status(_friendly_error(e, "Copying the settings"))

    # -------------------------------------------------------------------------
    # Paste Settings: the reverse of Copy Settings - read a settings JSON and
    # push every parameter it carries back into the widgets.
    #
    # The JSON is the get_stac_layers config (same file SLURM runs), so a few
    # GUI-only choices simply are not in it and cannot be restored: the COG
    # export mode (its JSON "output" is null by design), the animation / GIF
    # settings and the Result-panel date ticks. Those are reported back to the
    # user instead of being guessed.
    # -------------------------------------------------------------------------
    # Mission alias ("s2") -> full mission name ("sentinel_2_l2a"), from the
    # missions table itself, so it stays correct when missions are added.
    def _mission_name_from_json(value):
        if value is None:
            return None
        v = str(value)
        if v in mission_meta:
            return v
        for name, meta in mission_meta.items():
            alias = meta.get("alias", meta.get("allias"))
            if alias is not None and str(alias) == v:
                return name
        return None

    def _option_values(widget):
        """The selectable VALUES of a widget, whatever shape its options take:
        ipywidgets accepts both plain entries (stats, metadata) and
        (label, value) pairs (missions, bands, dropdowns)."""
        return [
            opt[1] if (isinstance(opt, tuple) and len(opt) == 2) else opt
            for opt in widget.options
        ]

    def _set_dropdown_if_valid(widget, value, label, warnings):
        """Set a dropdown only to a value it actually offers; warn otherwise."""
        valid = _option_values(widget)
        if value in valid:
            widget.value = value
            return True
        warnings.append(f"{label}: '{value}' is not available here - kept '{widget.value}'.")
        return False

    def _apply_daterange_from_json(dr, warnings):
        if dr is None:
            warnings.append("Time period: not set in the settings - kept as is.")
            return
        # ["YYYY-MM-DD", "YYYY-MM-DD"] -> the simple From / To pickers.
        if (
            isinstance(dr, (list, tuple))
            and len(dr) == 2
            and all(isinstance(d, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", d) for d in dr)
        ):
            advanced_dates_w.value = False
            date_from_w.value = _date.fromisoformat(dr[0])
            date_to_w.value = _date.fromisoformat(dr[1])
            return
        # {"season": [...], "years": ...} -> the seasonal text field.
        if isinstance(dr, dict) and "season" in dr and "years" in dr:
            years = dr["years"]
            if years == "all":
                mode = "seasonal_all"
            elif isinstance(years, str):
                mode = "seasonal_range"
            else:
                mode = "seasonal_selected"
            advanced_dates_w.value = True
            daterange_mode_w.value = mode          # may rewrite the text field...
            daterange_w.value = json.dumps(dr)     # ...so write the text after it
            return
        warnings.append(
            f"Time period: unsupported daterange {dr!r} - kept the current dates."
        )

    def _apply_cloud_settings_from_json(p, warnings):
        """Restore the cloud block. The three guided presets are exact shortcuts
        for particular raw settings, so a pasted config that matches one selects
        that preset; anything else lands on 'Free Settings' with the raw values."""
        cloud_masking = p.get("cloud_masking")
        keep_clouds = bool(p.get("keep_clouds"))
        max_cc = p.get("max_cc")
        mask_out = p.get("cloud_mask_output")

        preset_1 = cloud_masking is True and keep_clouds and max_cc in (None, 100)
        preset_2 = (
            cloud_masking is True
            and not keep_clouds
            and max_cc in (None, 100)
            and not mask_out
        )

        if preset_1:
            _select_cloud_preset(1)
        elif preset_2:
            _select_cloud_preset(2)
        else:
            _select_cloud_preset(3)   # free settings: unlock the raw widgets
            # A config that simply omits cloud_masking (hand-written SLURM JSON)
            # keeps whatever the mission defaults to, without a warning.
            if cloud_masking is not None:
                _set_dropdown_if_valid(
                    cloud_masking_w, cloud_masking, "Cloud detection", warnings
                )
            keep_clouds_w.value = "keep" if keep_clouds else "mask"
            if max_cc is not None and not max_cc_w.disabled:
                max_cc_w.value = int(max_cc)

        # Export mask (+ its path) is the user's under presets 1 and 3.
        if mask_out and cloud_masking is True:
            export_mask_w.value = True
            cloud_mask_output_w.value = str(mask_out)
        elif not export_mask_w.disabled:
            export_mask_w.value = False
        _sync_export_mask_visibility()
        _sync_export_mask_path_enabled()

    def _apply_json_settings(payload):
        """Push a parsed settings payload into the widgets. Returns the list of
        things that could not be restored, so the user is told rather than left
        with a silently half-applied form."""
        params = payload.get("parameters") if isinstance(payload, dict) else None
        if not isinstance(params, dict):
            raise ValueError(
                'The pasted JSON has no "parameters" object. Paste the settings '
                'exactly as Copy Settings produced them.'
            )
        warnings = []

        # 1) Mission first: it rebuilds the band / index / stats / source lists
        #    every later step writes into.
        # Output projection. A stored CRS goes into the free-text box rather than
        # the dropdown: the detected list is only populated after a search, and
        # the box overrides it anyway, so this restores exactly what was saved.
        _crs_json = params.get("crs")
        crs_user_w.value = "" if _crs_json in (None, "") else str(_crs_json)
        _sync_crs_controls()

        m_name = _mission_name_from_json(params.get("mission"))
        if m_name is None:
            warnings.append(
                f"Mission: '{params.get('mission')}' is unknown - kept "
                f"'{mission_dd.value}'."
            )
        elif m_name not in _SELECTABLE_MISSIONS:
            # The dropdown would snap back anyway (see _guard_mission_choice);
            # say so here instead of silently building another mission.
            warnings.append(
                f"Mission: '{m_name}' is not available yet - kept "
                f"'{mission_dd.value}'."
            )
            _update_from_mission()
        elif m_name != mission_dd.value:
            mission_dd.value = m_name       # fires _update_from_mission
        else:
            _update_from_mission()          # re-assert the mission defaults

        # 2) Data source (Sentinel-2 only; the dropdown is "Not applicable"
        #    elsewhere, so an unmatched value is reported).
        if params.get("source") is not None and not source_w.disabled:
            _set_dropdown_if_valid(source_w, params["source"], "Data source", warnings)

        # 3) Area of interest: a path string, or a bbox list written back as text.
        polygon = params.get("polygon")
        if isinstance(polygon, (list, tuple)):
            polygon_w.value = json.dumps(list(polygon))
        elif polygon is not None:
            polygon_w.value = str(polygon)

        # 4) Resolution (greyed for fixed-resolution missions).
        if params.get("resolution") is not None and not resolution_w.disabled:
            resolution_w.value = int(params["resolution"])

        # 5) Time period.
        _apply_daterange_from_json(params.get("daterange"), warnings)

        # 6) Bands, then indices: index availability follows the band selection.
        band_values = _option_values(bands_w)
        wanted_bands = [str(b) for b in (params.get("bands") or [])]
        bands_w.value = tuple(b for b in wanted_bands if b in band_values)
        missing_bands = [b for b in wanted_bands if b not in band_values]
        if missing_bands:
            warnings.append(
                "Bands not offered by this mission: " + ", ".join(missing_bands)
            )
        _refresh_index_availability()

        wanted_idx = {str(i) for i in (params.get("indices") or [])}
        for idx, cb in _index_rows.items():
            cb.value = (idx in wanted_idx) and not cb.disabled
        unavailable_idx = [
            i for i in wanted_idx
            if i not in _index_rows or _index_rows[i].disabled
        ]
        if unavailable_idx:
            warnings.append(
                "Indices that need unselected bands (or are not available): "
                + ", ".join(sorted(unavailable_idx))
            )

        # 7) Clipping + the whole cloud block (presets included).
        if not clip_raster_w.disabled:
            clip_raster_w.value = bool(params.get("clip_raster"))
        _apply_cloud_settings_from_json(params, warnings)

        # 8) Shadow masking: its gate (Mask Clouds + nir + S2 L2A) has just been
        #    re-evaluated by the cloud block, so only set it when allowed.
        _sync_shadow_masking_enabled()
        want_shadow = bool(params.get("shadow_masking"))
        if want_shadow and shadow_masking_w.disabled:
            warnings.append(
                "Shadow masking: needs Mask Clouds + the nir band on Sentinel-2 "
                "L2A - left off."
            )
        else:
            shadow_masking_w.value = want_shadow
        if params.get("nir_dark_threshold") is not None:
            shadow_nir_dark_w.value = float(params["nir_dark_threshold"])
        if params.get("shadow_proj_distance") is not None:
            shadow_proj_dist_w.value = float(params["shadow_proj_distance"])

        # 9) Remaining advanced parameters.
        if params.get("resampling_method") is not None:
            _set_dropdown_if_valid(
                resampling_w, params["resampling_method"], "Resampling", warnings
            )
        if params.get("tile_handling") is not None and not tile_handling_w.disabled:
            _set_dropdown_if_valid(
                tile_handling_w, params["tile_handling"], "Tile handling", warnings
            )
        if params.get("partial_scene_handling") is not None and not partial_scene_w.disabled:
            _set_dropdown_if_valid(
                partial_scene_w, params["partial_scene_handling"],
                "Scene coverage", warnings,
            )
        if params.get("min_scene_coverage") is not None:
            min_coverage_w.value = int(round(float(params["min_scene_coverage"]) * 100))
        # Footprint prefilter: absent / null means "off", which is 0 in the box.
        # Restored unconditionally so loading a config that does NOT use it also
        # clears a value left over from a previous load.
        try:
            _fp_cfg = params.get("min_footprint_coverage")
            skip_footprint_w.value = (
                round(float(_fp_cfg) * 100.0, 2) if _fp_cfg else 0.0
            )
        except (TypeError, ValueError):
            warnings.append(
                "min_footprint_coverage was not a number; left at 0 (off)."
            )

        meta_values = _option_values(scene_metadata_w)
        wanted_meta = [str(m) for m in (params.get("scene_metadata") or [])]
        scene_metadata_w.value = tuple(m for m in wanted_meta if m in meta_values)
        missing_meta = [m for m in wanted_meta if m not in meta_values]
        if missing_meta:
            warnings.append(
                "Scene metadata not published by this source: " + ", ".join(missing_meta)
            )

        # 10) Temporal Composites. stats carries the tokens; the two promoted
        #     ones map to their checkboxes and the rest to the "More composites"
        #     list. A legacy config written with the retired aggregator dropdown
        #     is translated into the equivalent composite (aggregator="median"
        #     == median_timeseries without the time series).
        #     Custom composites are dicts in the same list and must be split off
        #     BEFORE the mission-option filter below, which only knows tokens and
        #     would otherwise report every one of them as unavailable.
        _raw_stats = params.get("stats") or []
        wanted_stats = [str(s) for s in _raw_stats if not isinstance(s, dict)]
        _custom_stats = [s for s in _raw_stats if isinstance(s, dict)]

        _custom_clear_rows()
        for _spec in _custom_stats:
            try:
                _norm = _parse_custom_composite(_spec)
            except ValueError as _exc:
                warnings.append(f"Custom composite skipped - {_exc}")
                continue
            _custom_add_row(
                values={
                    "mode": _norm["mode"],
                    "start": _norm["start"],
                    "end": _norm["end"],
                    "op": _norm["op"],
                    "name": _norm["name"],
                }
            )
            if _norm["years"]:
                # The rows have no year picker: a season covers every year the
                # cube holds. Say so rather than dropping the restriction quietly.
                warnings.append(
                    f"Custom composite '{_norm['name']}' was limited to years "
                    f"{_norm['years']}; the interface applies it to every year "
                    "of the cube."
                )

        _legacy_agg = params.get("aggregator")
        _legacy_used = False
        if _legacy_agg and f"{_legacy_agg}_timeseries" not in wanted_stats:
            wanted_stats.append(f"{_legacy_agg}_timeseries")
            _legacy_used = True

        comp_mean_w.value = "mean_timeseries" in wanted_stats
        comp_median_w.value = "median_timeseries" in wanted_stats

        stats_values = _option_values(stats_w)
        stats_w.value = tuple(s for s in wanted_stats if s in stats_values)
        missing_stats = [
            s
            for s in wanted_stats
            if s not in stats_values and s not in _COMMON_COMPOSITES
        ]
        if missing_stats:
            warnings.append(
                "Composites not available for this mission: " + ", ".join(missing_stats)
            )

        # keep_timeseries: explicit in current settings; a legacy aggregator
        # config implies it (the dropdown always replaced the time series).
        _keep_ts_val = params.get("keep_timeseries")
        if _keep_ts_val is None:
            _keep_ts_val = not _legacy_used
        _sync_keep_timeseries()  # un-grey it first if a composite is selected
        keep_ts_w.value = bool(_keep_ts_val) or keep_ts_w.disabled
        _sync_keep_timeseries()
        if _legacy_used:
            warnings.append(
                f"These settings use the retired Temporal Composite "
                f"'{_legacy_agg}' - applied as '{_legacy_agg}_timeseries' with "
                "the time series dropped."
            )

        # 11) Export. export_format names the mode outright (netcdf / zarr /
        #     cogs); a hand-written SLURM config may omit it, in which case the
        #     format falls back to the file extension, as get_stac_layers does.
        output = params.get("output")
        if output:
            fmt = str(params.get("export_format") or "").strip().lower()
            if fmt not in ("netcdf", "zarr", "cogs"):
                fmt = "zarr" if str(output).lower().endswith(".zarr") else "netcdf"
            export_mode_w.value = fmt
            export_target_w.value = str(output)
        else:
            warnings.append(
                "Export mode / output path: not stored in the settings - "
                "kept the current choice."
            )
        export_compress_w.value = bool(params.get("compress"))
        # Absent from an older/hand-written config -> keep the default (on)
        # rather than silently turning the sidecar off.
        if "vrt" in params:
            export_vrt_w.value = bool(params.get("vrt"))
        # Absent from a hand-written config -> leave the box as it is.
        if "export_settings" in params:
            export_settings_w.value = bool(params.get("export_settings"))
        if "statistics_csv" in params:
            export_csv_w.value = bool(params.get("statistics_csv"))
        # Re-run the gate last: a hand-written config can ask for a side output
        # the pasted export mode cannot produce, and the box must not be left
        # ticked-but-greyed. Copy Settings never emits such a pair (both flags
        # are pinned to their modes), so this only ever fires on hand edits.
        _apply_compress_visibility()
        if not export_granule_meta_w.disabled:
            export_granule_meta_w.value = params.get("metadata_output") is not None

        # 12) Result-panel cloud filter (greyed until a build produces cloud %).
        scc = params.get("scene_cloud_coverage")
        result_cloud_max_w.value = 100 if scc is None else int(scc)

        # The coverage filter came back as partial_scene_handling /
        # min_scene_coverage and was applied to the Advanced widgets above, so
        # the rebuild drops the same scenes. Reset the Result panel's own box to
        # 0 - leaving it set would filter the already-filtered cube a second
        # time (same scenes, but a threshold shown in two places at once).
        _cov_filter_guard["busy"] = True
        try:
            result_coverage_min_w.value = 0
        finally:
            _cov_filter_guard["busy"] = False

        # 13) Date selection. The picker is filled from a BUILT cube's time axis,
        #     so on the first paste there is nothing to tick - the user builds and
        #     pastes again, and this second paste restores the exact selection.
        #     Timestamps are normalised through numpy exactly as
        #     get_stac_layers._normalize_dates does, so a config written from a
        #     microsecond time axis still matches a nanosecond one.
        _dates_json = params.get("dates")
        if _dates_json:
            def _iso_ns(v):
                try:
                    return str(np.datetime64(v, "ns"))
                except Exception:
                    return None

            _picker_ready = bool(result_date_w.options) and not result_date_w.disabled
            if not _picker_ready:
                warnings.append(
                    f"Date selection: {len(_dates_json)} date(s) are stored in "
                    "these settings. In order to select them exactly, build the "
                    "cube first, then paste the settings json AGAIN. You can "
                    "ignore this same status message after pasting for the "
                    "second time."
                )
            else:
                _wanted = {n for n in (_iso_ns(d) for d in _dates_json) if n}
                _sel = tuple(
                    v for v in _result_date_all_values() if _iso_ns(v) in _wanted
                )
                if not _sel:
                    warnings.append(
                        f"Date selection: none of the {len(_dates_json)} stored "
                        "date(s) is in the current cube - kept the current "
                        "selection."
                    )
                else:
                    # Guarded: one re-render below, after the selection and the
                    # scene filters agree.
                    _date_filter_guard["busy"] = True
                    try:
                        result_date_w.value = _sel
                    finally:
                        _date_filter_guard["busy"] = False
                    state["date_user_selection"] = set(_sel)
                    _sync_date_picker_to_cloud()
                    _on_result_date_change()
                    _missing = len(_wanted) - len(_sel)
                    if _missing:
                        warnings.append(
                            f"Date selection: {_missing} of {len(_wanted)} stored "
                            "date(s) are not in the current cube - ticked the "
                            f"other {len(_sel)}."
                        )

        return warnings

    def _paste_settings_clicked(_):
        """Show / hide the paste box. The kernel cannot read the browser
        clipboard, so the user pastes into a text area and the settings are
        applied from its content."""
        showing = paste_json_box.layout.display != "none"
        paste_json_box.layout.display = "none" if showing else ""
        if not showing:
            paste_json_area_w.value = ""
            _show_status("📋 Paste the copied settings into the box below.")

    def _on_paste_area_change(change):
        text = (change["new"] or "").strip()
        if not text:
            return
        try:
            payload = json.loads(text)
        except Exception:
            # Half a paste, or free text: stay quiet until it parses.
            return
        try:
            warnings = _apply_json_settings(payload)
        except Exception as e:
            _show_status(_friendly_error(e, "Applying the pasted settings"))
            return

        paste_json_area_w.value = ""
        paste_json_box.layout.display = "none"
        if warnings:
            _show_status(
                "✅ Settings applied, except:\n  - " + "\n  - ".join(warnings)
            )
        else:
            _show_status("✅ Settings applied.")

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

        elif mode == "zarr":
            suggestion = current or _auto_netcdf_suggestion_from_polygon()
            start_dir = _existing_dir_or_parent(suggestion)
            suggested_name = Path(suggestion).name or "test.zarr"
            if not suggested_name.lower().endswith(".zarr"):
                suggested_name = f"{Path(suggested_name).stem}.zarr"

            try:
                output_fc.reset(path=start_dir, filename=suggested_name)
            except Exception:
                try:
                    output_fc.default_path = start_dir
                    output_fc.default_filename = suggested_name
                except Exception:
                    pass

            output_fc.title = "Select Zarr output store (name ending in .zarr)"
            output_fc.show_only_dirs = False
            output_fc.filter_pattern = ["*.zarr", "*"]
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

        elif mode == "zarr":
            selected = getattr(chooser, "selected", None)
            if selected:
                s = str(selected)
                if s.lower().endswith(".nc"):
                    s = s[: s.rfind(".")]
                if not s.lower().endswith(".zarr"):
                    s += ".zarr"
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

        # The COG default is a FOLDER, so it is meaningless as a file/store path
        # and must be cleared when switching to netcdf or zarr. Listed once
        # here because all three branches need to recognise it - keeping the
        # list in only one of them is what left "./results/cogs" sitting in the
        # field after a cogs -> zarr switch.
        cog_defaults = ("./results/cogs", "results/cogs", r"results\cogs")
        if current in cog_defaults and mode in ("netcdf", "zarr"):
            current = ""
            export_target_w.value = ""

        # Re-gate the options that only some export modes can honour.
        _apply_compress_visibility()

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

            # (a leftover COG folder path was already cleared above)
            if current.lower().endswith(".zarr"):
                # Switching zarr -> netcdf: keep the name, swap the extension.
                export_target_w.value = f"{os.path.splitext(current)[0]}.nc"

            _update_netcdf_output_suggestion()
            _sync_output_filechooser_from_mode_and_text()

        elif mode == "zarr":
            export_target_w.disabled = False
            browse_output_btn.disabled = False
            export_target_w.description = "Export store:"
            export_target_w.placeholder = "./results/test.zarr"

            # Reuse the NetCDF path suggestion, swapping the extension to .zarr,
            # so switching netcdf <-> zarr keeps the same name with the right
            # extension. Only replace an empty field or a prior auto suggestion.
            base = _auto_netcdf_suggestion_from_polygon()
            zarr_suggestion = f"{os.path.splitext(base)[0]}.zarr"
            if current == "" or current == state.get("last_auto_netcdf_suggestion") \
                    or current.lower().endswith(".nc"):
                stem_current = os.path.splitext(current)[0] if current else ""
                export_target_w.value = (
                    f"{stem_current}.zarr" if stem_current else zarr_suggestion
                )
            _sync_output_filechooser_from_mode_and_text()

        elif mode == "cogs":
            export_target_w.disabled = False
            browse_output_btn.disabled = False
            export_target_w.description = "Export dir:"
            export_target_w.placeholder = "./results/cogs"
            # COGs write to a DIRECTORY, so any ".nc"/".zarr" path left over
            # from another mode is meaningless here - it would create a folder
            # literally named "*.nc". Drop it whether it was suggested or typed
            # (a file name is simply the wrong kind of thing) and offer the COGs
            # default. A folder path the user typed has no cube extension, so it
            # survives. Same rule as the Data Cube Editor.
            if current.lower().endswith((".nc", ".zarr")):
                export_target_w.value = ""

            if not export_target_w.value:
                export_target_w.value = "./results/cogs"

            _sync_output_filechooser_from_mode_and_text()

    def _apply_aggregator_stats_logic(*_):
        """Enable the Temporal Composites controls for missions that support
        them (i.e. that have a time axis to reduce).

        The old mutual-exclusion rule is gone: the mean/median Temporal
        Composite dropdown has been replaced by the two promoted checkboxes in
        this same section, so there is no second control to contradict.
        """
        meta = mission_meta[mission_dd.value]
        supported = len(_to_list_or_empty(meta.get("stats"))) > 0

        if not supported:
            if comp_mean_w.value:
                comp_mean_w.value = False
            if comp_median_w.value:
                comp_median_w.value = False
            if len(stats_w.value) > 0:
                stats_w.value = ()
            # A mission with no time axis has nothing to reduce over a period.
            _custom_clear_rows()

        comp_mean_w.disabled = not supported
        comp_median_w.disabled = not supported
        stats_w.disabled = not supported
        stats_all_btn.disabled = not supported
        stats_none_btn.disabled = not supported
        custom_add_btn.disabled = not supported
        for _row in _custom_rows:
            for _key in ("mode", "start", "end", "op", "name", "remove"):
                _row[_key].disabled = not supported
        _sync_keep_timeseries()

    def _sync_keep_timeseries(*_):
        """"Keep the full time series" only becomes a real choice once at least
        one composite is selected - otherwise unticking it would leave an empty
        cube, so it is force-ticked and greyed with the reason."""
        has_composite = bool(_selected_composites())
        if not has_composite and not keep_ts_w.value:
            keep_ts_w.value = True
        keep_ts_w.disabled = not has_composite
        if not has_composite:
            keep_ts_note.value = (
                "<div style='font-size:12px; color:#6b7280;'>"
                "Select a composite above to be able to export it without the "
                "time series.</div>"
            )
        elif not keep_ts_w.value:
            keep_ts_note.value = (
                "<div style='font-size:12px; color:#1e40af; background:#eff6ff; "
                "border:1px solid #bfdbfe; border-radius:6px; padding:6px 8px;'>"
                "The export will contain the selected composites only - no "
                "per-date time series. Such a cube <b>cannot</b> be updated, "
                "co-registered or cloud-masked afterwards, but <b>can</b> be super-resolved.</div>"
            )
        else:
            keep_ts_note.value = ""

    def _update_from_mission(*_):
        m_name = mission_dd.value
        meta = mission_meta[m_name]

        # Data Source (only applicable for Sentinel-2 L2A)
        if m_name == "sentinel_2_l2a":
            source_w.options = [
                ("Planetary Computer (Microsoft)", "planetary_computer"),
                ("Element84 (Earth Search)", "element84"),
                ("terrabyte (DLR)", "terrabyte"),
                ("Copernicus Data Space Ecosystem (Copernicus)", "cdse"),
            ]
            source_w.disabled = False
            # Keep whatever the user already picked if it is still offered;
            # otherwise fall back to the default catalogue (Planetary Computer).
            if source_w.value not in [v for _, v in source_w.options]:
                source_w.value = "planetary_computer"
        elif m_name == "sentinel_2_l1c":
            source_w.options = [
                ("Element84 (Earth Search)", "element84"),
                ("Copernicus Data Space Ecosystem (Copernicus)", "cdse"),
            ]
            source_w.value = "element84"
            source_w.disabled = False
        else:
            source_w.options = [("Not applicable", None)]
            source_w.value = None
            source_w.disabled = True

        # Refilter the Scene Specific Metadata list against the (possibly
        # unchanged) source value - the source_w observer alone misses mission
        # switches that keep the same source selected.
        _sync_scene_metadata_options()
        # Tile handling availability (Sentinel-2 L2A only).
        _sync_tile_handling()
        # Partial-scene (across-track) availability (optical missions).
        _sync_partial_scene()

        # The "Check data availability" button compares catalogues, which only
        # makes sense for the multi-catalogue Sentinel-2 missions.
        check_avail_btn.disabled = m_name not in (
            "sentinel_2_l2a", "sentinel_2_l1c"
        )

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

        # Indices (checkbox rows; availability follows the band selection)
        indices = _to_list_or_empty(meta.get("indices"))
        _set_index_options(m_name, indices)
        indices_all_btn.disabled = len(indices) == 0
        indices_none_btn.disabled = len(indices) == 0

        # Clip-to-polygon — availability depends on the mission. Keep the user's
        # current choice when the new mission supports clipping; force it OFF and
        # disable the box when it doesn't.
        clip_cfg = _bool_dropdown_from_metadata(meta.get("clip_raster"), default=False)
        _clip_supported = (not clip_cfg["disabled"]) and (
            True in [v for _, v in clip_cfg["options"]]
        )
        if _clip_supported:
            clip_raster_w.disabled = False
        else:
            clip_raster_w.value = False
            clip_raster_w.disabled = True

        # Cloud masking availability (mission capability). The disabled/greyed
        # state and values are governed by the active cloud preset, re-applied at
        # the end of this block - here we only set the options and the capability.
        cm_meta = meta.get("cloud_masking")
        if cm_meta is False:
            cloud_masking_w.options = [("Not available", None)]
            cloud_masking_w.value = None
            _cloud_caps["masking"] = False
        else:
            cm_cfg = _bool_dropdown_from_metadata(cm_meta, default=False)
            cloud_masking_w.options = cm_cfg["options"]
            cloud_masking_w.value = cm_cfg["value"]
            _cloud_caps["masking"] = True

        # Max CC availability
        max_cc_meta = meta.get("max_cc")
        if max_cc_meta is False:
            max_cc_w.value = 0
            _cloud_caps["max_cc"] = False
        else:
            try:
                max_cc_w.value = int(max_cc_meta)
            except Exception:
                max_cc_w.value = 100
            _cloud_caps["max_cc"] = True

        # Re-apply the active cloud preset so its values + greying survive the
        # mission switch, clamped to what this mission actually supports.
        _apply_cloud_preset(_cloud_preset_state["n"])

        # Stats
        stats_list = _to_list_or_empty(meta.get("stats"))

        # Hide *_all shortcuts in GUI (users can multi-select directly), and
        # the two promoted to their own checkboxes above the list, so no
        # composite can be selected from two places at once.
        stats_list = [
            s
            for s in stats_list
            if not (isinstance(s, str) and s.endswith("_all"))
            and str(s) not in _COMMON_COMPOSITES
        ]

        stats_w.options = stats_list

        stats_w.value = ()
        comp_mean_w.value = False
        comp_median_w.value = False
        stats_w.disabled = len(stats_list) == 0
        stats_all_btn.disabled = len(stats_list) == 0
        stats_none_btn.disabled = len(stats_list) == 0

        # Aggregator
        agg_list = _to_list_or_empty(meta.get("aggregator"))
        agg_options = [("None", None)] + [(str(x), x) for x in agg_list]
        aggregator_w.options = agg_options
        aggregator_w.value = None
        aggregator_w.label = "None"  # show "None", not a blank label (see widget def)
        aggregator_w.disabled = len(agg_list) == 0

        _apply_export_mode_defaults()
        _apply_aggregator_stats_logic()
        _update_daterange_placeholder()

    # -------------------------------------------------------------------------
    # Visualization callbacks
    # -------------------------------------------------------------------------
    def _on_viz_dropdown_clicked(_):
        try:
            cube = _active_result_cube()
            if cube is None:
                with viz_out:
                    clear_output()
                    print("ℹ️ Build a data cube first.")
                return

            # Honour the viewing-resolution pick. A coarse preview is a fresh
            # (smaller) read from the archive, so it can fail where the cached
            # cube would not - fall back to full detail and say so rather than
            # leaving the user with no view at all.
            try:
                da = _viz_cube_for_display()
            except Exception as _e:
                da = _pick_dataarray_for_visualization(cube)
                with viz_out:
                    clear_output()
                    print(
                        "Could not build the coarse preview "
                        f"({type(_e).__name__}) - showing full detail instead."
                    )

            with viz_out:
                clear_output()
                display(
                    widgets.HTML(
                        f"{_INFO_BOX}ℹ️ <b>PLEASE BE PATIENT</b> while the "
                        "scenes are being loaded! This data cube is not "
                        "computed and the loading speed depends on your local "
                        "machine.<br>For exported cubes, use Data Cube Editor "
                        "Visualization.</div>"
                    )
                )
                out = interactive_time_view(
                    stac=da,
                    widget_type="dropdown",
                    renderer=str(viz_renderer_w.value),
                )
                if out is not None:
                    display(out)

        except Exception as e:
            with viz_out:
                clear_output()
                print(_friendly_error(e, "Visualization"))

    def _current_gif_render_kwargs():
        """save_timeseries_gif kwargs for the active animation section."""
        sec = gif_section_w.value
        if sec == "band":
            if not gif_band_dd.value:
                raise ValueError("Select a band for the single-band animation.")
            kwargs = {"display_mode": "band", "band": str(gif_band_dd.value)}
        elif sec == "custom":
            if not (gif_r_dd.value and gif_g_dd.value and gif_b_dd.value):
                raise ValueError("Select R, G and B bands for the custom animation.")
            kwargs = {
                "display_mode": "custom",
                "rgb_bands": (
                    str(gif_r_dd.value),
                    str(gif_g_dd.value),
                    str(gif_b_dd.value),
                ),
            }
        else:
            return {"display_mode": gif_display_mode_w.value}

        p_lo, p_hi = (float(v) for v in gif_stretch_w.value)
        if p_hi <= p_lo:
            p_lo, p_hi = 2.0, 98.0
        kwargs.update(p_low=p_lo, p_high=p_hi)
        return kwargs

    def _on_viz_make_gif_clicked(_):
        try:
            # composite=False: the animation needs the per-date frames, which
            # only the time series has (the button is greyed for a composite
            # layer anyway - see _refresh_viz_layers).
            cube = _active_result_cube(composite=False)
            if cube is None:
                with anim_out:
                    clear_output()
                    print("ℹ️ Build a data cube first.")
                return

            da = _pick_dataarray_for_visualization(cube)

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

            gif_kwargs = _current_gif_render_kwargs()

            with anim_out:
                clear_output()
                print("Generating animation GIF...")
                # Animation is generated only (no preview inside GUI)
                save_timeseries_gif(
                    da=da,
                    out_path=out_path,
                    fps=fps_val,
                    label=gif_label_w.value,
                    **gif_kwargs,
                )
                print(f"✅ Animation saved: {out_path}")

        except Exception as e:
            with anim_out:
                clear_output()
                print(_friendly_error(e, "Animation"))

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
        # Only the indices whose required bands are selected can be ticked.
        for cb in _index_rows.values():
            if not cb.disabled:
                cb.value = True

    def _clear_indices(_):
        for cb in _index_rows.values():
            cb.value = False

    def _select_all_stats(_):
        if not stats_w.disabled:
            stats_w.value = tuple(stats_w.options)

    def _clear_stats(_):
        stats_w.value = ()

    def _on_generate_clicked(_):
        with result_out:
            clear_output()
        # The notices describe the cube that was just wiped, so they must go with
        # it: left up during a rebuild they read as if they applied to the run in
        # progress, which is misleading exactly when the user has changed the
        # settings that produced them. _show_result_summary puts back whatever the
        # NEW cube deserves.
        _set_coreg_warning("")
        _set_result_notes([])
        _set_result_viz_note(False)

        try:
            params, export_mode, export_target = _prepare_get_stac_layers_params()
            state["last_call_params"] = params

            # Co-registration size warning is no longer shown in Status (that is
            # now the "preview ready" message). It is stored and rendered in the
            # Result panel, below the visualize/export note (see _show_result_summary).
            state["coreg_size_hint"] = _polygon_coreg_size_hint(
                params.get("polygon"), params.get("resolution")
            )

            with status_out:
                clear_output()
                # Browser-side CSS animation so the dots keep moving even while the
                # kernel is blocked on a slow STAC build - shows the user it isn't
                # stuck. It's removed by clear_output(wait=True) once the build
                # returns, just before the final status message is printed.
                display(HTML(_busy_bear_html("Generating data cube")))

                # Ensure parent directory exists for direct NetCDF export
                if params["output"] is not None:
                    Path(params["output"]).parent.mkdir(parents=True, exist_ok=True)

                # If get_stac_layers(output=...) internally calls export_stac(),
                # Dask ProgressBar output will print inside this status box.
                result = get_stac_layers(**params)

                # When the binary mask was requested, get_stac_layers returns
                # (cube, mask); hold the mask in memory so we can write it at
                # export time (never during a lazy preview).
                if params.get("return_cloud_mask"):
                    result, _mask_result = result
                    state["cloud_mask_result"] = _mask_result
                else:
                    state["cloud_mask_result"] = None

                # Build done — clear the animated indicator (and any transient
                # progress) right before the final status is printed. wait=True
                # swaps the content in without a flash of empty space.
                clear_output(wait=True)

                # A build that returns but holds no data (e.g. an empty cube) must
                # be reported as a failure, not shown as a 'ready' cube. Raising
                # here routes it through the same reset+error handling below.
                if _result_is_empty(result):
                    raise ValueError(
                        "The build produced no data (no scenes or pixels for the "
                        "given parameters). Try a wider date range, a higher max "
                        "cloud coverage, or verify the polygon / AOI."
                    )

                state["result"] = result
                # Multi-swath hint (GUI-only): flag AOIs that sit across swaths
                # so some scenes are partial. Single cubes only; never fatal.
                state["multiswath_hint"] = (
                    None if isinstance(result, list)
                    else _compute_multiswath_hint(result)
                )
                # Projection note: only when scenes were actually re-drawn into
                # the cube's CRS (the attrs are absent otherwise).
                state["projection_hint"] = (
                    None if isinstance(result, list)
                    else _compute_projection_hint(result)
                )
                # The built cube already knows its projections, so fill the
                # Advanced dropdown from it instead of making the user press
                # Detect for information we now hold.
                _populate_detected_crs_from_cube(result)
                export_result_btn.disabled = False
                _set_visualization_enabled(True)
                _refresh_viz_feature_options()
                # A preview belongs to the cube it was read for; a fresh build
                # invalidates it (different dates/AOI/bands entirely).
                state["viz_preview"] = None
                _refresh_viz_resolution()
                _refresh_gif_band_options()
                _update_gif_output_suggestion()

                # Fresh build shows everything; enable the Max cloud % filter only
                # when this build carries cloud_percentage (cloud masking was on).
                # Guard the reset so it doesn't re-trigger the change handler here.
                _cloud_filter_guard["busy"] = True
                try:
                    result_cloud_max_w.value = 100
                finally:
                    _cloud_filter_guard["busy"] = False
                _sync_cloud_filter_enabled()

                # Same for the Min coverage % filter: back to 0 (keep every
                # scene), enabled only when this build carries ready per-scene
                # coverage numbers.
                _cov_filter_guard["busy"] = True
                try:
                    result_coverage_min_w.value = 0
                finally:
                    _cov_filter_guard["busy"] = False
                _sync_coverage_filter_enabled()

                # Fresh build: repopulate the date picker (all dates ticked) for a
                # single-cube result; hidden/disabled for lists or time-less cubes.
                _populate_result_dates(state["result"])

                # For a multi-feature batch, some features may have failed (kept in
                # the list as markers). Count real cubes so messages don't claim a
                # failed feature was produced/exported.
                if isinstance(result, list):
                    n_ok = sum(
                        1 for c in result
                        if isinstance(c, (xr.DataArray, xr.Dataset))
                    )
                    n_fail = len(result) - n_ok
                else:
                    n_ok, n_fail = 1, 0

                # Build always yields a lazy in-memory preview; nothing is written
                # here. The user inspects the result, then clicks
                # "Export Current Result" to write the NetCDF / COGs.
                print("✅ Data cube preview ready. Result stored in memory.")
                print("ℹ️ Inspect it, then click 'Export Current Result' to save it.")

                if n_fail:
                    print(
                        f"⚠️ {n_fail} feature(s) could not be generated and were "
                        "skipped - see the Result panel for which ones and why."
                    )

            # Show preview in Result panel (not in Status)
            _show_result_summary(_effective_result(state["result"]))

            # Auto-open Result accordion after generation
            try:
                result_acc.selected_index = 0
            except Exception:
                pass

        except Exception as e:
            # A failed build must not leave a stale 'ready' cube in the Result
            # panel. Reset the result/state and overwrite the panel with a clear
            # 'no data' message, then show the error in Status.
            state["result"] = None
            state["cloud_mask_result"] = None
            try:
                export_result_btn.disabled = True
                _set_visualization_enabled(False)
                result_cloud_max_w.disabled = True
                result_coverage_min_w.disabled = True
                _populate_result_dates(None)
            except Exception:
                pass
            _show_result_summary(None)
            _show_status(_friendly_error(e, "Building"))

    def _on_coreg_resize_clicked(_):
        """'Resize and Re-build Data Cube' (shown next to the co-registration
        size warning): enlarge the AOI to the suggested minimum edge - for a
        polygon file a new '<stem>_enlarged' file is written next to the
        original, for a bbox the values are widened in place - then re-run
        the normal build. Only the polygon changes; every other parameter is
        re-read from the widgets exactly like a manual build."""
        try:
            polygon = _parse_polygon_input(polygon_w.value)
            if polygon is None:
                raise ValueError(
                    "No polygon/bbox is set in the builder - nothing to resize."
                )
            resolution = None if resolution_w.disabled else int(resolution_w.value)
            enlarged = _enlarge_polygon_for_coreg(polygon, resolution)

            if isinstance(enlarged, list):
                polygon_w.value = (
                    "[" + ", ".join(f"{v:.6f}" for v in enlarged) + "]"
                )
            else:
                polygon_w.value = enlarged

            _on_generate_clicked(None)

            # _on_generate_clicked owns (and clears) the Status panel, so the
            # pointer to the new polygon is appended after it finishes - it is
            # true whether the rebuild succeeded or failed.
            with status_out:
                if isinstance(enlarged, list):
                    print(f"ℹ️ Enlarged bbox now in use: {polygon_w.value}")
                else:
                    print(f"ℹ️ Enlarged polygon saved and now in use: {enlarged}")
        except Exception as e:
            _show_status(_friendly_error(e, "Resize and re-build"))

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

                # export_stac() already prints "Export is done: ..." for file
                # exports (netcdf/zarr); only add a line for the COG folder case.
                if info.get("mode") not in ("netcdf", "zarr"):
                    print(f"✅ Export finished: {info['target']}")

                # Write the held binary mask alongside the cube (skipped for COG).
                _mask_written = _write_held_cloud_mask()
                if _mask_written:
                    print(f"✅ Binary cloud mask exported: {_mask_written}")

                # Granule metadata XMLs (own try: the cube export above already
                # succeeded, so an XML download problem must not read as a
                # failed export - report it as its own warning instead).
                try:
                    _meta_written = _write_granule_metadata_xmls()
                    if _meta_written:
                        print(f"✅ Granule metadata exported: {_meta_written}")
                except Exception as _me:
                    print(f"⚠️ Granule metadata export failed: {_me}")

                # Settings JSON beside the cube (own try for the same reason:
                # the cube is already written, a sidecar problem is a warning).
                try:
                    _settings_written = _write_settings_sidecar()
                    if _settings_written:
                        print(f"✅ Settings exported: {_settings_written}")
                except Exception as _se:
                    print(f"⚠️ Settings export failed: {_se}")

                # Statistics CSV, read back from the cube(s) just written (own
                # try for the same reason: the cube is already on disk).
                try:
                    _csv_written = _write_statistics_csv(info)
                    if _csv_written and len(_csv_written) == 1:
                        print(f"✅ Statistics report exported: {_csv_written[0]}")
                    elif _csv_written:
                        print(
                            f"✅ Statistics reports exported: {len(_csv_written)} "
                            f"files, one per area ({_csv_written[0]}, ...)"
                        )
                except Exception as _ce:
                    print(f"⚠️ Statistics report failed: {_ce}")

        except Exception as e:
            _show_status(_friendly_error(e, "Export"))

    # -------------------------------------------------------------------------
    # Wire callbacks
    # -------------------------------------------------------------------------
    bands_all_btn.on_click(_select_all_bands)
    bands_none_btn.on_click(_clear_bands)
    indices_all_btn.on_click(_select_all_indices)
    indices_none_btn.on_click(_clear_indices)
    # Re-grey the index rows whenever the band selection changes.
    bands_w.observe(_refresh_index_availability, names="value")
    stats_all_btn.on_click(_select_all_stats)
    stats_none_btn.on_click(_clear_stats)

    browse_polygon_btn.on_click(_on_browse_polygon_clicked)
    browse_output_btn.on_click(_on_browse_output_clicked)
    browse_gif_out_btn.on_click(_on_browse_gif_out_clicked)

    generate_btn.on_click(_on_generate_clicked)
    coreg_resize_btn.on_click(_on_coreg_resize_clicked)
    export_result_btn.on_click(_on_export_result_clicked)
    copy_json_btn.on_click(_copy_json_to_clipboard)
    paste_json_btn.on_click(_paste_settings_clicked)
    paste_json_area_w.observe(_on_paste_area_change, names="value")

    viz_dropdown_btn.on_click(_on_viz_dropdown_clicked)
    viz_make_gif_btn.on_click(_on_viz_make_gif_clicked)

    mission_dd.observe(_update_from_mission, names="value")

    def _on_composites_change(*_):
        """A composite selection changed: re-check the custom rows, re-sync the
        keep-time-series choice and re-render the Result panel so the layer list
        updates immediately. No rebuild - composites are derived from the built
        time series. The re-render no-ops when no cube has been built yet."""
        _custom_validate()
        _sync_keep_timeseries()
        _on_result_cloud_max_change()

    comp_mean_w.observe(_on_composites_change, names="value")
    comp_median_w.observe(_on_composites_change, names="value")
    stats_w.observe(_on_composites_change, names="value")
    keep_ts_w.observe(_on_composites_change, names="value")
    export_mode_w.observe(lambda change: _apply_export_mode_defaults(), names="value")
    daterange_mode_w.observe(
        lambda change: _update_daterange_placeholder(), names="value"
    )
    polygon_w.observe(
        lambda change: (
            _update_netcdf_output_suggestion(),
            _update_mask_binary_suggestion(),
            _update_gif_output_suggestion(),
        ),
        names="value",
    )
    gif_display_mode_w.observe(
        lambda change: _update_gif_output_suggestion(), names="value"
    )
    gif_section_w.observe(
        lambda change: (
            _sync_gif_section_visibility(),
            _update_gif_output_suggestion(),
        ),
        names="value",
    )
    gif_band_dd.observe(lambda change: _update_gif_output_suggestion(), names="value")
    # Different features of a multi-feature build may carry different bands.
    viz_feature_w.observe(lambda change: _refresh_gif_band_options(), names="value")

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------
    FORM_WIDTH = "96%"
    FORM_MAX_WIDTH = "950px"

    css_patch = _gui_css_widget()

    header = widgets.HTML(
        "<div style='margin:0 0 4px 0; font-size:28px; font-weight:700;'>Data Cube Builder</div>"
    )

    subtitle = widgets.HTML(
        "<div style='display:flex; flex-wrap:wrap; gap:10px; margin:14px 0 8px 0;'>"
        # Step 1 - blue "set up"
        "<div style='flex:1 1 200px; background:#f8fafc; border:1px solid #e5e7eb; "
        "border-left:4px solid #3b82f6; border-radius:8px; padding:8px 10px;'>"
        "<div style='font-weight:700; color:#1e3a8a; font-size:13px;'>1 &nbsp; Set up</div>"
        "<div style='font-size:12px; color:#475569; margin-top:2px;'>Fill <b>Basic</b> + "
        "<b>Advanced Parameters</b> and pick a <b>Data Source</b>.</div></div>"
        # Step 2 - green "build", matches the Build button
        "<div style='flex:1 1 200px; background:#f0fdf4; border:1px solid #dcfce7; "
        "border-left:4px solid #16a34a; border-radius:8px; padding:8px 10px;'>"
        "<div style='font-weight:700; color:#166534; font-size:13px;'>2 &nbsp; Build &amp; inspect</div>"
        "<div style='font-size:12px; color:#475569; margin-top:2px;'>Click "
        "<b>Build Data Cube Preview</b>, then check the <b>Result</b>. Filter by "
        "cloud percentage or date if necessary.</div></div>"
        # Step 3 - orange "export", matches the Export button
        "<div style='flex:1 1 200px; background:#fff7ed; border:1px solid #fed7aa; "
        "border-left:4px solid #f97316; border-radius:8px; padding:8px 10px;'>"
        "<div style='font-weight:700; color:#9a3412; font-size:13px;'>3 &nbsp; Export</div>"
        "<div style='font-size:12px; color:#475569; margin-top:2px;'>Choose a format in "
        "<b>Export Options</b>, then click <b>Export Current Result</b>.</div></div>"
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

    bands_box = _field_group(
        "Bands",
        [
            _boxed(bands_w),
            widgets.HBox(
                [bands_all_btn, bands_none_btn], layout=widgets.Layout(gap="6px")
            ),
        ],
        subtitle="Which spectral bands to include.",
        collapsible=True,
        open=False,
    )

    indices_box = _field_group(
        "Indices",
        [
            _boxed(indices_w),
            widgets.HBox(
                [indices_all_btn, indices_none_btn], layout=widgets.Layout(gap="6px")
            ),
        ],
        subtitle=(
            "Spectral indices to compute. Each index lists the bands it needs; "
            "an index is greyed out until those bands are selected above."
        ),
        collapsible=True,
        open=False,
    )

    # Stats help toggle built manually (not via field_group's help_html) so the
    # "Stats Explanation ?" row sits AFTER the blue info box, not above it.
    _stats_help_btn = _help_button()
    _stats_help_box = widgets.HTML(
        value=PARAM_HELP_HTML.get("stats", ""),
        layout=widgets.Layout(
            display="none",
            border="1px solid #dbeafe",
            padding="8px",
            border_radius="8px",
            margin="2px 0 2px 0",
            width="100%",
        ),
    )

    def _toggle_stats_help(_):
        _stats_help_box.layout.display = (
            "" if _stats_help_box.layout.display == "none" else "none"
        )

    _stats_help_btn.on_click(_toggle_stats_help)

    _stats_explain_row = widgets.HBox(
        [
            widgets.HTML(
                "<div style='font-weight:500; font-size:12px; color:#374151;'>"
                "Composite Explanation</div>"
            ),
            _stats_help_btn,
        ],
        layout=widgets.Layout(align_items="center", gap="6px"),
    )

    # Wrap a set of widgets in a lighter, boxed sub-panel (see the
    # .stac2cube-subpanel CSS). ``accent`` picks the coloured left bar:
    # "blue", "green" or "amber".
    def _subpanel(children, accent=None):
        box = widgets.VBox(
            list(children),
            layout=widgets.Layout(width="100%", gap="6px"),
        )
        box.add_class("stac2cube-subpanel")
        if accent:
            box.add_class(f"stac2cube-subpanel-{accent}")
        return box

    # A single titled parameter rendered as a sub-panel: a bold title, an
    # optional "?" help toggle, and the control below. Same look as the Polygon
    # option panels; used for the four raw cloud parameters.
    def _param_panel(title, control, accent=None, help_html=None):
        title_html = widgets.HTML(
            f"<div style='font-weight:600; font-size:13px; color:#374151; "
            f"margin:0 0 2px 0;'>{title}</div>"
        )
        if help_html:
            btn = _help_button()
            help_box = widgets.HTML(
                value=help_html,
                layout=widgets.Layout(
                    display="none", border="1px solid #dbeafe", padding="8px",
                    border_radius="8px", margin="2px 0 2px 0", width="100%",
                ),
            )

            def _toggle(_):
                help_box.layout.display = (
                    "" if help_box.layout.display == "none" else "none"
                )

            btn.on_click(_toggle)
            header = widgets.HBox(
                [title_html, btn],
                layout=widgets.Layout(align_items="center", gap="6px"),
            )
            kids = [header, help_box, control]
        else:
            kids = [title_html, control]
        return _subpanel(kids, accent=accent)

    # A prominent "OR" separator, reused in the Polygon group and here.
    def _or_divider():
        return widgets.HTML(
            "<div style='display:flex; align-items:center; gap:12px; margin:16px 0 12px 0;'>"
            "<span style='flex:1; height:2px; background:#cbd5e1;'></span>"
            "<span style='font-size:14px; font-weight:700; color:#6b7280; "
            "letter-spacing:2px;'>OR</span>"
            "<span style='flex:1; height:2px; background:#cbd5e1;'></span>"
            "</div>"
        )

    # A plain horizontal separator (no "OR"), with generous vertical spacing so the
    # panels above and below don't touch.
    def _line_divider():
        return widgets.HTML(
            "<div style='height:2px; background:#cbd5e1; border-radius:1px; "
            "margin:18px 0 16px 0;'></div>"
        )

    # Column captions for the Custom Composites rows. The widths mirror the row
    # widgets in _custom_row_widgets, so the captions sit above their fields.
    def _custom_caption(text, width):
        return widgets.HTML(
            f"<div style='font-size:11px; font-weight:600; color:#6b7280;'>{text}</div>",
            layout=widgets.Layout(width=width),
        )

    _custom_header_row = widgets.HBox(
        [
            _custom_caption("Period", "130px"),
            _custom_caption("From", "105px"),
            _custom_caption("To", "105px"),
            _custom_caption("Statistic", "105px"),
            _custom_caption("Name", "150px"),
            _custom_caption("", "38px"),
        ],
        layout=widgets.Layout(width="100%", gap="4px", flex_flow="row wrap"),
    )

    # --- Temporal Composites section -----------------------------------------
    # Two promoted checkboxes (what most people want) in an accented sub-panel,
    # then the long list of the remaining ops and the monthly / annual variants
    # in a collapsed group, the user's own periods, then the keep-or-drop choice.
    composites_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#475569; margin:0 0 2px 0;'>"
                "Statistics calculated over the dates kept in the Result "
                "section above."
                "</div>"
            ),
            _subpanel([comp_mean_w, comp_median_w], accent="blue"),
            widgets.HTML("<div style='height:10px;'></div>"),
            _field_group(
                "More Composites",
                [
                    _stats_explain_row,
                    _stats_help_box,
                    _boxed(stats_w),
                    widgets.HBox(
                        [stats_all_btn, stats_none_btn],
                        layout=widgets.Layout(gap="6px"),
                    ),
                ],
                subtitle="Minimum, maximum and standard deviation, plus "
                "monthly and annual composites.",
                collapsible=True,
                open=False,
            ),
            widgets.HTML("<div style='height:6px;'></div>"),
            _field_group(
                "Custom Composites",
                [
                    _custom_header_row,
                    custom_rows_box,
                    custom_add_btn,
                    custom_error_note,
                    widgets.HTML(
                        "<div style='font-size:12px; color:#6b7280;'>"
                        "<b>Every year</b> repeats the period in every year of "
                        "the cube and saves one image per year, named "
                        "<code>name_2024</code>, <code>name_2025</code>, ... "
                        "<b>Single window</b> saves one image."
                        "</div>"
                    ),
                ],
                subtitle="Your own period, for example a growing season.",
                collapsible=True,
                open=False,
                help_html=PARAM_HELP_HTML.get("custom_composites", ""),
            ),
            _line_divider(),
            keep_ts_w,
            keep_ts_note,
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    # --- Time period: simple From/To pickers, with advanced modes tucked away ---
    # The OR divider lives inside date_simple_box so it hides together with the
    # From/To pickers when "Use a seasonal date range" is ticked.
    date_simple_box = widgets.VBox(
        [
            _stacked_field(date_from_w, "From"),
            _stacked_field(date_to_w, "To"),
            _or_divider(),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    # Daterange field with the red MM-DD hint between its label and the input box.
    _dr_field = _stacked_field(daterange_w, "Daterange")
    _dr_field.children = [
        _dr_field.children[0],
        widgets.HTML(
            "<div style='font-size:12px; color:#b91c1c; margin:0;'>"
            "season &rarr; <code>\"MM-DD\" - \"MM-DD\"</code></div>"
        ),
        _dr_field.children[1],
    ]
    date_advanced_box = widgets.VBox(
        [
            _with_help_left(daterange_mode_w, "daterange_mode", label_text="Season mode"),
            _dr_field,
        ],
        layout=widgets.Layout(width="100%", gap="6px", display="none"),
    )

    def _update_date_inputs_visibility(*_):
        advanced = advanced_dates_w.value
        date_simple_box.layout.display = "none" if advanced else ""
        date_advanced_box.layout.display = "" if advanced else "none"

    advanced_dates_w.observe(lambda change: _update_date_inputs_visibility(), names="value")
    _update_date_inputs_visibility()

    date_section = _field_group(
        "Time Period",
        [date_simple_box, advanced_dates_w, date_advanced_box],
        subtitle="Which dates should the data cube cover?",
        collapsible=True,
        open=False,
    )

    # --- Optional: draw the area of interest on an interactive map ---
    draw_polygon_w = widgets.Checkbox(
        value=False,
        description="Draw the area on a map",
        indent=False,
    )
    use_drawn_btn = widgets.Button(
        description="Use drawn area",
        button_style="success",
        icon="check",
        layout=widgets.Layout(width="auto", min_width="220px", height="44px"),
        style=widgets.ButtonStyle(font_weight="bold", font_size="15px"),
    )
    draw_status = widgets.Output()
    draw_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%", gap="6px"))

    def _ensure_draw_map():
        """Create the leafmap map on first use. Lazy so the GUI loads fast and
        still works if leafmap is not installed."""
        if state.get("draw_map") is not None:
            return state["draw_map"]
        try:
            import leafmap
        except Exception:
            return None
        # Default view: the whole world, centred slightly north of the equator
        # so most of the land sits in frame ([lat, lon]).
        m = leafmap.Map(center=[20.0, 0.0], zoom=2)
        m.layout.height = "60vh"
        m.layout.min_height = "320px"
        m.layout.max_height = "640px"
        m.layout.width = "100%"
        # Hybrid-style background, stacked by hand: satellite imagery at the
        # bottom, place names on top. The label tile is ~97% transparent, so it
        # adds names without hiding any imagery - an OpenStreetMap tile is fully
        # painted and would wash the imagery out even at reduced opacity.
        # leafmap's own "HYBRID" basemap needs a GOOGLE_MAPS_API_KEY and
        # silently falls back to label-free Esri imagery without one.
        try:
            # Drop leafmap's default OSM base layer; the imagery replaces it
            # (layers render in the order they are added).
            for layer in list(m.layers):
                if getattr(layer, "name", "") == "OpenStreetMap":
                    m.remove(layer)
            m.add_basemap("Esri.WorldImagery")             # background
            m.add_basemap("CartoDB.DarkMatterOnlyLabels")  # labels only
        except Exception:
            pass
        state["draw_map"] = m
        return m

    def _update_draw_visibility(*_):
        if not draw_polygon_w.value:
            draw_box.layout.display = "none"
            return
        if not state.get("draw_box_built"):
            m = _ensure_draw_map()
            if m is None:
                draw_box.children = [
                    widgets.HTML(
                        "<div style='color:#b91c1c; font-size:12px;'>Drawing needs the "
                        "<code>leafmap</code> package, which isn't available here. You can "
                        "still type a path or bbox above.</div>"
                    )
                ]
            else:
                draw_box.children = [
                    widgets.HTML(
                        "<div style='font-size:12px; color:#6b7280; margin:0 0 4px 0;'>"
                        "• Use the tools at the <b>top-left</b> of the map to draw your area.<br>"
                        "&nbsp;&nbsp;Tip: choose <b>“Draw a rectangle”</b> to maximize "
                        "co-registration and super-resolution efficiency.<br>"
                        "• Default background is <b>Satellite (Esri)</b> with <b>place "
                        "names</b> on top. To change it, open the toolbar at the "
                        "<b>top-right</b> of the map and click <b>Layers</b> (next to "
                        "Toolbar button).</div>"
                    ),
                    m,
                    widgets.HBox([use_drawn_btn], layout=widgets.Layout(gap="6px")),
                    draw_status,
                ]
            state["draw_box_built"] = True
        draw_box.layout.display = ""

    def _on_use_drawn_clicked(_):
        with draw_status:
            clear_output()
            m = state.get("draw_map")
            if m is None:
                print("❌ The map isn't ready yet.")
                return
            roi = getattr(m, "user_roi", None)
            if not roi:
                print(
                    "✋ Draw a rectangle or polygon on the map first (tools at the "
                    "top-left of the map), then click “Use drawn area”."
                )
                return
            try:
                geom = roi.get("geometry", roi) if isinstance(roi, dict) else roi
                gtype = (geom or {}).get("type")
                coords = (geom or {}).get("coordinates")
                if gtype == "Polygon":
                    ring = coords[0]
                elif gtype == "MultiPolygon":
                    ring = coords[0][0]
                else:
                    print(
                        "✋ Please draw an area — a rectangle or polygon — not a point "
                        f"or line (this looks like a {gtype})."
                    )
                    return
                xs = [pt[0] for pt in ring]
                ys = [pt[1] for pt in ring]
                bbox = [min(xs), min(ys), max(xs), max(ys)]  # [xmin, ymin, xmax, ymax] WGS84
            except Exception as e:
                print(f"❌ Couldn't read the drawn shape: {e}")
                return

            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                print(
                    "✋ The drawn area has no width or height (it came out as a line or "
                    f"point):\n{bbox}\n"
                    "Please draw a box/polygon that covers an actual area — make sure to "
                    "drag both sideways and up/down — then click “Use drawn area” again."
                )
                return

            # Save the EXACT drawn outline as a WGS84 GeoJSON file and point Polygon
            # at it, so ticking "Clip data cube to polygon boundaries" clips to the
            # true shape (not just the bbox). get_stac still derives the bbox from it
            # for the search.
            try:
                feature = (
                    roi
                    if (isinstance(roi, dict) and roi.get("type") == "Feature")
                    else {"type": "Feature", "properties": {}, "geometry": geom}
                )
                fc = {"type": "FeatureCollection", "features": [feature]}
                out_dir = Path("polygons")
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    out_dir = Path(tempfile.gettempdir())
                out_path = out_dir / f"drawn_{_datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson"
                out_path = out_path.resolve()  # store an absolute path, not relative
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(fc, f)
            except Exception as e:
                print(f"❌ Couldn't save the drawn shape: {e}")
                return

            polygon_w.value = out_path.as_posix()  # overwrites any path/bbox set above
            # Clipping is left to the user (the "Output shape" section): a drawn
            # area defaults to its bounding box like any other input, and the user
            # can tick "Clip to exact polygon outline" if they want the exact shape.
            print(
                f'✅ Area from your drawing is saved to "{out_path.as_posix()}" and will be '
                "used as the new polygon of the data cube (◡ ‿ ◡ ✿)"
            )

    draw_polygon_w.observe(lambda c: _update_draw_visibility(), names="value")
    use_drawn_btn.on_click(_on_use_drawn_clicked)

    # Two boxes beside the opt-in clip control, following the file's blue/amber
    # split (see _INFO_BOX): blue states the default and always shows, amber
    # appears only once the user has actually ticked the box - the cost of
    # clipping is only worth spelling out to someone who chose it.
    clip_info_html = widgets.HTML(
        f"{_INFO_BOX}"
        "ℹ️ By default the cube is returned as the polygon's <b>bounding box</b> for "
        "more efficient <b>co-registration</b> and <b>super-resolution</b> results. "
        "Raster processing of a clipped area is <b>not</b> recommended."
        "</div>"
    )

    clip_warning_html = widgets.HTML(
        "<div style='font-size:12px; color:#92400e; background:#fffbeb; "
        "border:1px solid #fde68a; border-radius:6px; padding:6px 8px;'>"
        "⚠️ <b>Warning:</b> Yes, pixels outside the shape become NaN, but this does not "
        "reduce the data size significantly and it even increases the overall processing "
        "time. Selecting compression when exporting NetCDF, or exporting as Zarr, gives "
        "some size reduction.<br>"
        "Therefore, clipping is recommended in 'Data Cube Editor' as the last step. For "
        "wide elongated polygons, split your AOI and activate batch processing so later "
        "you can mosaic them in 'Data Cube Editor'."
        "</div>",
        layout=widgets.Layout(display="none"),
    )

    def _update_clip_warning(*_):
        clip_warning_html.layout.display = "" if clip_raster_w.value else "none"

    clip_raster_w.observe(lambda change: _update_clip_warning(), names="value")
    _update_clip_warning()

    # -------------------------------------------------------------------------
    # "Check area coverage": the last block of the Polygon group. Reads the STAC
    # catalogue (metadata only, no pixels) over the chosen area and date range and
    # shows which orbits see the area, how much of it each one covers per date, and
    # the footprints on a map. Purely informational - it changes no parameter.
    # -------------------------------------------------------------------------
    coverage_btn = widgets.Button(
        description="Check area coverage",
        button_style="info",
        icon="map",
        layout=widgets.Layout(width="220px"),
    )
    coverage_out = widgets.Output()
    # Hidden until the first run, then revealed and opened by the handler, so the
    # tall map never stretches the Polygon group before it is asked for.
    coverage_result_acc = widgets.Accordion(
        children=[coverage_out], selected_index=None
    )
    coverage_result_acc.set_title(0, "Area Coverage Result")
    coverage_result_acc.layout = widgets.Layout(width="99%", display="none")

    def _on_check_coverage(_btn):
        coverage_result_acc.layout.display = ""
        coverage_result_acc.selected_index = 0
        with coverage_out:
            clear_output()
            try:
                polygon = _parse_polygon_input(polygon_w.value)
                daterange = _resolve_daterange()
            except Exception as e:
                print("⚠️ " + _friendly_error(e, "Reading the parameters"))
                return
            if polygon is None:
                print("✋ Please set a polygon file or a bounding box first.")
                return

            # Measure against whatever the cube will actually be: the bounding box
            # by default, the exact outline when the user ticked clipping.
            clipped = bool(clip_raster_w.value)
            display(HTML(_busy_bear_html(
                "Reading scene footprints for your area",
                "(the whole date range is read, so a long range takes longer)",
            )))
            try:
                m, df, info = preview_scene_footprints(
                    mission_dd.value,
                    polygon,
                    source=source_w.value,
                    daterange=daterange,
                    coverage_geometry="polygon" if clipped else "bbox",
                    q=True,
                )
            except Exception as e:
                clear_output()
                print("⚠️ " + _friendly_error(e, "Checking area coverage"))
                return

            clear_output()
            if df.empty:
                print(
                    "No scenes found for this area and date range. Try a wider "
                    "date range, or check the area."
                )
                return

            span = info.get("window")
            area_word = "exact outline" if clipped else "bounding box"
            # The source is named because each catalogue publishes its OWN footprint
            # outlines for the same acquisition: measured on a 6 km area sitting on a
            # swath edge, Element84 read 4-8 points lower than Planetary Computer and
            # CDSE (which agreed exactly). So these numbers describe the catalogue the
            # cube will actually be built from, and only that one.
            source_label = dict(
                (v, k) for k, v in source_w.options
            ).get(source_w.value, source_w.value)
            display(HTML(
                "<div style='font-size:12px; color:#374151; margin:0 0 6px 0;'>"
                f"Dates read: <b>{span[0]}</b> to <b>{span[1]}</b> "
                f"({info['n_dates']} dates, {info['n_scenes']} scenes) from "
                f"<b>{source_label}</b>. "
                f"Coverage is the share of your area's <b>{area_word}</b> that one "
                "date holds.<br>"
                "<span style='color:#6b7280;'>Estimated from scene outlines, so "
                "values can differ by a few percent from the built cube and between "
                "data sources.</span>"
                "</div>"
            ))
            # --- Multi-feature (batch) files --------------------------------
            # One cube is built PER feature, each against its own bounding box,
            # so the union table above answers a question nobody asked. Show the
            # verdict per feature instead and keep the 47-row detail collapsed.
            fdf = info.get("per_feature")
            finfo = info.get("per_feature_info") or {}
            if fdf is not None and not fdf.empty:
                n_feat = int(finfo.get("n_features", len(fdf)))
                grazing = int(finfo.get("features_with_grazing", 0))
                never_full = int(finfo.get("features_never_full", 0))
                no_dates = int(finfo.get("features_with_no_dates", 0))
                graze_pct = int(round(float(finfo.get("graze_threshold", 0.1)) * 100))

                # Only findings worth acting on. A zero count is not news, and
                # printing "0 of 47" for every check buried the one line that
                # actually mattered.
                lines = [
                    f"This file holds <b>{n_feat} areas</b>, so it builds "
                    f"<b>{n_feat} separate cubes</b>. Each is measured against "
                    "its own area below."
                ]
                if grazing:
                    lines.append(
                        f"Areas with dates covering under {graze_pct}%: "
                        f"<b>{grazing}</b> of {n_feat}."
                    )
                if never_full:
                    # NOT "areas with no data": these are areas no SINGLE date
                    # covers completely, so every timestep of those cubes is a
                    # partial scene.
                    lines.append(
                        f"Areas that no single date covers completely: "
                        f"<b>{never_full}</b> of {n_feat}."
                    )
                if no_dates:
                    lines.append(
                        f"⚠️ <b>{no_dates}</b> area(s) get no scenes at all in "
                        "this date range."
                    )
                display(HTML(
                    f"{_INFO_BOX}" + "<br>".join(lines) + "</div>"
                ))

                _feat_out = widgets.Output()
                with _feat_out:
                    display(HTML(fdf.to_html(border=0, na_rep="-")))
                _feat_acc = widgets.Accordion(
                    children=[_feat_out], selected_index=None
                )
                _feat_acc.set_title(0, f"Per-area details ({n_feat} rows)")
                _feat_acc.layout = widgets.Layout(width="99%")
                display(_feat_acc)

                _union_out = widgets.Output()
                with _union_out:
                    display(HTML(
                        "<div style='font-size:12px; color:#6b7280; margin:0 0 6px 0;'>"
                        "Measured on the combined extent of all areas. Useful for "
                        "seeing which orbits exist, but no cube is built on this "
                        "geometry.</div>"
                    ))
                    display(HTML(df.to_html(border=0, na_rep="NaN")))
                _union_acc = widgets.Accordion(
                    children=[_union_out], selected_index=None
                )
                _union_acc.set_title(0, "Combined extent (all areas together)")
                _union_acc.layout = widgets.Layout(width="99%")
                display(_union_acc)
            else:
                display(HTML(df.to_html(border=0, na_rep="NaN")))

            # The one finding worth acting on, phrased against the threshold the
            # user's own Across-track setting would apply. On a batch file the
            # per-date series is the union one, which is exactly what must NOT
            # drive the advice here - so the partial/graze warnings below are
            # skipped and the per-area verdict above stands in their place.
            if fdf is not None and not fdf.empty:
                if m is None:
                    print(
                        "The map needs the leafmap package, which isn't available "
                        "here. The tables above still apply."
                    )
                else:
                    display(m)
                return

            threshold = int(min_coverage_w.value)
            shares = [
                v for v in info.get("per_date_coverage", {}).values() if v is not None
            ]
            n_partial = sum(1 for v in shares if 100.0 * v < threshold)
            if not n_partial:
                display(HTML(
                    "<div style='font-size:12px; color:#166534; background:#f0fdf4; "
                    "border:1px solid #bbf7d0; border-radius:6px; padding:6px 8px; "
                    "margin:6px 0;'>"
                    f"✅ Every date covers at least {threshold}% of your area."
                    "</div>"
                ))
            elif partial_scene_w.value == "remove":
                display(HTML(
                    f"{_INFO_BOX}"
                    f"ℹ️ <b>{n_partial} of {len(shares)} dates</b> cover less than "
                    f"{threshold}% of your area. Your <b>Across-track</b> setting is "
                    "already set to <b>Remove partially missing scenes</b>, so these "
                    "will be dropped."
                    "</div>"
                ))
            else:
                display(HTML(
                    "<div style='font-size:12px; color:#92400e; background:#fffbeb; "
                    "border:1px solid #fde68a; border-radius:6px; padding:6px 8px; "
                    "margin:6px 0;'>"
                    f"⚠️ <b>{n_partial} of {len(shares)} dates</b> cover less than "
                    f"{threshold}% of your area (swath-edge scenes).<br>"
                    "To drop them, open <b>Advanced Parameters &rarr; Overlapping "
                    "Tile Handling &rarr; Across-track</b> and set <b>Coverage</b> to "
                    "<b>Remove partially missing scenes</b>."
                    "</div>"
                ))

            # Near-misses are a different finding from "partial": these orbits
            # bring nothing usable at all, yet each still costs a full-size time
            # step. Reported separately, right where the box that removes them
            # is, and only when there is something to act on.
            # Threshold shared with the per-feature summary (auxiliary.
            # GRAZE_THRESHOLD), so the two views can never suggest different
            # numbers for the same finding.
            _graze_pct = int(round(_GRAZE_THRESHOLD * 100))
            n_graze = sum(1 for v in shares if 100.0 * v < _graze_pct)
            if n_graze and skip_footprint_w.value <= 0:
                display(HTML(
                    f"{_INFO_BOX}"
                    f"💡 <b>{n_graze} of {len(shares)} dates</b> cover less than "
                    f"<b>{_graze_pct}%</b> of your area - they would be almost "
                    "entirely empty, but each one still costs a full time "
                    "step.<br>"
                    "Set <b>Skip scenes that barely touch your area</b> (just "
                    f"below) to <b>{_graze_pct}</b> to leave them out of the "
                    "download."
                    "</div>"
                ))

            if m is None:
                print(
                    "The map needs the leafmap package, which isn't available "
                    "here. The table above still applies."
                )
            else:
                display(m)

    coverage_btn.on_click(_on_check_coverage)

    # -------------------------------------------------------------------------
    # Pre-load footprint prefilter. Sits under "Check area coverage" because that
    # button is what shows the problem: the orbits at the bottom of its table are
    # exactly the ones this box removes. The number the user types is the number
    # they just read there.
    #
    # Worded as "barely touch" rather than as a second coverage percentage, on
    # purpose. The Advanced across-track filter asks "is this scene complete?"
    # and measures pixels; this one asks "is it worth downloading?" and reads
    # only the published outlines. Users should never have to compare the two.
    # -------------------------------------------------------------------------
    skip_footprint_w = widgets.BoundedFloatText(
        value=0.0,
        min=0.0,
        max=100.0,
        step=0.5,
        # No description: stacked_field_with_help clears it and renders the
        # label above the box.
        continuous_update=False,  # validate on blur, never per keystroke
        layout=widgets.Layout(width="110px"),
    )
    skip_footprint_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:13px; font-weight:600; color:#374151; "
                "margin:0 0 2px 0;'>Skip scenes that barely touch your area</div>"
                "<div style='font-size:12px; color:#6b7280; margin:0 0 4px 0;'>"
                "Some orbits clip only the edge of your area. Skipping them "
                "before downloading makes the cube smaller and much faster. "
                "0 keeps everything.</div>"
            ),
            _with_help_left(
                skip_footprint_w,
                "footprint_prefilter",
                "Skip scenes covering less than (% of your area)",
            ),
        ],
        layout=widgets.Layout(width="100%", gap="2px"),
    )

    # Warning shown only when the terrabyte STAC API is the chosen Data Source.
    terrabyte_warning_html = widgets.HTML(
        "<div style='font-size:12px; color:#92400e; background:#fffbeb; "
        "border:1px solid #fde68a; border-radius:6px; padding:6px 8px;'>"
        "⚠️ The <b>terrabyte</b> STAC API data cube can be built, but it <b>cannot</b> be "
        "exported nor visualized if the user credentials are not met. In this case, please "
        "use this repository @ <a href='https://portal.terrabyte.lrz.de/' target='_blank' "
        "rel='noopener noreferrer'>https://portal.terrabyte.lrz.de/</a>"
        "</div>",
        layout=widgets.Layout(display="none"),
    )

    def _update_terrabyte_warning(*_):
        show = (source_w.value == "terrabyte") and (not source_w.disabled)
        terrabyte_warning_html.layout.display = "" if show else "none"

    source_w.observe(
        lambda change: _update_terrabyte_warning(), names=["value", "disabled"]
    )
    _update_terrabyte_warning()

    # Warning shown only when the Copernicus (CDSE) STAC API is the chosen Data
    # Source: its pixels live on s3://eodata and need the user's S3 keys, so the
    # cube cannot be built/exported/visualized until those are set.
    cdse_warning_html = widgets.HTML(
        "<div style='font-size:12px; color:#92400e; background:#fffbeb; "
        "border:1px solid #fde68a; border-radius:6px; padding:6px 8px;'>"
        "⚠️ The <b>Copernicus</b> (CDSE) data cube <b>cannot</b> be built, "
        "exported nor visualized without credentials. They are <b>free</b> and "
        "take about <b>5 minutes</b> to set up - see "
        "<code>credentials/README.md</code> for the step-by-step instructions. This option is bad idea for long time-series!"
        "</div>",
        layout=widgets.Layout(display="none"),
    )

    def _update_cdse_warning(*_):
        show = (source_w.value == "cdse") and (not source_w.disabled)
        cdse_warning_html.layout.display = "" if show else "none"

    source_w.observe(
        lambda change: _update_cdse_warning(), names=["value", "disabled"]
    )
    _update_cdse_warning()

    # Bottom-of-section "collapse" buttons so a long, scrolled accordion can be
    # folded back up without scrolling to its title bar. Wired to the accordions
    # below (which don't exist yet at this point).
    def _collapse_button(tooltip):
        return widgets.Button(
            description="Collapse",
            icon="chevron-up",
            tooltip=tooltip,
            layout=widgets.Layout(width="auto"),
        )

    def _collapse_row(btn):
        return widgets.HBox(
            [btn],
            layout=widgets.Layout(
                width="100%", justify_content="flex-end", margin="2px 0 0 0"
            ),
        )

    basic_collapse_btn = _collapse_button("Collapse Basic Parameters")
    advanced_collapse_btn = _collapse_button("Collapse Advanced Parameters")
    source_collapse_btn = _collapse_button("Collapse Data Source")
    viz_collapse_btn = _collapse_button("Collapse Visualization")

    basic_box = widgets.VBox(
        [
            _field_group("Mission", [_boxed(mission_dd), mission_note],
                         subtitle="Satellite mission to use."),
            # Time Period sits BEFORE Polygon on purpose: the map tools in the
            # Polygon group (footprint preview) and the availability check in Data
            # Source both read the date range, so a user who meets them with no
            # dates set would get a silent fallback window instead of their own.
            # Nothing in Time Period reads the polygon, so there is no cycle.
            date_section,
            # Resolution moved to Advanced Parameters, where it sits with the
            # other two controls that define the output grid (Resampling and
            # Output Projection). It is the group that opens by default there,
            # because pixel size is the biggest lever on cube size and build time
            # and must not be something a user only discovers after a slow build.
            _field_group(
                "Polygon",
                [
                    # --- Option 1: provide a file or bbox (boxed sub-panel) ---
                    _subpanel(
                        [
                            widgets.HTML(
                                "<div style='font-size:13px; font-weight:600; color:#374151; "
                                "margin:0 0 2px 0;'>Option 1 - Polygon file or bounding box</div>"
                            ),
                            _boxed(polygon_input_box),
                            widgets.HTML(
                                "<div style='font-size:12px; color:#1e3a8a; background:#eff6ff; "
                                "border:1px solid #bfdbfe; border-radius:6px; padding:6px 8px;'>"
                                "<b>NOTE:</b> a polygon vector file with <b>multiple features</b> is "
                                "accepted and automatically activates <b>BATCH PROCESSING</b> (one "
                                "data cube per feature). A <i>multi-polygon</i>-type vector file is "
                                "<b>NOT</b> accepted."
                                "</div>"
                            ),
                        ],
                        accent="blue",
                    ),
                    _or_divider(),
                    # --- Option 2: draw on a map (boxed sub-panel) ---
                    _subpanel(
                        [
                            widgets.HTML(
                                "<div style='font-size:13px; font-weight:600; color:#374151; "
                                "margin:0 0 2px 0;'>Option 2 - Draw on a map</div>"
                            ),
                            draw_polygon_w,
                            draw_box,
                        ],
                        accent="green",
                    ),
                    _line_divider(),
                    # --- Output shape: clipping applies to whichever option above ---
                    _subpanel(
                        [
                            widgets.HTML(
                                "<div style='font-size:13px; font-weight:600; color:#374151; "
                                "margin:0 0 2px 0;'>Output shape</div>"
                            ),
                            clip_raster_w,
                            clip_info_html,
                            clip_warning_html,
                        ],
                        accent="amber",
                    ),
                    _line_divider(),
                    # --- Optional coverage check: reads the catalogue, sets nothing ---
                    widgets.HBox(
                        [coverage_btn], layout=widgets.Layout(margin="2px 0")
                    ),
                    coverage_result_acc,
                    skip_footprint_box,
                ],
                subtitle="The area to cover. Pick a polygon file or bounding box, or draw it on a map.",
                help_html=PARAM_HELP_HTML.get("polygon", ""),
                collapsible=True,
                open=False,
            ),
            bands_box,
            indices_box,
            _collapse_row(basic_collapse_btn),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    basic_acc = widgets.Accordion(children=[basic_box], selected_index=None)
    basic_acc.set_title(0, "Basic Parameters")
    basic_acc.layout = widgets.Layout(width="99%")

    # -------------------------------------------------------------------------
    # Cloud Masking section: collapsed by default, gathers ALL cloud-related
    # parameters (SCL masking, keep-clouds, tile max cloud cover) plus a plain
    # intro and a foldable comparison of the two masking methods. Most users
    # don't know the SCL-vs-s2cloudless distinction; this keeps it out of the way
    # but explained for the ones who care.
    # -------------------------------------------------------------------------
    _cm_about_html = widgets.HTML(
        "<div style='font-size:12px; color:#374151; line-height:1.5;'>"
        "stac2cube offers two different cloud masking methods.<br><br>"
        "<b>a) Scene Classification Layer (this one)</b><br>"
        "<u>Advantages</u>:"
        "<ul style='margin:2px 0 6px 18px; padding:0;'>"
        "<li>Super fast and lightweight</li>"
        "<li>Masks the clouds immediately when building the data cube</li>"
        "</ul>"
        "<u>Disadvantages</u>:"
        "<ul style='margin:2px 0 12px 18px; padding:0;'>"
        "<li>Result is static, the user cannot change cloud probability</li>"
        "<li>False positive possibilities (e.g. bright gravels)</li>"
        "</ul>"
        "<b>b) Probabilistic - s2cloudless (not here, in 'Analysis Ready Data Cube Tools')</b><br>"
        "<u>Advantages</u>:"
        "<ul style='margin:2px 0 6px 18px; padding:0;'>"
        "<li>Dynamic: generates probability maps (same result as Google's "
        "Sentinel-2: Cloud Probability)</li>"
        "<li>The user can select sensitivity thresholds</li>"
        "<li>The user can generate multiple binary cloud masks</li>"
        "</ul>"
        "<u>Disadvantages</u>:"
        "<ul style='margin:2px 0 2px 18px; padding:0;'>"
        "<li>Requires computation power and takes a long time to process</li>"
        "<li>Has to be applied in Analysis Ready Data Cube Tools after generating "
        "the initial data cube</li>"
        "</ul>"
        "</div>"
    )
    # Match the "About Data Sources" design: a real collapsed-by-default
    # Accordion (not the custom field_group collapse), 99% wide to avoid the
    # stray 1px horizontal scrollbar a nested accordion can otherwise push.
    _cm_about = widgets.Accordion(
        children=[_cm_about_html], selected_index=None
    )
    _cm_about.set_title(0, "About Cloud Masking Methods")
    # Top margin pushes the About box down from the "Cloud Detection & Masking"
    # title (otherwise it sits flush against it and is easy to miss); the bottom
    # margin keeps it off the preset sub-panel below.
    _cm_about.layout = widgets.Layout(width="99%", margin="10px 0 10px 0")

    # Content of the Cloud Detection & Masking group. Starts with the methods
    # comparison (unchanged), then the guided presets inside one accented
    # sub-panel, a gap, and finally the four raw parameters - each in its own
    # sub-panel so they read as separate fields on the olive group background.
    cloud_masking_children = [
        _cm_about,
        widgets.HTML(
            "<div style='font-size:12px; color:#1e40af; line-height:1.5; "
            "background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; "
            "padding:8px 10px; margin:0 0 8px 0;'>"
            "This method is super fast. However, the user cannot change the "
            "strength of the cloud detection and it often results in false "
            "positives. If this matters to you, skip this section and use "
            "s2cloudless probabilistic cloud masking with multiple custom "
            "thresholds in <b>3. Analysis Ready Data Cube Tools -&gt; Cloud "
            "and Shadow Masking Data Cube</b>. Optional cloud shadow masking "
            "is also provided there and you will find much better "
            "visualization tools to compare your results."
            "</div>"
        ),
        # "Select one of the options" + the four presets, in one boxed sub-panel.
        _subpanel([_cloud_preset_box], accent="blue"),
        # Breathing room between the preset selector and the raw parameters.
        widgets.HTML("<div style='height:8px;'></div>"),
        _param_panel("Cloud Detection with SCL", _boxed(cloud_masking_w)),
        _param_panel("Mask or Keep Clouds", _boxed(keep_clouds_w)),
        _param_panel(
            "Export Mask as Binary File",
            widgets.VBox(
                [_boxed(export_mask_w), export_mask_path_box],
                layout=widgets.Layout(width="100%", gap="6px"),
            ),
        ),
        _param_panel(
            "Sentinel 2 Tile Max Cloud Coverage",
            _boxed(max_cc_w),
            help_html=PARAM_HELP_HTML.get("max_cc", ""),
        ),
    ]
    # Olive collapsible group, matching the Polygon / Stats / Temporal Composite
    # sections (previously this was a plain white Accordion).
    cloud_masking_group = _field_group(
        "Cloud Detection & Masking",
        cloud_masking_children,
        collapsible=True,
        open=False,
        accent="turquoise",
    )

    # -------------------------------------------------------------------------
    # Cloud Shadow Masking group: collapsed by default, directly under Cloud
    # Detection & Masking. Plain-language intro (incl. the two honest
    # limitations), the enable checkbox with its gate note, and the two
    # parameters - each with a '?' help toggle - greyed until enabled.
    # -------------------------------------------------------------------------
    _shadow_about_html = widgets.HTML(
        "<div style='font-size:12px; color:#1e40af; line-height:1.5; "
        "background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; "
        "padding:8px 10px; margin:0 0 8px 0;'>"
        "Cloud shadow detection quality heavily depends on the cloud "
        "detection. For better shadow detection, use this tool with "
        "s2cloudless cloud detection in <b>3. Analysis Ready Data Cube "
        "Tools</b>! Also better visualization tools there :)"
        "</div>"
        "<div style='font-size:12px; color:#374151; line-height:1.5;'>"
        "Cloud shadow detection is done using the detected clouds, a non-water "
        "NIR darkness threshold and the Sun's direction of the scene. The "
        "method is provided by Google.<br><br>"
        "<b>Good to know:</b>"
        "<ul style='margin:2px 0 2px 18px; padding:0;'>"
        "<li>It works best on <b>large areas</b> (landscape scale). On very "
        "small areas most shadows come from clouds outside your cube.</li>"
        "<li>It <b>cannot</b> mask a shadow whose cloud is not present in the "
        "scene - the projection needs the cloud itself.</li>"
        "<li>Over dense cities, dark surfaces (asphalt, building shade) may be "
        "masked as cloud shadow too.</li>"
        "</ul>"
        "</div>"
    )
    shadow_masking_group = _field_group(
        "Cloud Shadow Masking",
        [
            _shadow_about_html,
            _subpanel(
                [shadow_masking_w, shadow_gate_note], accent="blue"
            ),
            widgets.HTML("<div style='height:8px;'></div>"),
            _param_panel(
                "NIR Dark Threshold",
                _boxed(shadow_nir_dark_w),
                help_html=PARAM_HELP_HTML.get("nir_dark_threshold", ""),
            ),
            _param_panel(
                "Projection Distance",
                _boxed(shadow_proj_dist_w),
                help_html=PARAM_HELP_HTML.get("shadow_proj_distance", ""),
            ),
        ],
        collapsible=True,
        open=False,
        accent="turquoise",
    )

    # --- Output grid: resolution, resampling, projection -----------------------
    # Open by default (the only Advanced group that is), see the note in basic_box.
    resolution_group = _field_group(
        "Resolution",
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#475569; margin:0 0 6px 0;'>"
                "Output pixel size, in metres. Changing it to 2.5 metres DOES NOT "
                "super-resolve the data :) It just resamples it.<br>"
                "</div>"
            ),
            _boxed(resolution_w),
        ],
        collapsible=True,
        open=True,
        accent="violet",
    )

    resampling_group = _field_group(
        "Resampling Method",
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#475569; margin:0 0 6px 0;'>"
                "Affects both upsampling and downsampling - e.g. building a "
                "20 m cube resamples the native 10 m bands with this method, "
                "and a 10 m cube resamples the native 20 m bands with it."
                "</div>"
            ),
            _param_panel(
                "Spectral Band Resampling",
                _boxed(resampling_w),
                help_html=PARAM_HELP_HTML.get("resampling_method", ""),
            ),
        ],
        collapsible=True,
        open=False,
        accent="violet",
    )

    # --- Output Projection (CRS) ---------------------------------------------
    # "Automatic" resolves at build time from the scenes themselves, so this group
    # can be ignored entirely. The dropdown exists to SHOW what was detected and
    # to override it; the free-text box is for a projection nobody's tiles use
    # (e.g. an equal-area CRS for reporting hectares).
    _CRS_AUTO = "auto"
    # "auto" sentinel, not None: ipywidgets treats value=None as "nothing
    # selected" even when None is one of the option values, so the box would
    # render blank instead of showing the Automatic entry.
    crs_detected_w = widgets.Dropdown(
        options=[("Automatic (best coverage of your area)", _CRS_AUTO)],
        value=_CRS_AUTO,
        layout=widgets.Layout(width="100%"),
    )
    crs_user_w = widgets.Text(
        value="",
        placeholder="EPSG:3035",
        # continuous_update=False on purpose: with the default, `value` fires on
        # every keystroke, so typing "3035" is seen as "3", "30", "303" - each an
        # incomplete EPSG code that validation rejects. The box now commits on
        # Enter or when it loses focus, like the rest of the builder's inputs.
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
    )
    crs_search_btn = widgets.Button(
        description="Detect projections",
        icon="search",
        layout=widgets.Layout(width="200px"),
    )
    crs_status_w = widgets.HTML("")

    crs_group = _field_group(
        "Output Projection (CRS)",
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#475569; margin:0 0 6px 0;'>"
                "The common coordinate reference system that the data cube is "
                "built on. Leave it on <b>Automatic</b> unless you need a "
                "specific projection."
                "</div>"
            ),
            _param_panel(
                "Detected projections",
                widgets.VBox(
                    [
                        _boxed(crs_detected_w),
                        widgets.HBox(
                            [crs_search_btn],
                            layout=widgets.Layout(margin="4px 0 0 0"),
                        ),
                        crs_status_w,
                    ],
                    layout=widgets.Layout(width="100%", overflow="hidden"),
                ),
                help_html=PARAM_HELP_HTML.get("crs", ""),
            ),
            _param_panel(
                "User-Defined CRS",
                _boxed(crs_user_w),
                help_html=PARAM_HELP_HTML.get("crs", ""),
            ),
        ],
        collapsible=True,
        open=False,
        accent="violet",
    )

    # --- Output Projection behaviour ------------------------------------------
    _CRS_AUTO_LABEL = "Automatic (best coverage of your area)"

    def _set_crs_status(html):
        crs_status_w.value = html or ""

    def _reset_detected_crs(*_):
        """The detected list belongs to ONE area/source/mission. Anything that
        changes which scenes would be found invalidates it, so clear it and fall
        back to Automatic instead of leaving a stale projection selected - which
        would silently reproject every scene of the new area."""
        crs_detected_w.options = [(_CRS_AUTO_LABEL, _CRS_AUTO)]
        crs_detected_w.value = _CRS_AUTO
        # Re-render the status rather than blanking it: a typed CRS is still in
        # effect after an AOI change, so wiping its confirmation would read as
        # though the setting had been cleared too.
        _sync_crs_controls()

    def _sync_crs_controls(*_):
        """A user-defined CRS overrides the dropdown, so grey the dropdown while
        the box has content, and check what was typed once the box is committed
        (Enter / focus loss - see continuous_update above) rather than letting a
        bad CRS fail only after a long build.

        Catches Exception, not just ValueError: this runs inside an ipywidgets
        message handler, where anything escaping is dumped as a raw traceback
        under the GUI."""
        text = (crs_user_w.value or "").strip()
        crs_detected_w.disabled = bool(text)
        crs_search_btn.disabled = bool(text)
        if not text:
            _set_crs_status("")
            return
        try:
            canonical = validate_target_crs(text)
        except Exception as exc:
            _set_crs_status(
                "<div style='font-size:12px; color:#991b1b; background:#fef2f2; "
                "border:1px solid #fecaca; border-radius:6px; padding:6px 8px;'>"
                f"✗ {exc}</div>"
            )
            return
        _set_crs_status(
            "<div style='font-size:12px; color:#166534;'>✓ building in "
            f"<b>{canonical}</b>.</div>"
        )

    def _effective_crs():
        """crs= for get_stac_layers: the typed CRS wins, then the dropdown, then
        None, which means Automatic (resolved from the scenes at build time)."""
        text = (crs_user_w.value or "").strip()
        if text:
            return validate_target_crs(text)
        chosen = crs_detected_w.value
        return None if chosen in (None, _CRS_AUTO) else chosen

    def _populate_detected_crs(entries):
        """Fill the dropdown, putting the reason in each label so the automatic
        default explains itself instead of having to be trusted."""
        opts = [(_CRS_AUTO_LABEL, _CRS_AUTO)]
        for e in entries:
            share = e.get("share")
            cover = (
                f" - covers {share * 100:.0f}% of your area"
                if share is not None else ""
            )
            opts.append((f"{e['crs']}{cover}", e["crs"]))
        previous = crs_detected_w.value
        crs_detected_w.options = opts
        if previous in [o[1] for o in opts]:
            crs_detected_w.value = previous

    def _populate_detected_crs_from_cube(obj):
        """Fill the dropdown from a cube that was just built - free, no query.

        A finished cube already states its projections: ``native_crs`` whenever
        anything had to be reprojected, and otherwise its own ``crs``, which is
        then by definition the single native one. So after a build the Detect
        button has nothing left to find.

        Skipped for a multi-feature batch: each feature can land in a different
        projection, so one list could not honestly describe them all.
        """
        try:
            if isinstance(obj, list):
                return
            da = obj
            if isinstance(obj, xr.Dataset):
                da = obj.get("Time_Series")
                if da is None and len(obj.data_vars):
                    da = obj[list(obj.data_vars)[0]]
            attrs = getattr(da, "attrs", {}) or {}
            target = str(attrs.get("crs", "") or "")
            natives = attrs.get("native_crs")

            if natives is None:
                if not target:
                    return
                entries = [{"crs": target, "share": 1.0}]
            else:
                natives = [str(c) for c in np.asarray(natives).ravel()]
                shares = attrs.get("native_crs_share")
                share_by = {}
                if shares is not None:
                    vals = np.asarray(shares).ravel().tolist()
                    if len(vals) == len(natives):
                        share_by = dict(zip(natives, vals))
                entries = [{"crs": c, "share": share_by.get(c)} for c in natives]

            if not entries:
                return
            _populate_detected_crs(entries)
            # Don't clobber the typed-CRS validation message, which is still the
            # setting actually in force.
            if not (crs_user_w.value or "").strip():
                _set_crs_status(
                    "<div style='font-size:12px; color:#166534;'>✓ filled in from "
                    f"the cube you just built (<b>{target}</b>).</div>"
                )
        except Exception:
            pass

    def _on_detect_crs(_btn):
        """Manual, never automatic: this is a network round trip and ipywidgets
        would freeze the GUI if it fired on every polygon edit. Automatic already
        works without it, so nothing is blocked if the user never clicks."""
        try:
            polygon = _parse_polygon_input(polygon_w.value)
        except Exception as exc:
            _set_crs_status(
                f"<div style='font-size:12px; color:#991b1b;'>✗ {exc}</div>"
            )
            return
        if polygon is None:
            _set_crs_status(
                "<div style='font-size:12px; color:#92400e;'>Set a polygon or "
                "bounding box first.</div>"
            )
            return

        crs_search_btn.disabled = True
        _set_crs_status(
            "<div style='font-size:12px; color:#475569;'>Searching the "
            "catalogue...</div>"
        )
        try:
            entries = probe_native_crs(
                mission_dd.value,
                polygon,
                source=source_w.value,
                daterange=_resolve_daterange(),
            )
            if not entries:
                _set_crs_status(
                    "<div style='font-size:12px; color:#92400e;'>No scenes found "
                    "for this area.</div>"
                )
                return
            _populate_detected_crs(entries)
            if len(entries) == 1:
                _set_crs_status(
                    "<div style='font-size:12px; color:#166534;'>✓ one projection "
                    f"covers this area: <b>{entries[0]['crs']}</b>. Automatic "
                    "uses it.</div>"
                )
            else:
                best = entries[0]
                _set_crs_status(
                    "<div style='font-size:12px; color:#1e3a8a;'>Found "
                    f"<b>{len(entries)}</b> projections. Automatic uses "
                    f"<b>{best['crs']}</b> - it covers most of your area. No "
                    "scenes are dropped.</div>"
                )
        except Exception as exc:
            _set_crs_status(
                "<div style='font-size:12px; color:#991b1b;'>✗ "
                f"{_friendly_error(exc, 'Detecting projections')}</div>"
            )
        finally:
            crs_search_btn.disabled = bool((crs_user_w.value or "").strip())

    crs_search_btn.on_click(_on_detect_crs)
    crs_user_w.observe(lambda change: _sync_crs_controls(), names="value")
    for _w in (polygon_w, source_w, mission_dd):
        _w.observe(lambda change: _reset_detected_crs(), names="value")
    _sync_crs_controls()

    tile_handling_group = _field_group(
        "Overlapping Tile Handling",
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#475569; margin:0 0 6px 0;'>"
                "Control how overlapping Sentinel-2 tiles are handled when your "
                "area falls across a tile boundary."
                "</div>"
            ),
            # Along-track (North-South): tiles from the SAME satellite pass,
            # stacked in latitude (e.g. 47TPK/47TPL).
            _param_panel(
                "Along-track (North-South)",
                widgets.VBox(
                    [
                        widgets.HTML(
                            "<div style='font-size:12px; color:#475569; "
                            "margin:0 0 4px 0;'>Two tiles from the same pass, "
                            "stacked north-south: <b>mosaic</b> them into one "
                            "image per date, or keep them <b>separate</b>.</div>"
                        ),
                        _boxed(tile_handling_w),
                        tile_handling_note,
                    ],
                    layout=widgets.Layout(width="100%", overflow="hidden"),
                ),
            ),
            # Breathing room so the two panels do not read as one block.
            widgets.HTML("<div style='height:10px;'></div>"),
            # Across-track (East-West): swath / orbit edge - a scene covers only
            # part of the AOI, the rest is missing (NaN).
            _param_panel(
                "Across-track (East-West)",
                widgets.VBox(
                    [
                        widgets.HTML(
                            "<div style='font-size:12px; color:#475569; "
                            "margin:0 0 4px 0;'>At a swath or orbit edge a scene "
                            "images only part of your area: <b>keep</b> these "
                            "partial scenes, or <b>remove</b> them.</div>"
                        ),
                        _boxed(partial_scene_w),
                        min_coverage_box,
                        partial_scene_note,
                    ],
                    layout=widgets.Layout(width="100%", overflow="hidden"),
                ),
            ),
        ],
        collapsible=True,
        open=False,
        accent="violet",
    )

    # ------------------------------------------------------------------------
    # Scene Specific Metadata group: LAST entry of Advanced Parameters. The
    # option list is rebuilt from the selected Data Source (see
    # _sync_scene_metadata_options), so a field the source does not publish
    # can never be selected. The Export Granule Metadata checkbox downloads
    # the per-scene MTD_TL.xml files at export time (greyed on terrabyte).
    # ------------------------------------------------------------------------
    scene_metadata_group = _field_group(
        "Scene Specific Metadata",
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#475569; margin:0 0 6px 0;'>"
                "Attach per-scene metadata from the STAC catalogue to the data "
                "cube as extra <b>time coordinates</b>, so scenes can be "
                "queried and filtered by them later. The list follows the "
                "selected <b>Data Source</b>: fields it does not publish are "
                "hidden."
                "</div>"
                "<div style='font-size:12px; color:#92400e; background:#fffbeb; "
                "border:1px solid #fde68a; border-radius:6px; padding:6px 8px; "
                "margin:0 0 6px 0;'>⚠️ When the polygon is covered by more "
                "than one Sentinel-2 granule on the same date (tile overlap), "
                "angles are stored as the per-date mean and acq_datetime as "
                "the earliest acquisition.</div>"
            ),
            _param_panel(
                "Scene Metadata Fields",
                widgets.VBox(
                    [
                        _boxed(scene_metadata_w),
                        widgets.HBox(
                            [scene_meta_all_btn, scene_meta_none_btn],
                            layout=widgets.Layout(gap="8px", margin="6px 0 0 0"),
                        ),
                        scene_meta_note,
                    ],
                    layout=widgets.Layout(width="100%"),
                ),
            ),
            # Plain checkbox row (no extra title/explanation by design); the
            # subpanel gives it the boxed look of the other fields. overflow
            # hidden clips the classic 1px horizontal sliver.
            _subpanel(
                [
                    widgets.VBox(
                        [export_granule_meta_w, export_granule_meta_note],
                        layout=widgets.Layout(width="100%", overflow="hidden"),
                    )
                ],
            ),
        ],
        collapsible=True,
        open=False,
    )

    advanced_box = widgets.VBox(
        [
            # Turquoise = what gets masked out of the pixel values.
            cloud_masking_group,
            shadow_masking_group,
            # Violet = how the cube itself is constructed. Resolution first and
            # open by default (biggest lever on size); then the rest of the
            # output grid, then which scenes go in.
            resolution_group,
            resampling_group,
            crs_group,
            tile_handling_group,
            # Uncoloured = optional additions that do not change the base cube.
            scene_metadata_group,
            # Stats moved out to the Temporal Composites card below the Result
            # section: they reduce over the dates kept there, not over the
            # build, and they need no rebuild to change.
            _collapse_row(advanced_collapse_btn),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    # The mean/median Temporal Composite dropdown that used to sit here has
    # been replaced by the Temporal Composites card (below the Result section),
    # where it is the "Mean/Median of the time series" checkbox plus "Keep the
    # full time series". Export Options is about format and path again.

    # Slim guidance line shown under the selector: the right source is study-area
    # and cloud-masking dependent, so nudge unsure users to the helper tools
    # below, and note the one fact that isn't obvious from the widgets
    # (cross-source spectral harmonization).
    source_info_bar = widgets.HTML(
        "<div style='font-size:12px; color:#475569; margin:0;'>"
        "<b>Note:</b> All data sources are scaled accordingly and matching Google's "
        "\"Harmonized Sentinel-2 L2A SR\"."
        "</div>"
    )

    # "About Data Sources": collapsed-by-default explanation of when to use each
    # catalogue. Plain hyphens only (project style), corrected spelling.
    about_sources_html = widgets.HTML(
        "<div style='font-size:12px; color:#374151; line-height:1.5;'>"
        "<b>1) Element84 (ready-to-use)</b><br>"
        "Good for long time-series, with some missing dates from the earlier "
        "years of the mission and some missing dates in 2023. It is free and "
        "needs no credentials or login. Note that Scene Classification Layer "
        "masking result is significantly poor comparing the other data sources."
        "<br><br>"
        "<b>2) Planetary Computer (ready-to-use)</b><br>"
        "Great for long time-series, usually with the full archive. It is free "
        "and needs no credentials or login.<br><br>"
        "<b>3) terrabyte (only at terrabyte portal)</b><br>"
        "Best for terrabyte users collecting long time-series. However, data "
        "availability is limited if the study area is outside Europe. Requires "
        "a terrabyte account. Full-archive L1C can be requested from DLR @ "
        "<a href='https://forum.terrabyte.lrz.de/c/data-requests' target='_blank' "
        "rel='noopener noreferrer'>https://forum.terrabyte.lrz.de/c/data-requests</a>"
        "<br><br>"
        "<b>4) Copernicus Data Space Ecosystem (credentials/README.md)</b><br>"
        "Copernicus' original data catalogue, with an always fully available "
        "archive. HOWEVER, the data is not served as cloud-optimized files. "
        "Therefore this source is good for a single date or a very short "
        "time-series, but quite bad for long time-series (you will hit "
        "data-reading issues). It also requires access keys (5-minute "
        "instructions in <code>credentials/README.md</code>)."
        "</div>"
    )
    about_sources_acc = widgets.Accordion(
        children=[about_sources_html], selected_index=None
    )
    about_sources_acc.set_title(0, "About Data Sources")
    # 99% (not 100%) leaves the same slack the top-level accordions use, so a
    # nested accordion can't overflow source_box and push a stray 1px horizontal
    # scrollbar onto the whole Data Source section.
    about_sources_acc.layout = widgets.Layout(width="99%")

    # "Check data availability": queries each catalogue (for the current mission)
    # over the chosen area + date range and tabulates scenes per acquisition day.
    check_avail_btn = widgets.Button(
        description="Check data availability",
        button_style="info",
        icon="search",
        layout=widgets.Layout(width="220px"),
    )
    check_avail_out = widgets.Output()

    # The result lives in its own accordion so it can be collapsed once generated.
    # Hidden until the first run, then revealed and opened by the click handler.
    avail_result_acc = widgets.Accordion(
        children=[check_avail_out], selected_index=None
    )
    avail_result_acc.set_title(0, "Availability Result")
    avail_result_acc.layout = widgets.Layout(width="99%", display="none")

    def _on_check_availability(_btn):
        # Reveal + open the result accordion so the progress message and the
        # results below are visible (the user can collapse it afterwards).
        avail_result_acc.layout.display = ""
        avail_result_acc.selected_index = 0
        with check_avail_out:
            clear_output()
            mission = mission_dd.value
            if mission not in ("sentinel_2_l2a", "sentinel_2_l1c"):
                print(
                    "ℹ️ Data availability comparison is only available for "
                    "Sentinel-2 L2A and Sentinel-2 L1C."
                )
                return
            try:
                polygon = _parse_polygon_input(polygon_w.value)
                daterange = _resolve_daterange()
            except Exception as e:
                print("⚠️ " + _friendly_error(e, "Reading the parameters"))
                return

            display(HTML(_busy_bear_html(
                "Querying catalogues for your area and date range",
                "(this can take a while for long or seasonal ranges)",
            )))
            try:
                df, errors = check_scene_availability(mission, polygon, daterange)
            except Exception as e:
                clear_output()
                print("⚠️ " + _friendly_error(e, "Checking data availability"))
                return

            clear_output()
            if df.empty:
                print(
                    "No scenes found in any working catalogue for this area and "
                    "date range."
                )
            else:
                # The cube loads with groupby="solar_day", so the number of time
                # steps equals the number of available dates, NOT the number of
                # scenes (one date can span several MGRS tiles). We therefore
                # report and show available DATES only.
                _check = ("<span style='color:#16a34a; font-weight:bold;'>"
                          "&#10003;</span>")   # green check = available
                _cross = ("<span style='color:#dc2626; font-weight:bold;'>"
                          "&#10007;</span>")   # red cross = not available

                # Explanation up top (NaN = a catalogue that could not be queried).
                display(HTML(
                    "<div style='font-size:12px; color:#374151; margin-bottom:6px;'>"
                    "<b>NaN</b> = catalogue query failed (e.g. credentials or "
                    "API issue).<br>"
                    "The <b>Max cloud %</b> filter is not applied here, so a "
                    "cube built with a filter below 100 will have fewer dates."
                    "</div>"
                ))

                # Table 1: available dates per catalogue (a fully-NaN column =
                # query failed, reported as NaN rather than 0).
                summary_rows = {
                    col: (
                        None
                        if df[col].isna().all()
                        else int((df[col].fillna(0) > 0).sum())
                    )
                    for col in df.columns
                }
                summary_df = pd.DataFrame(
                    # Dates, not scenes: the cube loads with groupby="solar_day",
                    # so one date is one time step however many MGRS tiles it spans.
                    {"Available Dates": list(summary_rows.values())},
                    index=list(summary_rows.keys()),
                )
                summary_df.index.name = "Mission"
                display(HTML(
                    summary_df.to_html(border=0, na_rep="NaN")
                ))

                # Table 2: per-date availability as icons (green check / red
                # cross), with failed catalogues left as NaN.
                def _icon(v):
                    if pd.isna(v):
                        return "NaN"
                    return _check if v > 0 else _cross

                icon_df = df.map(_icon)
                display(HTML(
                    "<div style='max-height:420px; overflow:auto; "
                    "border:1px solid #e5e7eb; border-radius:6px; margin-top:8px;'>"
                    + icon_df.to_html(border=0, escape=False)
                    + "</div>"
                ))

            if errors:
                lines = "<br>".join(
                    f"<b>{src}:</b> {msg}" for src, msg in errors.items()
                )
                display(HTML(
                    "<div style='font-size:12px; color:#9a3412; background:#fff7ed; "
                    "border:1px solid #fed7aa; border-radius:6px; padding:6px 8px; "
                    "margin-top:6px;'>Some catalogues could not be queried "
                    "(shown as NaN above):<br>" + lines + "</div>"
                ))

    check_avail_btn.on_click(_on_check_availability)

    source_box = widgets.VBox(
        [
            # Slim hint first, then the dropdown; the "About Data Sources"
            # explainer and the availability check follow. The two guided
            # suggestion checkboxes that used to sit above the dropdown are
            # gone - most users never read them, and the dropdown (defaulting
            # to Planetary Computer) plus "About Data Sources" is enough.
            source_info_bar,
            _field_group(
                "Data Source",
                [
                    _boxed(source_w),
                    terrabyte_warning_html,
                    cdse_warning_html,
                ],
                subtitle="Catalog to download from. First two are free, publicly available and does not require any login & credentials :)",
            ),
            about_sources_acc,
            widgets.HBox([check_avail_btn], layout=widgets.Layout(margin="2px 0")),
            avail_result_acc,
            _collapse_row(source_collapse_btn),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    # Export mode + Output share one olive group, matching the Temporal Composite
    # group above it.
    export_mode_group = _field_group(
        "Export mode & Output",
        [
            _with_help_left(export_mode_w, "export_mode", label_text="Export mode"),
            # Compression belongs to the MODE, not below: it changes the cube
            # file itself (NetCDF only) rather than adding a file beside it,
            # which is what separates it from the Side Outputs group.
            export_compress_w,
            export_compress_warn_html,
            _stacked_field(output_input_box, "Output"),
        ],
    )

    # Side Outputs: the extra files an export can drop next to the cube. All
    # three are named after the Output path above, none of them changes the
    # cube, and each is one boolean in the copied settings - so they read as
    # one group. Collapsed by default: the common export needs none of them.
    export_side_outputs_group = _field_group(
        "Side Outputs",
        [
            export_vrt_w,
            export_vrt_note_html,
            export_settings_w,
            export_settings_note_html,
            export_csv_w,
            export_csv_note_html,
        ],
        subtitle="Extra files written besides the cube. Format dependent.",
        collapsible=True,
        open=False,
    )

    export_box = widgets.VBox(
        [
            #widgets.HTML("<b>Export Options</b>"),
            export_mode_group,
            export_side_outputs_group,
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    viz_feature_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#6b7280;'>This result has several "
                "features - choose which one to view or animate and click 'Open Interactive View' each time you change the feature!:</div>"
            ),
            viz_feature_w,
        ],
        layout=widgets.Layout(width="100%", gap="4px", display="none"),
    )

    visualization_box = widgets.VBox(
        [
            viz_feature_box,
            # Same collapsible group styling as the Animation section below,
            # but open by default - the viewer is the primary tool here.
            _field_group(
                "1) Interactive View",
                # The output area the viewer renders into lives INSIDE the
                # group, so collapsing the header hides the opened map too.
                [
                    widgets.HTML(
                        f"{_INFO_BOX}ℹ️ Click <b>Open Interactive View</b> "
                        "everytime you change the layer.<br>"
                        "For enormous areas, lower the <b>View resolution</b> "
                        "for much faster experience that won't affect "
                        "visualization that much in huge extends.</div>"
                    ),
                    viz_layer_w,
                    viz_layer_note,
                    viz_resolution_w,
                    viz_renderer_box,
                    viz_dropdown_btn,
                    viz_out,
                ],
                collapsible=True,
                open=True,
            ),
            # The animation maker is a separate tool from the viewer, so it lives
            # in its own collapsed-by-default section (a custom collapse, not a
            # nested ipywidgets Accordion, which would push a stray scrollbar).
            _field_group(
                "2) Animation (GIF export)",
                [
                    gif_section_w,
                    gif_preset_box,
                    gif_band_box,
                    gif_custom_box,
                    gif_stretch_box,
                    _with_help_left(gif_fps_w, "fps", label_text="FPS"),
                    _with_help_left(gif_label_w, "anim_label", label_text="Label"),
                    _stacked_field(gif_output_input_box, "Output GIF"),
                    viz_make_gif_btn,
                    anim_out,
                ],
                subtitle="Renders the whole time series to an animated GIF on disk. "
                "Status is reported below the button.",
                collapsible=True,
                open=False,
            ),
            _collapse_row(viz_collapse_btn),
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )

    # Collapsible sections
    advanced_acc = widgets.Accordion(children=[advanced_box], selected_index=None)
    advanced_acc.set_title(0, "Advanced Parameters")
    advanced_acc.layout = widgets.Layout(width="99%")

    source_acc = widgets.Accordion(children=[source_box], selected_index=None)
    source_acc.set_title(0, "Data Source")
    source_acc.layout = widgets.Layout(width="99%")

    # Now that both accordions exist, wire the bottom collapse buttons to fold them.
    basic_collapse_btn.on_click(lambda _: setattr(basic_acc, "selected_index", None))
    advanced_collapse_btn.on_click(
        lambda _: setattr(advanced_acc, "selected_index", None)
    )
    source_collapse_btn.on_click(
        lambda _: setattr(source_acc, "selected_index", None)
    )

    export_acc = widgets.Accordion(children=[export_box], selected_index=None)
    export_acc.set_title(0, "Export Options")
    export_acc.layout = widgets.Layout(width="99%")

    viz_acc = widgets.Accordion(children=[visualization_box], selected_index=None)
    viz_acc.set_title(0, "Visualization")
    viz_acc.layout = widgets.Layout(width="99%")
    viz_collapse_btn.on_click(lambda _: setattr(viz_acc, "selected_index", None))

    # Max cloud % filter sits right under the date/cloud table, pushed to the
    # right so it lines up beneath the 'cloud %' column. Greyed out until a
    # cloud-masked build exists; changing it re-filters the table, visualization
    # and export together.
    result_cloud_filter_row = widgets.HBox(
        [result_cloud_max_w],
        layout=widgets.Layout(
            width="100%", justify_content="flex-end", padding="0 6px 2px 0"
        ),
    )

    # Min coverage % sits directly under Max cloud %, same right alignment: the
    # two scene filters read as one block, and a scene must pass both.
    result_coverage_filter_row = widgets.HBox(
        [result_coverage_min_w],
        layout=widgets.Layout(
            width="100%", justify_content="flex-end", padding="0 6px 2px 0"
        ),
    )

    # Per-date picker: the fine-grained companion to the Max cloud % filter. Its
    # accordion is hidden until a single-cube build populates it (see
    # _populate_result_dates); ticking dates re-renders the table, visualization
    # and export via _effective_result.
    result_date_box = widgets.VBox(
        [
            result_date_legend,
            result_date_w,
            widgets.HBox(
                [result_date_all_btn, result_date_clear_btn],
                layout=widgets.Layout(gap="6px"),
            ),
        ],
        layout=widgets.Layout(width="100%", gap="4px"),
    )
    result_date_acc = widgets.Accordion(children=[result_date_box], selected_index=None)
    result_date_acc.set_title(0, "Date Selection")
    result_date_acc.layout = widgets.Layout(width="45%")

    # Keep the Date Picker on the right half of the panel, right-aligned so it
    # lines up under the Max cloud % box above it. The row is hidden/shown as a
    # whole (see _populate_result_dates) so no stray gap is left when disabled.
    result_date_row = widgets.HBox(
        [result_date_acc],
        layout=widgets.Layout(
            width="100%", justify_content="flex-end", display="none"
        ),
    )

    # Warning text on the left, the "Resize and Re-build" button next to it on
    # the right; both are empty/hidden when the last build carried no warning.
    result_coreg_warn_row = widgets.HBox(
        [result_coreg_warn_w, coreg_resize_btn],
        layout=widgets.Layout(width="100%", gap="8px", align_items="center",
                              display="none"),
    )

    # Notices sit at the TOP of the Result section (user request), in this order:
    # the yellow Area Size warning with its Resize button (the only notice that
    # asks for a decision), then the collapsed blue notes strip holding
    # everything informational, and only then the cube summary and its controls.
    result_box = widgets.VBox(
        [result_coreg_warn_row, result_notes_row,
         result_out, result_cloud_filter_row, result_coverage_filter_row,
         result_date_row, result_viz_note_row],
        layout=widgets.Layout(width="99%", gap="6px"),
    )
    result_acc = widgets.Accordion(children=[result_box], selected_index=None)
    result_acc.set_title(0, "Result")
    result_acc.layout = widgets.Layout(width="99%")

    # Temporal Composites: its own accordion between Result and Visualization.
    # AFTER Result because the composites reduce over the dates kept there, and
    # BEFORE Visualization because the viewer's Layer dropdown lists whatever
    # this section produced.
    composites_acc = widgets.Accordion(children=[composites_box], selected_index=None)
    composites_acc.set_title(0, "Temporal Composites")
    composites_acc.layout = widgets.Layout(width="99%")

    # Top action row: build the preview. Exporting is a separate, deliberate
    # step that lives with the Export Options section further down.
    action_row = widgets.HBox(
        [generate_btn],
        layout=widgets.Layout(gap="8px", flex_flow="row wrap"),
    )

    # The Export Current Result button sits right under the Export Options
    # accordion so the "choose format -> export" flow reads top to bottom.
    # Copy Settings rides along here: both are ways of taking the result away.
    # Paste Settings sits next to it as its counterpart, with the paste box
    # unfolding underneath the row when it is clicked.
    export_action_row = widgets.HBox(
        [export_result_btn, copy_json_btn, paste_json_btn],
        layout=widgets.Layout(gap="8px", flex_flow="row wrap", margin="6px 0 0 0"),
    )


    # --- Cards (layout only) ---
    spacer_after_export = widgets.HTML("<div style='height:6px;'></div>")
    spacer_between_cards = widgets.HTML("<div style='height:12px;'></div>")

    builder_panel = widgets.VBox(
        [basic_acc, advanced_acc, source_acc, spacer_after_export, action_row],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    builder_panel.add_class("stac2cube-card")

    result_card = widgets.VBox([result_acc], layout=widgets.Layout(width="100%"))
    result_card.add_class("stac2cube-card")

    composites_card = widgets.VBox(
        [composites_acc], layout=widgets.Layout(width="100%")
    )
    composites_card.add_class("stac2cube-card")

    viz_card = widgets.VBox([viz_acc], layout=widgets.Layout(width="100%"))
    viz_card.add_class("stac2cube-card")

    # Export Options now lives below Visualization as its own card, with the
    # Export Current Result button attached directly beneath it.
    export_card = widgets.VBox(
        [export_acc, export_action_row, paste_json_box],
        layout=widgets.Layout(width="100%"),
    )
    export_card.add_class("stac2cube-card")

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
            composites_card,   # composites reduce over the dates kept above

            spacer_between_cards,
            viz_card,          # ✅ Visualization moved above Status

            spacer_between_cards,
            export_card,       # ✅ Export Options + Export Current Result

            spacer_between_cards,
            status_card,
        ],
        layout=widgets.Layout(
            width="100%",
            max_width=FORM_MAX_WIDTH,
            margin="0 auto",
            gap="8px",
        ),
    )
    ui.add_class("stac2cube-root")

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
            "source": source_w,
            "resolution": resolution_w,
            "crs_detected": crs_detected_w,
            "crs_user": crs_user_w,
            "crs_search_btn": crs_search_btn,
            "crs_status": crs_status_w,
            "polygon": polygon_w,
            "browse_polygon_btn": browse_polygon_btn,
            "daterange_mode": daterange_mode_w,
            "daterange": daterange_w,
            # The simple From/To pickers and the advanced-mode toggle, mirroring
            # what the editor already exposes for its update dates.
            "advanced_dates": advanced_dates_w,
            "date_from": date_from_w,
            "date_to": date_to_w,
            "bands": bands_w,
            "indices": indices_w,
            "index_checkboxes": _index_rows,
            "clip_raster": clip_raster_w,
            "max_cc": max_cc_w,
            "cloud_masking": cloud_masking_w,
            "keep_clouds": keep_clouds_w,
            "export_mask": export_mask_w,
            "cloud_mask_output": cloud_mask_output_w,
            "cloud_preset1": cloud_preset1_cb,
            "cloud_preset2": cloud_preset2_cb,
            "cloud_preset3": cloud_preset3_cb,
            # Temporal Composites: the two promoted checkboxes, the "More
            # composites" list ("stats") and the keep-or-drop choice.
            "stats": stats_w,
            "composite_mean": comp_mean_w,
            "composite_median": comp_median_w,
            # Custom Composites: the "Add composite" button and the container
            # whose children are the rows (each an HBox of period / from / to /
            # statistic / name / remove).
            "custom_add_btn": custom_add_btn,
            "custom_rows": custom_rows_box,
            "custom_error_note": custom_error_note,
            "keep_timeseries": keep_ts_w,
            "viz_layer": viz_layer_w,
            # Result-panel scene filters the composites reduce over.
            "result_cloud_max": result_cloud_max_w,
            "result_coverage_min": result_coverage_min_w,
            "result_dates": result_date_w,
            "scene_metadata": scene_metadata_w,
            "export_granule_metadata": export_granule_meta_w,
            "tile_handling": tile_handling_w,
            "partial_scene_handling": partial_scene_w,
            "min_scene_coverage": min_coverage_w,
            "min_footprint_coverage": skip_footprint_w,
            "aggregator": aggregator_w,
            "export_mode": export_mode_w,
            "export_target": export_target_w,
            "export_settings": export_settings_w,
            "export_vrt": export_vrt_w,
            "export_csv": export_csv_w,
            "export_compress": export_compress_w,
            "browse_output_btn": browse_output_btn,
            "generate_btn": generate_btn,
            "coreg_resize_btn": coreg_resize_btn,
            "export_result_btn": export_result_btn,
            "copy_json_btn": copy_json_btn,
            "paste_json_btn": paste_json_btn,
            "paste_json_area": paste_json_area_w,
            "viz_dropdown_btn": viz_dropdown_btn,
            "viz_renderer": viz_renderer_w,
            "viz_resolution": viz_resolution_w,
            "gif_section": gif_section_w,
            "gif_display_mode": gif_display_mode_w,
            "gif_band": gif_band_dd,
            "gif_r": gif_r_dd,
            "gif_g": gif_g_dd,
            "gif_b": gif_b_dd,
            "gif_stretch": gif_stretch_w,
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
            "animation": anim_out,
        },
    }





def datacube_editor():
    """
    Data Cube Editor GUI
    --------------------
    - Load NetCDF (.nc) or Zarr (.zarr) data cube
    - Work on a current in-memory result (starts with Time_Series)
    - Slice by time and band (chained)
    - Filter by cloud coverage using existing cloud_percentage coord (chained)
    - Clip raster (vector file or bbox list; applied via Edit button)
    - Reproject to another CRS via stac2cube.reproject_stac (applied via Edit button)
    - Temporal composites (stats) via stac2cube.calculate_statistics (applied via Edit button)
    - Visualize (interactive dropdown + GIF generation)
    - Export current result (NetCDF / Zarr / COGs)
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
        "scene_coverage_filter": """
        <b>filter by scene coverage</b><br>
        Uses the <code>scene_coverage</code> coordinate stored in the data cube
        (the fraction 0-100% of the AOI each scene actually images).<br><br>
        Keeps only time steps where <code>scene_coverage &gt;= min coverage</code>,
        dropping scenes that image only part of the area - across-track / swath
        edge, or a faulty / partially-missing acquisition.<br><br>
        Best used before clipping and before temporal composites.
        """,
        "clip_raster": """
        <b>clip raster</b><br>
        <b>1) Path to polygon</b><br>
        Polygon formats: <code>gpkg</code>, <code>geojson</code>, <code>kml</code>, <code>kmz</code>, <code>shp</code>.<br>
        Polygons can be geographic (WGS84) or projected (e.g., UTM).<br>
        <b>2) List of BBOX</b><br>
        Can also be a WGS84 bbox list: <code>[xmin, ymin, xmax, ymax]</code> (not projected coords). Useful tool: <code>http://bboxfinder.com/</code>
        """,
        "reproject": """
        <b>reproject data cube</b><br>
        Warps the cube into another projection with
        <code>reproject_stac()</code>.<br><br>
        <b>Target CRS</b>: an EPSG code (e.g. <code>EPSG:3035</code>), a WKT or a
        PROJ string. It must be a projected, metre-based CRS - the same rule the
        Data Cube Builder applies, because pixel sizes here are metres.<br><br>
        <b>Pixel size</b>: leave empty to keep roughly the current pixel size, or
        type a number in metres (e.g. <code>20</code>).<br><br>
        <b>Resampling</b>: <code>nearest</code> copies pixel values unchanged and
        is the safe default. <code>bilinear</code> / <code>cubic</code> /
        <code>average</code> compute new values and give a smoother image.
        Class layers (<code>scl</code>, QA, <code>cloud_mask_*</code>) always use
        <code>nearest</code>, whatever is selected - averaging class codes would
        produce classes that do not exist.<br><br>
        <b>What it costs</b>: reprojection resamples, so pixel values and the
        pixel grid both change and the step cannot be undone exactly. Reproject
        once, from the cube in its original projection. The result is the
        upright bounding box of the rotated cube, so the corners are empty
        (NaN). <code>cloud_percentage</code> and <code>scene_coverage</code> are
        kept as they were measured before the warp; they are not recalculated on
        the new grid. The whole cube is read into memory for this step.
        """,
        "spectral_indices": """
        <b>calculate spectral indices</b><br>
        Computes the selected spectral indices from the spectral bands already present in
        the loaded data cube and appends them as new bands.<br><br>
        The available indices depend on the cube's mission. Each index needs specific bands
        (e.g. <code>ndvi</code> needs <code>red</code> and <code>nir</code>). If a required band
        is missing from the cube, the Status box will report which band(s) are missing.<br><br>
        Indices that are already present in the cube are skipped (not recomputed).
        """,
        "stats": """
        <b>Temporal Composites</b><br>
        Each composite reduces the time axis into one image per band / index,
        added as its own layer next to the time series.<br><br>
        Examples:
        <ul style="margin:4px 0 0 18px; padding:0;">
            <li><code>mean_timeseries</code> -> mean of all time steps</li>
            <li><code>mean_monthly</code> -> mean of each month</li>
            <li><code>mean_annual</code> -> mean of each year</li>
        </ul>
        Untick <b>Keep the full time series</b> to keep only the composites -
        the "just give me the median" case.
        """,
        "custom_composites": """
        <b>Custom Composites</b><br>
        A period you define yourself, instead of a whole month or year.<br><br>
        <b>Every year</b> repeats the period in each year the cube covers. A
        spring of <code>04-01</code> to <code>06-21</code> named
        <code>spring_mean</code> gives <code>spring_mean_2024</code>,
        <code>spring_mean_2025</code>, and so on - one image per year.<br><br>
        <b>Single window</b> uses full dates and gives one image, named exactly
        as you type it.<br><br>
        Both dates are included. A period that starts later than it ends
        (<code>12-01</code> to <code>02-28</code>) runs over New Year and is
        named after the year it starts in.<br><br>
        Composites are calculated from the dates of the current result, after
        any filter or edit already applied - no new scenes are downloaded. A
        year with no scene in the period is skipped.<br><br>
        Cloudy pixels that were masked out are left out of the statistic, so a
        pixel can be built from fewer dates than the period contains, and a
        pixel masked on every date stays empty.
        """,
        "export_mode": """
        <b>Which format should I pick?</b><br>
        <b>NetCDF (.nc)</b> - one single file. Works with every Analysis Ready Data
        (ARD) cube tool here and is easy to copy or share. A solid all-round default.<br><br>
        <b>Zarr (.zarr)</b> - a chunked folder. Also works with every ARD cube tool,
        and is the quickest to read and write for very large cubes. To share it, zip
        the folder first.<br><br>
        <b>Cloud Optimized GeoTIFFs</b> - a folder with one GeoTIFF per date. The
        bands are already mapped, so you can drag them straight into QGIS (or another
        GIS) and view them right away. NOT accepted by the ARD cube tools. Best when
        your goal is viewing or sharing individual scenes.
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
        <b>season mode</b><br>
        Repeat a seasonal window (<code>MM-DD</code> to <code>MM-DD</code>) across years:<br><br>
        <b>1) All available years</b>: <code>{"season": ["MM-DD", "MM-DD"], "years": "all"}</code><br>
        <b>2) Year range</b>: <code>{"season": ["MM-DD", "MM-DD"], "years": "2019-2024"}</code><br>
        <b>3) Selected years only</b>: <code>{"season": ["MM-DD", "MM-DD"], "years": [2019, 2021]}</code><br><br>
        The text box below is prefilled with an editable example for the selected mode.
        """,
        "update_cube": """
        <b>update data cube</b><br>
        Uses <code>get_stac_layers(update=...)</code> with the loaded cube path to fetch only the
        missing dates and/or missing bands and return an updated <code>Time_Series</code>.
        The cube's cloud/shadow masking strategy is restored from its attributes, so new scenes and
        new bands are masked (or kept) exactly like the stored data.<br><br>
        <b>Important:</b> This replaces the current working result with the updated cube.<br>
        Use it first (or by itself), then continue with other editing features (slice, clip, stats, export).
        """,
        "mosaic": """
        <b>mosaic data cubes</b><br>
        Joins several cubes side by side into one covering their combined area -
        for putting back together an area that was built in parts.<br><br>
        No new data is downloaded and nothing is recalculated: the result holds
        the pixels your cubes already have, placed on one grid.<br><br>
        The cubes need not overlap. Where none reaches, the mosaic is empty.
        Where more than one does, <b>Where cubes overlap</b> decides.
        """,
        "mosaic_overlap": """
        <b>where cubes overlap</b><br>
        What to keep for a pixel more than one cube holds.<br><br>
        <b>First cube in the list</b> (default) keeps that cube's value - the
        only option that never creates a number none of your cubes held.<br><br>
        <b>Average / Middle / Lowest / Highest</b> combine the cubes covering the
        pixel. Smoother seams, but a new number.<br><br>
        Often the choice makes no difference: cubes built from the same scenes
        hold identical pixels where they meet. The result box reports how much
        yours actually disagree.
        """,
        "mosaic_crs": """
        <b>output projection</b><br>
        A mosaic is one grid, so it has one projection. Cubes in another one are
        warped into it, which resamples them.<br><br>
        <b>Automatic</b> picks the projection covering the largest area, so the
        fewest pixels are warped. The dropdown lists the projections your cubes
        are in.<br><br>
        <b>User-defined</b> takes any projected, metre-based CRS (e.g.
        <code>EPSG:3035</code>). One that none of your cubes uses means every
        cube is warped, so it needs <b>Allow resampling</b> under Advanced.
        """,
        "mosaic_time": """
        <b>dates</b><br>
        Neighbouring areas do not always share acquisition dates - a scene can
        cover one and miss the other.<br><br>
        <b>Keep every date</b> (default) keeps the dates from all cubes; on a
        date a cube lacks, that cube's area is empty.<br><br>
        <b>Only dates all cubes have</b> keeps a date only if every cube has it -
        complete everywhere, but fewer dates.
        """,
        "mosaic_bands": """
        <b>bands</b><br>
        <b>Only bands all cubes have</b> (default) keeps the bands common to
        every cube.<br><br>
        <b>All bands</b> keeps every band in any cube; a cube lacking one is
        empty for that band.
        """,
        "mosaic_layers": """
        <b>layers to merge</b><br>
        The layers found in <u>every</u> selected cube - the time series and any
        temporal composites saved with it. All are selected by default; a layer
        only some cubes have cannot be merged and is not listed.<br><br>
        <b>On composites:</b> merging saved composites is not the same as
        computing one from the merged cube, because each piece's median was
        taken over its own dates. If your cubes still hold their time series,
        merge that and compute the composite afterwards.
        """,
        "mosaic_resampling": """
        <b>resampling method</b><br>
        How pixel values are recalculated for a cube that has to be warped onto
        the mosaic grid. Only used when <b>Allow resampling</b> is ticked.<br><br>
        <code>nearest</code> copies each value unchanged and is the safe
        default. <code>bilinear</code>, <code>cubic</code> and
        <code>average</code> compute new values from the neighbours: smoother
        to look at, but the numbers are no longer the ones the cube held.
        """,
        "mosaic_pixel_size": """
        <b>pixel size</b><br>
        The ground size of one pixel in the mosaic, in metres.<br><br>
        Leave it empty to use the finest pixel size among your cubes - nothing
        is lost, and any coarser cube is enlarged to match. A larger number
        makes a smaller, coarser result and needs <b>Allow resampling</b>.
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
        - If Dataset: prefer 'Time_Series', otherwise first data var
        """
        if isinstance(obj, xr.DataArray):
            return obj

        if isinstance(obj, xr.Dataset):
            if "Time_Series" in obj.data_vars:
                return obj["Time_Series"]
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
        - Dataset containing 'Time_Series'
        """
        if isinstance(obj, xr.DataArray):
            return obj

        if isinstance(obj, xr.Dataset):
            if "Time_Series" in obj.data_vars:
                return obj["Time_Series"]
            raise ValueError(
                "Current result is a Dataset but does not contain 'Time_Series'."
            )

        raise TypeError(f"Unsupported object type for stats: {type(obj)}")

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

    def _gif_mode_token():
        """Filename token describing the current animation rendering choice."""
        sec = gif_section_w.value
        if sec == "band":
            b = str(gif_band_dd.value or "band")
            return re.sub(r"[^A-Za-z0-9._-]+", "_", f"band_{b}")
        if sec == "custom":
            return "customRGB"
        return (gif_display_mode_w.value or "rgb").strip()

    def _auto_gif_output_suggestion():
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", _loaded_stem_default()).strip("._-") or "cube"
        return f"./animations/{stem}_{_gif_mode_token()}.gif"

    def _refresh_gif_band_options():
        """Populate the animation band selectors from the working cube's bands."""
        obj = state.get("current")
        if obj is None:
            return
        try:
            da = _pick_dataarray_for_visualization(obj)
        except Exception:
            return

        if "band" in da.dims:
            bands = [str(b) for b in da.coords["band"].values]
        else:
            bands = [str(da.name) if da.name is not None else "layer"]
        lower = [b.lower() for b in bands]

        def _default(name, idx):
            if name in lower:
                return bands[lower.index(name)]
            return bands[min(idx, len(bands) - 1)]

        for dd, default in (
            (gif_band_dd, bands[0]),
            (gif_r_dd, _default("red", 0)),
            (gif_g_dd, _default("green", 1)),
            (gif_b_dd, _default("blue", 2)),
        ):
            old = dd.value
            dd.options = bands
            dd.value = old if old in bands else default

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

        # Show the zlib option only in NetCDF mode.
        _apply_editor_compress_visibility()

        if mode == "netcdf":
            export_target_w.disabled = False
            browse_export_btn.disabled = False
            export_target_w.description = "Export file:"
            export_target_w.placeholder = "./results/cube_edited.nc"

            if current in ["./results/cogs", "results/cogs", r"results\cogs"]:
                export_target_w.value = ""
            elif current.lower().endswith(".zarr"):
                # Switching zarr -> netcdf: keep the name, swap the extension
                # (mirrors the Data Cube Builder).
                export_target_w.value = f"{os.path.splitext(current)[0]}.nc"

            if not export_target_w.value:
                export_target_w.value = _auto_netcdf_export_suggestion()

            _sync_export_filechooser_from_mode_and_text()

        elif mode == "zarr":
            export_target_w.disabled = False
            browse_export_btn.disabled = False
            export_target_w.description = "Export store:"
            export_target_w.placeholder = "./results/cube_edited.zarr"

            # Switching netcdf -> zarr keeps the same name with the extension
            # swapped; a leftover COGs folder path is replaced by the default
            # suggestion (mirrors the Data Cube Builder).
            if current in ["./results/cogs", "results/cogs", r"results\cogs"]:
                export_target_w.value = ""
            elif current.lower().endswith(".nc"):
                export_target_w.value = f"{os.path.splitext(current)[0]}.zarr"

            if not export_target_w.value:
                base = _auto_netcdf_export_suggestion()
                export_target_w.value = f"{os.path.splitext(base)[0]}.zarr"

            _sync_export_filechooser_from_mode_and_text()

        elif mode == "cogs":
            export_target_w.disabled = False
            browse_export_btn.disabled = False
            export_target_w.description = "Export dir:"
            export_target_w.placeholder = "./results/cogs"

            # Switching from NetCDF/Zarr: a ".nc"/".zarr" path is meaningless as
            # a COGs output directory (it would create a folder literally named
            # "*.nc"), so drop the leftover path and offer the COGs default.
            if current.lower().endswith((".nc", ".zarr")):
                export_target_w.value = ""

            if not export_target_w.value:
                export_target_w.value = "./results/cogs"

            _sync_export_filechooser_from_mode_and_text()

    def _is_derived_path(path):
        """True when the working cube was produced in-session and has no file of
        its own on disk.

        A mosaic is handed to the ordinary load path under a placeholder name
        ('<mosaic of N cubes>', see _on_mosaic_run_clicked), so state['loaded_path']
        is a label rather than a path. The two archive actions - Update Data Cube
        and Build Cloud Mask Cube - re-open that path to recover the cube's build
        parameters, so neither can run on such a cube: they have to be pointed at
        an exported file instead.
        """
        s = str(path or "").strip()
        return s.startswith("<") and s.endswith(">")

    def _archive_note(action):
        return (
            "<div style='font-size:12px; color:#9a3412; background:#fff7ed; "
            "border:1px solid #fed7aa; border-radius:6px; padding:8px 10px;'>"
            f"ℹ️ {action} needs the cube's own file, to read the build parameters "
            "stored in it. This cube was mosaicked in this session and has no "
            "file yet. Export it below, then load the exported file to use this "
            "tool.</div>"
        )

    def _sync_archive_actions():
        """Gate the two archive-backed actions on the working cube having a file.

        Runs after the plain enabled/disabled pass in _set_editor_enabled, so it
        can only ever tighten what that pass allowed.
        """
        derived = _is_derived_path(state.get("loaded_path"))

        if derived:
            update_run_btn.disabled = True
            build_mask_btn.disabled = True
            build_mask_out_w.disabled = True
            browse_build_mask_btn.disabled = True

        update_archive_note_html.value = (
            _archive_note("Update Data Cube") if derived else ""
        )
        update_archive_note_html.layout.display = "" if derived else "none"
        build_mask_archive_note_html.value = (
            _archive_note("Building a binary cloud mask") if derived else ""
        )
        build_mask_archive_note_html.layout.display = "" if derived else "none"

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

        # Scene coverage filter
        enable_coverage_filter_w.disabled = not enabled
        coverage_min_w.disabled = not enabled

        # Clip widgets
        enable_clip_w.disabled = not enabled
        clip_geom_w.disabled = not enabled
        browse_clip_btn.disabled = (not enabled) or (not filechooser_available)

        # Reproject widgets
        enable_reproject_w.disabled = not enabled
        reproject_crs_w.disabled = not enabled
        reproject_res_w.disabled = not enabled
        reproject_resampling_w.disabled = not enabled

        # Mask clouds with binary file
        enable_mask_clouds_w.disabled = not enabled
        mask_file_w.disabled = not enabled
        browse_mask_file_btn.disabled = (not enabled) or (not filechooser_available)

        # Build Cloud Mask Cube (standalone export)
        build_mask_out_w.disabled = not enabled
        build_mask_btn.disabled = not enabled
        browse_build_mask_btn.disabled = (not enabled) or (not filechooser_available)

        # Generate CSV Report (standalone export). Deliberately absent from
        # _sync_archive_actions: it reads the cube in hand, not the archive, so
        # a mosaic with no file of its own can still be reported on.
        csv_report_out_w.disabled = not enabled
        csv_report_btn.disabled = not enabled
        browse_csv_report_btn.disabled = (not enabled) or (not filechooser_available)

        # Mosaic Data Cubes is deliberately NOT gated here. Every other feature
        # edits the loaded cube, so it needs one; this one PRODUCES a cube from
        # files on disk, and is the natural starting point when the pieces of an
        # area have to be joined before anything can be edited. Disabling it
        # until a cube is loaded would mean loading a cube just to be allowed to
        # replace it. Its own Run button is enabled by the list length instead
        # (see _sync_mosaic_controls).

        # Spectral indices widgets
        indices_select_w.disabled = not enabled
        indices_all_btn.disabled = not enabled
        indices_clear_btn.disabled = not enabled

        # Temporal Composites widgets
        stats_select_w.disabled = not enabled
        stats_all_btn.disabled = not enabled
        stats_clear_btn.disabled = not enabled
        comp_mean_w.disabled = not enabled
        comp_median_w.disabled = not enabled
        # Custom Composites rows follow the same gate (they read
        # stats_select_w.disabled, so this has to come after it is set).
        _custom_sync_enabled()
        # "Keep the full time series" additionally needs a selected composite.
        _sync_keep_timeseries()

        # Update widgets
        update_run_btn.disabled = not enabled
        update_date_from_w.disabled = not enabled
        update_date_to_w.disabled = not enabled
        update_advanced_dates_w.disabled = not enabled
        update_daterange_mode_w.disabled = not enabled
        update_daterange_w.disabled = not enabled
        # The band list stays disabled when the loaded cube has no missing
        # bands to offer (the note below the list says why).
        update_bands_w.disabled = (not enabled) or (len(update_bands_w.options) == 0)

        # Visualization
        viz_dropdown_btn.disabled = not enabled
        gif_section_w.disabled = not enabled
        gif_display_mode_w.disabled = not enabled
        gif_band_dd.disabled = not enabled
        gif_r_dd.disabled = not enabled
        gif_g_dd.disabled = not enabled
        gif_b_dd.disabled = not enabled
        gif_stretch_w.disabled = not enabled
        gif_fps_w.disabled = not enabled
        gif_label_w.disabled = not enabled
        gif_out_path_w.disabled = not enabled
        viz_make_gif_btn.disabled = not enabled
        browse_gif_btn.disabled = (not enabled) or (not filechooser_available)

        # Export widgets. Export mode is NetCDF / COGs only (the old "Quick
        # Result, no Export" lazy option is gone): the loaded cube is already a
        # real, exported cube and editing just shows the result in the Result
        # panel, so not exporting simply means the edits aren't saved. Export is a
        # deliberate click on the button below, never triggered by an edit.
        export_mode_w.disabled = not enabled
        aggregator_w.disabled = not enabled
        export_compress_w.disabled = not enabled
        export_vrt_w.disabled = not enabled
        if enabled:
            _set_export_mode_defaults()
        else:
            export_target_w.disabled = True
            export_target_w.value = ""
            browse_export_btn.disabled = True
            export_current_btn.disabled = True
            if filechooser_available:
                export_fc_box.layout.display = "none"
            with viz_out:
                clear_output()
                print("ℹ️ Load a cube first to activate visualization tools.")
            with anim_out:
                clear_output()

        # Last: may only tighten the states set above.
        _sync_archive_actions()
        _sync_edit_button()

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
        # Options changed, so the "is this a strict subset?" answer may have
        # changed even where .value did not and no observer fired.
        _sync_edit_button()

    # ---------------------------------------------------------------------
    # Spectral indices helpers
    # ---------------------------------------------------------------------
    _INDEX_FULLNAMES = {
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
        "vh/vv": "VH/VV Ratio",
        "vv/vh": "VV/VH Ratio",
        "rvi": "Radar Vegetation Index",
    }

    def _current_mission():
        """Best-effort mission name for the loaded cube (from xarray attrs)."""
        for obj in (state.get("current"), state.get("loaded_original")):
            try:
                m = obj.attrs.get("mission") if obj is not None else None
            except Exception:
                m = None
            if m:
                return str(m)
        return None

    def _allowed_indices_for_mission(mission_name):
        """Indices offered by missions() for this mission ([] if none/unknown)."""
        if not mission_name:
            return []
        try:
            df = missions()
            row = df.loc[df["name"] == mission_name]
            if row.empty:
                return []
            vals = row.iloc[0]["indices"]
        except Exception:
            return []
        if not vals:  # False or None
            return []
        return [str(v) for v in vals]

    def _index_options_with_fullname(index_list):
        options = []
        for idx in index_list:
            full = _INDEX_FULLNAMES.get(str(idx))
            label = f"{idx} ({full})" if full else str(idx)
            options.append((label, str(idx)))
        return options

    def _populate_indices_widget_from_current():
        mission_name = _current_mission()
        allowed = _allowed_indices_for_mission(mission_name)
        indices_select_w.options = _index_options_with_fullname(allowed)
        indices_select_w.value = ()

    def _allowed_bands_for_mission(mission_name):
        """Bands offered by missions() for this mission ([] if none/unknown)."""
        if not mission_name:
            return []
        try:
            df = missions()
            row = df.loc[df["name"] == mission_name]
            if row.empty:
                return []
            vals = row.iloc[0]["bands"]
        except Exception:
            return []
        if not vals:  # False or None
            return []
        return [str(v) for v in vals]

    def _populate_update_bands_from_current():
        """Fill the Band Update list with the bands the loaded cube's mission
        offers but the cube does not contain yet."""
        update_bands_w.options = []
        update_bands_w.value = ()

        da = state.get("loaded_original")
        if da is None:
            update_bands_note_html.value = ""
            return
        if state.get("loaded_var") == "Cloud_Stack":
            update_bands_note_html.value = (
                "<div style='font-size:12px; color:#6b7280;'>Band update is "
                "available for Time_Series cubes only.</div>"
            )
            return

        mission_name = _current_mission()
        available = _allowed_bands_for_mission(mission_name)
        try:
            stored = [
                str(b).lower()
                for b in np.asarray(da.attrs.get("spectral_bands", [])).ravel()
            ]
        except Exception:
            stored = []
        if not stored and "band" in da.coords:
            # Legacy cubes without the attr: fall back on the band coordinate
            # (indices land in the stored list too, which is fine - they are
            # not mission bands and never appear in `available`).
            stored = [str(b).lower() for b in da.band.values]

        if not available:
            update_bands_note_html.value = (
                "<div style='font-size:12px; color:#6b7280;'>No band table "
                f"available for mission '{mission_name}'.</div>"
            )
            return

        missing = [b for b in available if str(b).lower() not in stored]
        if not missing:
            update_bands_note_html.value = (
                "<div style='font-size:12px; color:#6b7280;'>The cube already "
                "contains every band its mission offers.</div>"
            )
            return

        update_bands_w.options = _band_options_with_resolution(mission_name, missing)
        update_bands_note_html.value = (
            f"<div style='font-size:12px; color:#6b7280;'>{len(missing)} band"
            f"{'s' if len(missing) != 1 else ''} available to add. Leave the "
            "selection empty for a date-only update.</div>"
        )

    def _update_gif_output_suggestion(force=False):
        new_suggestion = _auto_gif_output_suggestion()
        current = (gif_out_path_w.value or "").strip()
        prev_auto = state.get("last_auto_gif_suggestion")
        gif_out_path_w.placeholder = new_suggestion

        should_replace = force or (current == "") or (prev_auto is not None and current == prev_auto)
        if should_replace:
            gif_out_path_w.value = new_suggestion

        state["last_auto_gif_suggestion"] = new_suggestion

    def _show_result_current():
        """Render the Result panel from the current working cube.

        Temporal Composites are a chained Edit here (like every other editor
        feature), not an export-time view, so what the Result shows IS what
        export writes - no second collapse anywhere."""
        obj = state.get("current")
        if obj is None:
            return
        _show_preview(result_out, obj)

    def _export_current_result():
        if state["current"] is None:
            raise ValueError("No current result available. Load a cube first.")

        mode = export_mode_w.value
        target = None if export_target_w.disabled else ((export_target_w.value or "").strip() or None)

        if not target:
            raise ValueError("Please provide an export file/folder path.")

        # Composites are already baked into the working cube by the Edit button,
        # so export writes exactly what the Result panel shows.
        obj = state["current"]
        if not isinstance(obj, (xr.DataArray, xr.Dataset)):
            raise TypeError(f"Unsupported result type for export: {type(obj)}")

        if mode in ("netcdf", "zarr"):
            # Same export path for both: export_stac picks the container from
            # the extension (.zarr -> Zarr store, else NetCDF). zlib applies to
            # NetCDF only (Zarr always uses its own default codec).
            want_ext = ".zarr" if mode == "zarr" else ".nc"
            if not target.lower().endswith(want_ext):
                target = f"{os.path.splitext(target)[0]}{want_ext}"
                export_target_w.value = target

            Path(target).parent.mkdir(parents=True, exist_ok=True)
            compress = bool(export_compress_w.value) if mode == "netcdf" else False
            want_vrt = bool(export_vrt_w.value) if mode == "netcdf" else False

            if isinstance(obj, xr.DataArray):
                export_stac(
                    stac=obj,
                    output=target,
                    var_name=(obj.name or "Time_Series"),
                    compress=compress,
                    vrt=want_vrt,
                )
                return {"mode": mode, "target": target}

            # Dataset export (e.g. after calculate_statistics)
            crs_ref, transform_ref = _get_reference_crs_transform_from_loaded()
            export_stac(
                stac=obj,
                output=target,
                crs=crs_ref,
                transform=transform_ref,
                compress=compress,
                vrt=want_vrt,
            )
            return {"mode": mode, "target": target}

        elif mode == "cogs":
            Path(target).mkdir(parents=True, exist_ok=True)
            export_to_cogs(stac=obj, output_dir=target, prefix="", dtype="float32")
            return {"mode": "cogs", "target": target}

        else:
            raise ValueError(f"Unsupported export mode: {mode}")

    # ---------------------------------------------------------------------
    # Question mark help UI helpers
    # ---------------------------------------------------------------------
    def _stacked_field_with_help(widget, label_text, help_key):
        return _field_with_help(widget, label_text, HELP_HTML.get(help_key, ""))

    # ---------------------------------------------------------------------
    # File chooser helpers (optional)
    # ---------------------------------------------------------------------
    filechooser_available = FileChooser is not None

    load_fc = None
    export_fc = None
    gif_fc = None
    clip_fc = None
    mask_file_fc = None
    build_mask_fc = None
    csv_report_fc = None

    load_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))
    export_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))
    gif_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))
    clip_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))
    mask_file_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))
    build_mask_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))
    csv_report_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))

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

        elif mode == "zarr":
            suggestion = current or _auto_netcdf_export_suggestion()
            start_dir = _existing_dir_or_parent(suggestion)
            suggested_name = Path(suggestion).name or "cube_edited.zarr"
            if not suggested_name.lower().endswith(".zarr"):
                suggested_name = f"{Path(suggested_name).stem}.zarr"

            try:
                export_fc.reset(path=start_dir, filename=suggested_name)
            except Exception:
                try:
                    export_fc.default_path = start_dir
                    export_fc.default_filename = suggested_name
                except Exception:
                    pass

            export_fc.title = "Select Zarr export store (name ending in .zarr)"
            export_fc.show_only_dirs = False
            export_fc.filter_pattern = ["*.zarr", "*"]

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

    def _sync_mask_file_filechooser_from_text():
        if not filechooser_available or mask_file_fc is None:
            return
        current = (mask_file_w.value or "").strip()
        start_dir = _existing_dir_or_parent(current)
        suggested_name = Path(current).name if current else ""
        try:
            mask_file_fc.reset(path=start_dir, filename=suggested_name)
        except Exception:
            try:
                mask_file_fc.default_path = start_dir
                mask_file_fc.default_filename = suggested_name
            except Exception:
                pass

    def _sync_build_mask_filechooser_from_text():
        if not filechooser_available or build_mask_fc is None:
            return
        current = (build_mask_out_w.value or "").strip()
        start_dir = _existing_dir_or_parent(current)
        suggested_name = Path(current).name if current else ""
        try:
            build_mask_fc.reset(path=start_dir, filename=suggested_name)
        except Exception:
            try:
                build_mask_fc.default_path = start_dir
                build_mask_fc.default_filename = suggested_name
            except Exception:
                pass

    def _sync_csv_report_filechooser_from_text():
        if not filechooser_available or csv_report_fc is None:
            return
        current = (csv_report_out_w.value or "").strip()
        start_dir = _existing_dir_or_parent(current)
        suggested_name = Path(current).name if current else ""
        try:
            csv_report_fc.reset(path=start_dir, filename=suggested_name)
        except Exception:
            try:
                csv_report_fc.default_path = start_dir
                csv_report_fc.default_filename = suggested_name
            except Exception:
                pass

    if filechooser_available:
        try:
            load_fc = _CubeFileChooser(
                path=str(Path(".").resolve()),
                filename="",
                title="Select cube (.nc file or .zarr store)",
                show_only_dirs=False,
                select_default=False,
            )
            # "*" also lets a .zarr store be clicked directly (see
            # _CubeFileChooser) or a file inside one be picked - the load
            # handler resolves either to the store root.
            load_fc.filter_pattern = ["*.nc", "*.zarr", "*"]
            load_fc.use_dir_icons = True
            load_fc_box = widgets.VBox([load_fc], layout=widgets.Layout(display="none", width="100%"))

            export_fc = _CubeFileChooser(
                path=str(Path(".").resolve()),
                filename="",
                title="Select export output",
                show_only_dirs=False,
                select_default=False,
            )
            export_fc.use_dir_icons = True
            export_fc_box = widgets.VBox([export_fc], layout=widgets.Layout(display="none", width="100%"))

            gif_fc = _CubeFileChooser(
                path=str(Path(".").resolve()),
                filename="",
                title="Select animation output folder",
                show_only_dirs=True,
                select_default=False,
            )
            gif_fc.use_dir_icons = True
            gif_fc_box = widgets.VBox([gif_fc], layout=widgets.Layout(display="none", width="100%"))

            clip_fc = _CubeFileChooser(
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

            mask_file_fc = _CubeFileChooser(
                path=str(Path(".").resolve()),
                filename="",
                title="Select binary cloud-mask cube (.nc file or .zarr store)",
                show_only_dirs=False,
                select_default=False,
            )
            mask_file_fc.use_dir_icons = True
            try:
                mask_file_fc.filter_pattern = ["*.nc", "*.zarr", "*"]
            except Exception:
                pass
            mask_file_fc_box = widgets.VBox([mask_file_fc], layout=widgets.Layout(display="none", width="100%"))

            build_mask_fc = _CubeFileChooser(
                path=str(Path(".").resolve()),
                filename="",
                title="Select output for the binary cloud mask (.nc)",
                show_only_dirs=False,
                select_default=False,
            )
            build_mask_fc.use_dir_icons = True
            try:
                build_mask_fc.filter_pattern = ["*.nc", "*.zarr", "*"]
            except Exception:
                pass
            build_mask_fc_box = widgets.VBox([build_mask_fc], layout=widgets.Layout(display="none", width="100%"))

            csv_report_fc = _CubeFileChooser(
                path=str(Path(".").resolve()),
                filename="",
                title="Select output for the statistics report (.csv)",
                show_only_dirs=False,
                select_default=False,
            )
            csv_report_fc.use_dir_icons = True
            try:
                csv_report_fc.filter_pattern = ["*.csv", "*"]
            except Exception:
                pass
            csv_report_fc_box = widgets.VBox([csv_report_fc], layout=widgets.Layout(display="none", width="100%"))

            mosaic_fc = _CubeFileChooser(
                path=str(Path(".").resolve()),
                filename="",
                title="Select a cube to add (.nc file or .zarr store)",
                show_only_dirs=False,
                select_default=False,
            )
            mosaic_fc.use_dir_icons = True
            try:
                mosaic_fc.filter_pattern = ["*.nc", "*.zarr", "*"]
            except Exception:
                pass
            mosaic_fc_box = widgets.VBox([mosaic_fc], layout=widgets.Layout(display="none", width="100%"))

        except Exception:
            filechooser_available = False
            load_fc = export_fc = gif_fc = clip_fc = mask_file_fc = build_mask_fc = None
            csv_report_fc = None
            mosaic_fc = None
            mosaic_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))
            load_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))
            export_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))
            gif_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))
            clip_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))
            mask_file_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))
            build_mask_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))
            csv_report_fc_box = widgets.VBox([], layout=widgets.Layout(display="none", width="100%"))

    # ---------------------------------------------------------------------
    # Widgets
    # ---------------------------------------------------------------------
    # Loading
    load_path_w = widgets.Text(
        value="./results/test.nc",
        description="",
        placeholder="./results/test.nc or ./results/test.zarr",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "0px"},
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

    # Layer selection (shown only when the loaded NetCDF has multiple layers,
    # e.g. a time series exported together with temporal composites/stats)
    layer_select_w = widgets.Dropdown(
        options=[],
        value=None,
        layout=widgets.Layout(width="100%"),
    )
    layer_load_btn = widgets.Button(
        description="Load selected layer",
        icon="check",
        button_style="info",
        layout=widgets.Layout(width="180px"),
    )
    layer_select_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "This cube contains <b>multiple layers</b>. Select the layer "
                "you want to work on, then click <b>Load selected layer</b>. "
                "You can come back and load a different layer of the same file "
                "at any time."
                "</div>"
            ),
            _stacked_field(layer_select_w, "Layer"),
            layer_load_btn,
        ],
        layout=widgets.Layout(width="100%", gap="6px", display="none"),
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

    # Scene coverage filter feature (applied via Edit button). Same shape as the
    # cloud filter, but keyed on the scene_coverage coord: drops scenes imaging
    # less than the given % of the AOI (swath / orbit edge, or faulty / missing
    # acquisitions). Reuses the across-track coverage code (compute_scene_coverage).
    enable_coverage_filter_w = widgets.Checkbox(
        value=False,
        description="Enable filter",
        indent=False,
        layout=widgets.Layout(width="140px"),
        disabled=True,
    )

    coverage_min_w = widgets.BoundedIntText(
        value=90,
        min=0,
        max=100,
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

    # Reproject feature (applied via Edit button). Warps the working cube into
    # another CRS via stac2cube.reproject_stac. Text (not FloatText) for the
    # pixel size so "empty" is expressible - that is what keeps the current
    # resolution. Validation happens on commit (continuous_update=False), never
    # per keystroke.
    enable_reproject_w = widgets.Checkbox(
        value=False,
        description="Enable reprojection",
        indent=False,
        layout=widgets.Layout(width="180px"),
        disabled=True,
    )

    reproject_crs_w = widgets.Text(
        value="",
        description="",
        placeholder="EPSG:3035",
        continuous_update=False,
        layout=widgets.Layout(width="80%"),
        disabled=True,
    )

    reproject_crs_status_w = widgets.HTML(value="")

    reproject_res_w = widgets.Text(
        value="",
        description="",
        placeholder="empty = keep current pixel size",
        continuous_update=False,
        layout=widgets.Layout(width="80%"),
        disabled=True,
    )

    reproject_resampling_w = widgets.Dropdown(
        options=["nearest", "bilinear", "cubic", "average", "mode"],
        value="nearest",
        description="",
        layout=widgets.Layout(width="40%"),
        disabled=True,
    )

    # Mask clouds with a binary masking file (applied via Edit button). Masks the
    # already-loaded cube out with a Cloud_Stack (1=cloud, 0=clear) NetCDF, e.g.
    # one produced by 'Export Mask as Binary File' or the ARD cloud tools.
    enable_mask_clouds_w = widgets.Checkbox(
        value=False,
        description="Enable masking",
        indent=False,
        layout=widgets.Layout(width="150px"),
        disabled=True,
    )

    mask_file_w = widgets.Text(
        value="",
        description="",
        placeholder="./results/test_mask_binary.nc",
        layout=widgets.Layout(width="80%"),
        disabled=True,
    )

    browse_mask_file_btn = widgets.Button(
        description="",
        icon="folder-open",
        tooltip="Browse binary cloud-mask file (.nc)",
        layout=widgets.Layout(width="34px", min_width="34px", height="32px", padding="0px"),
        disabled=True,
    )
    browse_mask_file_btn.style.button_color = "#f3f4f6"

    # Build Cloud Mask Cube: reconstruct the SCL binary cloud mask of the loaded
    # cube - the same file the builder writes with "Export Mask as Binary File".
    # It re-queries STAC using the cube's stored parameters (get_stac_parameters)
    # and writes a Cloud_Stack (1=cloud, 0=clear). Standalone export; it does NOT
    # change the working cube, so it is not part of the Edit chain.
    build_mask_out_w = widgets.Text(
        value="",
        placeholder="./results/<cube>_mask_binary.nc",
        layout=widgets.Layout(width="100%"),
        disabled=True,
    )
    browse_build_mask_btn = widgets.Button(
        description="",
        icon="folder-open",
        tooltip="Browse output for the binary cloud mask (.nc)",
        layout=widgets.Layout(width="34px", min_width="34px", height="32px", padding="0px"),
        disabled=True,
    )
    browse_build_mask_btn.style.button_color = "#f3f4f6"
    build_mask_btn = widgets.Button(
        description="Build Binary Cloud Mask",
        button_style="primary",
        icon="cloud",
        layout=widgets.Layout(width="230px"),
        disabled=True,
    )
    build_mask_out = widgets.Output(layout=widgets.Layout(width="99%", overflow="auto"))
    # Shown instead of the button being silently dead when the loaded cube has
    # no file of its own (see _is_derived_path).
    build_mask_archive_note_html = widgets.HTML(
        value="", layout=widgets.Layout(display="none", width="99%")
    )

    # Generate CSV Report: per-band statistics of the WORKING cube (the result
    # shown above, edits included), written as one CSV. Reads no archive and
    # needs no build parameters, so unlike the two tools above it also works on
    # a mosaic that has no file of its own.
    csv_report_out_w = widgets.Text(
        value="",
        placeholder="./results/<cube>_statistics.csv",
        layout=widgets.Layout(width="100%"),
        disabled=True,
    )
    browse_csv_report_btn = widgets.Button(
        description="",
        icon="folder-open",
        tooltip="Browse output for the statistics report (.csv)",
        layout=widgets.Layout(width="34px", min_width="34px", height="32px", padding="0px"),
        disabled=True,
    )
    browse_csv_report_btn.style.button_color = "#f3f4f6"
    csv_report_btn = widgets.Button(
        description="Generate CSV Report",
        button_style="primary",
        icon="table",
        layout=widgets.Layout(width="230px"),
        disabled=True,
    )
    csv_report_out = widgets.Output(layout=widgets.Layout(width="99%", overflow="auto"))

    # ------------------------------------------------------------------
    # Mosaic Data Cubes
    # ------------------------------------------------------------------
    # The one feature that does NOT start from a loaded cube: it takes several
    # cubes and produces one, so its widgets stay live while the editor is
    # otherwise disabled (see _set_editor_enabled). When it succeeds it hands
    # the result to the ordinary load path, so from that point the mosaic IS the
    # loaded cube and every other feature works on it unchanged.
    mosaic_path_w = widgets.Text(
        value="",
        placeholder="./results/piece_01.nc  or  ./results/  (then Add whole folder)",
        # Mandatory now that `value` is what commits a typed path: with the
        # default, `value` fires on every keystroke, so "./a.nc" would be
        # submitted as "." then "./" then "./a" - each a miss, each an error
        # line. Commits on Enter / focus loss instead.
        continuous_update=False,
        # Flex, not width:100% - see mosaic_input_row for why the 100% version
        # brings the horizontal scrollbar back.
        layout=widgets.Layout(flex="1 1 auto", width="auto", min_width="0"),
    )
    browse_mosaic_btn = widgets.Button(
        description="",
        icon="folder-open",
        tooltip="Browse for a data cube to add",
        layout=widgets.Layout(width="34px", min_width="34px", height="32px", padding="0px"),
    )
    browse_mosaic_btn.style.button_color = "#f3f4f6"
    mosaic_add_folder_btn = widgets.Button(
        description="Add whole folder",
        icon="folder-plus",
        tooltip=(
            "Add every .nc / .zarr cube in a folder at once. Uses the folder in "
            "the box above, or the one currently open in the browser."
        ),
        layout=widgets.Layout(width="170px"),
    )
    mosaic_clear_btn = widgets.Button(
        description="Clear list",
        icon="trash",
        layout=widgets.Layout(width="120px"),
    )
    # The queued cubes, rebuilt as rows so each can be removed or moved. Order
    # is the priority order for overlap="first", which is why it is editable at
    # all rather than just a text dump.
    mosaic_list_box = widgets.VBox(
        [],
        layout=widgets.Layout(
            width="100%", gap="2px", overflow="hidden", min_width="0",
        ),
    )
    mosaic_count_w = widgets.HTML("")

    mosaic_overlap_w = widgets.Dropdown(
        options=[
            ("First cube in the list (keeps original values)", "first"),
            ("Average of the cubes that cover it", "mean"),
            ("Middle value of the cubes that cover it", "median"),
            ("Lowest value", "min"),
            ("Highest value", "max"),
        ],
        value="first",
        layout=widgets.Layout(width="100%"),
    )

    _MOSAIC_CRS_AUTO = "auto"
    _MOSAIC_CRS_AUTO_LABEL = "Automatic (largest area, fewest pixels warped)"
    mosaic_crs_detected_w = widgets.Dropdown(
        options=[(_MOSAIC_CRS_AUTO_LABEL, _MOSAIC_CRS_AUTO)],
        value=_MOSAIC_CRS_AUTO,
        layout=widgets.Layout(width="100%"),
    )
    mosaic_crs_user_w = widgets.Text(
        value="",
        placeholder="EPSG:3035",
        # Same reason as the builder's CRS box: with continuous_update the value
        # fires per keystroke, so "3035" is validated as "3", "30", "303" - each
        # an incomplete code that fails. Commit on Enter / focus loss instead.
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
    )
    mosaic_crs_status_w = widgets.HTML("")

    mosaic_time_join_w = widgets.Dropdown(
        options=[
            ("Keep every date", "outer"),
            ("Only dates all cubes have", "inner"),
        ],
        value="outer",
        layout=widgets.Layout(width="100%"),
    )
    mosaic_band_join_w = widgets.Dropdown(
        options=[
            ("Only bands all cubes have", "inner"),
            ("All bands", "union"),
        ],
        value="inner",
        layout=widgets.Layout(width="100%"),
    )
    # Filled from mosaic_layers() whenever the list changes - the layers every
    # selected cube has, all selected. A picker that cannot be populated before
    # the run would have to provoke an error to learn its own options.
    mosaic_layers_w = widgets.SelectMultiple(
        options=[],
        value=(),
        rows=4,
        layout=widgets.Layout(width="100%"),
    )
    mosaic_layers_note_w = widgets.HTML("")

    mosaic_allow_resample_w = widgets.Checkbox(
        value=False,
        description="Allow resampling for cubes not on the same grid",
        indent=False,
        layout=widgets.Layout(width="100%"),
    )
    mosaic_resampling_w = widgets.Dropdown(
        options=["nearest", "bilinear", "cubic", "average"],
        value="nearest",
        layout=widgets.Layout(width="100%"),
        disabled=True,
    )
    mosaic_resolution_w = widgets.Text(
        value="",
        placeholder="finest of the selected cubes",
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
    )
    mosaic_strict_w = widgets.Checkbox(
        value=False,
        description="Merge cubes built differently (different mission / masking)",
        indent=False,
        layout=widgets.Layout(width="100%"),
    )
    mosaic_check_btn = widgets.Button(
        description="Check cubes",
        icon="search",
        layout=widgets.Layout(width="150px"),
    )
    mosaic_run_btn = widgets.Button(
        description="Mosaic Data Cubes",
        button_style="primary",
        icon="th-large",
        layout=widgets.Layout(width="210px"),
    )
    mosaic_out = widgets.Output(layout=widgets.Layout(width="99%", overflow="auto"))

    # Update Data Cube is a standalone feature too, so it gets its own output
    # group (instead of printing into the shared Status section).
    update_out = widgets.Output(layout=widgets.Layout(width="99%", overflow="auto"))

    # Spectral indices (applied via Edit button). Options are populated from the
    # loaded cube's mission once a cube is loaded.
    indices_select_w = widgets.SelectMultiple(
        options=[],
        value=(),
        rows=8,
        layout=widgets.Layout(width="100%", height="210px"),
        disabled=True,
    )
    indices_all_btn = widgets.Button(
        description="All indices", layout=widgets.Layout(width="110px"), disabled=True
    )
    indices_clear_btn = widgets.Button(
        description="Clear", layout=widgets.Layout(width="70px"), disabled=True
    )

    # Temporal composites (stats) -- applied via Edit button
    # Temporal Composites, same model as the Data Cube Builder: the two most
    # used composites get their own highlighted checkboxes, everything else
    # lives in the "More Composites" list, and one checkbox decides whether the
    # time series is kept alongside them or dropped.
    _COMMON_COMPOSITES = ("mean_timeseries", "median_timeseries")

    comp_mean_w = widgets.Checkbox(
        value=False,
        description="Mean of the time series",
        indent=False,
        disabled=True,
        layout=widgets.Layout(width="99%"),
    )
    comp_median_w = widgets.Checkbox(
        value=False,
        description="Median of the time series",
        indent=False,
        disabled=True,
        layout=widgets.Layout(width="99%"),
    )

    stats_select_w = widgets.SelectMultiple(
        options=[s for s in STATS_OPTIONS if s not in _COMMON_COMPOSITES],
        value=(),
        rows=8,
        layout=widgets.Layout(width="50%", height="210px"),
        disabled=True,
    )
    stats_all_btn = widgets.Button(description="All", layout=widgets.Layout(width="70px"), disabled=True)
    stats_clear_btn = widgets.Button(description="Clear", layout=widgets.Layout(width="70px"), disabled=True)

    # Off -> the Edit drops the time series and keeps only the composites.
    # Force-ticked and greyed while no composite is selected, since dropping it
    # then would leave an empty cube.
    keep_ts_w = widgets.Checkbox(
        value=True,
        description="Keep the full time series",
        indent=False,
        disabled=True,
        layout=widgets.Layout(width="99%"),
    )
    keep_ts_note = widgets.HTML("")

    # -- Custom Composites: user-defined periods, one row each ------------------
    # Same model as the Data Cube Builder: a row is either a season ("Every
    # year", MM-DD, expanded to name_YYYY per year the cube covers) or a single
    # window (full dates, one variable). Rows are added and removed at runtime,
    # so the container's children are rebuilt rather than declared here.
    custom_rows_box = widgets.VBox(layout=widgets.Layout(width="100%", gap="4px"))
    custom_add_btn = widgets.Button(
        description="Add composite",
        icon="plus",
        layout=widgets.Layout(width="150px"),
        disabled=True,
    )
    custom_error_note = widgets.HTML("")

    _custom_rows = []

    def _custom_row_is_blank(row):
        """A freshly added row the user has not touched yet - ignored quietly
        instead of being reported as an error."""
        return not any(
            (row[k].value or "").strip() for k in ("start", "end", "name")
        )

    def _custom_row_spec(row):
        """The dict this row stands for, without validating it."""
        return {
            "op": row["op"].value,
            ("season" if row["mode"].value == "season" else "window"): [
                (row["start"].value or "").strip(),
                (row["end"].value or "").strip(),
            ],
            "name": (row["name"].value or "").strip(),
        }

    def _custom_row_error(row, names_seen):
        """A short, plain problem description for this row, or None when it is
        fine. Short messages for the everyday mistakes, then the shared parser
        as the final authority so nothing calculate_statistics rejects gets
        through here."""
        seasonal = row["mode"].value == "season"
        start = (row["start"].value or "").strip()
        end = (row["end"].value or "").strip()
        name = (row["name"].value or "").strip()
        fmt = "MM-DD" if seasonal else "YYYY-MM-DD"

        if not start or not end:
            return f"fill both dates as {fmt}."
        if not name:
            return "give the composite a name."
        if name in names_seen:
            return f"the name '{name}' is already used by another row."

        try:
            _parse_custom_composite(_custom_row_spec(row))
        except ValueError:
            checker = is_mmdd if seasonal else is_iso_date
            if not checker(start) or not checker(end):
                return f"dates must be written as {fmt}."
            if not seasonal and start > end:
                return "the start date is after the end date."
            return (
                "the name can only use letters, digits and _ , and cannot start "
                "with a digit."
            )
        return None

    def _custom_composite_specs():
        """The valid custom composites, in row order, in the input form
        calculate_statistics accepts. Blank and broken rows are left out -
        _custom_validate() is what tells the user about them."""
        specs = []
        names_seen = set()
        for row in _custom_rows:
            if _custom_row_is_blank(row) or row["op"].disabled:
                continue
            if _custom_row_error(row, names_seen) is not None:
                continue
            names_seen.add((row["name"].value or "").strip())
            spec = _custom_row_spec(row)
            _parse_custom_composite(spec)  # validated, but keep the input form
            specs.append(spec)
        return specs

    def _custom_validate(*_):
        """Refresh the red note under the rows. Rows listed here are skipped by
        the Edit, never silently applied."""
        problems = []
        names_seen = set()
        for i, row in enumerate(_custom_rows):
            if _custom_row_is_blank(row) or row["op"].disabled:
                continue
            err = _custom_row_error(row, names_seen)
            if err:
                problems.append(f"Row {i + 1}: {err}")
            else:
                names_seen.add((row["name"].value or "").strip())
        if problems:
            custom_error_note.value = (
                "<div style='font-size:12px; color:#b91c1c; background:#fef2f2; "
                "border:1px solid #fecaca; border-radius:6px; padding:6px 8px;'>"
                "Ignored until fixed:<br>" + "<br>".join(problems) + "</div>"
            )
        else:
            custom_error_note.value = ""

    def _custom_row_widgets():
        """One editable row: period type, the two dates, the statistic, the
        name, and the button that removes it."""
        mode = widgets.Dropdown(
            options=[("Every year", "season"), ("Single window", "window")],
            value="season",
            layout=widgets.Layout(width="130px"),
        )
        # continuous_update=False: the value only changes when the field is left
        # (or Enter is pressed), so a half-typed date is never checked.
        start = widgets.Text(
            value="", placeholder="MM-DD", continuous_update=False,
            layout=widgets.Layout(width="105px"),
        )
        end = widgets.Text(
            value="", placeholder="MM-DD", continuous_update=False,
            layout=widgets.Layout(width="105px"),
        )
        op = widgets.Dropdown(
            options=sorted(_COMPOSITE_OPS),
            value="mean",
            layout=widgets.Layout(width="105px"),
        )
        name = widgets.Text(
            value="", placeholder="name", continuous_update=False,
            layout=widgets.Layout(width="150px"),
        )
        remove = widgets.Button(
            icon="times",
            tooltip="Remove this composite",
            layout=widgets.Layout(width="38px"),
        )
        return {
            "mode": mode, "start": start, "end": end,
            "op": op, "name": name, "remove": remove,
        }

    def _custom_sync_placeholders(row):
        """Season rows take MM-DD, single windows take full dates."""
        hint = "MM-DD" if row["mode"].value == "season" else "YYYY-MM-DD"
        row["start"].placeholder = hint
        row["end"].placeholder = hint

    def _custom_render():
        custom_rows_box.children = tuple(row["box"] for row in _custom_rows)

    def _custom_add_row(_=None, values=None):
        row = _custom_row_widgets()
        row["box"] = widgets.HBox(
            [row["mode"], row["start"], row["end"], row["op"], row["name"],
             row["remove"]],
            layout=widgets.Layout(
                width="100%", gap="4px", flex_flow="row wrap", align_items="center"
            ),
        )
        if values:
            row["mode"].value = values.get("mode", "season")
            row["start"].value = values.get("start", "")
            row["end"].value = values.get("end", "")
            row["op"].value = values.get("op", "mean")
            row["name"].value = values.get("name", "")
        _custom_sync_placeholders(row)

        def _changed(*_a):
            _custom_sync_placeholders(row)
            _custom_validate()
            _sync_keep_timeseries()

        for key in ("mode", "start", "end", "op", "name"):
            row[key].observe(_changed, names="value")
        row["remove"].on_click(lambda _b: _custom_remove_row(row))

        # Inherit the enabled/disabled state of the section (no cube loaded yet).
        for key in ("mode", "start", "end", "op", "name", "remove"):
            row[key].disabled = stats_select_w.disabled

        _custom_rows.append(row)
        _custom_render()
        _custom_validate()
        _sync_keep_timeseries()
        return row

    def _custom_remove_row(row):
        if row in _custom_rows:
            _custom_rows.remove(row)
        _custom_render()
        _custom_validate()
        _sync_keep_timeseries()

    def _custom_clear_rows():
        _custom_rows.clear()
        _custom_render()
        custom_error_note.value = ""

    def _custom_sync_enabled():
        """Follow the rest of the section: greyed until a cube is loaded."""
        off = stats_select_w.disabled
        custom_add_btn.disabled = off
        for row in _custom_rows:
            for key in ("mode", "start", "end", "op", "name", "remove"):
                row[key].disabled = off
        # Custom rows are created and removed at runtime, so they cannot be
        # observed once up front like the fixed controls are.
        _sync_edit_button()

    custom_add_btn.on_click(_custom_add_row)

    def _selected_composites():
        """The composites chosen in the Temporal Composites section, in a stable
        order: the two promoted ones, then "More Composites", then the Custom
        Composites rows (dicts, understood by calculate_statistics)."""
        tokens = []
        if comp_mean_w.value and not comp_mean_w.disabled:
            tokens.append("mean_timeseries")
        if comp_median_w.value and not comp_median_w.disabled:
            tokens.append("median_timeseries")
        if not stats_select_w.disabled:
            tokens.extend(str(s) for s in stats_select_w.value)
        tokens.extend(_custom_composite_specs())
        return tokens

    def _sync_keep_timeseries(*_):
        """"Keep the full time series" is only a real choice once a composite is
        selected; otherwise force it on and say why."""
        has_composite = bool(_selected_composites())
        if not has_composite and not keep_ts_w.value:
            keep_ts_w.value = True
        # Never un-grey it while the whole section is disabled (no cube loaded).
        keep_ts_w.disabled = stats_select_w.disabled or not has_composite
        if not has_composite:
            keep_ts_note.value = (
                "<div style='font-size:12px; color:#6b7280;'>"
                "Select a composite above to be able to keep only it, without "
                "the time series.</div>"
            )
        elif not keep_ts_w.value:
            keep_ts_note.value = (
                "<div style='font-size:12px; color:#1e40af; background:#eff6ff; "
                "border:1px solid #bfdbfe; border-radius:6px; padding:6px 8px;'>"
                "The Edit will keep the selected composites only - no per-date "
                "time series. Such a cube cannot be updated, co-registered or "
                "cloud-masked afterwards (use <b>Reset to loaded cube</b> to "
                "get the time series back).</div>"
            )
        else:
            keep_ts_note.value = ""

    comp_mean_w.observe(_sync_keep_timeseries, names="value")
    comp_median_w.observe(_sync_keep_timeseries, names="value")
    stats_select_w.observe(_sync_keep_timeseries, names="value")
    keep_ts_w.observe(_sync_keep_timeseries, names="value")

    # Update Data Cube (fetch missing dates and/or missing bands for the
    # loaded cube path). Standalone action: its own button + output group,
    # not the shared Edit button.
    update_run_btn = widgets.Button(
        description="Update Data Cube",
        button_style="primary",
        icon="refresh",
        layout=widgets.Layout(width="200px"),
        disabled=True,
    )
    # Shown instead of the button being silently dead when the loaded cube has
    # no file of its own (see _is_derived_path).
    update_archive_note_html = widgets.HTML(
        value="", layout=widgets.Layout(display="none", width="99%")
    )

    # -- Date Update: mirrors the builder's Time Period group (simple From/To
    # pickers, with the seasonal modes behind an "advanced" checkbox).
    update_date_from_w = widgets.DatePicker(
        value=_date(2024, 4, 1),
        layout=widgets.Layout(width="100%"),
        disabled=True,
    )
    update_date_to_w = widgets.DatePicker(
        value=_date(2024, 4, 10),
        layout=widgets.Layout(width="100%"),
        disabled=True,
    )
    update_advanced_dates_w = widgets.Checkbox(
        value=False,
        description="Use a seasonal date range (repeating across years)",
        indent=False,
        # Full row width, 99% not 100% (see the builder's advanced_dates_w).
        layout=widgets.Layout(width="99%"),
        disabled=True,
    )
    update_daterange_mode_w = widgets.Dropdown(
        options=[
            ("Seasonal (all available years)", "seasonal_all"),
            ("Seasonal (year range)", "seasonal_range"),
            ("Seasonal (selected years only)", "seasonal_selected"),
        ],
        value="seasonal_all",
        description="",
        layout=widgets.Layout(width="99%"),
        disabled=True,
    )

    update_daterange_w = widgets.Text(
        value='{"season": ["04-01", "10-31"], "years": "all"}',
        description="",
        placeholder='{"season": ["04-01", "10-31"], "years": "all"}',
        layout=widgets.Layout(width="99%"),
        disabled=True,
    )

    # -- Band Update: bands the loaded cube's mission offers but the cube
    # lacks (populated on load; empty selection = no band update).
    update_bands_w = widgets.SelectMultiple(
        options=[],
        value=(),
        rows=8,
        layout=widgets.Layout(width="100%", height="180px"),
        disabled=True,
    )
    update_bands_note_html = widgets.HTML("")

    # Export options. NetCDF / Zarr / COGs - editing shows the result in the
    # Result panel, so there is no separate "no export" mode; exporting saves
    # the edits.
    export_mode_w = widgets.Dropdown(
        options=[
            ("NetCDF (accepted for ARD Cube Tools)", "netcdf"),
            ("Zarr (accepted for ARD Cube Tools)", "zarr"),
            ("Geotiffs, Cloud Optimized (select a folder, NOT accepted for ARD Cube Tools)", "cogs"),
        ],
        value="netcdf",
        description="Mode:",
        layout=widgets.Layout(width="99%"),
        style={"description_width": "90px"},
        disabled=True,
    )

    export_target_w = widgets.Text(
        value="",
        description="Output:",
        placeholder="Load a cube, then set an output path",
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

    # Lossless zlib compression for the exported cube. NetCDF only: COGs are
    # already compressed (deflate), so the checkbox is hidden in COG mode (see
    # _apply_editor_compress_visibility). Same behavior as the Data Cube Builder.
    export_compress_w = widgets.Checkbox(
        value=False,
        description="Lossless compression (zlib)",
        indent=False,
        layout=widgets.Layout(width="auto"),
        disabled=True,
    )
    export_compress_warn_html = widgets.HTML(
        "<div style='font-size:12px; color:#b00020;'>"
        "⚠️ <b>Warning:</b> compression shrinks the output file a further "
        "~20-40% (scene-dependent), but the export step takes roughly "
        "<b>10x longer</b>. Enable it only for archiving, when disk space "
        "matters more than your time.</div>"
    )
    export_compress_warn_html.layout.display = "none"

    # QGIS band-mapping sidecar, same option as the Data Cube Builder.
    export_vrt_w = widgets.Checkbox(
        value=False,
        description="Export Band Mapping for GIS Tools (.vrt)",
        indent=False,
        layout=widgets.Layout(width="auto"),
        disabled=True,
    )
    export_vrt_note_html = widgets.HTML(
        "<div style='font-size:12px; color:#555; margin-left:2px;'>"
        "&#8505;&#65039; Open the <b>.vrt</b> file in QGIS, not the .nc, "
        "and keep both files in the same folder.</div>"
    )
    export_vrt_note_html.layout.display = "none"

    def _apply_editor_compress_visibility(*_):
        # Both options are NetCDF-only; hide them for cogs / zarr. Each
        # explanation shows only when its own box is visible + ticked.
        show = export_mode_w.value == "netcdf"
        export_compress_w.layout.display = "" if show else "none"
        export_compress_warn_html.layout.display = (
            "" if (show and export_compress_w.value) else "none"
        )
        export_vrt_w.layout.display = "" if show else "none"
        export_vrt_note_html.layout.display = (
            "" if (show and export_vrt_w.value) else "none"
        )

    export_compress_w.observe(_apply_editor_compress_visibility, names="value")
    export_vrt_w.observe(_apply_editor_compress_visibility, names="value")
    # Set the initial visibility explicitly instead of relying on the export
    # mode happening to default to NetCDF.
    _apply_editor_compress_visibility()

    # Legacy: the mean/median Temporal Composite dropdown that used to sit in
    # Export Options. Replaced by the Temporal Composites section, where it is
    # the "Mean/Median of the time series" checkbox plus "Keep the full time
    # series". Kept as a hidden widget so the enable/disable wiring and the
    # returned widget registry stay intact.
    aggregator_w = widgets.Dropdown(
        options=[("None", None), ("mean", "mean"), ("median", "median")],
        value=None,
        description="Temporal Composite:",
        layout=widgets.Layout(width="99%", display="none"),
        style={"description_width": "150px"},
        disabled=True,
    )
    # ipywidgets shows a BLANK label for value=None even when a ("None", None)
    # option exists, so set the label explicitly to display "None".
    aggregator_w.label = "None"

    # Visualization
    viz_dropdown_btn = widgets.Button(
        description="Launch interactive viewer",
        icon="image",
        button_style="info",
        layout=widgets.Layout(width="260px"),
        disabled=True,
    )

    viz_renderer_w, viz_renderer_box = _make_viz_renderer_control()

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

    # Animation rendering sections mirror the interactive viewer: presets,
    # single band (grey levels) and custom RGB (free band mapping). Band
    # dropdowns are populated from the loaded/edited cube.
    gif_section_w = widgets.ToggleButtons(
        options=[
            ("Presets", "preset"),
            ("Single band", "band"),
            ("Custom RGB", "custom"),
        ],
        value="preset",
        style={"button_width": "110px"},
        disabled=True,
    )

    gif_band_dd = widgets.Dropdown(
        options=[],
        description="Band:",
        layout=widgets.Layout(width="260px"),
        disabled=True,
    )

    _gif_chan_layout = widgets.Layout(width="180px")
    _gif_chan_style = {"description_width": "24px"}
    gif_r_dd = widgets.Dropdown(options=[], description="R:", layout=_gif_chan_layout,
                                style=_gif_chan_style, disabled=True)
    gif_g_dd = widgets.Dropdown(options=[], description="G:", layout=_gif_chan_layout,
                                style=_gif_chan_style, disabled=True)
    gif_b_dd = widgets.Dropdown(options=[], description="B:", layout=_gif_chan_layout,
                                style=_gif_chan_style, disabled=True)

    gif_stretch_w = widgets.FloatRangeSlider(
        value=(2.0, 98.0),
        min=0.0,
        max=100.0,
        step=0.5,
        description="Stretch (%):",
        continuous_update=False,
        readout_format=".1f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="380px"),
        disabled=True,
    )

    # Contextual rows: only the active section's controls are visible.
    gif_preset_box = widgets.VBox(
        [_stacked_field(gif_display_mode_w, "Display mode")],
        layout=widgets.Layout(width="100%"),
    )
    gif_band_box = widgets.VBox(
        [gif_band_dd], layout=widgets.Layout(width="100%", display="none")
    )
    gif_custom_box = widgets.VBox(
        [widgets.HBox([gif_r_dd, gif_g_dd, gif_b_dd],
                      layout=widgets.Layout(gap="8px"))],
        layout=widgets.Layout(width="100%", display="none"),
    )
    gif_stretch_box = widgets.VBox(
        [gif_stretch_w],
        layout=widgets.Layout(width="100%", gap="0px", display="none"),
    )

    def _sync_gif_section_visibility():
        sec = gif_section_w.value
        gif_preset_box.layout.display = "" if sec == "preset" else "none"
        gif_band_box.layout.display = "" if sec == "band" else "none"
        gif_custom_box.layout.display = "" if sec == "custom" else "none"
        # Presets keep their fixed scaling; the stretch applies to band/custom.
        gif_stretch_box.layout.display = "" if sec in ("band", "custom") else "none"

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
        layout=widgets.Layout(width="210px"),
        disabled=True,
    )

    export_current_btn = widgets.Button(
        description="Export current result",
        icon="save",
        layout=widgets.Layout(width="190px"),
        disabled=True,
    )
    # Warm, energetic orange call-to-action, matching the Data Cube Builder's
    # export button (replaces the old red "danger" style).
    export_current_btn.style.button_color = "#f97316"

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
            max_height="260px",
            overflow="auto",
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

    # Animation status has its own box: the viewer and the animation maker are
    # different tools, so GIF prompts/errors must never land in the interactive
    # view's output.
    anim_out = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #e5e7eb",
            padding="10px",
            border_radius="8px",
            width="99%",
            min_height="40px",
        )
    )

    # ---------------------------------------------------------------------
    # State
    # ---------------------------------------------------------------------
    state = {
        "loaded_path": None,
        "loaded_ds": None,        # open (lazy) xr.Dataset; kept open for on-demand reads
        "loaded_var": None,       # name of the loaded layer (data variable)
        "loaded_original": None,  # untouched loaded layer DataArray
        "current": None,          # working result (DataArray or Dataset after stats)
        "pending_ds": None,       # multi-layer file waiting for a layer selection
        "pending_path": None,
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
            elif mode == "zarr":
                selected = getattr(chooser, "selected", None)
                if selected:
                    s = str(selected)
                    if s.lower().endswith(".nc"):
                        s = s[: s.rfind(".")]
                    if not s.lower().endswith(".zarr"):
                        s += ".zarr"
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

        def _on_mask_file_fc_selected(chooser):
            selected = getattr(chooser, "selected", None)
            if selected:
                mask_file_w.value = _normalize_ui_path(selected)
                mask_file_fc_box.layout.display = "none"

        def _on_build_mask_fc_selected(chooser):
            # Save path: append .nc when the user picked/typed a name without a
            # cube extension (mirrors the export chooser's NetCDF behaviour).
            selected = getattr(chooser, "selected", None)
            if selected:
                s = str(selected)
                if not (s.lower().endswith(".nc") or s.lower().endswith(".zarr")):
                    s += ".nc"
                build_mask_out_w.value = _normalize_ui_path(s)
                build_mask_fc_box.layout.display = "none"

        def _on_csv_report_fc_selected(chooser):
            # Save path: append .csv when the name carries no extension, the
            # same courtesy the mask chooser does for .nc.
            selected = getattr(chooser, "selected", None)
            if selected:
                s = str(selected)
                if not s.lower().endswith(".csv"):
                    s += ".csv"
                csv_report_out_w.value = _normalize_ui_path(s)
                csv_report_fc_box.layout.display = "none"

        try:
            load_fc.register_callback(_on_load_fc_selected)
            export_fc.register_callback(_on_export_fc_selected)
            gif_fc.register_callback(_on_gif_fc_selected)
            if clip_fc is not None:
                clip_fc.register_callback(_on_clip_fc_selected)
            if mask_file_fc is not None:
                mask_file_fc.register_callback(_on_mask_file_fc_selected)
            if build_mask_fc is not None:
                build_mask_fc.register_callback(_on_build_mask_fc_selected)
            if csv_report_fc is not None:
                csv_report_fc.register_callback(_on_csv_report_fc_selected)
        except Exception:
            filechooser_available = False

    def _on_browse_load_clicked(_):
        if not filechooser_available or load_fc is None:
            _show_status("ℹ️ Optional dependency 'ipyfilechooser' is not available. Install it to use Browse buttons.")
            return
        _sync_load_filechooser_from_text()
        _toggle_box_display(load_fc_box)

    def _on_browse_export_clicked(_):
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

    def _on_browse_mask_file_clicked(_):
        if state["current"] is None:
            _show_status("ℹ️ Load a cube first to enable editing features.")
            return
        if not filechooser_available or mask_file_fc is None:
            _show_status("ℹ️ Optional dependency 'ipyfilechooser' is not available. Install it to use Browse buttons.")
            return
        _sync_mask_file_filechooser_from_text()
        _toggle_box_display(mask_file_fc_box)

    def _on_browse_build_mask_clicked(_):
        if state["current"] is None:
            _show_status("ℹ️ Load a cube first to enable editing features.")
            return
        if not filechooser_available or build_mask_fc is None:
            _show_status("ℹ️ Optional dependency 'ipyfilechooser' is not available. Install it to use Browse buttons.")
            return
        _sync_build_mask_filechooser_from_text()
        _toggle_box_display(build_mask_fc_box)

    def _on_browse_csv_report_clicked(_):
        if state["current"] is None:
            _show_status("ℹ️ Load a cube first to enable editing features.")
            return
        if not filechooser_available or csv_report_fc is None:
            _show_status("ℹ️ Optional dependency 'ipyfilechooser' is not available. Install it to use Browse buttons.")
            return
        _sync_csv_report_filechooser_from_text()
        _toggle_box_display(csv_report_fc_box)

    # ---------------------------------------------------------------------
    # Feature helpers
    # ---------------------------------------------------------------------
    def _daterange_mode_example(mode_value: str):
        # Same seasonal modes/examples as the builder's Time Period group.
        if mode_value == "seasonal_all":
            return '{"season": ["04-01", "10-31"], "years": "all"}'
        elif mode_value == "seasonal_range":
            return '{"season": ["04-01", "10-31"], "years": "2019-2024"}'
        elif mode_value == "seasonal_selected":
            return '{"season": ["04-01", "10-31"], "years": [2019, 2021, 2023]}'
        return '{"season": ["04-01", "10-31"], "years": "all"}'

    def _update_update_daterange_example(force=False):
        new_example = _daterange_mode_example(update_daterange_mode_w.value)
        current = (update_daterange_w.value or "").strip()
        prev_auto = state.get("last_auto_update_daterange_example")

        update_daterange_w.placeholder = new_example
        should_replace = force or (current == "") or (prev_auto is not None and current == prev_auto)
        if should_replace:
            update_daterange_w.value = new_example

        state["last_auto_update_daterange_example"] = new_example

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
        If current result is Dataset (e.g. after stats), filter Time_Series and drop stale stats.
        """
        if not enable_cloud_filter_w.value:
            return obj, False, []

        max_cloud = int(cloud_max_w.value)
        if max_cloud < 0 or max_cloud > 100:
            raise ValueError("Max cloud % must be between 0 and 100.")

        # Dataset case -> filter time series and drop stats
        if isinstance(obj, xr.Dataset):
            if "Time_Series" not in obj.data_vars:
                raise ValueError(
                    "Current Dataset does not contain 'Time_Series' for cloud filtering."
                )
            da = obj["Time_Series"]
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

    def _apply_coverage_filter_feature(obj):
        """Filter by Scene Coverage: keep only time steps imaging at least the
        chosen percentage of the AOI, dropping partial / faulty / missing scenes.

        Reuses the across-track coverage code: the stored scene_coverage coord
        when present (cheap, no read), otherwise compute_scene_coverage() measured
        from the cube's own no-data pattern. Mirrors _apply_cloud_filter_feature -
        a Dataset input is filtered on Time_Series and stale stats are
        dropped; an empty result is a warning, not an error.
        """
        if not enable_coverage_filter_w.value:
            return obj, False, []

        min_cov = int(coverage_min_w.value)
        if min_cov < 0 or min_cov > 100:
            raise ValueError("Min scene coverage % must be between 0 and 100.")
        thr = min_cov / 100.0

        stats_dropped = isinstance(obj, xr.Dataset)
        if isinstance(obj, xr.Dataset):
            if "Time_Series" not in obj.data_vars:
                raise ValueError(
                    "Current Dataset does not contain 'Time_Series' "
                    "for scene-coverage filtering."
                )
            da = obj["Time_Series"]
        elif isinstance(obj, xr.DataArray):
            da = obj
        else:
            raise TypeError(
                f"Unsupported object type for scene-coverage filtering: {type(obj)}"
            )

        if "time" not in da.dims:
            raise ValueError("Scene-coverage filtering requires a 'time' dimension.")

        # Coverage source: the stored coord (default on new cubes, no read) or a
        # fresh measurement for cubes built before scene_coverage was default.
        if "scene_coverage" in da.coords:
            cov_vals = np.asarray(da["scene_coverage"].values, dtype=float)
            cov_src = "stored scene_coverage coord"
        else:
            cov = compute_scene_coverage(da)
            if cov is None:
                raise ValueError("Could not measure scene coverage (no time dim).")
            cov_vals = np.asarray(cov.values, dtype=float)
            cov_src = "measured now (no stored coord)"

        before_n = int(da.sizes.get("time", 0))
        keep = np.asarray(cov_vals >= thr)
        attrs_ref = dict(getattr(da, "attrs", {}) or {})
        if keep.any():
            filtered = da.isel(time=np.flatnonzero(keep))
            # Refresh the coord so survivors stay self-describing and re-runs are
            # 1:1 with the time axis.
            filtered = filtered.assign_coords(
                scene_coverage=("time", cov_vals[keep])
            )
        else:
            filtered = da.isel(time=slice(0, 0))
        try:
            filtered.attrs.update(attrs_ref)
        except Exception:
            pass
        after_n = int(keep.sum())

        msgs = [
            "scene coverage filter applied",
            f"min_coverage={min_cov}%  ({cov_src})",
            f"Scenes kept: {after_n} / {before_n}",
            f"Removed scenes: {max(0, before_n - after_n)}",
        ]
        if stats_dropped:
            msgs.append(
                "Previous stats were removed because coverage filtering changes "
                "the selected time steps."
            )
        if after_n == 0:
            msgs.append(
                "Warning: no scenes remain after filtering. Lower the Min coverage %."
            )
        return filtered, True, msgs

    def _resolve_update_daterange():
        """Builder-style Time Period resolution: the simple From/To pickers, or
        the seasonal text field when the 'advanced' checkbox is ticked. Returns
        None when no dates are given - the update mechanism then defaults to
        the cube's own time span (useful for a band-only update)."""
        if update_advanced_dates_w.value:
            return _parse_daterange_input(
                update_daterange_mode_w.value, update_daterange_w.value
            )
        d_from = update_date_from_w.value
        d_to = update_date_to_w.value
        if d_from is None and d_to is None:
            return None
        if d_from is None or d_to is None:
            raise ValueError("Please choose both a 'From' and a 'To' date.")
        if d_from > d_to:
            raise ValueError("The 'From' date is after the 'To' date - please swap them.")
        return [d_from.isoformat(), d_to.isoformat()]

    def _apply_update_feature(obj):
        """
        Update the loaded cube by requesting missing dates and/or bands via:
        - get_stac_layers(update=...) for Time_Series cubes
          (dates + bands; the cloud/shadow strategy is restored from the attrs)
        - update_cloud_mask_cube(...) for SCL binary masks (cloud_mask_scl band)
        - get_cloud_layers(update=..., threshold=None) for cloud-probability
          Cloud_Stack cubes (dates only, probability only)
        """
        loaded_path = state.get("loaded_path")
        if not loaded_path:
            raise ValueError("No loaded cube path available for update.")
        if _is_derived_path(loaded_path):
            raise ValueError(
                "Update Data Cube needs the cube's own file, to read the build "
                "parameters stored in it. This cube was mosaicked in this "
                "session and has no file yet. Export it below, then load the "
                "exported file and update that."
            )

        daterange = _resolve_update_daterange()
        add_bands = [str(b) for b in update_bands_w.value]
        if not daterange and not add_bands:
            raise ValueError(
                "Please provide a daterange (Date Update) and/or select bands "
                "to add (Band Update)."
            )

        # Which kind of cube was loaded?
        loaded_var = state.get("loaded_var")
        if not loaded_var:
            try:
                loaded_var = state.get("loaded_original").name
            except Exception:
                loaded_var = None

        # ------------------------------------------------------------------
        # SCL binary cloud-mask update (Cloud_Stack with a cloud_mask_scl band)
        # ------------------------------------------------------------------
        _mask_da = obj if isinstance(obj, xr.DataArray) else (
            obj["Cloud_Stack"] if isinstance(obj, xr.Dataset)
            and "Cloud_Stack" in obj.data_vars else None
        )
        _mask_bands = (
            [str(b) for b in _mask_da["band"].values]
            if _mask_da is not None and "band" in getattr(_mask_da, "dims", ())
            else []
        )
        if loaded_var == "Cloud_Stack" and "cloud_mask_scl" in _mask_bands:
            if add_bands:
                raise ValueError(
                    "Band update is not available for a binary cloud mask. Clear "
                    "the Band Update selection."
                )
            if not daterange:
                raise ValueError("Please provide a daterange for Update Data Cube.")
            merged = update_cloud_mask_cube(_mask_da, daterange, q=True)
            _old_days = set(
                np.asarray(_mask_da.time.values).astype("datetime64[D]")
            )
            added = sorted(
                set(np.asarray(merged.time.values).astype("datetime64[D]"))
                - _old_days
            )
            if not added:
                return merged, False, [
                    f"No new dates in {daterange} - the binary mask is already "
                    "up to date."
                ]
            msgs = [
                f"binary cloud mask updated (SCL): {len(added)} new date(s) "
                f"in {daterange}",
                "added: " + ", ".join(
                    np.datetime_as_string(d, unit="D") for d in added
                ),
            ]
            return merged, True, msgs

        # ------------------------------------------------------------------
        # Cloud cube update (Cloud_Stack) -> cloud probability only
        # ------------------------------------------------------------------
        if loaded_var == "Cloud_Stack":
            if add_bands:
                raise ValueError(
                    "Band update is available for Time_Series cubes "
                    "only. Clear the Band Update selection to update a "
                    "Cloud_Stack cube."
                )
            if not daterange:
                raise ValueError("Please provide a daterange for Update Data Cube.")
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
            ]
            return updated, True, msgs

        # ------------------------------------------------------------------
        # Spectral cube update (missing dates and/or missing bands)
        # ------------------------------------------------------------------
        # cloud_masking is intentionally NOT passed: get_stac_layers restores
        # the cube's exact cloud/shadow strategy from its attributes
        # (cloud_status, shadow params), so new scenes AND new bands arrive
        # masked - or kept - consistently with the stored data.
        updated = get_stac_layers(
            update=loaded_path,
            daterange=daterange,
            bands=add_bands or None,
            max_cc=100,
            clip_raster=False,
            stats=None,
            aggregator=None,
            output=None,  # return in memory
            q=True,       # silent for GUI
        )

        if isinstance(updated, xr.Dataset):
            if "Time_Series" not in updated.data_vars:
                raise ValueError("Update returned a Dataset without 'Time_Series'.")
            updated = updated["Time_Series"]

        if not isinstance(updated, xr.DataArray):
            raise TypeError(f"Update returned unsupported object type: {type(updated)}")

        msgs = [
            "update_data_cube applied (get_stac_layers(update=...))",
            f"daterange={daterange if daterange else 'cube time span (default)'}",
            f"bands added: {', '.join(add_bands) if add_bands else 'none'}",
            "cloud/shadow masking strategy restored from the cube's attributes",
            "Current working result was replaced with the updated Time_Series.",
        ]
        return updated, True, msgs
    
    
    
    
    def _apply_clip_feature(obj):
        """
        Apply clipping using stac2cube.clip_stac().

        Behavior:
        - If clip checkbox is disabled -> no change
        - If enabled but no clip input -> raises clear error
        - If current is DataArray -> clip directly
        - If current is Dataset with Time_Series -> clip time series and
          drop old stats (they become invalid after spatial clip)
        """
        if not enable_clip_w.value:
            return obj, False, []

        geom = _parse_clip_geometry_input(clip_geom_w.value)
        if geom is None:
            raise ValueError("Clip is enabled, but no polygon/bbox was provided.")

        if isinstance(obj, xr.Dataset):
            if "Time_Series" not in obj.data_vars:
                raise ValueError(
                    "Current Dataset does not contain 'Time_Series' for clipping."
                )
            da = obj["Time_Series"]
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

    def _parse_reproject_resolution():
        """Pixel size box -> float metres, or None for "keep the current one"."""
        text = (reproject_res_w.value or "").strip().replace(",", ".")
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            raise ValueError(
                f"Pixel size '{reproject_res_w.value}' is not a number. Type a "
                "size in metres (e.g. 20), or leave the box empty to keep the "
                "current pixel size."
            )
        if value <= 0:
            raise ValueError("Pixel size must be greater than 0.")
        return value

    def _sync_reproject_crs_status(*_):
        """Check the typed CRS when the box is committed (Enter / focus loss),
        so a bad code is reported here instead of failing mid-edit.

        Catches Exception, not just ValueError: this runs inside an ipywidgets
        message handler, where anything escaping is dumped as a raw traceback
        under the GUI.
        """
        text = (reproject_crs_w.value or "").strip()
        if not text:
            reproject_crs_status_w.value = ""
            return
        try:
            canonical = validate_target_crs(text)
        except Exception as exc:
            reproject_crs_status_w.value = (
                "<div style='font-size:12px; color:#991b1b; background:#fef2f2; "
                "border:1px solid #fecaca; border-radius:6px; padding:6px 8px;'>"
                f"✗ {exc}</div>"
            )
            return
        reproject_crs_status_w.value = (
            "<div style='font-size:12px; color:#166534;'>✓ reprojecting to "
            f"<b>{canonical}</b>.</div>"
        )

    reproject_crs_w.observe(_sync_reproject_crs_status, names="value")

    def _apply_reproject_feature(obj):
        """Reproject the working cube into another CRS via reproject_stac().

        A Dataset input is reprojected on Time_Series and the stats layers are
        dropped, exactly like clipping: a temporal composite computed on the old
        grid cannot be carried onto the new one. reproject_stac prints its own
        progress into Status (the warp is eager and can take a while), so only
        the summary lines are returned here.
        """
        if not enable_reproject_w.value:
            return obj, False, []

        crs_text = (reproject_crs_w.value or "").strip()
        if not crs_text:
            raise ValueError(
                "Reprojection is enabled, but no target CRS was given "
                "(e.g. EPSG:3035)."
            )
        resolution = _parse_reproject_resolution()
        method = reproject_resampling_w.value

        stats_dropped = isinstance(obj, xr.Dataset)
        if isinstance(obj, xr.Dataset):
            if "Time_Series" not in obj.data_vars:
                raise ValueError(
                    "Current Dataset does not contain 'Time_Series' for "
                    "reprojection."
                )
            da = obj["Time_Series"]
        elif isinstance(obj, xr.DataArray):
            da = obj
        else:
            raise TypeError(
                f"Unsupported object type for reprojection: {type(obj)}"
            )

        src_crs = da.attrs.get("crs") or da.rio.crs
        out = reproject_stac(
            da, crs_text, resolution=resolution, resampling=method
        )

        msgs = [
            "reprojection applied",
            f"CRS: {src_crs} -> {out.attrs.get('crs')}  ({method})",
            f"Grid: {int(out.sizes['y'])} x {int(out.sizes['x'])} pixels at "
            f"{abs(out.attrs['transform'].a):g} m",
        ]
        if resolution is None:
            msgs.append("Pixel size: kept (approximately) as it was.")
        if "cloud_percentage" in out.coords or "scene_coverage" in out.coords:
            msgs.append(
                "cloud_percentage / scene_coverage were kept as measured before "
                "the reprojection; they are not recalculated on the new grid."
            )
        msgs.append(
            "Reprojection resamples: pixel values and the pixel grid changed, "
            "and the empty corners are the rotated cube's bounding box."
        )
        if stats_dropped:
            msgs.append(
                "Previous stats were removed because reprojection changes the "
                "raster grid."
            )
        return out, True, msgs

    def _pick_mask_band(cloud):
        """Choose the binary cloud-mask band from a loaded Cloud_Stack: prefer the
        SCL mask, then any 'cloud_mask_*' band. None when there is no band dim."""
        if "band" not in getattr(cloud, "dims", ()) and "band" not in getattr(cloud, "coords", {}):
            return None
        bands = [str(b) for b in cloud["band"].values]
        if "cloud_mask_scl" in bands:
            return "cloud_mask_scl"
        mask_bands = [b for b in bands if b.startswith("cloud_mask")]
        if mask_bands:
            return mask_bands[0]
        raise ValueError(
            "The selected file has no binary cloud-mask band (expected 'cloud_mask_*', "
            "e.g. 'cloud_mask_scl'). It looks like a cloud probability cube - build a "
            "binary mask first (ARD Cloud tools), or use one from 'Export Mask as "
            "Binary File'."
        )

    def _apply_mask_clouds_feature(obj):
        """Mask the working cube out with a binary Cloud_Stack file (1=cloud,
        0=clear) via stac2cube.mask_stac_clouds(). Returns the masked DataArray;
        any stats are dropped because masking changes pixel validity.
        """
        if not enable_mask_clouds_w.value:
            return obj, False, []

        path = (mask_file_w.value or "").strip()
        if not path:
            raise ValueError("Masking is enabled, but no binary mask file was provided.")
        # A file picked inside a .zarr store resolves to the store root.
        path = resolve_cube_path(path)
        if not os.path.exists(path):
            raise ValueError(f"Binary mask file not found: {path}")

        # Load the Cloud_Stack (small binary mask; eager load for a clean align).
        with open_cube(path) as cds:
            if "Cloud_Stack" in cds.data_vars:
                cloud = cds["Cloud_Stack"].load()
            elif len(cds.data_vars) == 1:
                cloud = cds[list(cds.data_vars)[0]].load()
            else:
                raise ValueError(
                    "Binary mask file does not contain a 'Cloud_Stack' variable."
                )

        mask_layer = _pick_mask_band(cloud)
        if mask_layer is None:
            raise ValueError(
                "The mask file has no 'band' dimension; expected a Cloud_Stack with "
                "a binary 'cloud_mask_*' band."
            )

        # Align the mask to the cube's CURRENT dates by the time coordinate, so
        # masking still works after the cube has been date-sliced (this feature is
        # chainable). The binary mask normally comes from the same cube, so its
        # dates are a superset of whatever remains - we simply pick out the dates
        # the cube still has. Only a mask that is actually MISSING one of those
        # dates is a real error.
        da_ref = obj["Time_Series"] if isinstance(obj, xr.Dataset) else obj
        if "time" in getattr(da_ref, "dims", ()) and "time" in getattr(cloud, "dims", ()):
            cube_times = da_ref["time"].values
            mask_times = set(cloud["time"].values.tolist())
            missing = [t for t in cube_times.tolist() if t not in mask_times]
            if missing:
                raise ValueError(
                    f"The binary mask is missing {len(missing)} of the cube's "
                    f"{cube_times.size} current dates. The mask must come from the "
                    "same cube and cover every date still in the cube."
                )
            # Keep only the dates the cube currently has (in the cube's order).
            cloud = cloud.sel(time=cube_times)

        # obj may be a DataArray or a Dataset (with stats); mask_stac_clouds pulls
        # out the Time_Series and returns a masked DataArray.
        masked = mask_stac_clouds(obj, cloud, mask_layer)

        msgs = [
            "mask_stac_clouds applied (binary cloud mask)",
            f"mask file: {Path(path).name}",
            f"mask band: {mask_layer}",
            "Clouds masked out (pixels set to no-data); cloud_percentage recomputed.",
        ]
        if isinstance(obj, xr.Dataset):
            msgs.append("Previous stats were removed because masking changes pixel validity.")
        return masked, True, msgs

    def _apply_indices_feature(obj):
        """
        Calculate spectral indices via stac2cube.calculate_spectral_index() and
        append them as new bands.

        Behavior:
        - If nothing selected -> no change.
        - Indices already present in the cube are skipped (not recomputed).
        - Each requested index is computed on its own so that, if bands are
          missing, every missing band can be reported instead of only the first.
        - If any selected index is missing a required band, NOTHING is applied and
          a clear error is raised naming the missing band(s).
        """
        selected = list(indices_select_w.value)
        if not selected:
            return obj, False, []

        # Pick the spectral DataArray to compute from.
        if isinstance(obj, xr.Dataset):
            if "Time_Series" not in obj.data_vars:
                raise ValueError(
                    "Spectral indices require the 'Time_Series' cube. "
                    "Calculate indices before generating temporal composites (stats)."
                )
            da = obj["Time_Series"]
        elif isinstance(obj, xr.DataArray):
            da = obj
        else:
            raise TypeError(f"Unsupported object type for spectral indices: {type(obj)}")

        mission_name = _current_mission()
        if not mission_name:
            raise ValueError(
                "Cannot determine this cube's mission (missing 'mission' attribute), "
                "so spectral indices cannot be computed."
            )

        existing_bands = (
            [str(b) for b in da.coords["band"].values] if "band" in da.coords else []
        )
        existing_lower = {b.lower() for b in existing_bands}

        # Skip indices that are already present as bands.
        already_present = [i for i in selected if i.lower() in existing_lower]
        to_compute = [i for i in selected if i.lower() not in existing_lower]

        if not to_compute:
            return obj, False, [
                "calculate spectral indices: nothing to do "
                f"(already present: {', '.join(already_present)})"
            ]

        # Compute each index individually so we can collect ALL missing bands.
        computed = []
        missing = []  # list of (index, missing_band)
        for idx in to_compute:
            try:
                computed.append(calculate_spectral_index(da, mission_name, [idx]))
            except KeyError as e:
                band = _extract_missing_band(str(e))
                missing.append((idx, band))

        if missing:
            parts = []
            for idx, band in missing:
                parts.append(f"'{idx}' needs band '{band}'" if band else f"'{idx}'")
            avail = ", ".join(existing_bands) if existing_bands else "none"
            raise ValueError(
                "Cannot calculate spectral indices because required band(s) are "
                f"missing from this cube: {'; '.join(parts)}. "
                f"Available bands: {avail}."
            )

        indices_da = computed[0] if len(computed) == 1 else xr.concat(computed, dim="band")

        # Append the new index bands to the existing cube (mirrors the builder).
        combined = xr.concat([da, indices_da], dim="band")
        if "time" in combined.dims:
            combined = combined.transpose("time", "band", "y", "x")
        combined = combined.rename(da.name) if da.name is not None else combined
        combined.attrs = dict(da.attrs)
        prev_idx = [str(i) for i in (da.attrs.get("indices") or [])]
        combined.attrs["indices"] = prev_idx + list(to_compute)

        msgs = [f"spectral indices calculated: {', '.join(to_compute)}"]
        if already_present:
            msgs.append(f"skipped (already present): {', '.join(already_present)}")

        if isinstance(obj, xr.Dataset):
            new_ds = obj.copy()
            new_ds["Time_Series"] = combined
            return new_ds, True, msgs

        return combined, True, msgs

    def _extract_missing_band(error_text):
        """Pull the band name out of _require_band's 'please include "x"' message."""
        m = re.search(r'please include "([^"]+)"', error_text)
        return m.group(1) if m else None

    def _apply_stats_feature(obj):
        """
        Apply the Temporal Composites selection using calculate_statistics(),
        then optionally drop the time series ("Keep the full time series" off).
        Returns (new_obj, changed, messages).
        """
        selected = _selected_composites()
        if not selected:
            return obj, False, []

        da = _pick_timeseries_for_stats(obj)
        if "time" not in da.dims:
            raise ValueError(
                "Temporal composites require a 'time' dimension. "
                "Use 'Reset to loaded cube' if you are currently on a non-temporal result."
            )

        ds_stats = calculate_statistics(da, selected)
        keep_ts = bool(keep_ts_w.value) or keep_ts_w.disabled
        msgs = [f"composites={len(selected)} selection(s)"]

        if not keep_ts and "Time_Series" in ds_stats.data_vars:
            remaining = [v for v in ds_stats.data_vars if v != "Time_Series"]
            if remaining:
                ds_stats = ds_stats.drop_vars("Time_Series")
                # No variable uses the time axis once the series is gone; drop
                # the orphaned time coords so the cube does not advertise dates
                # it no longer holds (mirrors main._drop_timeseries).
                if not any("time" in ds_stats[v].dims for v in ds_stats.data_vars):
                    orphans = [
                        n for n, c in ds_stats.coords.items() if "time" in c.dims
                    ]
                    if orphans:
                        ds_stats = ds_stats.drop_vars(orphans)
                msgs.append("Composites only - the time series was dropped.")
        else:
            msgs.append("Temporal composites generated (time series + composites).")

        msgs.append("This should usually be the LAST step before exporting.")
        return ds_stats, True, msgs

    # ---------------------------------------------------------------------
    # Core callbacks
    # ---------------------------------------------------------------------
    def _finalize_load(path, ds_loaded, var_name):
        """Initialize the editor from one layer (data variable) of an already
        opened dataset. Called directly for single-layer files, or from the
        'Load selected layer' button for multi-layer files."""
        loaded = ds_loaded[var_name]

        state["loaded_path"] = path
        state["loaded_var"] = var_name
        state["loaded_original"] = loaded
        state["current"] = _safe_copy_xarray(loaded)

        _show_preview(loaded_summary_out, state["loaded_original"])
        _show_result_current()

        _populate_slice_widgets_from_current(select_all=True)
        _populate_indices_widget_from_current()
        _populate_update_bands_from_current()
        # Prefill the update From/To pickers with the cube's own span: a
        # natural base to extend for new dates, and exactly right as-is for a
        # band-only update (0 new dates, bands filled on every stored date).
        try:
            _t = loaded["time"].values
            update_date_from_w.value = pd.Timestamp(_t.min()).date()
            update_date_to_w.value = pd.Timestamp(_t.max()).date()
        except Exception:
            pass
        _set_editor_enabled(True)
        _refresh_gif_band_options()
        _update_gif_output_suggestion(force=True)
        _update_update_daterange_example(force=True)

        if export_mode_w.value == "netcdf" and not export_target_w.value:
            export_target_w.value = _auto_netcdf_export_suggestion()
        elif export_mode_w.value == "zarr" and not export_target_w.value:
            base = _auto_netcdf_export_suggestion()
            export_target_w.value = f"{os.path.splitext(base)[0]}.zarr"

        # Build Cloud Mask output: suggest <cube>_mask_binary.<ext> on load, but
        # keep a manual edit (same freshening rule as the GIF/export suggestions).
        _bm_new = _suggest_build_mask_path()
        if _bm_new:
            build_mask_out_w.placeholder = _bm_new
            _bm_cur = (build_mask_out_w.value or "").strip()
            _bm_prev = state.get("last_auto_build_mask_suggestion")
            if _bm_cur == "" or (_bm_prev is not None and _bm_cur == _bm_prev):
                build_mask_out_w.value = _bm_new
            state["last_auto_build_mask_suggestion"] = _bm_new

        # CSV report output: same freshening rule again.
        _csv_new = _suggest_csv_report_path()
        if _csv_new:
            csv_report_out_w.placeholder = _csv_new
            _csv_cur = (csv_report_out_w.value or "").strip()
            _csv_prev = state.get("last_auto_csv_report_suggestion")
            if _csv_cur == "" or (_csv_prev is not None and _csv_cur == _csv_prev):
                csv_report_out_w.value = _csv_new
            state["last_auto_csv_report_suggestion"] = _csv_new

        print(f"✅ Loaded cube: {path}")
        print(f"   Working layer: {_layer_display_name(var_name)}")
        _print_working_note()

        try:
            loaded_summary_acc.selected_index = 0
        except Exception:
            pass
        try:
            result_acc.selected_index = 0
        except Exception:
            pass

    def _on_load_cube_clicked(_):
        path = (load_path_w.value or "").strip()
        if not path:
            _show_status("❌ Please provide a NetCDF (.nc) or Zarr (.zarr) cube path.")
            return
        # A file picked INSIDE a .zarr store (e.g. zarr.json) resolves to the
        # store root, so the file chooser can be used for Zarr cubes too.
        resolved = resolve_cube_path(path)
        if resolved != path:
            path = resolved
            load_path_w.value = path
        if not (path.lower().endswith(".nc") or is_zarr_path(path)):
            _show_status("❌ Please select a NetCDF file (.nc) or a Zarr store (.zarr).")
            return
        if not Path(path).exists():
            _show_status(f"❌ File not found: {path}")
            return

        try:
            with status_out:
                clear_output()
                print("Loading data cube...")

                # Close any previously opened cube to release its file handle
                # (important on Windows, where an open handle locks the file).
                _prev_ds = state.get("loaded_ds")
                if _prev_ds is not None:
                    try:
                        _prev_ds.close()
                    except Exception:
                        pass
                    state["loaded_ds"] = None

                # Open lazily (Dask-backed) so large cubes are read from disk on
                # demand instead of being copied into RAM at load time. The lazy
                # array keeps reading from this file during preview/edit/export,
                # so the handle must stay open -- do NOT wrap this in a closing
                # `with` block. open_cube dispatches .zarr -> open_zarr (always
                # lazy), else NetCDF with one chunk per scene ("frames"): the
                # viewer, the GIF and every preview read one date at a time, and
                # dask's "auto" sizing made each of those reads pull ~1 GB off
                # disk for a 4.8 MB frame (~80x slower - see
                # _frame_chunked_netcdf).
                ds_open = open_cube(path, chunks="frames")
                # Keep small coordinates in memory; only the data variables stay
                # lazy. Otherwise chunked non-dimension coords (e.g.
                # cloud_percentage) become dask arrays, which breaks boolean-indexer
                # ops such as cloud_filter's .where(cond, drop=True).
                ds_loaded = ds_open.assign_coords(
                    {name: coord.compute() for name, coord in ds_open.coords.items()}
                )
                state["loaded_ds"] = ds_open

                layers = _raster_layer_names(ds_loaded)
                if not layers:
                    raise ValueError(
                        "Cube contains no raster layers (data variables with "
                        f"'y'/'x' dims). Found data_vars: {list(ds_loaded.data_vars)}"
                    )

                if len(layers) == 1:
                    # Single layer (e.g. cube exported without stats): load it
                    # directly, no matter how the variable is named.
                    state["pending_ds"] = None
                    state["pending_path"] = None
                    layer_select_w.options = []
                    layer_select_box.layout.display = "none"
                    _finalize_load(path, ds_loaded, layers[0])
                else:
                    # Multiple layers (e.g. time series + temporal composites):
                    # list them and let the user pick before initializing.
                    state["pending_ds"] = ds_loaded
                    state["pending_path"] = path
                    # The previous working result may reference the file handle
                    # that was just closed above, so drop it until a layer is
                    # confirmed.
                    state["loaded_path"] = None
                    state["loaded_var"] = None
                    state["loaded_original"] = None
                    state["current"] = None
                    _populate_update_bands_from_current()  # clears the band list
                    _set_editor_enabled(False)

                    layer_select_w.options = _layer_dropdown_options(ds_loaded, layers)
                    layer_select_w.value = (
                        "Time_Series"
                        if "Time_Series" in layers
                        else layers[0]
                    )
                    layer_select_box.layout.display = ""

                    for out_w, note in (
                        (loaded_summary_out, "No layer loaded yet."),
                        (result_out, "No layer loaded yet."),
                    ):
                        with out_w:
                            clear_output()
                            print(note)

                    print(f"ℹ️ This cube contains {len(layers)} layers:")
                    for name in layers:
                        dims = ", ".join(
                            f"{d}: {ds_loaded[name].sizes[d]}"
                            for d in ds_loaded[name].dims
                        )
                        print(f"   - {_layer_display_name(name)}  ({dims})")
                    print(
                        "Select the layer to work on in the 'Layer' dropdown, "
                        "then click 'Load selected layer'."
                    )

        except Exception as e:
            _show_status(_friendly_error(e, "Loading"))

    def _on_layer_load_clicked(_):
        ds_loaded = state.get("pending_ds")
        path = state.get("pending_path")
        var_name = layer_select_w.value
        if ds_loaded is None or not path:
            _show_status("❌ Load a cube first.")
            return
        if not var_name:
            _show_status("❌ Please select a layer to load.")
            return

        try:
            with status_out:
                clear_output()
                print(f"Loading layer '{var_name}'...")
                _finalize_load(path, ds_loaded, var_name)
        except Exception as e:
            _show_status(_friendly_error(e, "Loading"))

    def _on_reset_clicked(_):
        if state["loaded_original"] is None:
            _show_status("ℹ️ No loaded cube to reset to yet.")
            return

        state["current"] = _safe_copy_xarray(state["loaded_original"])
        _populate_slice_widgets_from_current(select_all=True)
        _refresh_gif_band_options()
        _show_result_current()

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
        enable_coverage_filter_w.value = False
        enable_mask_clouds_w.value = False
        enable_clip_w.value = False
        enable_reproject_w.value = False
        _sync_edit_button()

    def _staged_features():
        """Names of the Edit-card tools that would do something right now.

        Each entry mirrors the early-return gate of the matching _apply_*_feature
        function, so the count on the button cannot promise a change that the
        Edit run then skips. Slice is special: it has no on/off switch, and an
        empty OR complete selection means 'keep everything' (see
        _apply_slice_feature), so only a strict subset counts as staged.
        """
        staged = []

        def _is_subset(w):
            sel, opts = list(w.value), list(w.options)
            return 0 < len(sel) < len(opts)

        if _is_subset(slice_time_w) or _is_subset(slice_band_w):
            staged.append("Slice")
        if enable_cloud_filter_w.value:
            staged.append("Cloud filter")
        if enable_coverage_filter_w.value:
            staged.append("Coverage filter")
        if enable_mask_clouds_w.value:
            staged.append("Mask with file")
        if enable_clip_w.value:
            staged.append("Clip")
        if enable_reproject_w.value:
            staged.append("Reproject")
        # Indices already present as bands are skipped by _apply_indices_feature,
        # and the selection survives an Edit run, so a plain "is anything
        # selected?" would keep claiming work after the indices were added.
        selected_idx = [str(i) for i in indices_select_w.value]
        if selected_idx:
            obj = state.get("current")
            da = (
                obj["Time_Series"]
                if isinstance(obj, xr.Dataset) and "Time_Series" in obj.data_vars
                else obj
            )
            try:
                present = {str(b).lower() for b in da.coords["band"].values}
            except Exception:
                present = set()
            if any(i.lower() not in present for i in selected_idx):
                staged.append("Indices")
        if _selected_composites():
            staged.append("Composites")
        return staged

    def _sync_edit_button(*_):
        """Put the number of staged tools on the Edit button.

        A multi-select pipeline's usual failure is a switch left on from the
        previous run; showing the count makes that visible before the click
        rather than in the Applied list afterwards.
        """
        try:
            n = len(_staged_features())
        except Exception:
            n = 0
        edit_btn.description = (
            f"Edit data cube ({n} staged)" if n else "Edit data cube"
        )
        edit_btn.tooltip = (
            "Will apply: " + ", ".join(_staged_features()) if n
            else "Switch on at least one tool above"
        )

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

                # 1) Slice
                current_obj, changed_slice, slice_msgs = _apply_slice_feature(current_obj)
                changed_any = changed_any or changed_slice
                messages.extend(slice_msgs)

                # 2) Filter by Cloud Coverage
                current_obj, changed_cloud, cloud_msgs = _apply_cloud_filter_feature(current_obj)
                changed_any = changed_any or changed_cloud
                messages.extend(cloud_msgs)

                # 2b) Filter by Scene Coverage (drop partial / faulty scenes)
                current_obj, changed_cov, cov_msgs = _apply_coverage_filter_feature(current_obj)
                changed_any = changed_any or changed_cov
                messages.extend(cov_msgs)

                # 3) Mask Clouds with Binary Masking File
                current_obj, changed_mask, mask_msgs = _apply_mask_clouds_feature(current_obj)
                changed_any = changed_any or changed_mask
                messages.extend(mask_msgs)

                # 4) Clip Raster
                current_obj, changed_clip, clip_msgs = _apply_clip_feature(current_obj)
                changed_any = changed_any or changed_clip
                messages.extend(clip_msgs)

                # 4b) Reproject (after clipping: the polygon is cut in the
                # cube's own projection, then the smaller result is warped)
                current_obj, changed_reproj, reproj_msgs = _apply_reproject_feature(current_obj)
                changed_any = changed_any or changed_reproj
                messages.extend(reproj_msgs)

                # 5) Calculate Spectral Indices
                current_obj, changed_idx, idx_msgs = _apply_indices_feature(current_obj)
                changed_any = changed_any or changed_idx
                messages.extend(idx_msgs)

                # 6) Temporal composites (stats)
                current_obj, changed_stats, stats_msgs = _apply_stats_feature(current_obj)
                changed_any = changed_any or changed_stats
                messages.extend(stats_msgs)

                state["current"] = current_obj

                # Refresh UI from updated current cube
                _populate_slice_widgets_from_current(select_all=True)
                _refresh_gif_band_options()
                _show_result_current()

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

                # Editing only previews the result in the Result panel - it never
                # writes a file. Saving is a deliberate click on 'Export current
                # result' (with the Temporal Composite, if any, applied there).

            try:
                result_acc.selected_index = 0
            except Exception:
                pass

        except Exception as e:
            _show_status(_friendly_error(e, "Editing"))

    def _on_update_clicked(_):
        """Standalone Update Data Cube action: fetch missing dates/bands for the
        loaded cube via _apply_update_feature, printing into the Update group."""
        if state["current"] is None:
            with update_out:
                clear_output()
                print("❌ Load a cube first.")
            return

        try:
            with update_out:
                clear_output()
                print("Updating data cube...")

                current_obj, changed, msgs = _apply_update_feature(state["current"])
                state["current"] = current_obj

                _populate_slice_widgets_from_current(select_all=True)
                _refresh_gif_band_options()
                _show_result_current()

                if changed:
                    print("✅ Update finished.")
                    if msgs:
                        print("Applied:")
                        for m in msgs:
                            print(f"- {m}")
                    # The update only extends the working cube in memory. Users
                    # read "Update finished" as "written to disk" and stop here,
                    # so spell out that the file is still the old one.
                    print(
                        "\nℹ️ Not written to disk yet. The new dates/bands are "
                        "in the working result only, the loaded file is "
                        "unchanged.\n"
                        "Next: open 'Result' below to check the updated cube, "
                        "then write it out with 'Export current result'."
                    )
                else:
                    print("✅ Update finished (no changes applied).")

                update_bands_w.value = ()
                _print_working_note()

            try:
                result_acc.selected_index = 0
            except Exception:
                pass

        except Exception as e:
            with update_out:
                clear_output()
                print(_friendly_error(e, "Update"))

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

                # export_stac() already prints "Export is done: ..." for the
                # netcdf and zarr modes; only add a line for the COG folder.
                if info.get("mode") not in ("netcdf", "zarr"):
                    print(f"✅ Export finished: {info['target']}")

                _print_working_note()

        except Exception as e:
            _show_status(_friendly_error(e, "Export"))

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
                if isinstance(state["current"], xr.Dataset) and da.name != "Time_Series":
                    print(f"ℹ️ Visualizing dataset variable: {da.name}")
                out = interactive_time_view(
                    stac=da,
                    widget_type="dropdown",
                    renderer=str(viz_renderer_w.value),
                )
                if out is not None:
                    display(out)
        except Exception as e:
            with viz_out:
                clear_output()
                print(_friendly_error(e, "Visualization"))

    def _current_gif_render_kwargs():
        """save_timeseries_gif kwargs for the active animation section."""
        sec = gif_section_w.value
        if sec == "band":
            if not gif_band_dd.value:
                raise ValueError("Select a band for the single-band animation.")
            kwargs = {"display_mode": "band", "band": str(gif_band_dd.value)}
        elif sec == "custom":
            if not (gif_r_dd.value and gif_g_dd.value and gif_b_dd.value):
                raise ValueError("Select R, G and B bands for the custom animation.")
            kwargs = {
                "display_mode": "custom",
                "rgb_bands": (
                    str(gif_r_dd.value),
                    str(gif_g_dd.value),
                    str(gif_b_dd.value),
                ),
            }
        else:
            return {"display_mode": gif_display_mode_w.value}

        p_lo, p_hi = (float(v) for v in gif_stretch_w.value)
        if p_hi <= p_lo:
            p_lo, p_hi = 2.0, 98.0
        kwargs.update(p_low=p_lo, p_high=p_hi)
        return kwargs

    def _on_make_gif_clicked(_):
        if state["current"] is None:
            with anim_out:
                clear_output()
                print("ℹ️ Load a cube first.")
            return

        try:
            da = _pick_dataarray_for_visualization(state["current"])

            if "time" not in da.dims:
                raise ValueError(
                    f"Animation generation requires a 'time' dimension. Found dims: {da.dims}. "
                    "If you are viewing a stats-only variable, use 'Reset to loaded cube' "
                    "or visualize 'Time_Series'."
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

            gif_kwargs = _current_gif_render_kwargs()

            with anim_out:
                clear_output()
                print("Generating animation GIF...")
                save_timeseries_gif(
                    da=da,
                    out_path=out_path,
                    fps=fps_val,
                    label=gif_label_w.value,
                    **gif_kwargs,
                )
                print(f"✅ Animation saved: {out_path}")

        except Exception as e:
            with anim_out:
                clear_output()
                print(_friendly_error(e, "Animation"))

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

    def _select_all_indices(_):
        indices_select_w.value = tuple(v for _, v in indices_select_w.options)

    def _clear_indices(_):
        indices_select_w.value = ()

    # ---------------------------------------------------------------------
    # Observe / wire
    # ---------------------------------------------------------------------
    browse_load_btn.on_click(_on_browse_load_clicked)
    browse_export_btn.on_click(_on_browse_export_clicked)
    browse_gif_btn.on_click(_on_browse_gif_clicked)
    browse_clip_btn.on_click(_on_browse_clip_clicked)
    browse_mask_file_btn.on_click(_on_browse_mask_file_clicked)

    load_cube_btn.on_click(_on_load_cube_clicked)
    layer_load_btn.on_click(_on_layer_load_clicked)
    reset_btn.on_click(_on_reset_clicked)
    edit_btn.on_click(_on_edit_clicked)
    export_current_btn.on_click(_on_export_current_clicked)

    # Keep the staged count on the Edit button in step with the controls it
    # reads. Every fixed control that _staged_features() looks at is listed here;
    # the runtime-created Custom Composites rows are covered by the call inside
    # _custom_sync_enabled instead.
    for _w in (
        slice_time_w, slice_band_w,
        enable_cloud_filter_w, enable_coverage_filter_w, enable_mask_clouds_w,
        enable_clip_w, enable_reproject_w,
        indices_select_w,
        comp_mean_w, comp_median_w, stats_select_w,
    ):
        _w.observe(_sync_edit_button, names="value")

    slice_time_all_btn.on_click(_select_all_dates)
    slice_time_clear_btn.on_click(_clear_dates)
    slice_band_all_btn.on_click(_select_all_bands)
    slice_band_clear_btn.on_click(_clear_bands)

    stats_all_btn.on_click(_select_all_stats)
    stats_clear_btn.on_click(_clear_stats)

    indices_all_btn.on_click(_select_all_indices)
    indices_clear_btn.on_click(_clear_indices)

    viz_dropdown_btn.on_click(_on_viz_dropdown_clicked)
    viz_make_gif_btn.on_click(_on_make_gif_clicked)

    export_mode_w.observe(lambda change: _set_export_mode_defaults(), names="value")
    gif_display_mode_w.observe(lambda change: _update_gif_output_suggestion(), names="value")
    gif_section_w.observe(
        lambda change: (
            _sync_gif_section_visibility(),
            _update_gif_output_suggestion(),
        ),
        names="value",
    )
    gif_band_dd.observe(lambda change: _update_gif_output_suggestion(), names="value")
    update_daterange_mode_w.observe(lambda change: _update_update_daterange_example(), names="value")

    # ---------------------------------------------------------------------
    # Layout helpers
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # Build layout
    # ---------------------------------------------------------------------
    header = widgets.HTML(
        "<div style='margin:0 0 4px 0; font-size:28px; font-weight:700;'>Data Cube Editor</div>"
    )

    subtitle = widgets.HTML(
        "<div style='display:flex; flex-wrap:wrap; gap:10px; margin:14px 0 8px 0;'>"
        # Step 1 - blue "load"
        "<div style='flex:1 1 200px; background:#f8fafc; border:1px solid #e5e7eb; "
        "border-left:4px solid #3b82f6; border-radius:8px; padding:8px 10px;'>"
        "<div style='font-weight:700; color:#1e3a8a; font-size:13px;'>1 &nbsp; Source</div>"
        "<div style='font-size:12px; color:#475569; margin-top:2px;'>Open a "
        "<b>NetCDF</b> or <b>Zarr</b> data cube, or mosaic several into one.</div></div>"
        # Step 2 - green "edit", matches the green Edit button
        "<div style='flex:1 1 200px; background:#f0fdf4; border:1px solid #dcfce7; "
        "border-left:4px solid #16a34a; border-radius:8px; padding:8px 10px;'>"
        "<div style='font-weight:700; color:#166534; font-size:13px;'>2 &nbsp; Edit &amp; inspect</div>"
        "<div style='font-size:12px; color:#475569; margin-top:2px;'>Switch on the "
        "tools you want in <b>Edit</b>, click <b>Edit data cube</b>, then check "
        "the <b>Result</b>. Repeat to chain edits.</div></div>"
        # Step 3 - orange "export", matches the orange Export button
        "<div style='flex:1 1 200px; background:#fff7ed; border:1px solid #fed7aa; "
        "border-left:4px solid #f97316; border-radius:8px; padding:8px 10px;'>"
        "<div style='font-weight:700; color:#9a3412; font-size:13px;'>3 &nbsp; Export</div>"
        "<div style='font-size:12px; color:#475569; margin-top:2px;'>Choose a format in "
        "<b>Export Options</b>, then click <b>Export Current Result</b>.</div></div>"
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

    # Body of the "Open a cube" source mode. The card around it (mode switch,
    # heading, and the shared Layer dropdown both modes hand over to) is
    # assembled further down, next to the mosaic body - see source_box.
    open_cube_box = widgets.VBox(
        [
            widgets.HTML("<div style='font-size:12px; color:#666;'>NetCDF and Zarr only (Geotiffs are not supported as editor input).</div>"),
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

    # Same two layout helpers the builder uses, so the Temporal Composites
    # section looks identical in both GUIs (both are pure CSS-class wrappers -
    # see .stac2cube-subpanel in gui_common's stylesheet).
    def _subpanel(children, accent=None):
        box = widgets.VBox(
            list(children),
            layout=widgets.Layout(width="100%", gap="6px"),
        )
        box.add_class("stac2cube-subpanel")
        if accent:
            box.add_class(f"stac2cube-subpanel-{accent}")
        return box

    def _line_divider():
        return widgets.HTML(
            "<div style='height:2px; background:#cbd5e1; border-radius:1px; "
            "margin:18px 0 16px 0;'></div>"
        )

    # Accent colours for the Edit card's stage headers. Deliberately the SAME
    # meanings the builder's field_group accents already carry, so nothing new
    # has to be learned (see .stac2cube-group-* in gui_common's stylesheet):
    #   turquoise = what gets taken out (scenes, bands, cloudy pixels)
    #   violet    = how the cube is laid out on the ground (grid, projection)
    #   green     = what gets added on top (new bands)
    #   blue      = temporal composites (already the accent of the stats panel
    #               in both GUIs)
    # Amber and red are deliberately absent: they mean warning and error here.
    _STAGE_ACCENTS = {
        "turquoise": "#14b8a6",
        "violet": "#8b5cf6",
        "green": "#16a34a",
        "blue": "#3b82f6",
    }

    def _card_title(text):
        """Heading of a top-level card (Source, Extend, Edit, ...).

        Deliberately larger than the 13px stage headers and group titles inside
        the cards: these five are the spine of the page, and at plain <b> size
        they were easy to scroll straight past.
        """
        return widgets.HTML(
            f"<div class='stac2cube-card-title' style='font-size:17px; "
            f"font-weight:700; color:#374151; margin:0 0 2px 0; "
            f"line-height:1.3;'>{text}</div>"
        )

    def _stage_header(number, title, subtitle, accent):
        """Header strip introducing one stage of the Edit pipeline.

        Plain HTML rather than a field_group wrapper on purpose. The items below
        it are Accordions, which already bring their own container and sage
        header bar; boxing them again would stack three tinted backgrounds
        (card, group, header) whose two innermost tints are nearly the same hue,
        and nesting containers is exactly what has raised stray horizontal
        scrollbars elsewhere in this file.

        The NUMBER carries the running order (the Edit button applies the stages
        top to bottom, see _on_edit_clicked); the COLOUR carries what kind of
        change the stage makes. Two channels, two separate jobs - a colour alone
        cannot express a sequence.
        """
        bar = _STAGE_ACCENTS[accent]
        return widgets.HTML(
            f"<div style='border-left:4px solid {bar}; padding:2px 0 2px 10px; "
            f"margin:14px 0 6px 0;'>"
            f"<div style='font-weight:700; font-size:13px; color:#374151;'>"
            f"<span style='color:{bar};'>{number}</span> &nbsp;{title}</div>"
            f"<div style='font-size:12px; color:#6b7280; margin-top:1px;'>"
            f"{subtitle}</div>"
            f"</div>"
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
                "This is only possible if the data cube is already cloud-detected. "
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

    # Scene coverage filter feature (drop partial / faulty / missing scenes)
    coverage_filter_controls = widgets.VBox(
        [
            enable_coverage_filter_w,
            _stacked_field_with_help(
                coverage_min_w, "Min coverage %", "scene_coverage_filter"
            ),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    coverage_filter_feature_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Keeps only scenes imaging at least the chosen % of the area, "
                "dropping partial ones (across-track / swath edge, or a faulty / "
                "partially-missing acquisition). Uses the stored "
                "<code>scene_coverage</code>, or measures it if the cube predates it."
                "</div>"
            ),
            coverage_filter_controls,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )

    coverage_filter_acc = widgets.Accordion(
        children=[coverage_filter_feature_box], selected_index=None
    )
    coverage_filter_acc.set_title(0, "Filter by Scene Coverage")
    coverage_filter_acc.layout = widgets.Layout(width="99%")

    # ------------------------------------------------------------------
    # Build Cloud Mask Cube (standalone export - same as builder's
    # "Export Mask as Binary File")
    # ------------------------------------------------------------------
    def _suggest_build_mask_path():
        lp = state.get("loaded_path")
        if not lp:
            return ""
        p = Path(lp)
        ext = ".zarr" if is_zarr_path(lp) else ".nc"
        # Strip a trailing _cr/_sr etc.? No - keep the cube's own stem so the mask
        # is unambiguously tied to this cube.
        return (p.parent / f"{p.stem}_mask_binary{ext}").as_posix()

    def _on_build_mask_clicked(_):
        with build_mask_out:
            clear_output()

        if state.get("loaded_path") is None:
            with build_mask_out:
                print("❌ Load a cube first.")
            return
        if _is_derived_path(state.get("loaded_path")):
            with build_mask_out:
                print(
                    "❌ Building a binary cloud mask needs the cube's own file, to "
                    "read the build parameters stored in it. This cube was "
                    "mosaicked in this session and has no file yet. Export it "
                    "below, then load the exported file and build the mask from "
                    "that."
                )
            return
        if state.get("loaded_var") != "Time_Series":
            with build_mask_out:
                print(
                    "❌ Building a binary cloud mask needs the 'Time_Series' "
                    f"cube (the loaded layer is '{state.get('loaded_var')}')."
                )
            return

        out_path = (build_mask_out_w.value or "").strip()
        if not out_path:
            out_path = _suggest_build_mask_path()
            build_mask_out_w.value = out_path

        p_out = Path(out_path)
        try:
            # All the work happens in stac2cube.cloud_masking.build_cloud_mask_cube
            # (validation, STAC re-query, exact date match to the cube, metadata
            # stamping, export); the GUI only wires it to the interface.
            with build_mask_out:
                print(
                    "Rebuilding the binary cloud mask from the cube's parameters..."
                )
                mask_out = build_cloud_mask_cube(
                    state["loaded_path"], output=out_path, q=True
                )

            if not p_out.exists():
                with build_mask_out:
                    print("❌ Build failed: the mask file was not created.")
                return

            # Report: date coverage vs the loaded cube.
            cube_days = set(
                np.asarray(state["loaded_original"]["time"].values)
                .astype("datetime64[D]")
            )
            mask_days = set(
                np.asarray(mask_out["time"].values).astype("datetime64[D]")
            )
            missing_days = sorted(cube_days - mask_days)
            m_bands = [str(b) for b in mask_out["band"].values]
            # Hand the file straight to its consumer: "Mask Clouds with Binary
            # Masking File" in stage 1 of the Edit card is the reason this tool
            # exists, and copying the path across by hand was the only link
            # between them. Never overwrite a path the user typed themselves.
            filled_mask_field = False
            if not (mask_file_w.value or "").strip():
                mask_file_w.value = out_path
                filled_mask_field = True

            with build_mask_out:
                clear_output()
                print(f"✅ Binary cloud mask exported: {out_path}")
                if filled_mask_field:
                    print(
                        "   Filled in as the mask file of 'Mask Clouds with "
                        "Binary Masking File' above - tick 'Enable masking' "
                        "there to apply it."
                    )
                print(
                    f"   Scenes: {int(mask_out.sizes.get('time', 0))} (matched to "
                    f"the cube's {len(cube_days)} date(s)) | bands: {', '.join(m_bands)}"
                )
                if missing_days:
                    print(
                        f"   ⚠️ {len(missing_days)} cube date(s) had no scene in the "
                        "fresh STAC query and are not in the mask:"
                    )
                    for d in missing_days:
                        print(f"      - {np.datetime_as_string(d, unit='D')}")
        except Exception as e:
            with build_mask_out:
                clear_output()
                print(_friendly_error(e, "Build binary cloud mask"))

    build_mask_btn.on_click(_on_build_mask_clicked)
    browse_build_mask_btn.on_click(_on_browse_build_mask_clicked)

    build_mask_input_row = widgets.HBox(
        [browse_build_mask_btn, build_mask_out_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    build_mask_input_box = widgets.VBox(
        [build_mask_input_row, build_mask_fc_box],
        layout=widgets.Layout(width="100%", gap="4px"),
    )

    build_mask_feature_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Build the <b>binary cloud mask</b> of the loaded data cube "
                "(SCL-based, 1=cloud, 0=clear). Use it to co-register or mask a "
                "data cube that isn't masked yet."
                "</div>"
            ),
            build_mask_archive_note_html,
            _stacked_field(build_mask_input_box, "Output binary mask (NetCDF/Zarr)"),
            build_mask_btn,
            build_mask_out,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    build_mask_acc = widgets.Accordion(children=[build_mask_feature_box], selected_index=None)
    build_mask_acc.set_title(0, "Build Cloud Mask Cube")
    build_mask_acc.layout = widgets.Layout(width="99%")

    # ------------------------------------------------------------------
    # Generate CSV Report (standalone export)
    # ------------------------------------------------------------------
    def _suggest_csv_report_path():
        lp = state.get("loaded_path")
        # A mosaic's "path" is a label, not a file (see _is_derived_path), so
        # there is nothing to name the report after; the user types one.
        if not lp or _is_derived_path(lp):
            return ""
        p = Path(lp)
        return (p.parent / f"{p.stem}_statistics.csv").as_posix()

    def _on_csv_report_clicked(_):
        with csv_report_out:
            clear_output()

        if state.get("current") is None:
            with csv_report_out:
                print("❌ Load a cube first.")
            return

        # Both refusals below are also raised by export_cube_statistics, but in
        # the words of a headless build ("rebuild with keep_timeseries=True").
        # Caught here so the advice names what to do IN THE EDITOR.
        current = state["current"]
        if state.get("loaded_var") not in (None, "Time_Series"):
            with csv_report_out:
                print(
                    "❌ A CSV report needs the 'Time_Series' cube (the loaded "
                    f"layer is '{state.get('loaded_var')}')."
                )
            return
        has_time_series = (
            "Time_Series" in current.data_vars
            if isinstance(current, xr.Dataset)
            else "time" in getattr(current, "dims", ())
        )
        if not has_time_series:
            with csv_report_out:
                print(
                    "❌ The current result holds temporal composites only, so it "
                    "has no dates to report. Tick 'Keep the full time series' in "
                    "Temporal Composites and edit again, or press Reset."
                )
            return

        out_path = (csv_report_out_w.value or "").strip()
        if not out_path:
            out_path = _suggest_csv_report_path()
            csv_report_out_w.value = out_path
        if not out_path:
            with csv_report_out:
                print("❌ Choose where to write the CSV report.")
            return

        try:
            with csv_report_out:
                print("Calculating statistics...")
                # The WORKING cube, not the file on disk: what the report
                # describes is what the Result panel shows, edits included.
                # q=True - the row counts are printed GUI-style below instead.
                df = export_cube_statistics(
                    state["current"], csv_path=out_path, q=True
                )

            # Distinct periods, not rows: every period contributes one row PER
            # BAND, so counting rows would report 24 dates for a 4-date cube.
            counts = {
                p: int(df.loc[df["period"] == p, "label"].nunique())
                for p in ("date", "year", "month")
            }
            n_bands = int(df["band"].nunique()) if len(df) else 0
            with csv_report_out:
                clear_output()
                print(f"✅ CSV report written: {out_path}")
                print(
                    f"   {len(df)} rows: {counts['date']} date(s), "
                    f"{counts['year']} year(s), {counts['month']} month(s) "
                    f"x {n_bands} band(s)."
                )
                print("   Describes the current result, with every edit applied.")
        except Exception as e:
            with csv_report_out:
                clear_output()
                print(_friendly_error(e, "Generate CSV report"))

    csv_report_btn.on_click(_on_csv_report_clicked)
    browse_csv_report_btn.on_click(_on_browse_csv_report_clicked)

    csv_report_input_row = widgets.HBox(
        [browse_csv_report_btn, csv_report_out_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    csv_report_input_box = widgets.VBox(
        [csv_report_input_row, csv_report_fc_box],
        layout=widgets.Layout(width="100%", gap="4px"),
    )

    csv_report_feature_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Per-band <b>statistics</b> of the current result as a table: "
                "mean, median, min, max and standard deviation, one row per "
                "date, per year and per month. Time series only - a cube of "
                "temporal composites alone has no dates to report."
                "</div>"
            ),
            _stacked_field(csv_report_input_box, "Output report (CSV)"),
            csv_report_btn,
            csv_report_out,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    csv_report_acc = widgets.Accordion(children=[csv_report_feature_box], selected_index=None)
    csv_report_acc.set_title(0, "Generate CSV Report")
    csv_report_acc.layout = widgets.Layout(width="99%")

    # Mask Clouds with Binary Masking File
    mask_file_input_row = widgets.HBox(
        [browse_mask_file_btn, mask_file_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    mask_file_input_box = widgets.VBox(
        [mask_file_input_row, mask_file_fc_box],
        layout=widgets.Layout(width="100%", gap="4px"),
    )
    mask_clouds_controls = widgets.VBox(
        [
            enable_mask_clouds_w,
            _stacked_field(mask_file_input_box, "Binary mask file (.nc)"),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )
    mask_clouds_feature_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Mask the loaded cube's clouds out using a binary cloud-mask file "
                "that was exported along (e.g. <code>test_mask_binary.nc</code>). "
                "The mask must come from the same cube (same grid); it is matched to "
                "the cube's current dates, so it still works after slicing."
                "</div>"
            ),
            mask_clouds_controls,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    mask_clouds_acc = widgets.Accordion(children=[mask_clouds_feature_box], selected_index=None)
    mask_clouds_acc.set_title(0, "Mask Clouds with Binary Masking File")
    mask_clouds_acc.layout = widgets.Layout(width="99%")

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

    # Reproject feature
    reproject_crs_box = widgets.VBox(
        [reproject_crs_w, reproject_crs_status_w],
        layout=widgets.Layout(width="100%", gap="4px"),
    )

    reproject_controls_box = widgets.VBox(
        [
            enable_reproject_w,
            _stacked_field_with_help(reproject_crs_box, "Target CRS", "reproject"),
            _stacked_field(reproject_res_w, "Pixel size (m)"),
            _stacked_field(reproject_resampling_w, "Resampling"),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    reproject_feature_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Put the data cube into another projection, for example "
                "<code>EPSG:3035</code>. Reprojection resamples the pixels, so "
                "do it once and keep the original cube."
                "</div>"
            ),
            reproject_controls_box,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )

    reproject_acc = widgets.Accordion(
        children=[reproject_feature_box], selected_index=None
    )
    reproject_acc.set_title(0, "Reproject Data Cube")
    reproject_acc.layout = widgets.Layout(width="99%")

    # Spectral indices feature
    indices_inner_widget = widgets.VBox(
        [
            indices_select_w,
            widgets.HBox([indices_all_btn, indices_clear_btn], layout=widgets.Layout(gap="6px")),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    indices_feature_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Calculate spectral indices from the cube's spectral bands and append them as new bands. "
                "Available indices depend on the cube's mission; missing required bands are reported in Status."
                "</div>"
            ),
            _stacked_field_with_help(indices_inner_widget, "Indices", "spectral_indices"),
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )

    indices_acc = widgets.Accordion(children=[indices_feature_box], selected_index=None)
    indices_acc.set_title(0, "Calculate Spectral Indices")
    indices_acc.layout = widgets.Layout(width="99%")

    # Column captions for the Custom Composites rows. The widths mirror the row
    # widgets in _custom_row_widgets, so the captions sit above their fields.
    def _custom_caption(text, width):
        return widgets.HTML(
            f"<div style='font-size:11px; font-weight:600; color:#6b7280;'>{text}</div>",
            layout=widgets.Layout(width=width),
        )

    _custom_header_row = widgets.HBox(
        [
            _custom_caption("Period", "130px"),
            _custom_caption("From", "105px"),
            _custom_caption("To", "105px"),
            _custom_caption("Statistic", "105px"),
            _custom_caption("Name", "150px"),
            _custom_caption("", "38px"),
        ],
        layout=widgets.Layout(width="100%", gap="4px", flex_flow="row wrap"),
    )

    # Temporal Composites: the mean/median Temporal Composite dropdown that used
    # to live in Export Options is now the two promoted checkboxes here, plus
    # "Keep the full time series" - one section for one concept, matching the
    # Data Cube Builder.
    stats_inner_widget = widgets.VBox(
        [
            stats_select_w,
            widgets.HBox([stats_all_btn, stats_clear_btn], layout=widgets.Layout(gap="6px")),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    # Stats help toggle built manually (not via _stacked_field_with_help) so the
    # "Stats Explanation ?" row sits AFTER the blue info box, matching the builder.
    _stats_help_btn = _help_button()
    _stats_help_box = widgets.HTML(
        value=HELP_HTML.get("stats", ""),
        layout=widgets.Layout(
            display="none",
            border="1px solid #dbeafe",
            padding="8px",
            border_radius="8px",
            margin="2px 0 2px 0",
            width="100%",
        ),
    )

    def _toggle_stats_help(_):
        _stats_help_box.layout.display = (
            "" if _stats_help_box.layout.display == "none" else "none"
        )

    _stats_help_btn.on_click(_toggle_stats_help)

    _stats_explain_row = widgets.HBox(
        [
            widgets.HTML(
                "<div style='font-weight:500; font-size:12px; color:#374151;'>"
                "Composite Explanation</div>"
            ),
            _stats_help_btn,
        ],
        layout=widgets.Layout(align_items="center", gap="6px"),
    )

    stats_feature_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#475569; margin:0 0 2px 0;'>"
                "Statistics calculated over the dates of the current result."
                "</div>"
            ),
            _subpanel([comp_mean_w, comp_median_w], accent="blue"),
            widgets.HTML("<div style='height:10px;'></div>"),
            _field_group(
                "More Composites",
                [
                    _stats_explain_row,
                    _stats_help_box,
                    stats_inner_widget,
                ],
                subtitle="Minimum, maximum and standard deviation, plus "
                "monthly and annual composites.",
                collapsible=True,
                open=False,
            ),
            widgets.HTML("<div style='height:6px;'></div>"),
            _field_group(
                "Custom Composites",
                [
                    _custom_header_row,
                    custom_rows_box,
                    custom_add_btn,
                    custom_error_note,
                    widgets.HTML(
                        "<div style='font-size:12px; color:#6b7280;'>"
                        "<b>Every year</b> repeats the period in every year of "
                        "the cube and saves one image per year, named "
                        "<code>name_2024</code>, <code>name_2025</code>, ... "
                        "<b>Single window</b> saves one image."
                        "</div>"
                    ),
                ],
                subtitle="Your own period, for example a growing season.",
                collapsible=True,
                open=False,
                help_html=HELP_HTML.get("custom_composites", ""),
            ),
            _line_divider(),
            keep_ts_w,
            keep_ts_note,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )

    stats_acc = widgets.Accordion(children=[stats_feature_box], selected_index=None)
    stats_acc.set_title(0, "Temporal Composites")
    stats_acc.layout = widgets.Layout(width="99%")



    # Update Data Cube feature: two always-visible groups (Date Update / Band
    # Update) below the enable checkbox, mirroring the builder's Time Period
    # and Bands fields.
    # The OR divider lives inside the simple box so it hides together with the
    # From/To pickers when "Use a seasonal date range" is ticked (builder-style).
    _upd_date_simple_box = widgets.VBox(
        [
            _stacked_field(update_date_from_w, "From"),
            _stacked_field(update_date_to_w, "To"),
            widgets.HTML(
                "<div style='display:flex; align-items:center; gap:12px; margin:16px 0 12px 0;'>"
                "<span style='flex:1; height:2px; background:#cbd5e1;'></span>"
                "<span style='font-size:14px; font-weight:700; color:#6b7280; "
                "letter-spacing:2px;'>OR</span>"
                "<span style='flex:1; height:2px; background:#cbd5e1;'></span>"
                "</div>"
            ),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    # Daterange field with the red MM-DD hint between its label and the input
    # box (exactly like the builder's advanced Time Period field).
    _upd_dr_field = _stacked_field(update_daterange_w, "Daterange")
    _upd_dr_field.children = [
        _upd_dr_field.children[0],
        widgets.HTML(
            "<div style='font-size:12px; color:#b91c1c; margin:0;'>"
            "season &rarr; <code>\"MM-DD\" - \"MM-DD\"</code></div>"
        ),
        _upd_dr_field.children[1],
    ]
    _upd_date_advanced_box = widgets.VBox(
        [
            _stacked_field_with_help(update_daterange_mode_w, "Season mode", "daterange_mode"),
            _upd_dr_field,
        ],
        layout=widgets.Layout(width="100%", gap="6px", display="none"),
    )

    def _update_update_date_inputs_visibility(*_):
        advanced = update_advanced_dates_w.value
        _upd_date_simple_box.layout.display = "none" if advanced else ""
        _upd_date_advanced_box.layout.display = "" if advanced else "none"

    update_advanced_dates_w.observe(
        lambda change: _update_update_date_inputs_visibility(), names="value"
    )
    _update_update_date_inputs_visibility()

    update_date_group = _field_group(
        "Date Update",
        [_upd_date_simple_box, update_advanced_dates_w, _upd_date_advanced_box],
        subtitle="Fetch the scenes missing from the cube in this period.",
    )

    update_band_group = _field_group(
        "Band Update",
        [update_bands_note_html, update_bands_w],
        subtitle="Add mission bands the loaded cube does not contain yet.",
    )

    update_feature_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Fetch missing dates and bands for the loaded data cube in the given date range. "
                "Instead of re-building the entire data cube, it only computes the new scenes and bands.<br>"
                "<b>This feature is recommended to be used alone without in sequence with other features.</b>"
                "</div>"
            ),
            update_archive_note_html,
            update_date_group,
            update_band_group,
            update_run_btn,
            update_out,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    update_run_btn.on_click(_on_update_clicked)

    update_acc = widgets.Accordion(children=[update_feature_box], selected_index=None)
    update_acc.set_title(0, "Update Data Cube (Date and/or band)")
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
            # The Temporal Composite dropdown moved into the Temporal
            # Composites feature above, so this section is about format and
            # path again.
            _stacked_field_with_help(export_mode_w, "Export mode", "export_mode"),
            # NetCDF-only switches sit with the mode that owns them, above the
            # shared Output field (same order as the Data Cube Builder).
            export_compress_w,
            export_compress_warn_html,
            export_vrt_w,
            export_vrt_note_html,
            _stacked_field(export_input_box, "Output"),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    export_acc = widgets.Accordion(children=[export_box], selected_index=None)
    export_acc.set_title(0, "Export Options")
    export_acc.layout = widgets.Layout(width="99%")

    # Features group
    # ------------------------------------------------------------------
    # Mosaic Data Cubes - behaviour
    # ------------------------------------------------------------------
    def _mosaic_paths():
        return list(state.get("mosaic_paths") or [])

    def _mosaic_move(idx, delta):
        paths = _mosaic_paths()
        j = idx + delta
        if 0 <= idx < len(paths) and 0 <= j < len(paths):
            paths[idx], paths[j] = paths[j], paths[idx]
            state["mosaic_paths"] = paths
            _mosaic_refresh()

    def _mosaic_remove(idx):
        paths = _mosaic_paths()
        if 0 <= idx < len(paths):
            paths.pop(idx)
            state["mosaic_paths"] = paths
            _mosaic_refresh()

    def _mosaic_add_text(raw):
        """Add one or many cubes. Several paths can be pasted at once.

        Split on newlines and semicolons but NOT commas: a Windows path cannot
        contain either of the first two, while a comma is legal in a folder
        name and splitting on it would quietly break such a path in half.
        """
        if not raw:
            return
        candidates = []
        for chunk in str(raw).replace(";", "\n").splitlines():
            chunk = chunk.strip().strip('"').strip("'")
            if chunk:
                candidates.append(chunk)

        added, skipped, bad = [], [], []
        paths = _mosaic_paths()
        for cand in candidates:
            resolved = resolve_cube_path(cand)
            # A plain FOLDER is not a mistake here - it is how "Add whole
            # folder" is aimed, and it is what the file browser hands back when
            # a directory is selected. Passing silently instead of reporting it
            # as a bad cube: complaining about a path the user was told to pick
            # reads as an error when nothing is wrong.
            if Path(resolved).is_dir() and not is_zarr_path(resolved):
                continue
            if not (resolved.lower().endswith(".nc") or is_zarr_path(resolved)):
                bad.append(f"{cand} (not a .nc file or .zarr store)")
                continue
            if not Path(resolved).exists():
                bad.append(f"{cand} (not found)")
                continue
            if resolved in paths:
                skipped.append(resolved)
                continue
            paths.append(resolved)
            added.append(resolved)

        state["mosaic_paths"] = paths
        if added:
            # Where the browser should reopen next (see
            # _sync_mosaic_filechooser_from_text).
            state["mosaic_last_dir"] = str(Path(added[-1]).parent)
        _mosaic_refresh()

        msgs = []
        if added:
            msgs.append(f"✅ Added {len(added)} cube(s).")
        if skipped:
            msgs.append(f"ℹ️ Already in the list: {len(skipped)}.")
        for b in bad:
            msgs.append(f"❌ {b}")
        if msgs:
            with mosaic_out:
                clear_output()
                for m in msgs:
                    print(m)

    def _on_mosaic_path_submitted(change=None):
        """Enter in the path box adds that cube.

        The Add button is gone (the browser is the normal way in), so the box
        needs its own commit or a typed path would have no way to reach the
        list. A folder is left in the box instead - that is what "Add whole
        folder" reads.
        """
        raw = (mosaic_path_w.value or "").strip()
        if not raw:
            return
        resolved = resolve_cube_path(raw)
        if Path(resolved).is_dir() and not is_zarr_path(resolved):
            return
        _mosaic_add_text(raw)
        mosaic_path_w.value = ""

    def _on_mosaic_add_folder_clicked(_):
        """Add every cube in the folder of whatever is in the box.

        The reason this button exists: an area split into pieces is a folder of
        dozens of cubes, and adding them one at a time is not a workflow.
        """
        raw = (mosaic_path_w.value or "").strip()
        if not raw:
            # Fall back to whatever folder the browser is showing, so the button
            # works after browsing instead of demanding the path be typed too.
            try:
                raw = (mosaic_fc.selected_path or "") if mosaic_fc else ""
            except Exception:
                raw = ""
        if not raw:
            with mosaic_out:
                clear_output()
                print(
                    "ℹ️ Type a folder path above (or open one with 📂) first, "
                    "then press this to add every cube in it."
                )
            return
        p = Path(resolve_cube_path(raw))
        folder = p if p.is_dir() and not is_zarr_path(str(p)) else p.parent
        if not folder.exists():
            with mosaic_out:
                clear_output()
                print(f"❌ Folder not found: {folder}")
            return
        found = sorted(
            [str(f) for f in folder.glob("*.nc")]
            + [str(f) for f in folder.glob("*.zarr")]
        )
        if not found:
            with mosaic_out:
                clear_output()
                print(f"ℹ️ No .nc or .zarr cubes in {folder}")
            return
        _mosaic_add_text("\n".join(found))
        mosaic_path_w.value = ""

    def _on_mosaic_clear_clicked(_):
        state["mosaic_paths"] = []
        _mosaic_refresh()
        with mosaic_out:
            clear_output()

    def _mosaic_refresh_list():
        """Rebuild the queued-cube rows.

        Each row carries its own position, and the arrows change it, because the
        order IS the priority order for the default overlap setting - a list the
        user cannot reorder would make that setting unusable.
        """
        paths = _mosaic_paths()
        rows = []
        for i, p in enumerate(paths):
            up = widgets.Button(
                icon="arrow-up", tooltip="Move up (higher priority)",
                layout=widgets.Layout(width="32px", height="26px", padding="0px"),
                disabled=(i == 0),
            )
            down = widgets.Button(
                icon="arrow-down", tooltip="Move down",
                layout=widgets.Layout(width="32px", height="26px", padding="0px"),
                disabled=(i == len(paths) - 1),
            )
            rm = widgets.Button(
                icon="times", tooltip="Remove from the list",
                layout=widgets.Layout(width="32px", height="26px", padding="0px"),
            )
            for btn in (up, down, rm):
                btn.style.button_color = "#f3f4f6"
            # idx bound as a default argument: without it every handler would
            # close over the LAST i of the loop and every row would edit the
            # same entry.
            up.on_click(lambda _b, idx=i: _mosaic_move(idx, -1))
            down.on_click(lambda _b, idx=i: _mosaic_move(idx, +1))
            rm.on_click(lambda _b, idx=i: _mosaic_remove(idx))
            label = widgets.HTML(
                f"<div style='font-size:12px; color:#334155; overflow:hidden; "
                f"text-overflow:ellipsis; white-space:nowrap;' title='{p}'>"
                f"<b>{i + 1}.</b> {Path(p).name}</div>",
                # min_width:0 lets a long file name be ellipsised instead of
                # forcing the row - and the whole card - wider than the panel.
                layout=widgets.Layout(
                    flex="1 1 auto", width="auto", overflow="hidden",
                    min_width="0",
                ),
            )
            rows.append(widgets.HBox(
                [label, up, down, rm],
                layout=widgets.Layout(
                    width="100%", gap="4px", align_items="center",
                    overflow="hidden", min_width="0",
                ),
            ))
        mosaic_list_box.children = rows
        if not paths:
            mosaic_count_w.value = (
                "<div style='font-size:12px; color:#64748b;'>Add at least two "
                "cubes.</div>"
            )
        else:
            mosaic_count_w.value = (
                f"<div style='font-size:12px; color:#334155;'><b>{len(paths)}</b> "
                "cubes - the first wins where they overlap.</div>"
            )

    def _mosaic_refresh_metadata():
        """Fill the layer and projection pickers from the selected cubes.

        Metadata only - mosaic_layers opens each cube lazily and closes it - so
        this can run on every list change without touching a pixel.
        """
        paths = _mosaic_paths()
        if len(paths) < 2:
            mosaic_layers_w.options = []
            mosaic_layers_w.value = ()
            mosaic_layers_note_w.value = ""
            mosaic_crs_detected_w.options = [(_MOSAIC_CRS_AUTO_LABEL, _MOSAIC_CRS_AUTO)]
            mosaic_crs_detected_w.value = _MOSAIC_CRS_AUTO
            return
        try:
            info = mosaic_layers(paths)
        except Exception as exc:
            mosaic_layers_w.options = []
            mosaic_layers_w.value = ()
            mosaic_layers_note_w.value = (
                "<div style='font-size:12px; color:#991b1b;'>Could not read the "
                f"selected cubes: {exc}</div>"
            )
            return
        # Kept for _sync_mosaic_controls, which uses the projections actually
        # present to tell a typed CRS apart from one nothing uses. Stored HERE,
        # on every list change, rather than only in the Check handler - reading
        # it there alone meant a user who typed a CRS without pressing Check
        # first got no warning and no automatic resampling.
        state["mosaic_scan"] = info

        common = info["common"]
        previous = set(mosaic_layers_w.value or ())
        mosaic_layers_w.options = [
            (_layer_display_name(name), name) for name in common
        ]
        # All selected by default; a selection already made is kept so a user
        # who unticked a layer does not get it back when another cube is added.
        keep = tuple(n for n in common if n in previous) if previous else tuple(common)
        mosaic_layers_w.value = keep or tuple(common)

        notes = []
        if info["only_some"]:
            notes.append(
                "Not in every cube, so not available to merge: "
                + ", ".join(info["only_some"]) + "."
            )
        if not common:
            notes.append("These cubes share no layer, so they cannot be merged.")
        mosaic_layers_note_w.value = (
            "<div style='font-size:12px; color:#92400e;'>" + " ".join(notes) + "</div>"
            if notes else ""
        )

        opts = [(_MOSAIC_CRS_AUTO_LABEL, _MOSAIC_CRS_AUTO)]
        for crs_name, n in (info["crs_counts"] or {}).items():
            opts.append((f"{crs_name} - {n} of {len(paths)} cube(s)", crs_name))
        previous_crs = mosaic_crs_detected_w.value
        mosaic_crs_detected_w.options = opts
        if previous_crs in [o[1] for o in opts]:
            mosaic_crs_detected_w.value = previous_crs

    def _mosaic_refresh():
        _mosaic_refresh_list()
        _mosaic_refresh_metadata()
        _sync_mosaic_controls()

    def _sync_mosaic_controls(*_):
        """Grey out what cannot apply, and say what a typed CRS would do.

        A user-defined CRS overrides the dropdown, and one that none of the
        cubes uses means every cube must be warped - which the default refusal
        would reject. Saying so here (and ticking the box) is the difference
        between a setting that works and an error the user cannot act on.
        """
        text = (mosaic_crs_user_w.value or "").strip()
        mosaic_crs_detected_w.disabled = bool(text)
        mosaic_resampling_w.disabled = not bool(mosaic_allow_resample_w.value)
        mosaic_run_btn.disabled = len(_mosaic_paths()) < 2

        if not text:
            mosaic_crs_status_w.value = ""
            return
        try:
            canonical = validate_target_crs(text)
        except Exception as exc:
            mosaic_crs_status_w.value = (
                "<div style='font-size:12px; color:#991b1b; background:#fef2f2; "
                "border:1px solid #fecaca; border-radius:6px; padding:6px 8px;'>"
                f"✗ {exc}</div>"
            )
            return
        in_use = set()
        try:
            info = state.get("mosaic_scan") or {}
            in_use = set((info.get("crs_counts") or {}).keys())
        except Exception:
            pass
        extra = ""
        if in_use and canonical not in in_use:
            extra = (
                " None of your cubes uses it, so every cube will be resampled - "
                "<b>Allow resampling</b> has been switched on under Advanced."
            )
            if not mosaic_allow_resample_w.value:
                mosaic_allow_resample_w.value = True
        mosaic_crs_status_w.value = (
            "<div style='font-size:12px; color:#166534;'>✓ mosaicking into "
            f"<b>{canonical}</b>.{extra}</div>"
        )

    def _sync_mosaic_filechooser_from_text():
        """Open the browser wherever the path box points.

        Without this the chooser always reopened at the folder it was
        constructed with (the working directory), so pasting a path and then
        clicking 📂 ignored what had just been typed - the one flow this tool
        is built around. Every other browse button in the editor does the same
        sync; this one was simply missing it.
        """
        if not filechooser_available or mosaic_fc is None:
            return
        current = (mosaic_path_w.value or "").strip()
        if not current:
            # The box empties itself once a cube path has been added, so falling
            # back to the folder last used keeps the browser where the user was
            # working instead of jumping back to the working directory.
            current = state.get("mosaic_last_dir") or ""
        if not current:
            return
        start_dir = _existing_dir_or_parent(current)
        # A folder in the box IS the target, so no filename to preselect; a
        # cube path preselects that file inside its folder.
        p = Path(current)
        suggested_name = "" if p.is_dir() and not is_zarr_path(current) else p.name
        try:
            mosaic_fc.reset(path=start_dir, filename=suggested_name)
        except Exception:
            try:
                mosaic_fc.default_path = start_dir
                mosaic_fc.default_filename = suggested_name
            except Exception:
                pass

    def _on_browse_mosaic_clicked(_):
        if not filechooser_available or mosaic_fc is None:
            with mosaic_out:
                clear_output()
                print(
                    "ℹ️ Optional dependency 'ipyfilechooser' is not available. "
                    "Type the cube paths instead (one per line)."
                )
            return
        _sync_mosaic_filechooser_from_text()
        _toggle_box_display(mosaic_fc_box)

    def _on_mosaic_fc_selected(chooser=None):
        """A picked CUBE goes straight into the list; a picked FOLDER goes into
        the box, ready for "Add whole folder".

        Two selections, two meanings, and the browser is the only way in now
        that the Add button is gone - so it has to handle both rather than
        rejecting one of them.
        """
        try:
            selected = (mosaic_fc.selected or "").strip()
        except Exception:
            selected = ""
        if selected:
            resolved = resolve_cube_path(selected)
            if Path(resolved).is_dir() and not is_zarr_path(resolved):
                mosaic_path_w.value = resolved
                with mosaic_out:
                    clear_output()
                    print(
                        f"📂 {resolved}\n"
                        "Press 'Add whole folder' to add every cube in it."
                    )
            else:
                _mosaic_add_text(selected)
        mosaic_fc_box.layout.display = "none"

    def _mosaic_effective_crs():
        text = (mosaic_crs_user_w.value or "").strip()
        if text:
            return validate_target_crs(text)
        chosen = mosaic_crs_detected_w.value
        return None if chosen in (None, _MOSAIC_CRS_AUTO) else chosen

    def _on_mosaic_check_clicked(_):
        """Report what the cubes are, before anything is merged.

        Metadata only. It exists because the union grid is rectangular while
        the cubes usually are not, so mosaicking pieces that were split to stay
        small can rebuild a very large, mostly empty cube - and that is worth
        seeing as a number before starting, not after.
        """
        paths = _mosaic_paths()
        with mosaic_out:
            clear_output()
            if len(paths) < 2:
                print("ℹ️ Add at least two cubes first.")
                return
            try:
                info = mosaic_layers(paths)
                state["mosaic_scan"] = info
                print(f"{len(paths)} cube(s) selected.\n")

                print("Layers that can be merged: "
                      + (", ".join(info["common"]) or "(none)"))
                if info["only_some"]:
                    print("Not in every cube (cannot be merged): "
                          + ", ".join(info["only_some"]))

                counts = info["crs_counts"] or {}
                if len(counts) <= 1:
                    print("\nProjection: "
                          + (next(iter(counts), "unknown"))
                          + " (all cubes)")
                else:
                    print("\nProjections (a mosaic uses one; the others are "
                          "resampled into it):")
                    for crs_name, n in counts.items():
                        print(f"   {crs_name}: {n} cube(s)")

                # Grid + size estimate, read from the coordinates only.
                import numpy as _np

                target = _mosaic_effective_crs() or next(iter(counts), None)
                xmin = ymin = _np.inf
                xmax = ymax = -_np.inf
                res = None
                same_grid = 0
                px_sum = 0
                # Per-LAYER shape of the non-spatial dimensions, and the longest
                # time axis any cube has. Sized per layer rather than once for
                # all of them because a cube can hold a time series AND a
                # composite: multiplying the composite by the number of dates
                # too would overstate the result substantially.
                layer_extra = {}
                n_time = 1
                for p in paths:
                    ds = open_cube(p, chunks="frames")
                    try:
                        x = _np.asarray(ds["x"].values, dtype="float64")
                        y = _np.asarray(ds["y"].values, dtype="float64")
                        rx = abs(float(x[1] - x[0]))
                        res = rx if res is None else min(res, rx)
                        px_sum += x.size * y.size
                        if "time" in ds.dims:
                            n_time = max(n_time, int(ds.sizes["time"]))
                        for name in info["common"]:
                            da = ds.get(name)
                            if da is None:
                                continue
                            extra = 1
                            for d in da.dims:
                                if d not in ("y", "x", "time"):
                                    extra *= int(da.sizes[d])
                            layer_extra[name] = (
                                extra, "time" in da.dims
                            )
                        cube_crs = info["crs"].get(Path(p).name or p, None)
                        if target is None or cube_crs == target:
                            same_grid += 1
                            xmin, xmax = min(xmin, x.min()), max(xmax, x.max())
                            ymin, ymax = min(ymin, y.min()), max(ymax, y.max())
                    finally:
                        try:
                            ds.close()
                        except Exception:
                            pass

                print(f"\nCubes already on the target projection: "
                      f"{same_grid} of {len(paths)}")
                if same_grid < len(paths):
                    print("   The rest need 'Allow resampling' under Advanced.")

                if res and _np.isfinite(xmin):
                    nx = int(round((xmax - xmin) / res)) + 1
                    ny = int(round((ymax - ymin) / res)) + 1
                    wanted = list(mosaic_layers_w.value or info["common"])
                    total = 0
                    for name in wanted:
                        extra, has_time = layer_extra.get(name, (1, False))
                        total += (
                            nx * ny * extra * (n_time if has_time else 1) * 4
                        )
                    fill = 100.0 * px_sum / max(nx * ny, 1)
                    print(f"\nMosaic grid: {ny} x {nx} pixels "
                          f"({nx * ny / 1e6:.1f} megapixels)")
                    print(f"Estimated size: {_human_readable_bytes(total)}")
                    print(f"Your cubes fill about {fill:.0f}% of it - the rest "
                          "is empty area between them.")
                    if fill < 40:
                        print("   ⚠️ Mostly empty. Pieces spread over a wide "
                              "area rebuild a large, sparse cube.")
                print("\nℹ️ Nothing is mosaicked, this is only a check.")
            except Exception as e:
                print(_friendly_error(e, "Check cubes"))

    def _on_mosaic_run_clicked(_):
        paths = _mosaic_paths()
        with mosaic_out:
            clear_output()
            if len(paths) < 2:
                print("ℹ️ Add at least two cubes to mosaic.")
                return
            try:
                chosen_layers = list(mosaic_layers_w.value or ())
                res_text = (mosaic_resolution_w.value or "").strip()
                resolution = float(res_text) if res_text else None

                print(f"Mosaicking {len(paths)} cubes...\n", flush=True)
                merged = mosaic_cubes(
                    paths,
                    overlap=mosaic_overlap_w.value,
                    time_join=mosaic_time_join_w.value,
                    band_join=mosaic_band_join_w.value,
                    layers=chosen_layers or None,
                    crs=_mosaic_effective_crs(),
                    resolution=resolution,
                    resampling=mosaic_resampling_w.value,
                    on_grid_mismatch=(
                        "resample" if mosaic_allow_resample_w.value else "raise"
                    ),
                    # source_map is deliberately NOT offered here. Its two extra
                    # rasters turn every mosaic into a multi-layer cube, which
                    # forces the result through the Layer-dropdown handoff
                    # instead of loading straight into the editor - a second
                    # loading step for a diagnostic most runs do not need. The
                    # headless mosaic_cubes(source_map=True) still provides it.
                    strict=not bool(mosaic_strict_w.value),
                    report=True,
                )

                # Hand the result to the ORDINARY load path. The editor works on
                # one layer at a time (state["current"] is a DataArray), so a
                # multi-layer mosaic goes through the same Layer dropdown a
                # multi-layer file does. From here on the mosaic simply IS the
                # loaded cube and every other feature applies to it.
                _prev = state.get("loaded_ds")
                if _prev is not None:
                    try:
                        _prev.close()
                    except Exception:
                        pass
                state["loaded_ds"] = merged
                # The mosaic is lazy and still reads from the input files, so
                # their handles must stay open for as long as it is in use.
                state["mosaic_inputs"] = list(paths)

                layers_found = _raster_layer_names(merged)
                pseudo_path = f"<mosaic of {len(paths)} cubes>"
                if len(layers_found) == 1:
                    layer_select_w.options = []
                    layer_select_box.layout.display = "none"
                    _finalize_load(pseudo_path, merged, layers_found[0])
                    print(
                        f"\n✅ Mosaic ready: {layers_found[0]}."
                        "\n\nℹ️ Not written to disk yet. The mosaic is the "
                        "working result only, no file has been created.\n"
                        "Next: open 'Result' below to check it, then write it "
                        "out with 'Export current result'."
                    )
                else:
                    layer_select_w.options = _layer_dropdown_options(
                        merged, layers_found
                    )
                    layer_select_w.value = (
                        "Time_Series" if "Time_Series" in layers_found
                        else layers_found[0]
                    )
                    layer_select_box.layout.display = ""
                    # The names 'Load selected layer' actually reads. Setting
                    # loaded_ds/loaded_path instead left that button reporting
                    # "Load a cube first" on a mosaic that was sitting right
                    # there, with no way to reach any of its layers.
                    state["pending_ds"] = merged
                    state["pending_path"] = pseudo_path
                    state["current"] = None
                    print(
                        f"\n✅ Mosaic ready with {len(layers_found)} layers: "
                        + ", ".join(layers_found)
                        + "\nPick one in the 'Layer' dropdown under Loading and "
                        "click 'Load selected layer' to start editing it."
                        "\n\nℹ️ Not written to disk yet. The mosaic exists in "
                        "this session only, so export it once the layer is "
                        "loaded and ready."
                    )
            except Exception as e:
                print(_friendly_error(e, "Mosaic data cubes"))

    browse_mosaic_btn.on_click(_on_browse_mosaic_clicked)
    mosaic_add_folder_btn.on_click(_on_mosaic_add_folder_clicked)
    mosaic_path_w.observe(_on_mosaic_path_submitted, names="value")
    mosaic_clear_btn.on_click(_on_mosaic_clear_clicked)
    mosaic_check_btn.on_click(_on_mosaic_check_clicked)
    mosaic_run_btn.on_click(_on_mosaic_run_clicked)
    mosaic_crs_user_w.observe(_sync_mosaic_controls, names="value")
    mosaic_allow_resample_w.observe(_sync_mosaic_controls, names="value")
    if filechooser_available and mosaic_fc is not None:
        try:
            mosaic_fc.register_callback(_on_mosaic_fc_selected)
        except Exception:
            pass

    # The text box FLEXES instead of claiming width:100%. With three children in
    # the row - browse, box, Add - a 100%-wide middle child plus two fixed
    # buttons adds up to more than the row, and the overflow is what puts the
    # horizontal scrollbar back under the card. min_width:0 is the other half:
    # without it a flex item refuses to shrink below its content width, so a
    # long path re-introduces the same overflow.
    mosaic_input_row = widgets.HBox(
        [browse_mosaic_btn, mosaic_path_w],
        layout=widgets.Layout(
            width="100%", gap="6px", align_items="center",
            overflow="hidden", min_width="0",
        ),
    )
    mosaic_input_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "📂 pick a cube either one by one, or select a folder and click "
                "<b>Add whole folder</b>."
                "</div>"
            ),
            mosaic_input_row,
            mosaic_fc_box,
            widgets.HBox(
                [mosaic_add_folder_btn, mosaic_clear_btn],
                layout=widgets.Layout(
                    gap="6px", flex_flow="row wrap", width="100%",
                    overflow="hidden", min_width="0",
                ),
            ),
        ],
        layout=widgets.Layout(
            width="100%", gap="4px", overflow="hidden", min_width="0",
        ),
    )

    mosaic_advanced_box = widgets.VBox(
        [
            mosaic_allow_resample_w,
            _stacked_field_with_help(
                mosaic_resampling_w, "Resampling method", "mosaic_resampling"
            ),
            _stacked_field_with_help(
                mosaic_resolution_w, "Pixel size (m)", "mosaic_pixel_size"
            ),
            mosaic_strict_w,
        ],
        layout=widgets.Layout(
            width="100%", gap="6px", overflow="hidden", min_width="0",
        ),
    )
    mosaic_advanced_acc = widgets.Accordion(
        children=[mosaic_advanced_box], selected_index=None
    )
    mosaic_advanced_acc.set_title(0, "Advanced")
    mosaic_advanced_acc.layout = widgets.Layout(width="100%", overflow="hidden")

    mosaic_feature_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Join several cubes into one. The result becomes the cube you "
                "are editing."
                "</div>"
            ),
            # No help icon: "Data cubes to join" needs no explaining, and the
            # tool's own '?' would only repeat the line above it.
            _stacked_field(mosaic_input_box, "Data cubes to join"),
            mosaic_count_w,
            mosaic_list_box,
            _stacked_field_with_help(
                mosaic_overlap_w, "Where cubes overlap", "mosaic_overlap"
            ),
            _stacked_field_with_help(
                mosaic_layers_w, "Layers to merge", "mosaic_layers"
            ),
            mosaic_layers_note_w,
            _stacked_field_with_help(
                mosaic_time_join_w, "Dates", "mosaic_time"
            ),
            _stacked_field_with_help(
                mosaic_band_join_w, "Bands", "mosaic_bands"
            ),
            _stacked_field_with_help(
                mosaic_crs_detected_w, "Output projection", "mosaic_crs"
            ),
            _stacked_field(mosaic_crs_user_w, "User-defined projection"),
            mosaic_crs_status_w,
            mosaic_advanced_acc,
            widgets.HBox(
                [mosaic_check_btn, mosaic_run_btn],
                layout=widgets.Layout(
                    gap="8px", flex_flow="row wrap", width="100%",
                    overflow="hidden", min_width="0",
                ),
            ),
            mosaic_out,
        ],
        layout=widgets.Layout(
            width="100%", gap="8px", overflow="hidden", min_width="0",
        ),
    )
    # ------------------------------------------------------------------
    # Source card: the two ways to get a working cube.
    #
    # Mosaicking used to sit in the feature list, among tools that CONSUME the
    # working cube - but it needs no loaded cube and PRODUCES one: it ends in
    # _finalize_load, exactly like opening a file does. So it belongs here, as
    # the second of two alternatives, and both modes hand over to the same Layer
    # dropdown below.
    # ------------------------------------------------------------------
    source_mode_w = widgets.ToggleButtons(
        options=[("Open a cube", "open"), ("Mosaic several cubes", "mosaic")],
        value="open",
        style={"button_width": "180px"},
        layout=widgets.Layout(margin="0 0 4px 0"),
    )

    open_mode_box = widgets.VBox([open_cube_box], layout=widgets.Layout(width="100%"))
    mosaic_mode_box = widgets.VBox(
        [mosaic_feature_box],
        layout=widgets.Layout(width="100%", display="none"),
    )

    def _sync_source_mode(*_):
        mosaic = source_mode_w.value == "mosaic"
        open_mode_box.layout.display = "none" if mosaic else ""
        mosaic_mode_box.layout.display = "" if mosaic else "none"

    source_mode_w.observe(_sync_source_mode, names="value")
    _sync_source_mode()

    source_box = widgets.VBox(
        [
            _card_title("Source"),
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>Open a data cube, or "
                "mosaic several into one. Either way the result becomes the cube "
                "you edit below.</div>"
            ),
            source_mode_w,
            open_mode_box,
            mosaic_mode_box,
            # Shared by both modes: a multi-layer file and a multi-layer mosaic
            # both stop here for the user to pick which layer to work on.
            layer_select_box,
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    # The Edit card holds ONLY the tools the Edit button applies. Their order
    # here is the order _on_edit_clicked runs them in, and the stage headers
    # number it, so what you read top to bottom is what actually happens. The
    # three tools that run on their own (Mosaic, Update, Build Cloud Mask) live
    # in their own cards - see the ui assembly at the end of this function.
    features_box = widgets.VBox(
        [
            _card_title("Edit"),
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>Chain as many tools as "
                "you like, then click <b>Edit data cube</b> once. They are "
                "applied in the order shown below, top to bottom. Don't forget "
                "to click <b>enable button</b> for some features.</div>"
            ),

            _stage_header(
                "1", "Select a subset",
                "Slice and filter out the data cube based on bands, time, "
                "clouds and coverage.",
                "turquoise",
            ),
            slice_acc,
            cloud_filter_acc,
            coverage_filter_acc,
            mask_clouds_acc,

            _stage_header(
                "2", "Change the geometry",
                "Reshapes the grid system of the pixels. Clip Raster is "
                "recommended to obtain the final product.",
                "violet",
            ),
            clip_acc,
            reproject_acc,

            _stage_header(
                "3", "Add new bands",
                "Derives extra bands from the ones the cube already has.",
                "green",
            ),
            indices_acc,

            _stage_header(
                "4", "Collapse the time axis",
                "Runs last, on whatever the stages above left behind.",
                "blue",
            ),
            stats_acc,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )

    # Actions. Export lives in its own card below Visualization now (matching the
    # Data Cube Builder), so this row only holds the Edit button.
    action_row = widgets.HBox(
        [edit_btn],
        layout=widgets.Layout(gap="8px", flex_flow="row wrap"),
    )

    # Update: its own card between the loaded summary and Edit. It fetches
    # missing dates and bands from the archive, which completes the SOURCE
    # rather than editing it, and it has always been the one tool that should
    # not be mixed into an Edit run.
    extend_box = widgets.VBox(
        [
            _card_title("Extend"),
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>Queries the catalog "
                "for scenes and bands the loaded cube is missing. Runs on its "
                "own, before editing.</div>"
            ),
            update_acc,
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )

    # Build Cloud Mask: the one tool whose product is a separate FILE rather
    # than the working cube, so it sits beside Export rather than in the Edit
    # list. Its output is the input of "Mask Clouds with Binary Masking File"
    # up in stage 1, which _on_build_mask_clicked now fills in automatically.
    side_outputs_box = widgets.VBox(
        [
            _card_title("Side Outputs"),
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>Writes a separate file "
                "alongside the cube. Does not change the result above.</div>"
            ),
            build_mask_acc,
            csv_report_acc,
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
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
            # Same collapsible group styling as the Animation section below,
            # but open by default - the viewer is the primary tool here.
            _field_group(
                "1) Interactive View",
                # The output area the viewer renders into lives INSIDE the
                # group, so collapsing the header hides the opened map too.
                [viz_renderer_box, viz_dropdown_btn, viz_out],
                subtitle="Explore the cube scene by scene: presets (RGB, false "
                "color, indices), any single band in grey levels, or a custom "
                "R/G/B band combination. The viewer opens below.",
                collapsible=True,
                open=True,
            ),
            # The animation maker is a separate tool from the viewer, so it lives
            # in its own collapsed-by-default section (a custom collapse, not a
            # nested ipywidgets Accordion, which would push a stray scrollbar).
            _field_group(
                "2) Animation (GIF export)",
                [
                    gif_section_w,
                    gif_preset_box,
                    gif_band_box,
                    gif_custom_box,
                    gif_stretch_box,
                    _stacked_field_with_help(gif_fps_w, "FPS", "fps"),
                    _stacked_field_with_help(gif_label_w, "Label", "gif_label"),
                    _stacked_field(gif_input_box, "Output GIF"),
                    viz_make_gif_btn,
                    anim_out,
                ],
                subtitle="Renders the whole time series to an animated GIF on disk "
                "(requires a time dimension). Status is reported below the button.",
                collapsible=True,
                open=False,
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
    source_card = widgets.VBox([source_box], layout=widgets.Layout(width="100%"))
    source_card.add_class("stac2cube-card")

    loaded_summary_card = widgets.VBox([loaded_summary_acc], layout=widgets.Layout(width="100%"))
    loaded_summary_card.add_class("stac2cube-card")

    extend_card = widgets.VBox([extend_box], layout=widgets.Layout(width="100%"))
    extend_card.add_class("stac2cube-card")

    side_outputs_card = widgets.VBox([side_outputs_box], layout=widgets.Layout(width="100%"))
    side_outputs_card.add_class("stac2cube-card")

    features_card = widgets.VBox(
        [features_box, widgets.HTML("<div style='height:6px;'></div>"), action_row],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    features_card.add_class("stac2cube-card")

    result_card = widgets.VBox([result_acc], layout=widgets.Layout(width="100%"))
    result_card.add_class("stac2cube-card")

    viz_card = widgets.VBox([viz_acc], layout=widgets.Layout(width="100%"))
    viz_card.add_class("stac2cube-card")

    # Export Options now lives below Visualization as its own card, with the
    # Export current result button attached directly beneath it (matching the
    # Data Cube Builder).
    export_action_row = widgets.HBox(
        [export_current_btn],
        layout=widgets.Layout(gap="8px", flex_flow="row wrap", margin="6px 0 0 0"),
    )
    export_card = widgets.VBox(
        [export_acc, export_action_row], layout=widgets.Layout(width="100%")
    )
    export_card.add_class("stac2cube-card")

    status_card = widgets.VBox(
        [_card_title("Status"), status_out],
        layout=widgets.Layout(width="100%", gap="6px"),
    )
    status_card.add_class("stac2cube-card")


    # Main UI
    # Spacers (keep them simple)
    spacer_small = widgets.HTML("<div style='height:6px;'></div>")
    spacer_med = widgets.HTML("<div style='height:12px;'></div>")

    # One card per verb, in the order the work happens:
    #   get a cube -> complete it -> edit it -> look at it -> write it out.
    ui = widgets.VBox(
        [
            header,
            subtitle,

            source_card,
            spacer_small,
            loaded_summary_card,

            spacer_med,
            extend_card,

            spacer_med,
            features_card,

            spacer_med,
            result_card,

            spacer_med,
            viz_card,

            spacer_med,
            export_card,

            spacer_med,
            side_outputs_card,

            spacer_med,
            status_card,
        ],
        layout=widgets.Layout(
            width="100%",
            max_width="980px",
            margin="0 auto",
            gap="0px",
        ),
    )

    ui.add_class("stac2cube-root")

    # ---------------------------------------------------------------------
    # Initialize + styling  (inject CSS BEFORE display)
    # ---------------------------------------------------------------------
    display(_gui_css_widget())

    outer = widgets.HBox([ui], layout=widgets.Layout(width="100%", justify_content="center"))

    _set_editor_enabled(False)
    _show_status(
        "ℹ️ Load a NetCDF or Zarr cube to start editing or use "
        "'Mosaic Data Cubes' to join several into one first."
    )
    _update_gif_output_suggestion(force=True)
    _update_update_daterange_example(force=True)
    # Renders the empty-list hint and disables Run until two cubes are queued.
    _mosaic_refresh()

    display(outer)

    return {
        "ui": ui,
        "outer": outer,
        "state": state,
        "widgets": {
            "load_path": load_path_w,
            "load_cube_btn": load_cube_btn,
            "layer_select": layer_select_w,
            "layer_load_btn": layer_load_btn,
            "reset_btn": reset_btn,
            "slice_time": slice_time_w,
            "slice_band": slice_band_w,
            "enable_cloud_filter": enable_cloud_filter_w,
            "cloud_max": cloud_max_w,
            "enable_coverage_filter": enable_coverage_filter_w,
            "coverage_min": coverage_min_w,
            "enable_mask_clouds": enable_mask_clouds_w,
            "mask_file": mask_file_w,
            "enable_clip": enable_clip_w,
            "clip_geom": clip_geom_w,
            "enable_reproject": enable_reproject_w,
            "reproject_crs": reproject_crs_w,
            "reproject_resolution": reproject_res_w,
            "reproject_resampling": reproject_resampling_w,
            "indices_select": indices_select_w,
            # Temporal Composites: the two promoted checkboxes, the "More
            # Composites" list ("stats_select") and the keep-or-drop choice.
            "stats_select": stats_select_w,
            "composite_mean": comp_mean_w,
            "composite_median": comp_median_w,
            # Custom Composites: the "Add composite" button and the container
            # whose children are the rows (each an HBox of period / from / to /
            # statistic / name / remove).
            "custom_add_btn": custom_add_btn,
            "custom_rows": custom_rows_box,
            "custom_error_note": custom_error_note,
            "keep_timeseries": keep_ts_w,
            "edit_btn": edit_btn,
            "update_run_btn": update_run_btn,
            "update_date_from": update_date_from_w,
            "update_date_to": update_date_to_w,
            "update_advanced_dates": update_advanced_dates_w,
            "update_daterange_mode": update_daterange_mode_w,
            "update_daterange": update_daterange_w,
            "update_bands": update_bands_w,
            "aggregator": aggregator_w,
            "export_mode": export_mode_w,
            "export_target": export_target_w,
            "export_current_btn": export_current_btn,
            "viz_dropdown_btn": viz_dropdown_btn,
            "viz_renderer": viz_renderer_w,
            "gif_section": gif_section_w,
            "gif_display_mode": gif_display_mode_w,
            "gif_band": gif_band_dd,
            "gif_r": gif_r_dd,
            "gif_g": gif_g_dd,
            "gif_b": gif_b_dd,
            "gif_stretch": gif_stretch_w,
            "gif_fps": gif_fps_w,
            "gif_label": gif_label_w,
            "gif_out_path": gif_out_path_w,
            "viz_make_gif_btn": viz_make_gif_btn,
            # Source card: which of the two ways to get a cube is showing.
            "source_mode": source_mode_w,
            "mosaic_run_btn": mosaic_run_btn,
            # Side outputs card.
            "build_mask_btn": build_mask_btn,
            "build_mask_out_path": build_mask_out_w,
            "csv_report_btn": csv_report_btn,
            "csv_report_out_path": csv_report_out_w,
        },
        "outputs": {
            "loaded_summary": loaded_summary_out,
            "result": result_out,
            "status": status_out,
            "visualization": viz_out,
            "animation": anim_out,
            # The three standalone tools report into their own panels rather
            # than into Status.
            "update": update_out,
            "build_mask": build_mask_out,
            "csv_report": csv_report_out,
            "mosaic": mosaic_out,
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
    css_patch = _gui_css_widget()

    # -----------------------------------------
    # State
    # -----------------------------------------
    state = {
        "loaded_path": None,
        "loaded_ds": None,           # open (lazy) xr.Dataset; kept open for on-demand reads
        "loaded_var": None,          # name of the loaded layer (data variable)
        "loaded_obj": None,
        "pending_ds": None,          # multi-layer file waiting for a layer selection
        "pending_path": None,
        "current_result_path": None,
    }

    # -----------------------------------------
    # Outputs
    # -----------------------------------------
    loaded_summary_out = widgets.Output(layout=widgets.Layout(width="99%", max_height="420px", overflow="auto"))
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

    def _output_stat(p: Path):
        """(mtime, size) fingerprint of an exported output.

        For a directory (a Zarr store) aggregate over all files inside: the
        directory's own mtime/size do not reliably change when chunk files
        inside are rewritten, so the flat stat() the NetCDF checks use would
        false-fail the "was the output updated" verification."""
        if p.is_dir():
            mtime, size = 0.0, 0
            for f in p.rglob("*"):
                if f.is_file():
                    st = f.stat()
                    mtime = max(mtime, st.st_mtime)
                    size += st.st_size
            return mtime, size
        st = p.stat()
        return st.st_mtime, st.st_size

    # -----------------------------------------
    # File chooser helper (optional)
    # -----------------------------------------
    def _guess_dir_from_text(text_value: str) -> str:
        s = (text_value or "").strip()
        if not s:
            return os.getcwd()
        p = Path(s).expanduser()
        # A .zarr store is one cube, not a folder to browse into: open its
        # parent so the store itself is the entry to click.
        if p.is_dir() and not is_zarr_path(p):
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
                fc = _CubeFileChooser(start_dir)
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

    def _ext_from_loaded():
        """Output extension matching the loaded cube's format, so every tool
        preserves the container: zarr in -> zarr out, nc in -> nc out."""
        lp = state.get("loaded_path")
        if lp and is_zarr_path(lp):
            return ".zarr"
        return ".nc"

    def _suggest_masked_path(threshold: int):
        base = _stem_from_loaded()
        outdir = _dir_from_loaded()
        return (outdir / f"{base}_masked_{int(threshold)}{_ext_from_loaded()}").as_posix()

    def _suggest_cr_path():
        base = _stem_from_loaded()
        outdir = _dir_from_loaded()
        return (outdir / f"{base}_cr{_ext_from_loaded()}").as_posix()

    def _suggest_sr_path():
        base = _stem_from_loaded()
        outdir = _dir_from_loaded()
        return (outdir / f"{base}_sr{_ext_from_loaded()}").as_posix()

    def _suggest_clouds_path_from_loaded():
        p = Path(state["loaded_path"])
        return (p.parent / f"{p.stem}_cloud{_ext_from_loaded()}").as_posix()

    # -----------------------------------------
    # Header
    # -----------------------------------------
    header = widgets.HTML("<div style='margin:0 0 4px 0; font-size:28px; font-weight:700;'>Analysis Ready Data Cube Tools</div>")
    subtitle = widgets.HTML(
        "<div style='display:flex; flex-wrap:wrap; gap:10px; margin:14px 0 8px 0;'>"
        # Step 1 - blue "load"
        "<div style='flex:1 1 200px; background:#f8fafc; border:1px solid #e5e7eb; "
        "border-left:4px solid #3b82f6; border-radius:8px; padding:8px 10px;'>"
        "<div style='font-weight:700; color:#1e3a8a; font-size:13px;'>1 &nbsp; Load</div>"
        "<div style='font-size:12px; color:#475569; margin-top:2px;'>Load a "
        "<b>NetCDF</b> or <b>Zarr</b> data cube.</div></div>"
        # Step 2 - green "select your tool"
        "<div style='flex:1 1 200px; background:#f0fdf4; border:1px solid #dcfce7; "
        "border-left:4px solid #16a34a; border-radius:8px; padding:8px 10px;'>"
        "<div style='font-weight:700; color:#166534; font-size:13px;'>2 &nbsp; Select your tool</div>"
        "<div style='font-size:12px; color:#475569; margin-top:2px;'>Select "
        "<b>one</b> of the tools below. For each feature, you should load a "
        "separate data cube.</div></div>"
        # Step 3 - orange "execute and export"
        "<div style='flex:1 1 200px; background:#fff7ed; border:1px solid #fed7aa; "
        "border-left:4px solid #f97316; border-radius:8px; padding:8px 10px;'>"
        "<div style='font-weight:700; color:#9a3412; font-size:13px;'>3 &nbsp; Execute &amp; Export</div>"
        "<div style='font-size:12px; color:#475569; margin-top:2px;'>Each tool has its "
        "own export system and keeps the loaded cube's format: zarr in -&gt; zarr out , "
        "netcdf in -&gt; netcdf out .</div></div>"
        "</div>"
    )

    # -----------------------------------------
    # Loading card (same pattern as editor)
    # -----------------------------------------
    load_path_w = widgets.Text(value="./results/test.nc", layout=widgets.Layout(width="100%"))
    browse_load_btn = widgets.Button(icon="folder-open", description="", layout=widgets.Layout(width="36px"))
    load_cube_btn = widgets.Button(description="Load cube", button_style="primary", icon="upload", layout=widgets.Layout(width="140px"))
    #reset_btn = widgets.Button(description="Reset to loaded cube", icon="undo", layout=widgets.Layout(width="180px"), disabled=True)

    load_fc_box = _attach_filechooser(
        browse_load_btn,
        load_path_w,
        title="Select cube (.nc file or .zarr store)",
        pattern=["*.nc", "*.zarr", "*"],
        select_dirs=False,
    )

    # Layer selection (shown only when the loaded NetCDF has multiple layers,
    # e.g. a time series exported together with temporal composites/stats)
    layer_select_w = widgets.Dropdown(
        options=[],
        value=None,
        layout=widgets.Layout(width="100%"),
    )
    layer_load_btn = widgets.Button(
        description="Load selected layer",
        icon="check",
        button_style="primary",
        layout=widgets.Layout(width="180px"),
    )
    layer_select_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "This cube contains <b>multiple layers</b>. Select the layer "
                "you want to work on, then click <b>Load selected layer</b>. "
                "Super-resolution runs on both time series and temporal "
                "composites; cloud masking and co-registration always need only "
                "time series."
                "</div>"
            ),
            _stacked_field(layer_select_w, "Layer"),
            layer_load_btn,
        ],
        layout=widgets.Layout(width="100%", gap="6px", display="none"),
    )




    COREG_HELP = {
        "grid_size": """
    <b>Full scan size (for difficult scenes)</b><br>
    How many small windows are used to measure each scene's shift. More windows = more votes,
    so clouds or a shifting river bar get outvoted and the shift is more reliable - but it takes
    longer.<br>
    The adaptive scan uses only a few windows on easy scenes and this full number only on hard ones,
    so raising it mainly costs time on cloudy dates. The default is fine for most cubes.
    """,
        "max_cc": """
    <b>Max Cloud Coverage</b><br>
    Scenes with more cloud than this (%) are dropped before co-registration.<br>
    Very cloudy scenes are already detected and removed automatically, but dropping them early
    can help in very cloudy regions. Needs a cloud-masked cube to know the cloud %.
    """,
        "time_period": """
    <b>Time Period</b><br>
    Limit co-registration to a date range: <code>["YYYY-MM-DD", "YYYY-MM-DD"]</code>.<br>
    Handy to leave out hard periods (e.g. snow &amp; ice). Leave empty to use all dates.
    """,
        "match_band": """
    <b>Matching Band</b><br>
    The band used to line up the scenes. <code>auto</code> picks a good native 10-m band
    (nir, then red, green, blue).<br>
    Avoid the resampled 20-m bands (rededge*, nir08, swir16, swir22) - they match poorly.
    """,
        "cloud_mask": """
    <b>Cloud Mask Cube</b><br>
    A binary cloud mask (1 = cloud) for cubes that still keep their clouds.<br>
    The clouds are hidden while measuring the shift, but stay in the exported scenes.<br>
    The mask can have extra dates, but every date in the cube must be present.
    """,
        "min_inliers_keep": """
    <b>Matching Points to Keep a Scene</b><br>
    Each scene's shift is measured at many points. This is the minimum number that must agree
    for the scene to be kept; scenes with fewer (usually very cloudy) are dropped.<br>
    <code>auto</code> scales with the scan size and is recommended.
    """,
        "min_inliers_update_ref": """
    <b>Matching Points to Trust as Reference</b><br>
    A scene is only used to align the next one when at least this many points agree on its shift.
    Scenes below this stay in the cube but are not used as a reference.<br>
    <code>auto</code> scales with the scan size and is recommended.
    """,
        "max_cloud_update_ref": """
    <b>Max Cloud % for Reference Scenes</b><br>
    A scene cloudier than this (%) is never used to align the next scene, even if its shift
    looked confident.
    """,
        "first_scene_mode": """
    <b>Reference Scene</b><br>
    The scene everything else is aligned to.<br>
    <code>auto</code> picks a clear, high-contrast scene for you; <code>select date</code> lets you
    name one (or pick it visually with Browse Scenes); <code>composite</code> uses a median of the
    first few days; <code>first</code> uses the first scene.<br>
    With <code>auto</code> or a chosen date, alignment runs both forward and backward from it.
    """,
        "reference_date": """
    <b>Reference Date</b><br>
    Used only with <code>select date</code>: the scene to align everything to (<code>YYYY-MM-DD</code>,
    nearest scene is used).<br>
    Tip: click <b>Browse Scenes</b> to look through the cube and pick a clear, high-contrast scene -
    it fills this field for you.
    """,
        "composite_window_days": """
    <b>Composite Window (days)</b><br>
    Only for the <code>composite</code> reference: how many days after the first scene to median.<br>
    Example: first scene 2020-01-15 with 30 days = median of 2020-01-15 to 2020-02-15.
    """,
        "iteration": """
    <b>Iteration</b><br>
    How many times the shift is re-measured. The pixels are resampled only once at the end, so
    extra passes just sharpen the estimate without degrading the data.<br>
    <code>auto</code> keeps refining only while it actually helps the cube (max 5). Usually 1 pass
    is enough.
    """,
    }

    def _stacked_field_with_help(widget, label_text, help_key):
        return _field_with_help(widget, label_text, COREG_HELP.get(help_key, ""))







    load_input_row = widgets.HBox([browse_load_btn, load_path_w], layout=widgets.Layout(width="100%", gap="6px", align_items="center"))
    load_input_box = widgets.VBox([load_input_row, load_fc_box], layout=widgets.Layout(width="100%", gap="4px"))

    loading_box = widgets.VBox(
        [
            widgets.HTML("<b>Loading</b>"),
            widgets.HTML("<div style='font-size:12px; color:#666;'>NetCDF and Zarr only (Geotiffs are not supported as input).</div>"),
            _stacked_field(load_input_box, "Data cube path"),
            widgets.HBox([load_cube_btn], layout=widgets.Layout(gap="8px", flex_flow="row wrap")),
            layer_select_box,
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

    # Umbrella NetCDF output options for the whole cloud/shadow tool. This
    # section writes from six places ((a)'s masked cube and cloud layers, plus
    # (b) i-iv), so per-export checkboxes would mean repeating the same pair six
    # times; one group above (a) and (b) applies to all of them.
    cm_compress_w = widgets.Checkbox(
        value=False,
        description="Lossless compression (zlib)",
        indent=False,
        layout=widgets.Layout(width="auto"),
    )
    cm_compress_warn_html = widgets.HTML(
        "<div style='font-size:12px; color:#b00020;'>"
        "⚠️ <b>Warning:</b> compression shrinks the output file a further "
        "~20-40% (scene-dependent), but the export step takes roughly "
        "<b>10x longer</b>. Enable it only for archiving, when disk space "
        "matters more than your time.</div>"
    )
    cm_compress_warn_html.layout.display = "none"
    cm_vrt_w = widgets.Checkbox(
        value=False,
        description="Export Band Mapping for GIS Tools (.vrt)",
        indent=False,
        layout=widgets.Layout(width="auto"),
    )
    cm_vrt_note_html = widgets.HTML(
        "<div style='font-size:12px; color:#555; margin-left:2px;'>"
        "&#8505;&#65039; Open the <b>.vrt</b> file in QGIS, not the .nc, "
        "and keep both files in the same folder.</div>"
    )
    cm_vrt_note_html.layout.display = "none"
    cm_scope_html = widgets.HTML(
        "<div style='font-size:12px; color:#555;'>"
        "&#8505;&#65039; These settings will be applied to all exportings.</div>"
    )

    def _sync_cm_output_options(*_):
        """Hide the pair for a Zarr cube (same rule as the other two tools)."""
        lp = state.get("loaded_path")
        zarr_loaded = bool(lp) and is_zarr_path(lp)
        if zarr_loaded:
            cm_compress_w.value = False
            cm_vrt_w.value = False
        for w in (cm_compress_w, cm_vrt_w):
            w.layout.display = "none" if zarr_loaded else ""
            w.disabled = lp is None
        cm_scope_html.layout.display = "none" if zarr_loaded else ""
        cm_compress_warn_html.layout.display = (
            "none" if zarr_loaded or lp is None or not cm_compress_w.value else ""
        )
        cm_vrt_note_html.layout.display = (
            "none" if zarr_loaded or lp is None or not cm_vrt_w.value else ""
        )

    cm_compress_w.observe(_sync_cm_output_options, names="value")
    cm_vrt_w.observe(_sync_cm_output_options, names="value")

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

    # Optional cloud SHADOW masking for the automated workflow: shadows are
    # projected from the s2cloudless mask at the same threshold, and the
    # masked cube removes cloud AND shadow pixels. The two parameters stay
    # greyed until shadow masking is switched on.
    auto_shadow_w = widgets.Checkbox(
        value=False,
        description="Also mask cloud shadows",
        indent=False,
    )
    auto_nir_dark_w = widgets.FloatText(
        value=0.18, step=0.01, layout=widgets.Layout(width="200px"), disabled=True
    )
    auto_proj_dist_w = widgets.FloatText(
        value=1.0, step=0.5, layout=widgets.Layout(width="200px"), disabled=True
    )

    def _sync_auto_shadow(change=None):
        on = bool(auto_shadow_w.value)
        auto_nir_dark_w.disabled = not on
        auto_proj_dist_w.disabled = not on

    auto_shadow_w.observe(_sync_auto_shadow, names="value")

    def _suggest_clouds_path():
        base = _stem_from_loaded()
        outdir = _dir_from_loaded()
        return (outdir / f"{base}_cloud{_ext_from_loaded()}").as_posix()

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

    def _ensure_cube_suffix(path_str: str) -> str:
        """Keep a .nc/.zarr extension as typed; anything else gets the loaded
        cube's extension (format preserved: zarr in -> zarr out, netcdf in -> netcdf out)."""
        p = Path(path_str)
        if p.suffix.lower() not in (".nc", ".zarr"):
            p = p.with_suffix(_ext_from_loaded())
        p.parent.mkdir(parents=True, exist_ok=True)
        return p.as_posix()

    def _on_mask_and_export_clicked(_):
        if state["loaded_obj"] is None or not state["loaded_path"]:
            _status("❌ Load a cube first.")
            return
        if state.get("loaded_var") != "Time_Series":
            _status(
                "❌ Cloud masking applies to the 'Time_Series' time series, "
                f"but the loaded layer is '{state.get('loaded_var')}'.",
                "Composite layers (e.g. median) have no time dimension, so per-date "
                "cloud masks cannot be applied to them.",
                "Reload the cube and select the 'Time_Series' layer to mask.",
            )
            return
        if get_cloud_layers is None:
            _status("❌ get_cloud_layers is not available. Check your stac2cube installation/imports.")
            return

        threshold = int(mask_threshold_w.value)

        out_masked = (masked_out_w.value or "").strip()
        if not out_masked:
            out_masked = _suggest_masked_path(threshold)
            masked_out_w.value = out_masked
        out_masked = _ensure_cube_suffix(out_masked)

        out_clouds = None
        if export_clouds_w.value:
            tmp = (clouds_out_w.value or "").strip()
            if not tmp:
                tmp = _suggest_clouds_path()
                clouds_out_w.value = tmp
            out_clouds = _ensure_cube_suffix(tmp)

        shadow_on = bool(auto_shadow_w.value)
        if shadow_on:
            # Shadow detection needs the nir band of the loaded cube.
            _loaded_bands = []
            try:
                _loaded_bands = [
                    str(b).lower() for b in state["loaded_obj"]["band"].values
                ]
            except Exception:
                pass
            if "nir" not in _loaded_bands:
                _status(
                    "❌ Cloud shadow masking needs the 'nir' band in the loaded "
                    "cube (dark-pixel test). Rebuild the cube with nir, or "
                    "untick 'Also mask cloud shadows'."
                )
                return

        p_masked = Path(out_masked)
        existed_before = p_masked.exists()
        old_mtime, old_size = _output_stat(p_masked) if existed_before else (None, None)

        _status(
            "Masking and exporting...",
            f"masking (input) = {state['loaded_path']}",
            f"threshold = {threshold}",
            f"shadow masking = {shadow_on}",
            f"output_masked = {out_masked}",
            f"output_clouds = {out_clouds if out_clouds else 'None'}",
        )

        # Release our persistent handle on the loaded cube: the tools below
        # open the same file themselves, and two concurrent netCDF/HDF5
        # handles to one file crash the kernel on Windows (same avoidance as
        # the manual Mask-out step; the handle is reopened in finally).
        _prev_loaded_ds = state.get("loaded_ds")
        if _prev_loaded_ds is not None:
            try:
                _prev_loaded_ds.close()
            except Exception:
                pass
            state["loaded_ds"] = None

        try:
            # Run your tool (exports inside)
            with status_out:
                # keep the lines you've already printed, then run the tool so its prints show here
                if not shadow_on:
                    get_cloud_layers(
                        masking=state["loaded_path"],
                        output_clouds=out_clouds,
                        output_masked=out_masked,
                        threshold=threshold,
                        compress=bool(cm_compress_w.value),
                        vrt=bool(cm_vrt_w.value),
                    )
                else:
                    # 1) probability + binary mask at the threshold (in memory)
                    _cloud_stack = get_cloud_layers(
                        input_cube=state["loaded_path"],
                        threshold=threshold,
                    )
                    # 2) shadows from that exact mask band; mask cloud AND
                    #    shadow out of the cube in one go.
                    _shadow_stack, _ = get_shadow_layers(
                        state["loaded_path"],
                        cloud=_cloud_stack,
                        threshold=threshold,
                        nir_dark_threshold=float(auto_nir_dark_w.value),
                        proj_distance=float(auto_proj_dist_w.value),
                        masking=True,
                        output_masked=out_masked,
                        compress=bool(cm_compress_w.value),
                        vrt=bool(cm_vrt_w.value),
                    )
                    # 3) optional cloud-layers export: probability + mask +
                    #    the two shadow bands, matching the manual workflow's
                    #    file layout (shadow_mask_<thr> / cloudshadow_mask_<thr>).
                    if out_clouds:
                        _new = _shadow_stack.sel(
                            band=["shadow_mask", "cloudshadow_mask"]
                        ).assign_coords(
                            band=[
                                f"shadow_mask_{threshold}",
                                f"cloudshadow_mask_{threshold}",
                            ]
                        )
                        _comb = xr.concat(
                            [_cloud_stack, _new.sel(time=_cloud_stack.time)],
                            dim="band",
                            coords="minimal",
                        ).transpose("time", "band", "y", "x")
                        _comb.name = "Cloud_Stack"
                        export_stac(
                            _comb,
                            out_clouds,
                            _comb.attrs.get("crs"),
                            _comb.attrs.get("transform"),
                            var_name="Cloud_Stack",
                            compress=bool(cm_compress_w.value),
                            vrt=bool(cm_vrt_w.value),
                        )

            # --- Verify export actually happened (prevents false ✅).
            # NOTE: on failure we print into status_out (append) instead of
            # calling _status (which clears), so the tool's own error stays
            # visible and is never overwritten by a success message.
            if not p_masked.exists():
                with status_out:
                    print("❌ Cloud masking failed: output file was not created.")
                return

            new_mtime, new_size = _output_stat(p_masked)
            if existed_before and (new_mtime == old_mtime) and (new_size == old_size):
                with status_out:
                    print(
                        "❌ Cloud masking failed: output file was not updated "
                        "(e.g. the cloud cube could not be built for all of the "
                        "loaded cube's dates)."
                    )
                return

            try:
                with open_cube(p_masked) as _:
                    pass
            except Exception as e:
                with status_out:
                    print(f"❌ Cloud masking failed: output file is not readable ({type(e).__name__}: {e})")
                return

            state["current_result_path"] = out_masked

            lines = [f"✅ Mask and export finished: {out_masked}"]
            if out_clouds:
                lines.append(f"✅ Cloud layers exported: {out_clouds}")
            _status(*lines)

        except Exception as e:
            _status(f"❌ {type(e).__name__}: {e}")
        finally:
            # Restore the session handle (defined in the manual Mask-out
            # section below; safe here because it only runs on click).
            try:
                _reopen_loaded_cube_handle()
            except Exception:
                pass

    mask_and_export_btn.on_click(_on_mask_and_export_clicked)

    # Sub-accordions inside Tool 1
    # --- Pretty layout for Tool 1a: three boxed sub-panels (cloud masking /
    # optional shadow masking / exporting setup) instead of a flat list.
    threshold_row = widgets.HBox(
        [
            widgets.HTML("<div style='font-weight:500;'>Threshold (%):</div>"),
            mask_threshold_w,
        ],
        layout=widgets.Layout(width="100%", gap="8px", align_items="center"),
    )

    mask_a_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Masks out the loaded data cube with a single known threshold value.<br> Optionally, exports time series of 'cloud probability + selected threshold binary maps', <br> Cloud probability time series can be used at in step (ii) of the manual workflow to experiment with different thresholds without re-computing probabilities."
                "</div>"
            ),
            _field_group("Cloud Masking", [threshold_row]),
            _field_group(
                "Cloud Shadow Masking (Optional)",
                [
                    auto_shadow_w,
                    _field_with_help(
                        auto_nir_dark_w,
                        "NIR dark threshold",
                        PARAM_HELP_HTML.get("nir_dark_threshold", ""),
                    ),
                    _field_with_help(
                        auto_proj_dist_w,
                        "Projection distance (km)",
                        PARAM_HELP_HTML.get("shadow_proj_distance", ""),
                    ),
                ],
                subtitle=(
                    "Projects each detected cloud along the Sun's direction and "
                    "masks the dark pixels it explains. Works best on large "
                    "areas; shadows of clouds outside the scene cannot be "
                    "detected."
                ),
            ),
            _field_group(
                "Exporting Setup",
                [
                    _stacked_field(masked_out_box, "Output masked cube (NetCDF/Zarr)"),
                    export_clouds_row,
                    _stacked_field(clouds_out_box, "Output cloud layers (NetCDF/Zarr)"),
                ],
            ),
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
        title="Select output cube (.nc or .zarr) for cloud probability cube",
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


    def _ensure_cube_suffix(path_str: str) -> str:
        """Keep a .nc/.zarr extension as typed; anything else gets the loaded
        cube's extension (format preserved: zarr in -> zarr out)."""
        p = Path(path_str)
        if p.suffix.lower() not in (".nc", ".zarr"):
            p = p.with_suffix(_ext_from_loaded())
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
            # default: same folder, *_cloud + the loaded cube's extension
            out_cloud = _suggest_clouds_path_from_loaded()
            b1_cloud_out_w.value = out_cloud
        out_cloud = _ensure_cube_suffix(out_cloud)
        b2_prob_in_w.value = out_cloud
        # Also pre-fill the shadow step and the visualization step so the user
        # can continue with the freshly built cloud cube without re-browsing.
        s3_cloud_path_w.value = out_cloud
        viz_cloud_path_w.value = out_cloud

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
            
        p_cloud = Path(out_cloud)
        existed_before = p_cloud.exists()
        old_mtime, old_size = _output_stat(p_cloud) if existed_before else (None, None)

        _status(
            "Building cloud probability data cube...",
            f"loaded cube = {state['loaded_path']}",
            f"output_clouds = {out_cloud}",
            f"threshold = {thresholds}",
        )

        try:
            with status_out:
                # Capture progress prints from get_cloud_layers.
                # input_cube -> probability is computed on the loaded cube's EXACT
                # dates (seasonal-safe), without masking the cube. polygon/daterange
                # are derived from the cube inside get_cloud_layers.
                cloud_da = get_cloud_layers(
                    input_cube=state["loaded_path"],
                    output_clouds=out_cloud,
                    output_masked=None,
                    threshold=thresholds,          # None => probability only
                    clip_raster=False,
                    masking=None,            # IMPORTANT: do not trigger masking branch
                    update=None,
                    compress=bool(cm_compress_w.value),
                    vrt=bool(cm_vrt_w.value),
                )

            # --- Verify export actually happened (prevents false ✅).
            # On failure, append into status_out (never call _status, which
            # clears) so the tool's own error is never overwritten by success.
            if not p_cloud.exists():
                with status_out:
                    print("❌ Build failed: output file was not created.")
                return

            new_mtime, new_size = _output_stat(p_cloud)
            if existed_before and (new_mtime == old_mtime) and (new_size == old_size):
                with status_out:
                    print(
                        "❌ Build failed: output file was not updated "
                        "(e.g. the cloud cube could not be built for all of the "
                        "loaded cube's dates)."
                    )
                return

            try:
                with open_cube(p_cloud) as _:
                    pass
            except Exception as e:
                with status_out:
                    print(f"❌ Build failed: output file is not readable ({type(e).__name__}: {e})")
                return

            state["current_result_path"] = out_cloud
            _status(f"✅ Cloud probability cube exported: {out_cloud}")

        except Exception as e:
            _status(f"❌ {type(e).__name__}: {e}")

    b1_build_btn.on_click(_on_b1_build_clicked)





    b1_box = widgets.VBox(
        [
            widgets.HTML("<div style='font-size:12px; color:#666;'>"
                        "Builds cloud data cube.<br> If threshold is not given, only cloud probability cube will be built. "
                        "In that case, binary mask(s) with threshold(s) can be generated in step (ii).<br>"
                        "Any cube date with no Element84 L1C scene is filled with the SCL cloud mask "
                        "from the cube's own source (reported below), so cubes from any source can be masked."
                        "</div>"),
            _stacked_field(b1_thresholds_w, "Threshold(s) (Optional)"),
            _stacked_field(b1_cloud_out_box, "Output cloud probability cube (NetCDF/Zarr)"),
            b1_build_btn,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    b1_acc = widgets.Accordion(children=[b1_box], selected_index=None)
    b1_acc.set_title(0, "i) Build Cloud Mask Data Cube")
    b1_acc.layout = widgets.Layout(width="99%")

    # ii) (Optional) Generate Masks from Probability Map
    b2_prob_in_w = widgets.Text(value="", layout=widgets.Layout(width="100%"))
    browse_b2_prob_in_btn = widgets.Button(icon="folder-open", description="", layout=widgets.Layout(width="36px"))
    b2_prob_in_fc_box = _attach_filechooser(
        browse_b2_prob_in_btn,
        b2_prob_in_w,
        title="Select cloud probability cube (.nc or .zarr)",
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
        prob_path = resolve_cube_path(prob_path)
        b2_prob_in_w.value = str(Path(prob_path).as_posix())

        thresholds = _parse_thresholds_text(b2_thresholds_w.value)
        if thresholds is None:
            _status("ℹ️ No thresholds provided. Nothing to do (leaving probability cube unchanged).")
            return

        p = Path(prob_path)
        if not p.exists():
            _status(f"❌ File not found: {p.as_posix()}")
            return

        old_mtime, old_size = _output_stat(p)

        _status(
            "Generating masks from probability map...",
            f"input/overwrite file = {p.as_posix()}",
            f"thresholds = {thresholds}",
        )

        try:
            # Load existing cloud cube
            with open_cube(p) as ds:
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

            band_names = [str(b) for b in combined["band"].values]
            combined = combined.assign_coords(band=np.array(band_names, dtype=object))

            combined.encoding = {}
            combined["band"].encoding = {}
            for coord in combined.coords:
                combined[coord].encoding = {}

            # --- Overwrite the same file (no need to pass crs/transform) ---
            with status_out:
                export_stac(combined, p.as_posix(), overwrite=True,
                            var_name="Cloud_Stack",
                            compress=bool(cm_compress_w.value),
                            vrt=bool(cm_vrt_w.value))

            # --- Verify the file was actually overwritten (prevents false ✅).
            # On failure, append into status_out (never call _status, which
            # clears) so the tool's own error is never overwritten by success.
            if not p.exists() or _output_stat(p) == (old_mtime, old_size):
                with status_out:
                    print("❌ Mask generation failed: file was not overwritten.")
                return
            try:
                with open_cube(p) as _:
                    pass
            except Exception as e:
                with status_out:
                    print(f"❌ Mask generation failed: file is not readable ({type(e).__name__}: {e})")
                return

            state["current_result_path"] = p.as_posix()
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
            _stacked_field(b2_prob_in_box, "Input probability cube (NetCDF/Zarr)"),
            _stacked_field(b2_thresholds_w, "Threshold(s)"),
            b2_generate_btn,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    b2_acc = widgets.Accordion(children=[b2_box], selected_index=None)
    b2_acc.set_title(0, "ii) (Optional) Generate Masks from Probability Map")
    b2_acc.layout = widgets.Layout(width="99%")

    # -----------------------------------------------------------------
    # iii) (Optional) Build Cloud Shadow Mask
    # Projects shadows from ONE selected binary cloud mask band (sun
    # direction + dark-NIR test) and appends shadow_mask_<xx> and
    # cloudshadow_mask_<xx> bands to the cloud cube, overwriting the file.
    # Step (iv) can then mask with clouds only, shadows only, or both.
    # -----------------------------------------------------------------
    s3_cloud_path_w = widgets.Text(value="", layout=widgets.Layout(width="100%"))
    browse_s3_cloud_btn = widgets.Button(icon="folder-open", description="", layout=widgets.Layout(width="36px"))
    s3_cloud_fc_box = _attach_filechooser(
        browse_s3_cloud_btn,
        s3_cloud_path_w,
        title="Select cloud cube (.nc or .zarr with Cloud_Stack)",
        pattern=["*.nc", "*"],
        select_dirs=False,
    )
    s3_cloud_row = widgets.HBox(
        [browse_s3_cloud_btn, s3_cloud_path_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    s3_cloud_box = widgets.VBox(
        [s3_cloud_row, s3_cloud_fc_box], layout=widgets.Layout(width="100%", gap="4px")
    )

    s3_load_btn = widgets.Button(
        description="Load cloud cube",
        button_style="primary",
        icon="upload",
        layout=widgets.Layout(width="160px"),
    )
    s3_mask_band_w = widgets.Dropdown(
        options=[], value=None, layout=widgets.Layout(width="60%"), disabled=True
    )
    s3_nir_dark_w = widgets.FloatText(
        value=0.18, step=0.01, layout=widgets.Layout(width="200px")
    )
    s3_proj_dist_w = widgets.FloatText(
        value=1.0, step=0.5, layout=widgets.Layout(width="200px")
    )
    s3_generate_btn = widgets.Button(
        description="Generate and Overwrite",
        button_style="success",
        icon="play",
        layout=widgets.Layout(width="210px"),
    )

    def _on_s3_load_cloud_clicked(_):
        cloud_path = (s3_cloud_path_w.value or "").strip()
        if not cloud_path:
            _status("❌ Please select a cloud cube path (.nc or .zarr).")
            return
        cloud_path = resolve_cube_path(cloud_path)
        s3_cloud_path_w.value = str(Path(cloud_path).as_posix())

        p = Path(cloud_path)
        if not p.exists():
            _status(f"❌ File not found: {p.as_posix()}")
            return

        try:
            _status("Loading cloud cube (for shadow projection)...", f"path = {p.as_posix()}")
            with open_cube(p) as ds:
                if "Cloud_Stack" not in ds.data_vars:
                    raise ValueError("Cube does not contain 'Cloud_Stack'.")
                cloud = ds["Cloud_Stack"]
                if "band" not in cloud.dims:
                    raise ValueError("Cloud_Stack has no 'band' dimension.")
                bands = [str(b) for b in cloud["band"].values]

            # Shadows are projected from a binary CLOUD mask band - the
            # probability band and already-existing shadow bands are not
            # valid projection sources.
            cloud_bands = [
                b for b in bands
                if b != "cloud_prob"
                and not b.startswith("shadow_mask")
                and not b.startswith("cloudshadow_mask")
            ]
            if not cloud_bands:
                raise ValueError(
                    "No binary cloud mask bands found. Generate masks from the "
                    "probability map in step (ii) first."
                )

            s3_mask_band_w.options = cloud_bands
            s3_mask_band_w.value = cloud_bands[0]
            s3_mask_band_w.disabled = False
            _status("✅ Cloud cube loaded for shadow masking.", f"Cloud mask bands: {cloud_bands}")

        except Exception as e:
            _status(f"❌ {type(e).__name__}: {e}")

    s3_load_btn.on_click(_on_s3_load_cloud_clicked)

    def _on_s3_generate_clicked(_):
        if state.get("loaded_path") is None:
            _status("❌ Load the main data cube first (loader above).")
            return
        if state.get("loaded_var") != "Time_Series":
            _status(
                "❌ Shadow masking applies to the 'Time_Series' time series, "
                f"but the loaded layer is '{state.get('loaded_var')}'.",
                "Reload the cube and select the 'Time_Series' layer.",
            )
            return
        if not s3_mask_band_w.value:
            _status("❌ Load a cloud cube and select a cloud mask band first.")
            return
        _loaded_bands = []
        try:
            _loaded_bands = [str(b).lower() for b in state["loaded_obj"]["band"].values]
        except Exception:
            pass
        if "nir" not in _loaded_bands:
            _status(
                "❌ Cloud shadow masking needs the 'nir' band in the loaded cube "
                "(dark-pixel test)."
            )
            return

        cloud_path = (s3_cloud_path_w.value or "").strip()
        p = Path(resolve_cube_path(cloud_path))
        if not p.exists():
            _status(f"❌ File not found: {p.as_posix()}")
            return

        mask_band = str(s3_mask_band_w.value)
        old_mtime, old_size = _output_stat(p)

        _status(
            "Building cloud shadow masks...",
            f"data cube = {state['loaded_path']}",
            f"cloud cube (overwrite) = {p.as_posix()}",
            f"cloud mask band = {mask_band}",
            f"nir_dark_threshold = {float(s3_nir_dark_w.value)}",
            f"proj_distance = {float(s3_proj_dist_w.value)} km",
        )

        # Release the persistent handle on the loaded cube: the shadow tool
        # opens the same file itself (two concurrent netCDF/HDF5 handles to
        # one file crash the kernel on Windows). Reopened in finally.
        _prev_loaded_ds = state.get("loaded_ds")
        if _prev_loaded_ds is not None:
            try:
                _prev_loaded_ds.close()
            except Exception:
                pass
            state["loaded_ds"] = None

        try:
            with status_out:
                add_shadow_masks_to_cloud_stack(
                    input_cube=state["loaded_path"],
                    cloud=p.as_posix(),
                    mask_band=mask_band,
                    nir_dark_threshold=float(s3_nir_dark_w.value),
                    proj_distance=float(s3_proj_dist_w.value),
                )

            # --- Verify the file was actually overwritten (prevents false ✅).
            if not p.exists() or _output_stat(p) == (old_mtime, old_size):
                with status_out:
                    print("❌ Shadow mask generation failed: file was not overwritten.")
                return
            try:
                with open_cube(p) as _:
                    pass
            except Exception as e:
                with status_out:
                    print(f"❌ Shadow mask generation failed: file is not readable ({type(e).__name__}: {e})")
                return

            # Refresh the Mask-out step's band list if it has this cloud cube
            # loaded (the new shadow bands become selectable immediately).
            try:
                if state.get("cloud_path") and Path(state["cloud_path"]).resolve() == p.resolve():
                    with open_cube(p) as ds:
                        _bands_new = [str(b) for b in ds["Cloud_Stack"]["band"].values]
                    _mask_bands = [b for b in _bands_new if b != "cloud_prob"]
                    _cur = b3_mask_band_w.value
                    b3_mask_band_w.options = _mask_bands
                    b3_mask_band_w.value = _cur if _cur in _mask_bands else _mask_bands[0]
                    state["cloud_mask_bands"] = _mask_bands
            except Exception:
                pass

            # Pre-fill the visualization step for immediate comparison.
            viz_cloud_path_w.value = p.as_posix()

            _sfx = mask_band[len("cloud_mask_"):] if mask_band.startswith("cloud_mask_") else mask_band
            state["current_result_path"] = p.as_posix()
            _status(
                f"✅ Shadow masks generated and file overwritten: {p.as_posix()}",
                f"New layers: shadow_mask_{_sfx}, cloudshadow_mask_{_sfx}",
                "In step (iv) you can now mask with clouds only, shadows only, or both (cloudshadow).",
            )

        except Exception as e:
            _status(f"❌ {type(e).__name__}: {e}")
        finally:
            try:
                _reopen_loaded_cube_handle()
            except Exception:
                pass

    s3_generate_btn.on_click(_on_s3_generate_clicked)

    s3_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Detects cloud shadows using a selected binary cloud mask band, the "
                "Sun's direction of each scene and a dark-NIR test, then appends "
                "<b>shadow_mask_xx</b> and <b>cloudshadow_mask_xx</b> layers "
                "(xx = the mask's threshold).<br>"
                "<b>Warning:</b> This overwrites the cloud data cube (keeps all "
                "existing bands). In step (iv) you can then mask with clouds only, "
                "shadows only, or both."
                "</div>"
            ),
            _stacked_field(s3_cloud_box, "Cloud data cube (NetCDF/Zarr)"),
            s3_load_btn,
            _stacked_field(s3_mask_band_w, "Cloud mask band (shadows are projected from it)"),
            _field_with_help(
                s3_nir_dark_w, "NIR dark threshold",
                PARAM_HELP_HTML.get("nir_dark_threshold", ""),
            ),
            _field_with_help(
                s3_proj_dist_w, "Projection distance (km)",
                PARAM_HELP_HTML.get("shadow_proj_distance", ""),
            ),
            s3_generate_btn,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    s3_acc = widgets.Accordion(children=[s3_box], selected_index=None)
    s3_acc.set_title(0, "iii) (Optional) Build Cloud Shadow Mask")
    s3_acc.layout = widgets.Layout(width="99%")

    



    # iii) Mask out Data Cube (by single threshold value) — NEW design

    # Cloud cube selector (separate from main loaded cube)
    b3_cloud_path_w = widgets.Text(value="", layout=widgets.Layout(width="100%"))
    browse_b3_cloud_btn = widgets.Button(icon="folder-open", description="", layout=widgets.Layout(width="36px"))
    b3_cloud_fc_box = _attach_filechooser(
        browse_b3_cloud_btn,
        b3_cloud_path_w,
        title="Select cloud cube (.nc or .zarr with Cloud_Stack)",
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
        title="Select output masked cube (.nc or .zarr)",
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
        out = p.parent / f"{p.stem}_masked_{thr}{_ext_from_loaded()}"
        return out.as_posix()

    def _on_load_cloud_clicked(_):
        if state.get("loaded_path") is None:
            _status("❌ Load the main data cube first.")
            return
        if state.get("loaded_var") != "Time_Series":
            _status(
                "❌ Cloud masking applies to the 'Time_Series' time series, "
                f"but the loaded layer is '{state.get('loaded_var')}'.",
                "Composite layers (e.g. median) have no time dimension, so per-date "
                "cloud masks cannot be applied to them.",
                "Reload the cube and select the 'Time_Series' layer to mask.",
            )
            return

        cloud_path = (b3_cloud_path_w.value or "").strip()
        if not cloud_path:
            _status("❌ Please select a cloud cube path (.nc or .zarr).")
            return
        cloud_path = resolve_cube_path(cloud_path)
        b3_cloud_path_w.value = str(Path(cloud_path).as_posix())

        p = Path(cloud_path)
        if not p.exists():
            _status(f"❌ File not found: {p.as_posix()}")
            return

        try:
            _status("Loading cloud cube (for masks)...", f"path = {p.as_posix()}")

            with open_cube(p) as ds:
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

    def _reopen_loaded_cube_handle():
        """Reopen the persistent (lazy) handle on the loaded cube from
        state['loaded_path'] and rebuild state['loaded_obj'].

        Used to restore the session handle after an operation that needs
        exclusive file access to the loaded cube (see the masking handler):
        two concurrent netCDF/HDF5 handles to one file crash the kernel on
        Windows, so we close ours during the op and reopen here. The reopen
        stays lazy (chunks='frames'), so nothing is force-loaded into RAM."""
        lp = state.get("loaded_path")
        if not lp:
            return
        ds_open = open_cube(lp, chunks="frames")
        ds = ds_open.assign_coords(
            {name: coord.compute() for name, coord in ds_open.coords.items()}
        )
        state["loaded_ds"] = ds_open
        lv = state.get("loaded_var")
        if lv and lv in ds.data_vars:
            state["loaded_obj"] = ds[lv]
        elif "Time_Series" in ds.data_vars:
            state["loaded_obj"] = ds["Time_Series"]
        else:
            state["loaded_obj"] = ds

    def _on_b3_mask_export_clicked(_):
        if state.get("loaded_path") is None:
            _status("❌ Load the main data cube first.")
            return
        if state.get("loaded_var") != "Time_Series":
            _status(
                "❌ Cloud masking applies to the 'Time_Series' time series, "
                f"but the loaded layer is '{state.get('loaded_var')}'.",
                "Reload the cube and select the 'Time_Series' layer to mask.",
            )
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

        out_path = _ensure_cube_suffix(out_path)
        mask_layer = str(b3_mask_band_w.value)

        p_out = Path(out_path)
        existed_before = p_out.exists()
        old_mtime, old_size = _output_stat(p_out) if existed_before else (None, None)

        _status(
            "Masking and exporting...",
        )

        # Release our persistent handle on the loaded cube so mask_stac_clouds is
        # the ONLY handle to that file while it reads + exports. Two concurrent
        # netCDF/HDF5 handles to a single file crash the kernel on Windows; that
        # is why masking worked once from a fresh load but crashed on the repeat.
        _prev_loaded_ds = state.get("loaded_ds")
        if _prev_loaded_ds is not None:
            try:
                _prev_loaded_ds.close()
            except Exception:
                pass
            state["loaded_ds"] = None

        try:
            with status_out:
                res = mask_stac_clouds(
                    stac=state["loaded_path"],
                    cloud=state["cloud_path"],
                    mask_layer=mask_layer,
                    output=out_path,
                    compress=bool(cm_compress_w.value),
                    vrt=bool(cm_vrt_w.value),
                )

            # --- Verify export actually happened (prevents false ✅).
            # On failure, append into status_out (never call _status, which
            # clears) so the tool's own error is never overwritten by success.
            if not p_out.exists():
                with status_out:
                    print("❌ Masking failed: output file was not created.")
                return

            new_mtime, new_size = _output_stat(p_out)
            if existed_before and (new_mtime == old_mtime) and (new_size == old_size):
                with status_out:
                    print(
                        "❌ Masking failed: output file was not updated "
                        "(e.g. the cloud cube does not cover all of the loaded "
                        "cube's dates)."
                    )
                return

            try:
                with open_cube(p_out) as _:
                    pass
            except Exception as e:
                with status_out:
                    print(f"❌ Masking failed: output file is not readable ({type(e).__name__}: {e})")
                return

            state["current_result_path"] = out_path
            _status(f"✅ Masked cube exported: {out_path}")

        except Exception as e:
            _status(f"❌ {type(e).__name__}: {e}")
        finally:
            # Restore the session handle so the rest of the UI keeps working
            # and a subsequent Mask and Export can run again.
            try:
                _reopen_loaded_cube_handle()
            except Exception:
                pass

    b3_mask_btn.on_click(_on_b3_mask_export_clicked)

    b3_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Masks the <b>loaded data cube</b> using a selected binary mask band from a <b>Cloud_Stack</b> cube.<br>"
                "Load a cloud cube, pick a mask band (not cloud_prob), then export a masked cube.<br>"
                "With shadow layers from step (iii) you can mask <b>clouds only</b> (cloud_mask_xx), "
                "<b>shadows only</b> (shadow_mask_xx) or <b>both</b> (cloudshadow_mask_xx)."
                "</div>"
            ),
            _stacked_field(b3_cloud_box, "Cloud data cube (NetCDF/Zarr)"),
            load_cloud_btn,
            _stacked_field(b3_mask_band_w, "Mask band"),
            _stacked_field(b3_masked_out_box, "Output masked cube (NetCDF/Zarr)"),
            b3_mask_btn,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )

    b3_acc = widgets.Accordion(children=[b3_box], selected_index=None)
    b3_acc.set_title(0, "iv) Mask out Data Cube")
    b3_acc.layout = widgets.Layout(width="99%")


    # iv) Compare Masks vs RGB (Visualization)
    # Lets the user overlay the cloud probability map and each binary threshold
    # mask onto the true-color (RGB) scene, date by date, to decide which
    # threshold best removes clouds without deleting good pixels. The main
    # spectral cube loaded above supplies the RGB; the cloud cube is chosen here.
    viz_cloud_path_w = widgets.Text(value="", layout=widgets.Layout(width="100%"))
    browse_viz_cloud_btn = widgets.Button(icon="folder-open", description="", layout=widgets.Layout(width="36px"))
    viz_cloud_fc_box = _attach_filechooser(
        browse_viz_cloud_btn,
        viz_cloud_path_w,
        title="Select cloud probability/mask cube (.nc or .zarr with Cloud_Stack)",
        pattern=["*.nc", "*"],
        select_dirs=False,
    )
    viz_cloud_row = widgets.HBox(
        [browse_viz_cloud_btn, viz_cloud_path_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    viz_cloud_box = widgets.VBox([viz_cloud_row, viz_cloud_fc_box], layout=widgets.Layout(width="100%", gap="4px"))

    viz_btn = widgets.Button(
        description="Visualize",
        button_style="primary",
        icon="image",
        layout=widgets.Layout(width="160px"),
    )
    viz_out = widgets.Output(layout=widgets.Layout(width="99%", overflow="auto"))

    def _on_visualize_masks_clicked(_):
        with viz_out:
            clear_output()

        if state.get("loaded_obj") is None or not state.get("loaded_path"):
            with viz_out:
                print("❌ Load the main spectral data cube first (loader above).")
            return

        cloud_path = (viz_cloud_path_w.value or "").strip()
        if cloud_path:
            cloud_path = resolve_cube_path(cloud_path)
            viz_cloud_path_w.value = str(Path(cloud_path).as_posix())
        # Fall back to the loaded cube's default *_cloud.nc/.zarr if it exists.
        if not cloud_path and state.get("loaded_path"):
            cand = _suggest_clouds_path_from_loaded()
            if Path(cand).exists():
                cloud_path = cand
                viz_cloud_path_w.value = cloud_path
        if not cloud_path:
            with viz_out:
                print("❌ Select a cloud cube (.nc or .zarr, with Cloud_Stack) to compare.")
            return

        p = Path(cloud_path)
        if not p.exists():
            with viz_out:
                print(f"❌ File not found: {p.as_posix()}")
            return

        try:
            with viz_out:
                print("Loading cloud cube and preparing viewer...")

            spectral = state["loaded_obj"]
            if isinstance(spectral, xr.Dataset):
                if "Time_Series" not in spectral.data_vars:
                    raise ValueError("Loaded cube has no 'Time_Series'.")
                spectral = spectral["Time_Series"]

            spec_bands = [str(b).lower() for b in spectral["band"].values]
            for need in ("red", "green", "blue"):
                if need not in spec_bands:
                    raise ValueError(
                        f"Loaded cube is missing the '{need}' band; the RGB view needs red, green and blue."
                    )

            # Cloud cube is small; load it fully so no file handle lingers.
            with open_cube(p) as ds:
                if "Cloud_Stack" not in ds.data_vars:
                    raise ValueError("Selected NetCDF has no 'Cloud_Stack'.")
                cloud = ds["Cloud_Stack"].load()

            if "band" not in cloud.dims:
                raise ValueError("Cloud_Stack has no 'band' dimension.")

            # Spatial grid must match (same AOI + resolution).
            if (spectral.sizes.get("y"), spectral.sizes.get("x")) != (
                cloud.sizes.get("y"),
                cloud.sizes.get("x"),
            ):
                raise ValueError(
                    "The spectral cube and the cloud cube have different y/x grids. "
                    "They must cover the same area at the same resolution."
                )

            # Compare only on the dates both cubes share.
            sp_times = pd.to_datetime(spectral["time"].values)
            cl_times = pd.to_datetime(cloud["time"].values)
            common = sp_times.intersection(cl_times)
            if len(common) == 0:
                raise ValueError("The spectral and cloud cubes share no common dates.")

            spectral_c = spectral.sel(time=common.values)
            cloud_c = cloud.sel(time=common.values)

            with viz_out:
                clear_output()
                interactive_cloud_overlay_view(spectral_c, cloud_c, widget_type="dropdown")

        except Exception as e:
            with viz_out:
                clear_output()
                print(_friendly_error(e, "Cloud mask visualization"))

    viz_btn.on_click(_on_visualize_masks_clicked)

    viz_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Compare the cloud <b>probability</b> map and each binary <b>threshold mask</b> against the "
                "true-color (RGB) scene, date by date, to decide which threshold best removes clouds without "
                "deleting good pixels."
                "</div>"
            ),
            _stacked_field(viz_cloud_box, "Cloud probability/mask cube (NetCDF/Zarr)"),
            viz_btn,
            viz_out,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    viz_acc = widgets.Accordion(children=[viz_box], selected_index=None)
    viz_acc.set_title(0, "Visualization")
    viz_acc.layout = widgets.Layout(width="99%")
    # Vivid green header (shared stylesheet) so the section that lets users
    # SEE their masks stands out and welcomes them in.
    viz_acc.add_class("stac2cube-acc-vivid")







    # Tool 1b container
    mask_b_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#1e40af; line-height:1.5; "
                "background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; "
                "padding:8px 10px; margin:0 0 4px 0;'>"
                "Remember to use the <b>Visualization</b> section to compare "
                "your cloud and shadow masks with the RGB image."
                "</div>"
            ),
            b1_acc,
            b2_acc,
            s3_acc,
            b3_acc,
            viz_acc,
        ],
        layout=widgets.Layout(width="100%", gap="10px"),
    )

    mask_b_acc = widgets.Accordion(children=[mask_b_box], selected_index=None)
    mask_b_acc.set_title(0, "b) Manually Build Cloud Masking Data Cube")
    mask_b_acc.layout = widgets.Layout(width="99%")


    mask_tool_box = widgets.VBox(
        [
            #widgets.HTML("<b>1) Cloud Masking Data Cube</b>"),
            widgets.HTML("<div style='font-size:12px; color:#666;'>If you already know the threshold value, proceed with Fully Automated Workflow. <br>If not, build your cloud data cube manually and inspect the result.</div>"),
            widgets.HTML(
                "<div style='font-size:12px; color:#1e40af; line-height:1.5; "
                "background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; "
                "padding:8px 10px; margin:0 0 4px 0;'>"
                "s2cloudless needs L1C data, which may not be available for every "
                "date. Dates without L1C are masked with the Scene "
                "Classification Layer instead of being dropped from the time "
                "series, so your cube keeps all its dates. "
                "These dates can be filtered out later if you wish by scene's cloud masking method."
                "</div>"
            ),
            # Applies to every export in (a) and (b), so it sits above both.
            _field_group(
                "Output options (NetCDF)",
                [
                    cm_scope_html,
                    cm_compress_w,
                    cm_compress_warn_html,
                    cm_vrt_w,
                    cm_vrt_note_html,
                ],
            ),
            mask_a_acc,
            mask_b_acc,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    mask_tool_acc = widgets.Accordion(children=[mask_tool_box], selected_index=None)
    mask_tool_acc.set_title(0, "1) Cloud and Shadow Masking Data Cube")
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

    cr_min_inliers_keep_w = widgets.Text(value="auto", placeholder="auto", layout=widgets.Layout(width="200px"))
    cr_min_inliers_update_ref_w = widgets.Text(value="auto", placeholder="auto", layout=widgets.Layout(width="200px"))
    cr_max_cloud_update_ref_w = widgets.BoundedFloatText(value=20.0, min=0.0, max=100.0, step=1.0, layout=widgets.Layout(width="200px"))
    cr_adaptive_w = widgets.Checkbox(
        value=True,
        description="Adaptive window scan",
        indent=False,
        layout=widgets.Layout(width="auto"),
    )

    def _parse_inlier_value(txt):
        s = str(txt or "").strip().lower()
        if s in ("", "auto"):
            return "auto"
        return int(s)  # raises ValueError on junk; surfaced by the caller
    cr_match_band_w = widgets.Dropdown(
        options=["auto", "nir", "red", "green", "blue"],
        value="auto",
        layout=widgets.Layout(width="200px"),
    )

    cr_first_scene_mode_w = widgets.Dropdown(
        options=[
            ("auto", "auto"),
            ("select date", "date"),
            ("composite", "composite"),
            ("first", "first"),
        ],
        value="auto",
        layout=widgets.Layout(width="200px"),
    )

    cr_composite_window_days_w = widgets.BoundedIntText(value=30, min=1, max=365, step=1, layout=widgets.Layout(width="200px"))
    cr_composite_window_days_w.disabled = True

    cr_ref_date_w = widgets.Text(
        value="",
        placeholder="YYYY-MM-DD",
        layout=widgets.Layout(width="200px"),
        disabled=True,
    )

    # scene browser: reuses the standard cube viewer so the user can look
    # through the scenes and pick the reference date visually
    cr_browse_scenes_btn = widgets.Button(
        description="Browse Scenes",
        icon="eye",
        layout=widgets.Layout(width="150px", margin="18px 0 0 0"),
    )
    cr_pick_out = widgets.Output()

    def _on_browse_scenes_clicked(_):
        spectral = state.get("loaded_obj")
        if spectral is None:
            _status("❌ Load a cube first.")
            return
        if isinstance(spectral, xr.Dataset):
            if "Time_Series" not in spectral.data_vars:
                _status("❌ Loaded cube has no 'Time_Series'.")
                return
            spectral = spectral["Time_Series"]
        if "time" not in spectral.dims:
            _status(
                "❌ The loaded layer is a single composite image; "
                "there are no scenes to browse."
            )
            return

        with cr_pick_out:
            clear_output()
            time_w = interactive_time_view(
                stac=spectral, widget_type="dropdown", return_time_widget=True
            )

            select_btn = widgets.Button(
                description="Select Scene as Reference",
                button_style="info",
                icon="check",
                layout=widgets.Layout(width="230px"),
            )
            close_btn = widgets.Button(
                description="Close",
                layout=widgets.Layout(width="90px"),
            )

            def _on_select(_btn):
                idx = int(time_w.value)
                date_str = str(spectral.time.values[idx])[:10]
                cr_first_scene_mode_w.value = "date"  # enables the date field
                cr_ref_date_w.value = date_str
                _status(f"✅ Reference scene set to {date_str}.")

            def _on_close(_btn):
                cr_pick_out.clear_output()

            select_btn.on_click(_on_select)
            close_btn.on_click(_on_close)
            display(widgets.HBox([select_btn, close_btn],
                                 layout=widgets.Layout(gap="8px")))

    cr_browse_scenes_btn.on_click(_on_browse_scenes_clicked)

    cr_iteration_w = widgets.Dropdown(
        options=[("auto", "auto"), ("1", 1), ("2", 2), ("3", 3), ("4", 4), ("5", 5)],
        value="auto",
        layout=widgets.Layout(width="200px"),
    )

    # output path
    cr_out_w = widgets.Text(value="", layout=widgets.Layout(width="100%"))
    browse_cr_out_btn = widgets.Button(icon="folder-open", description="", layout=widgets.Layout(width="36px"))
    cr_out_fc_box = _attach_filechooser(
        browse_cr_out_btn,
        cr_out_w,
        title="Select output cube (.nc or .zarr) for co-registered cube",
        pattern=["*.nc", "*"],
        select_dirs=False,
    )
    cr_out_row = widgets.HBox(
        [browse_cr_out_btn, cr_out_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    cr_out_box = widgets.VBox([cr_out_row, cr_out_fc_box], layout=widgets.Layout(width="100%", gap="4px"))

    # NetCDF-only output options, same pair as the Data Cube Builder. Hidden for
    # a Zarr cube (zlib is a no-op there and a VRT cannot read a Zarr store's
    # pixels back) - see _sync_cr_output_options.
    cr_compress_w = widgets.Checkbox(
        value=False,
        description="Lossless compression (zlib)",
        indent=False,
        layout=widgets.Layout(width="auto"),
    )
    cr_compress_warn_html = widgets.HTML(
        "<div style='font-size:12px; color:#b00020;'>"
        "⚠️ <b>Warning:</b> compression shrinks the output file a further "
        "~20-40% (scene-dependent), but the export step takes roughly "
        "<b>10x longer</b>. Enable it only for archiving, when disk space "
        "matters more than your time.</div>"
    )
    cr_compress_warn_html.layout.display = "none"
    cr_vrt_w = widgets.Checkbox(
        value=False,
        description="Export Band Mapping for GIS Tools (.vrt)",
        indent=False,
        layout=widgets.Layout(width="auto"),
    )
    cr_vrt_note_html = widgets.HTML(
        "<div style='font-size:12px; color:#555; margin-left:2px;'>"
        "&#8505;&#65039; Open the <b>.vrt</b> file in QGIS, not the .nc, "
        "and keep both files in the same folder.</div>"
    )
    cr_vrt_note_html.layout.display = "none"

    def _sync_cr_output_options(*_):
        """Show the NetCDF-only switches only when the output really is NetCDF.

        Keyed off the OUTPUT path rather than the loaded cube: co-registration
        picks its container from the output extension, so a NetCDF cube can be
        written to a .zarr store and vice versa.
        """
        out = (cr_out_w.value or "").strip()
        lp = state.get("loaded_path")
        target_zarr = is_zarr_path(out) if out else (bool(lp) and is_zarr_path(lp))
        if target_zarr:
            cr_compress_w.value = False
            cr_vrt_w.value = False
        for w in (cr_compress_w, cr_vrt_w):
            w.layout.display = "none" if target_zarr else ""
            w.disabled = lp is None
        cr_compress_warn_html.layout.display = (
            "none" if target_zarr or lp is None or not cr_compress_w.value else ""
        )
        cr_vrt_note_html.layout.display = (
            "none" if target_zarr or lp is None or not cr_vrt_w.value else ""
        )

    cr_compress_w.observe(_sync_cr_output_options, names="value")
    cr_vrt_w.observe(_sync_cr_output_options, names="value")
    cr_out_w.observe(_sync_cr_output_options, names="value")

    # optional binary cloud mask (for co-registering unmasked / keep-clouds cubes)
    cr_cloud_mask_w = widgets.Text(value="", layout=widgets.Layout(width="100%"))
    browse_cr_mask_btn = widgets.Button(icon="folder-open", description="", layout=widgets.Layout(width="36px"))
    cr_mask_fc_box = _attach_filechooser(
        browse_cr_mask_btn,
        cr_cloud_mask_w,
        title="Select binary cloud mask cube (.nc or .zarr, Cloud_Stack)",
        pattern=["*.nc", "*"],
        select_dirs=False,
    )
    cr_mask_row = widgets.HBox(
        [browse_cr_mask_btn, cr_cloud_mask_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    cr_mask_box = widgets.VBox([cr_mask_row, cr_mask_fc_box], layout=widgets.Layout(width="100%", gap="4px"))

    cr_run_btn = widgets.Button(
        description="Co-register and Export",
        button_style="success",
        icon="play",
        layout=widgets.Layout(width="210px"),
    )

    cr_copy_json_btn = widgets.Button(
        description="Copy Settings",
        icon="copy",
        layout=widgets.Layout(width="150px"),  # colorless, like the data cube builder
    )

    def _on_first_scene_mode_change(change):
        if change.get("name") != "value":
            return
        cr_composite_window_days_w.disabled = (cr_first_scene_mode_w.value != "composite")
        cr_ref_date_w.disabled = (cr_first_scene_mode_w.value != "date")

    cr_first_scene_mode_w.observe(_on_first_scene_mode_change, names="value")

    def _resolve_first_scene_mode():
        """GUI mode -> coregister_cube's first_scene_mode value."""
        mode = str(cr_first_scene_mode_w.value)
        if mode != "date":
            return mode
        date_txt = (cr_ref_date_w.value or "").strip()
        if not date_txt:
            raise ValueError(
                'First Reference Scene is "select date" but no date is given. '
                "Enter YYYY-MM-DD (inspect the cube in the Data Cube Editor to "
                "pick a clean scene)."
            )
        return date_txt

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
        if state.get("loaded_var") != "Time_Series":
            _status(
                "❌ Co-registration needs the 'Time_Series' time series, "
                f"but the loaded layer is '{state.get('loaded_var')}'.",
                "Composite layers (e.g. median) are single images and cannot be "
                "co-registered over time.",
                "Reload the cube and select the 'Time_Series' layer.",
            )
            return
        if coregister_cube is None:
            _status("❌ coregister_cube is not available. Check stac2cube imports.")
            return

        out_path = (cr_out_w.value or "").strip()
        if not out_path:
            out_path = _suggest_cr_path()
            cr_out_w.value = out_path
        out_path = _ensure_cube_suffix(out_path)

        try:
            time_period = _parse_time_period(cr_time_period_w.value)
            first_scene_mode = _resolve_first_scene_mode()
            inliers_keep = _parse_inlier_value(cr_min_inliers_keep_w.value)
            inliers_update = _parse_inlier_value(cr_min_inliers_update_ref_w.value)
        except Exception as e:
            _status(f"❌ ValueError: {e}")
            return

        p_out = Path(out_path)
        existed_before = p_out.exists()
        old_mtime, old_size = _output_stat(p_out) if existed_before else (None, None)

        _status(
            "Co-registering...",
        )

        # Capture coregister_cube's printed "Co-registration summary" so it can be
        # shown BELOW the final status line (the finished/exported message comes
        # first, the summary after it). tqdm progress bars use display() and still
        # render live in status_out; only stdout prints are captured here.
        summary_buf = io.StringIO()

        def _flush_summary():
            summary_text = summary_buf.getvalue().strip()
            if summary_text:
                with status_out:
                    print()
                    print(summary_text)

        try:
            with status_out:
                with redirect_stdout(summary_buf):
                    coregister_cube(
                        input_path=state["loaded_path"],
                        grid_size=int(cr_grid_size_w.value),
                        max_cc=int(cr_max_cc_w.value),
                        time_period=time_period,
                        cloud_mask=((cr_cloud_mask_w.value or "").strip() or None),
                        match_band=str(cr_match_band_w.value),
                        min_inliers_keep=inliers_keep,
                        min_inliers_update_ref=inliers_update,
                        adaptive=bool(cr_adaptive_w.value),
                        max_cloud_update_ref=float(cr_max_cloud_update_ref_w.value),
                        first_scene_mode=first_scene_mode,
                        composite_window_days=int(cr_composite_window_days_w.value),
                        compress=bool(cr_compress_w.value),
                        vrt=bool(cr_vrt_w.value),
                        iteration=cr_iteration_w.value,  # "auto" or int
                        output_path=out_path,
                    )

            # --- Verify export actually happened (prevents false ✅). The tool's
            # own summary is captured above, so it is flushed after the message.
            if not p_out.exists():
                _status("❌ Co-registration failed: output file was not created.")
                _flush_summary()
                return

            new_mtime, new_size = _output_stat(p_out)
            if existed_before and (new_mtime == old_mtime) and (new_size == old_size):
                _status("❌ Co-registration failed: output file was not updated.")
                _flush_summary()
                return

            try:
                with open_cube(p_out) as _:
                    pass
            except Exception as e:
                _status(f"❌ Co-registration failed: output file is not readable ({type(e).__name__}: {e})")
                _flush_summary()
                return

            state["current_result_path"] = out_path
            # Spectral profiler: point the "after" path at the fresh co-registered
            # cube, and default the "before" path to the loaded cube if unset.
            cr_prof_after_w.value = out_path
            if not (cr_prof_before_w.value or "").strip() and state.get("loaded_path"):
                cr_prof_before_w.value = str(Path(state["loaded_path"]).as_posix())

            # Finished/exported message FIRST, then the co-registration summary.
            _status(f"✅ Co-registration finished and exported: {out_path}")
            _flush_summary()

        except Exception as e:
            _status(f"❌ {type(e).__name__}: {e}")
            _flush_summary()

    cr_run_btn.on_click(_on_coregister_clicked)

    def _build_cr_json_syntax_text():
        """
        Build JSON syntax for HPC/SLURM coregistration config from current UI state.
        Mirrors stac2cube.coregistration.coregister_cube(**parameters) and the
        example in slurm/3_coregistration/README.md. Uses null/true/false via json.dumps.
        """
        try:
            time_period = _parse_time_period(cr_time_period_w.value)
        except Exception:
            # Don't block copying on an invalid time_period; emit null and let
            # the run/validation surface the error.
            time_period = None

        input_path = state.get("loaded_path")
        input_for_json = str(input_path) if input_path else None

        out_path = (cr_out_w.value or "").strip()
        output_for_json = out_path or None

        # composite_window_days only applies in "composite" mode; the widget is
        # disabled for "first", so emit null there rather than a stale value.
        composite_window_for_json = (
            None
            if cr_composite_window_days_w.disabled
            else int(cr_composite_window_days_w.value)
        )

        # "select date" mode emits the date string itself; fall back to the
        # raw mode when the date is missing and let the run surface the error.
        try:
            first_scene_mode_for_json = _resolve_first_scene_mode()
        except Exception:
            first_scene_mode_for_json = str(cr_first_scene_mode_w.value)

        # agreement thresholds: "auto" or an integer; emit the raw text as
        # typed when unparsable and let the run surface the error
        try:
            inliers_keep_for_json = _parse_inlier_value(cr_min_inliers_keep_w.value)
        except Exception:
            inliers_keep_for_json = str(cr_min_inliers_keep_w.value)
        try:
            inliers_update_for_json = _parse_inlier_value(cr_min_inliers_update_ref_w.value)
        except Exception:
            inliers_update_for_json = str(cr_min_inliers_update_ref_w.value)

        # Parameter order follows the GUI layout (top -> bottom):
        # max_cc, time_period, grid_size, iteration, match_band,
        # min_inliers_keep, min_inliers_update_ref, max_cloud_update_ref,
        # first_scene_mode, composite_window_days. Paths first, as in the
        # SLURM README.
        json_payload = {
            "parameters": {
                "input_path": input_for_json,
                "output_path": output_for_json,
                "max_cc": int(cr_max_cc_w.value),
                "time_period": time_period,
                "grid_size": int(cr_grid_size_w.value),
                "iteration": cr_iteration_w.value,
                "match_band": str(cr_match_band_w.value),
                "min_inliers_keep": inliers_keep_for_json,
                "min_inliers_update_ref": inliers_update_for_json,
                "adaptive": bool(cr_adaptive_w.value),
                "max_cloud_update_ref": float(cr_max_cloud_update_ref_w.value),
                "first_scene_mode": first_scene_mode_for_json,
                "composite_window_days": composite_window_for_json,
                "cloud_mask": ((cr_cloud_mask_w.value or "").strip() or None),
                # NetCDF-only; pinned to False for a Zarr output so a copied
                # config reproduces what the GUI would actually do.
                "compress": bool(cr_compress_w.value) and not is_zarr_path(out_path),
                "vrt": bool(cr_vrt_w.value) and not is_zarr_path(out_path),
            }
        }

        return json.dumps(json_payload, indent=2, ensure_ascii=False)

    def _copy_cr_json_to_clipboard(_):
        """
        Build current coregistration JSON syntax and copy it to clipboard.
        """
        try:
            text = _build_cr_json_syntax_text()
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

            _status("✅ JSON syntax copied to clipboard.")

        except Exception as e:
            _status(f"❌ {type(e).__name__}: {e}")

    cr_copy_json_btn.on_click(_copy_cr_json_to_clipboard)


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

    # --- Layout: a welcoming front (output + run is all a casual user
    # needs; everything else is auto) with collapsed sections for the rest.

    cr_ref_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "The reference scene anchors the alignment; every other scene is "
                "aligned to it (in both time directions). <b>auto</b> picks a clean, "
                "high-contrast scene near the middle of the series for you - or use "
                "<b>Browse Scenes</b> to pick one yourself."
                "</div>"
            ),
            _row(
                [
                    _stacked_field_with_help(cr_first_scene_mode_w, "Reference Scene", "first_scene_mode"),
                    _stacked_field_with_help(cr_ref_date_w, "Reference Date", "reference_date"),
                    cr_browse_scenes_btn,
                    _stacked_field_with_help(cr_composite_window_days_w, "Composite Window (days)", "composite_window_days"),
                ],
                gap_px=20,
            ),
            cr_pick_out,
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )
    cr_ref_acc = widgets.Accordion(children=[cr_ref_box], selected_index=None)
    cr_ref_acc.set_title(0, "Reference Scene")
    cr_ref_acc.layout = widgets.Layout(width="99%")

    cr_filter_box = widgets.VBox(
        [
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
    cr_filter_acc = widgets.Accordion(children=[cr_filter_box], selected_index=None)
    cr_filter_acc.set_title(0, "Filters (optional)")
    cr_filter_acc.layout = widgets.Layout(width="99%")

    cr_adv_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "All of these have safe automatic defaults - open the (?) of a field "
                "before changing it."
                "</div>"
            ),
            _row(
                [
                    _stacked_field_with_help(cr_grid_size_w, "Full scan size (for difficult scenes)", "grid_size"),
                    _stacked_field_with_help(cr_match_band_w, "Matching Band", "match_band"),
                    _stacked_field_with_help(cr_iteration_w, "Iteration", "iteration"),
                ],
                gap_px=20,
            ),
            _row(
                [
                    _stacked_field_with_help(cr_min_inliers_keep_w, "Matching Points to Keep a Scene", "min_inliers_keep"),
                    _stacked_field_with_help(cr_min_inliers_update_ref_w, "Matching Points to Trust as Reference", "min_inliers_update_ref"),
                    _stacked_field_with_help(cr_max_cloud_update_ref_w, "Max Cloud % for Reference Scenes", "max_cloud_update_ref"),
                ],
                gap_px=20,
            ),
            cr_adaptive_w,
            widgets.HTML(
                "<div style='font-size:11px; color:#666; margin-top:-4px;'>"
                "Uses a quick coarse scan first and the full scan size only when a "
                "scene is difficult - fast without losing accuracy."
                "</div>"
            ),
        ],
        layout=widgets.Layout(width="100%", gap="6px"),
    )
    cr_adv_acc = widgets.Accordion(children=[cr_adv_box], selected_index=None)
    cr_adv_acc.set_title(0, "Advanced Settings")
    cr_adv_acc.layout = widgets.Layout(width="99%")

    # ------------------------------------------------------------------
    # Cloud Mask Cube (standalone, before Reference Scene)
    # ------------------------------------------------------------------
    # Only relevant for cubes that still keep their clouds. Shown (with a yellow
    # warning) for unmasked / clouds-detected cubes; hidden for cloud-masked
    # cubes, where co-registration already ignores the masked pixels.
    cr_mask_warn_w = widgets.HTML(value="")
    cr_mask_section = widgets.VBox(
        [
            cr_mask_warn_w,
            _stacked_field(cr_mask_box, "Cloud Mask Cube (binary mask)"),
            widgets.HTML(
                "<div style='font-size:11px; color:#666; margin-top:-6px;'>"
                "The binary cloud mask (builder's mask export) lets the algorithm ignore clouds "
                "while measuring the shift; the clouds themselves stay in the exported scenes."
                "</div>"
            ),
        ],
        layout=widgets.Layout(width="99%", gap="6px", display="none"),
    )

    def _cube_is_cloud_masked():
        """True when the loaded cube's cloud_status marks it as cloud-masked
        (e.g. scl_masked, scl_shadow_masked, cloud_mask_50). clouds_detected /
        clouds_not_detected count as NOT masked."""
        for src in (state.get("loaded_obj"), state.get("loaded_ds")):
            attrs = getattr(src, "attrs", None) or {}
            cs = str(attrs.get("cloud_status", "") or "").strip().lower()
            if "mask" in cs:
                return True
        return False

    def _refresh_cr_cloud_mask_ui():
        """Show/hide the Cloud Mask Cube selector from the loaded cube's state."""
        if state.get("loaded_obj") is None:
            cr_mask_section.layout.display = "none"
            cr_mask_warn_w.value = ""
            return
        if _cube_is_cloud_masked():
            # Already masked: the masked (NaN) pixels are ignored automatically,
            # so no external cloud mask is needed.
            cr_mask_section.layout.display = "none"
            cr_mask_warn_w.value = ""
            cr_cloud_mask_w.value = ""
        else:
            cr_mask_section.layout.display = ""
            cr_mask_warn_w.value = (
                "<div style='font-size:12px; color:#92400e; "
                "background:#fef3c7; border:1px solid #fcd34d; "
                "border-radius:8px; padding:8px 10px; margin:0 0 4px 0;'>"
                "⚠️ Clouds are not masked in this data cube. Please provide a binary "
                "cloud mask file to co-register the time series with higher efficiency! "
                "If you don't have any, use the <b>Build Cloud Mask Cube</b> feature "
                "from the <b>Data Cube Editor</b>."
                "</div>"
            )

    # ------------------------------------------------------------------
    # Spectral Profiler (after Output NetCDF)
    # ------------------------------------------------------------------
    # Click a pixel on a median-RGB base image and compare the chosen band's
    # time series before (loaded cube) vs after (co-registered cube).
    cr_prof_before_w = widgets.Text(value="", layout=widgets.Layout(width="100%"))
    browse_cr_prof_before_btn = widgets.Button(icon="folder-open", description="", layout=widgets.Layout(width="36px"))
    cr_prof_before_fc = _attach_filechooser(
        browse_cr_prof_before_btn, cr_prof_before_w,
        title="Select the BEFORE cube (loaded, non-co-registered)",
        pattern=["*.nc", "*"], select_dirs=False,
    )
    cr_prof_before_row = widgets.HBox(
        [browse_cr_prof_before_btn, cr_prof_before_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    cr_prof_before_box = widgets.VBox([cr_prof_before_row, cr_prof_before_fc], layout=widgets.Layout(width="100%", gap="4px"))

    cr_prof_after_w = widgets.Text(value="", layout=widgets.Layout(width="100%"))
    browse_cr_prof_after_btn = widgets.Button(icon="folder-open", description="", layout=widgets.Layout(width="36px"))
    cr_prof_after_fc = _attach_filechooser(
        browse_cr_prof_after_btn, cr_prof_after_w,
        title="Select the AFTER cube (co-registered)",
        pattern=["*.nc", "*"], select_dirs=False,
    )
    cr_prof_after_row = widgets.HBox(
        [browse_cr_prof_after_btn, cr_prof_after_w],
        layout=widgets.Layout(width="100%", gap="6px", align_items="center"),
    )
    cr_prof_after_box = widgets.VBox([cr_prof_after_row, cr_prof_after_fc], layout=widgets.Layout(width="100%", gap="4px"))

    cr_prof_band_w = widgets.Dropdown(options=["ndvi"], value="ndvi", layout=widgets.Layout(width="200px"))
    cr_prof_btn = widgets.Button(
        description="Show Spectral Profiler",
        button_style="primary",
        icon="chart-line",
        layout=widgets.Layout(width="220px"),
    )
    cr_prof_out = widgets.Output(layout=widgets.Layout(width="99%", overflow="auto"))

    def _cube_band_list(obj):
        """Band names of a loaded cube (DataArray or Dataset), or []."""
        try:
            if obj is None:
                return []
            da = obj
            if isinstance(obj, xr.Dataset):
                da = obj.get("Time_Series")
                if da is None:
                    return []
            return [str(b) for b in da["band"].values]
        except Exception:
            return []

    def _refresh_cr_prof_band_options():
        bands = _cube_band_list(state.get("loaded_obj"))
        if not bands:
            cr_prof_band_w.options = ["ndvi"]
            cr_prof_band_w.value = "ndvi"
            return
        cur = cr_prof_band_w.value
        cr_prof_band_w.options = bands
        if cur in bands:
            cr_prof_band_w.value = cur
        elif "ndvi" in bands:
            cr_prof_band_w.value = "ndvi"
        else:
            cr_prof_band_w.value = bands[0]

    def _refresh_cr_prof_paths():
        """Default the before path to the loaded cube and the after path to the
        co-registered output; only fill blanks so user edits are kept."""
        if not (cr_prof_before_w.value or "").strip() and state.get("loaded_path"):
            cr_prof_before_w.value = str(Path(state["loaded_path"]).as_posix())
        after_default = state.get("current_result_path") or (cr_out_w.value or "").strip()
        if not (cr_prof_after_w.value or "").strip() and after_default:
            cr_prof_after_w.value = str(after_default)

    def _on_show_profiler_clicked(_):
        with cr_prof_out:
            clear_output()
        before = (cr_prof_before_w.value or "").strip()
        after = (cr_prof_after_w.value or "").strip()
        band = str(cr_prof_band_w.value or "ndvi").strip()
        if not before or not after:
            with cr_prof_out:
                print("❌ Set both the before (loaded) and after (co-registered) cube paths.")
            return
        for label, pth in (("before", before), ("after", after)):
            if not Path(pth).exists():
                with cr_prof_out:
                    print(f"❌ {label} cube not found: {pth}")
                return
        try:
            with cr_prof_out:
                print("Loading cubes and preparing the spectral profiler...")
            with cr_prof_out:
                clear_output()
                spectral_profiler(before, after, band=band, rgb_time="median")
        except Exception as e:
            with cr_prof_out:
                clear_output()
                print(_friendly_error(e, "Spectral profiler"))

    cr_prof_btn.on_click(_on_show_profiler_clicked)

    cr_prof_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Click a pixel on the median-RGB base image and compare the chosen band's "
                "time series <b>before</b> (loaded cube) vs <b>after</b> (co-registered cube). "
                "The paths are pre-filled with your loaded and co-registered cubes."
                "</div>"
            ),
            _stacked_field(cr_prof_before_box, "Before cube (loaded)"),
            _stacked_field(cr_prof_after_box, "After cube (co-registered)"),
            _stacked_field(cr_prof_band_w, "Band"),
            cr_prof_btn,
            cr_prof_out,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    cr_prof_acc = widgets.Accordion(children=[cr_prof_box], selected_index=None)
    cr_prof_acc.set_title(0, "Spectral Profiler")
    cr_prof_acc.layout = widgets.Layout(width="99%")
    # Same vivid green header as the mask tool's Visualization section.
    cr_prof_acc.add_class("stac2cube-acc-vivid")

    def _on_prof_acc_open(change):
        if change.get("name") == "selected_index" and change.get("new") == 0:
            _refresh_cr_prof_band_options()
            _refresh_cr_prof_paths()
    cr_prof_acc.observe(_on_prof_acc_open, names="selected_index")

    cr_tool_box = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:12px; color:#666;'>"
                "Aligns all scenes of the data cube to each other with sub-pixel "
                "precision, so time series become clean and comparable.<br>"
                "The default settings work well for most cubes: <b>set the output "
                "file and click Co-register</b>. Larger, varied areas align best."
                "</div>"
            ),
            cr_mask_section,
            cr_ref_acc,
            cr_filter_acc,
            cr_adv_acc,
            widgets.HTML("<div style='height:10px;'></div>"),
            # Co-registration picks the container from the extension (a .zarr
            # path writes a Zarr store), so the label must not say NetCDF only.
            _stacked_field(cr_out_box, "Output cube (NetCDF/Zarr)"),
            cr_compress_w,
            cr_compress_warn_html,
            cr_vrt_w,
            cr_vrt_note_html,
            widgets.HTML("<div style='height:10px;'></div>"),
            widgets.HBox(
                [cr_run_btn, cr_copy_json_btn],
                layout=widgets.Layout(gap="8px", align_items="center"),
            ),
            widgets.HTML("<div style='height:10px;'></div>"),
            cr_prof_acc,
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
            return (p.parent / f"{p.stem}_sr{_ext_from_loaded()}").as_posix()
        return "./results/cube_sr.nc"

    sr_mode_w = widgets.Dropdown(
        options=[
            ("10-m RGBN to 2.5-m", "rgbn"),
            ("10-m and 20-m Full Spectral to 2.5-m", "full_spectral"),
            ("20-m Bands to 10-m", "20to10"),
        ],
        value="rgbn",
        layout=widgets.Layout(width="320px"),
    )

    SR_REQUIRED_BANDS = {
        "rgbn": ["blue", "green", "red", "nir"],
        "full_spectral": [
            "blue", "green", "red", "nir", "nir08",
            "rededge1", "rededge2", "rededge3", "swir16", "swir22",
        ],
        "20to10": [
            "blue", "green", "red", "nir", "nir08",
            "rededge1", "rededge2", "rededge3", "swir16", "swir22",
        ],
    }

    # Bullet lines after the required-band line (which is built dynamically).
    SR_DESC_REST = {
        "rgbn": (
            "- If exist, indices must be only 10-meter resolution ones, e.g., ndvi, ndwi<br>"
            "- Use this model if you don't have 20-m bands. Much faster model!"
        ),
        "full_spectral": (
            "- If exist, indices can be both 10 and 20-meter resolution ones, e.g., ndvi, ndwi, ndmi<br>"
            "- Use this model only if you need to super resolve 20-meter bands.<br>"
            "- Even if you need to super-resolve one of the 20-meter bands, still need to include all of the required ones."
        ),
        "20to10": (
            "- Sharpens the six 20-m bands (rededge1/2/3, nir08, swir16, swir22) to true 10-m detail; the 10-m bands are used as reference and pass through unchanged.<br>"
            "- The output keeps the cube's 10-m grid (no pixel-size change), so the cube must be built at 10-m resolution.<br>"
            "- If exist, indices can be both 10 and 20-meter resolution ones, e.g., ndvi, ndwi, ndmi"
        ),
    }

    def _sr_required_bands_html(mode):
        """Required-band list for a mode. Once a cube is loaded, each band is
        coloured green when the cube has it and red when it is missing, so the
        user sees at a glance whether the mode is usable."""
        req = SR_REQUIRED_BANDS[mode]
        have = {b.lower() for b in _cube_band_list(state.get("loaded_obj"))}
        if not have:
            return "<code>" + ", ".join(req) + "</code>"
        parts = [
            "<span style='color:{c}; font-weight:600;'>{b}</span>".format(
                c="#2e7d32" if b.lower() in have else "#c62828", b=b
            )
            for b in req
        ]
        return "<code>" + ", ".join(parts) + "</code>"

    def _sr_desc_value(mode):
        return (
            "<div style='font-size:12px; color:#666;'>"
            f"- Required band setup -> {_sr_required_bands_html(mode)}<br>"
            f"{SR_DESC_REST[mode]}</div>"
        )

    sr_desc_html = widgets.HTML(_sr_desc_value(sr_mode_w.value))

    def _refresh_sr_desc():
        sr_desc_html.value = _sr_desc_value(sr_mode_w.value)

    # Output NetCDF
    sr_out_w = widgets.Text(value="", layout=widgets.Layout(width="100%"))
    browse_sr_out_btn = widgets.Button(icon="folder-open", description="", layout=widgets.Layout(width="36px"))
    sr_out_fc_box = _attach_filechooser(
        browse_sr_out_btn,
        sr_out_w,
        title="Select output cube (.nc or .zarr) for super-resolved cube",
        pattern=["*.nc", "*"],
        select_dirs=False,
    )
    sr_out_row = widgets.HBox([browse_sr_out_btn, sr_out_w], layout=widgets.Layout(width="100%", gap="6px", align_items="center"))
    sr_out_box = widgets.VBox([sr_out_row, sr_out_fc_box], layout=widgets.Layout(width="100%", gap="4px"))

    sr_compress_w = widgets.Checkbox(
        value=False,
        description="Lossless compression (zlib)",
        indent=False,
        layout=widgets.Layout(width="auto"),
    )
    sr_compress_warn_html = widgets.HTML(
        "<div style='font-size:12px; color:#b00020;'>"
        "⚠️ <b>Warning:</b> compression shrinks the output file a further "
        "~20-40% (scene-dependent), but the export step takes roughly "
        "<b>10x longer</b>. Enable it only for archiving, when disk space "
        "matters more than your time.</div>"
    )
    sr_compress_warn_html.layout.display = "none"

    # Shown INSTEAD of the checkbox+warning when the loaded cube is Zarr: zlib
    # is a NetCDF-only knob, and a Zarr store is always written with Zarr's own
    # default codec, so the flag would be a silent no-op (verified: compress
    # True vs False yields a bit-identical Zarr store). Greying the checkbox out
    # keeps it from implying an effect it does not have.
    sr_zarr_note_html = widgets.HTML(
        "<div style='font-size:12px; color:#666;'>"
        "ℹ️ Zarr stores are always compressed with Zarr's default codec, so the "
        "zlib option does not apply to a Zarr output (it is a NetCDF-only "
        "setting).</div>"
    )
    sr_zarr_note_html.layout.display = "none"

    sr_vrt_w = widgets.Checkbox(
        value=False,
        description="Export Band Mapping for GIS Tools (.vrt)",
        indent=False,
        layout=widgets.Layout(width="auto"),
    )
    sr_vrt_note_html = widgets.HTML(
        "<div style='font-size:12px; color:#555; margin-left:2px;'>"
        "&#8505;&#65039; Open the <b>.vrt</b> file in QGIS, not the .nc, "
        "and keep both files in the same folder.</div>"
    )
    sr_vrt_note_html.layout.display = "none"

    def _on_sr_compress_change(change):
        if change.get("name") != "value":
            return
        _sync_sr_compress_for_format()

    sr_compress_w.observe(_on_sr_compress_change, names="value")
    sr_vrt_w.observe(_on_sr_compress_change, names="value")

    def _sync_sr_compress_for_format():
        """Both options are NetCDF-only, so hide them for a Zarr cube.

        zlib is a no-op on a Zarr store (always written with Zarr's own codec)
        and a VRT cannot read a Zarr store's pixels back, so neither switch has
        anything to offer there - showing them disabled only invites the
        question of why. For a NetCDF cube they behave normally; with nothing
        loaded yet they stay visible but disabled.
        """
        lp = state.get("loaded_path")
        zarr_loaded = bool(lp) and is_zarr_path(lp)
        if zarr_loaded:
            sr_compress_w.value = False
            sr_vrt_w.value = False
        for w in (sr_compress_w, sr_vrt_w):
            w.layout.display = "none" if zarr_loaded else ""
            w.disabled = lp is None
        sr_zarr_note_html.layout.display = "none"
        sr_compress_warn_html.layout.display = (
            "none" if zarr_loaded or lp is None or not sr_compress_w.value else ""
        )
        sr_vrt_note_html.layout.display = (
            "none" if zarr_loaded or lp is None or not sr_vrt_w.value else ""
        )

    sr_run_btn = widgets.Button(
        description="Super-resolve and Export",
        button_style="success",
        icon="play",
        layout=widgets.Layout(width="220px"),
    )

    def _on_sr_mode_change(change):
        if change.get("name") != "value":
            return
        _refresh_sr_desc()

        # enabled state depends on load status via _set_enabled_after_load
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
        out_path = (sr_out_w.value or "").strip()
        if not out_path:
            out_path = _suggest_sr_path_from_loaded()
            sr_out_w.value = out_path
        out_path = _ensure_cube_suffix(out_path)

        p_out = Path(out_path)
        existed_before = p_out.exists()
        old_mtime, old_size = _output_stat(p_out) if existed_before else (None, None)

        sr_var_name = state.get("loaded_var") or "Time_Series"

        _status(
            "Super-resolving and exporting...",
            "",
            "ℹ️ Note: the progress bar below tracks how many time steps have been "
            "processed — not the progress of super-resolving each individual image. "
            "A single time step can take a while, so the bar may sit still for a bit. "
            "That's expected, not a bug.",
            "",
        )

        try:
            # Capture progress prints from the tool
            with status_out:
                super_resolve_cube(
                    input_path=state["loaded_path"],
                    output_path=out_path,
                    var_name=sr_var_name,
                    model_type=mode,  # "rgbn" | "full_spectral" | "20to10"
                    compress=bool(sr_compress_w.value),
                    vrt=bool(sr_vrt_w.value),
                )

            # --- Verify export actually happened (prevents false ✅) ---
            if not p_out.exists():
                with status_out:
                    print("❌ Super-resolution failed: output file was not created.")
                return

            new_mtime, new_size = _output_stat(p_out)

            if existed_before and (new_mtime == old_mtime) and (new_size == old_size):
                with status_out:
                    print(
                        "❌ Super-resolution failed: output file was not updated "
                        "(likely missing required bands for the selected mode)."
                    )
                return

            # Ensure file is readable
            try:
                with open_cube(p_out) as _:
                    pass
            except Exception as e:
                with status_out:
                    print(f"❌ Super-resolution failed: output file is not readable ({type(e).__name__}: {e})")
                return

            # Success
            state["current_result_path"] = out_path
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
            _stacked_field(sr_out_box, "Output cube (NetCDF/Zarr)"),
            sr_compress_w,
            sr_compress_warn_html,
            sr_vrt_w,
            sr_vrt_note_html,
            sr_zarr_note_html,
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
            widgets.HTML("<div style='font-size:12px; color:#666;'>Each tool exports its result in the loaded cube's format - NetCDF or Zarr (no Geotiff export here).</div>"),
            mask_tool_acc,
            cr_tool_acc,
            sr_tool_acc,
        ],
        layout=widgets.Layout(width="100%", gap="8px"),
    )
    tools_card = widgets.VBox([tools_box], layout=widgets.Layout(width="100%"))
    tools_card.add_class("stac2cube-card")

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
        cr_match_band_w.disabled = not enabled
        cr_min_inliers_keep_w.disabled = not enabled
        cr_min_inliers_update_ref_w.disabled = not enabled
        cr_max_cloud_update_ref_w.disabled = not enabled
        cr_first_scene_mode_w.disabled = not enabled
        cr_composite_window_days_w.disabled = (not enabled) or (cr_first_scene_mode_w.value != "composite")
        cr_ref_date_w.disabled = (not enabled) or (cr_first_scene_mode_w.value != "date")
        cr_browse_scenes_btn.disabled = not enabled
        cr_adaptive_w.disabled = not enabled
        cr_iteration_w.disabled = not enabled
        cr_cloud_mask_w.disabled = not enabled
        browse_cr_mask_btn.disabled = not enabled
        cr_out_w.disabled = not enabled
        browse_cr_out_btn.disabled = not enabled
        cr_run_btn.disabled = not enabled

        # Tool 3/3 Super-resolution
        sr_mode_w.disabled = not enabled
        sr_out_w.disabled = not enabled
        browse_sr_out_btn.disabled = not enabled

        # NetCDF-only switches on both tools: hidden for a Zarr target (reads
        # state['loaded_path'], which _finalize_load sets before calling this).
        _sync_sr_compress_for_format()
        _sync_cr_output_options()
        _sync_cm_output_options()

        # Colour the required-band list against the freshly loaded cube (and
        # reset it to neutral when a cube is unloaded).
        _refresh_sr_desc()

        sr_run_btn.disabled = not enabled


    _set_enabled_after_load(False)

    # -----------------------------------------
    # Events
    # -----------------------------------------
    def _finalize_load(path_posix, ds, var_name):
        """Initialize the tools from one layer (data variable) of an already
        opened dataset. Called directly for single-layer files, or from the
        'Load selected layer' button for multi-layer files."""
        obj = ds[var_name]

        state["loaded_path"] = path_posix
        state["loaded_var"] = var_name
        state["loaded_obj"] = obj
        #reset_btn.disabled = False
        _set_enabled_after_load(True)
        _refresh_output_suggestions()

        # Co-register tool: refresh the cloud-mask section (shown only for
        # unmasked cubes) and the spectral profiler defaults for this cube.
        _refresh_cr_cloud_mask_ui()
        _refresh_cr_prof_band_options()
        cr_prof_before_w.value = str(Path(path_posix).as_posix())
        cr_prof_after_w.value = ""

        _show_loaded_summary(obj)
        loaded_summary_acc.selected_index = 0
        _status(
            "✅ Cube loaded.",
            f"Loaded path: {state['loaded_path']}",
            f"Working layer: {_layer_display_name(var_name)}",
            "Select one of the listed tools to proceed.",
        )

    def _on_load_clicked(_):
        path = (load_path_w.value or "").strip()
        if not path:
            _status("❌ Please select a NetCDF (.nc) or Zarr (.zarr) cube path.")
            return
        # A file picked INSIDE a .zarr store (e.g. zarr.json) resolves to the
        # store root, so the file chooser can be used for Zarr cubes too.
        resolved = resolve_cube_path(path)
        if resolved != path:
            path = resolved
            load_path_w.value = str(Path(path).as_posix())
        p = Path(path)
        if not p.exists():
            _status(f"❌ File not found: {p.as_posix()}")
            return

        try:
            _status("Loading cube...")

            # Close any previously opened cube to release its file handle
            # (important on Windows, where an open handle locks the file).
            _prev_ds = state.get("loaded_ds")
            if _prev_ds is not None:
                try:
                    _prev_ds.close()
                except Exception:
                    pass
                state["loaded_ds"] = None

            # Open lazily (Dask-backed): large cubes are read on demand instead of
            # being copied into RAM. The handle must stay open for later reads, so
            # this is deliberately not a closing `with` block. One chunk per scene
            # ("frames"), because every interactive read here is one date at a
            # time; see _frame_chunked_netcdf for why "auto" is ~80x slower.
            ds_open = open_cube(p, chunks="frames")
            # Keep small coordinates in memory; only the data variables stay lazy
            # (chunked non-dimension coords otherwise break boolean-indexer ops).
            ds = ds_open.assign_coords(
                {name: coord.compute() for name, coord in ds_open.coords.items()}
            )
            state["loaded_ds"] = ds_open

            layers = _raster_layer_names(ds)
            if not layers:
                raise ValueError(
                    "Cube contains no raster layers (data variables with "
                    f"'y'/'x' dims). Found data_vars: {list(ds.data_vars)}"
                )

            if len(layers) == 1:
                # Single layer (e.g. cube exported without stats): load it
                # directly, no matter how the variable is named.
                state["pending_ds"] = None
                state["pending_path"] = None
                layer_select_w.options = []
                layer_select_box.layout.display = "none"
                _finalize_load(p.as_posix(), ds, layers[0])
            else:
                # Multiple layers (e.g. time series + temporal composites):
                # list them and let the user pick before initializing.
                state["pending_ds"] = ds
                state["pending_path"] = p.as_posix()
                # The previous loaded object may reference the file handle that
                # was just closed above, so drop it until a layer is confirmed.
                state["loaded_path"] = None
                state["loaded_var"] = None
                state["loaded_obj"] = None
                _set_enabled_after_load(False)

                layer_select_w.options = _layer_dropdown_options(ds, layers)
                layer_select_w.value = (
                    "Time_Series"
                    if "Time_Series" in layers
                    else layers[0]
                )
                layer_select_box.layout.display = ""

                with loaded_summary_out:
                    clear_output()
                    print("No layer loaded yet.")

                layer_lines = []
                for name in layers:
                    dims = ", ".join(
                        f"{d}: {ds[name].sizes[d]}" for d in ds[name].dims
                    )
                    layer_lines.append(f"   - {_layer_display_name(name)}  ({dims})")
                _status(
                    f"ℹ️ This cube contains {len(layers)} layers:",
                    *layer_lines,
                    "Select the layer to work on in the 'Layer' dropdown, "
                    "then click 'Load selected layer'.",
                )
        except Exception as e:
            _status(f"❌ {type(e).__name__}: {e}")

    def _on_layer_load_clicked(_):
        ds = state.get("pending_ds")
        path_posix = state.get("pending_path")
        var_name = layer_select_w.value
        if ds is None or not path_posix:
            _status("❌ Load a cube first.")
            return
        if not var_name:
            _status("❌ Please select a layer to load.")
            return

        try:
            _finalize_load(path_posix, ds, var_name)
        except Exception as e:
            _status(f"❌ {type(e).__name__}: {e}")

    def _on_reset_clicked(_):
        if state["loaded_obj"] is None:
            _status("❌ No loaded cube to reset to.")
            return
        # For this UI, reset just clears the last-exported-result pointer
        # (used as the spectral profiler's default "after" cube).
        state["current_result_path"] = None
        _status("✅ Reset done. (No exported result selected.)")


    load_cube_btn.on_click(_on_load_clicked)
    layer_load_btn.on_click(_on_layer_load_clicked)
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
            status_card,
        ],
        layout=widgets.Layout(width="100%", max_width="980px", margin="0 auto", gap="8px"),
    )
    ui.add_class("stac2cube-root")

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
            "layer_select": layer_select_w,
            "layer_load_btn": layer_load_btn,
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
            "status": status_out,
        },
    }
