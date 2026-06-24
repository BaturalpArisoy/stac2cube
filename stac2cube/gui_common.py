"""Shared, pure helpers for the stac2cube GUI tools (gui.py).

These utilities were previously duplicated inside datacube_builder() and
datacube_editor(). Centralising them keeps the GUIs consistent and shrinks
gui.py. They are intentionally UI-framework-light (no widget/closure state):
styling and help-text helpers differ per GUI and are unified separately.
"""
import ast
import os
import re
from pathlib import Path

import xarray as xr
import ipywidgets as widgets


def human_readable_bytes(n):
    if n is None:
        return "unknown"
    n = float(n)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}"


def estimated_data_size_bytes(obj):
    """Estimated uncompressed data size (shape * dtype), no compute triggered.

    This is NOT the final exported file size on disk.
    """
    try:
        if isinstance(obj, xr.DataArray):
            return int(getattr(obj, "nbytes", 0))
        if isinstance(obj, xr.Dataset):
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


def normalize_ui_path(path_str):
    if not path_str:
        return ""
    try:
        return os.path.normpath(str(path_str))
    except Exception:
        return str(path_str)


def existing_dir_or_parent(path_str):
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


def is_str_list_len2(obj):
    return (
        isinstance(obj, (list, tuple))
        and len(obj) == 2
        and all(isinstance(x, str) for x in obj)
    )


def validate_date_string(s: str, pattern: str, label: str):
    if not re.match(pattern, s):
        raise ValueError(f"Invalid {label}: '{s}'")


def parse_daterange_input(mode: str, text: str):
    """Parse the daterange UI field into the object get_stac_layers expects.

    Returns one of:
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
        if not is_str_list_len2(obj):
            raise ValueError('Standard mode expects: ["YYYY-MM-DD", "YYYY-MM-DD"]')
        for d in obj:
            validate_date_string(d, r"^\d{4}-\d{2}-\d{2}$", "date (YYYY-MM-DD)")
        return list(obj)

    if mode == "seasonal":
        if not is_str_list_len2(obj):
            raise ValueError('Seasonal mode expects: ["MM-DD", "MM-DD"]')
        for d in obj:
            validate_date_string(d, r"^\d{2}-\d{2}$", "season date (MM-DD)")
        return list(obj)

    if mode in ("seasonal_years", "seasonal_all", "seasonal_range", "seasonal_selected"):
        if not isinstance(obj, dict):
            raise ValueError(
                "Seasonal mode expects a dict, e.g. "
                '{"season": ["04-01", "10-31"], "years": [2019, 2020]}'
            )
        if "season" not in obj or "years" not in obj:
            raise ValueError(
                'Seasonal mode requires keys: "season" and "years"'
            )

        season = obj["season"]
        years = obj["years"]

        if not is_str_list_len2(season):
            raise ValueError('"season" must be ["MM-DD", "MM-DD"]')
        for d in season:
            validate_date_string(d, r"^\d{2}-\d{2}$", "season date (MM-DD)")

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


def friendly_error(e, action="The operation"):
    """One-line, plain-language error message for the GUI status box.

    Known failure families get a specific sentence with a suggested fix; the
    package's own validation ValueErrors (already written as human sentences)
    pass through without the exception-class jargon; anything unrecognized falls
    back to a generic sentence plus a single technical-detail line. Never shows
    a traceback. Classification is by exception type + conservative message
    patterns, so a misclassified error lands in the generic fallback rather
    than getting a wrong suggestion.
    """
    etype = type(e).__name__
    msg = str(e).strip() or etype
    low = msg.lower()

    def out(text):
        return f"❌ {action} failed — {text}"

    # Out of memory
    if isinstance(e, MemoryError) or "unable to allocate" in low or "out of memory" in low:
        return out(
            "not enough memory (RAM) for this data cube. Try selecting a shorter "
            "date range and use the update mechanism in the Data Cube Editor, which "
            "usually solves the issue. If even a smaller date range does not work, "
            "then select a significantly smaller area."
        )

    # Disk full
    if "no space left on device" in low or getattr(e, "errno", None) == 28:
        return out(
            "not enough disk space to write the file. Free some space or pick "
            "another location."
        )

    # Output file locked / write-protected
    if isinstance(e, PermissionError) or "permission denied" in low:
        return out(
            "the output file is in use or write-protected (is it open in another "
            "program?). Close it or choose a different output name."
        )

    # File not found (covers fiona/GDAL 'No such file or directory' too)
    if isinstance(e, FileNotFoundError) or "no such file or directory" in low:
        path = getattr(e, "filename", None)
        where = f": {path}" if path else f" ({msg})"
        return out(
            f"a file couldn't be found{where}. Check the path, or use the folder "
            "button to pick the file."
        )

    # Vector file exists but can't be read as polygon data
    if etype == "DriverError" or "not recognized as a supported file format" in low:
        return out(
            "this file couldn't be read as a polygon. Supported formats: "
            "gpkg, geojson, kml, kmz, shp."
        )

    # Invalid / degenerate geometry rejected by the STAC API
    if "invalid_shape" in low or "cannot determine orientation" in low:
        return out(
            "the selected area isn't a valid shape (it may have no width/height or "
            "self-crossing edges). Please re-draw the area or fix the polygon file, "
            "then try again."
        )

    # Access denied (e.g. terrabyte without credentials)
    if (
        "unauthorized" in low
        or "forbidden" in low
        or "authentication" in low
        or "invalid token" in low
        or "401 client error" in low
        or "403 client error" in low
    ):
        return out(
            "the data service refused access (credentials required). If you're "
            "using terrabyte, you need valid terrabyte credentials — see "
            "https://portal.terrabyte.lrz.de/."
        )

    # Catalog / network unreachable
    if (
        etype in (
            "ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout",
            "MaxRetryError", "NewConnectionError", "ProxyError", "SSLError",
        )
        or "max retries" in low
        or "timed out" in low
        or "getaddrinfo" in low
        or "connection refused" in low
        or "connection aborted" in low
        or "connection reset" in low
        or "failed to establish" in low
        or "temporary failure in name resolution" in low
    ):
        return out(
            "couldn't reach the data catalog. Check your internet connection, or "
            "the service may be temporarily down; try again in a few minutes."
        )

    # No scenes for the chosen parameters
    if "no scenes found" in low or "no scenes left" in low or "no scenes were kept" in low:
        return out(
            "no scenes were found for these settings. Try a wider time period, a "
            "higher Max CC, or double-check the polygon's location."
        )

    # The package's own validation messages are already human sentences — show
    # them as-is, just without the 'ValueError:' jargon (drop a leading 'Error:').
    if isinstance(e, ValueError):
        clean = re.sub(r"^error:\s*", "", msg, flags=re.IGNORECASE).strip()
        return out(clean if clean else "the input values are not valid.")

    # Unknown: generic sentence + one technical line (no traceback).
    detail = msg if len(msg) <= 300 else msg[:300] + "…"
    return (
        out("something unexpected went wrong and the operation couldn't be completed.")
        + f"\nTechnical detail (for bug reports): {etype}: {detail}"
    )

    raise ValueError(f"Unknown Date Range Mode: {mode}")


# ---------------------------------------------------------------------------
# Shared styling for all three GUIs
# ---------------------------------------------------------------------------
GUI_CSS = """
<style>
/* Root: kill the tiny horizontal overflow that otherwise shows a useless,
   non-functional scrollbar on the panels. */
.stac2cube-root {
    overflow-x: hidden !important;
    background: #f4f3ec;
    margin: 18px auto 20px auto !important;
    padding: 18px 14px 22px 14px;
    box-sizing: border-box;
    border: 1px solid #ddd9c8;
    border-radius: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* Cards */
.stac2cube-card {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 12px;
    background: #fbfaf4;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    box-sizing: border-box;
    overflow-x: hidden;
    min-width: 0;
}

/* A subtle sub-panel that groups related fields (e.g. the date inputs) and sets
   them apart from neighbouring fields inside the same card. */
.stac2cube-group {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 10px 12px;
    background: #eef0e3;
    box-sizing: border-box;
    min-width: 0;
    margin-bottom: 10px;
    /* Clip any sliver of horizontal overflow inside a group (e.g. the full-width
       header button) so a group never shows its own ugly horizontal scrollbar.
       'clip' (not 'hidden') does not force a vertical scrollbar, so the draw map
       and other tall content still display fully. */
    overflow-x: clip;
}

/* Keep accordion panels and nested widget boxes from overflowing the card */
.stac2cube-card .p-Accordion-child,
.stac2cube-card .p-Accordion-child > .p-Widget,
.stac2cube-card .widget-vbox,
.stac2cube-card .widget-hbox {
    min-width: 0 !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}

/* Header button for the custom collapsible groups (Polygon / Time Period / Bands
   / Indices / Stats): looks like a clickable section title, not a button. */
button.stac2cube-group-toggle,
button.stac2cube-group-toggle:hover,
button.stac2cube-group-toggle:focus,
button.stac2cube-group-toggle:active {
    background: transparent !important;
    color: #374151 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 2px 0 !important;
    border: 0 !important;
    box-shadow: none !important;
    width: 100% !important;
}

/* Let widget containers shrink so long option text / long values can't push a
   stray horizontal scrollbar onto the field (the symptom inside accordions). */
.stac2cube-card .widget-box,
.stac2cube-card .widget-checkbox,
.stac2cube-card .widget-dropdown,
.stac2cube-card .widget-text,
.stac2cube-card .widget-textarea,
.stac2cube-card .widget-select,
.stac2cube-card .widget-select-multiple,
.stac2cube-card .widget-inttext {
    min-width: 0 !important;
    max-width: 100% !important;
}

/* Constrain the inner controls themselves */
.stac2cube-card .widget-text input,
.stac2cube-card .widget-dropdown select,
.stac2cube-card .widget-select-multiple select {
    min-width: 0 !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    overflow-x: hidden !important;
}

/* "Details for Xarray-Nerds" toggle in the Result panel: bold label. */
.stac2cube-nerd-toggle label,
.stac2cube-nerd-toggle span {
    font-weight: 700 !important;
}

/* Outputs: wrap long lines (file paths, tracebacks, xarray repr) so Status /
   Result text is fully readable instead of clipped with no scrollbar. */
.stac2cube-root .widget-output { overflow-x: hidden !important; }
.stac2cube-root .widget-output pre {
    white-space: pre-wrap !important;
    overflow-wrap: anywhere !important;
}

/* One consistent help (?) button across all three GUIs: same size and colour,
   overriding any per-widget inline colour set in Python. */
button.stac2cube-help-btn,
button.stac2cube-help-btn:hover,
button.stac2cube-help-btn:active,
button.stac2cube-help-btn:focus {
    border-radius: 9999px !important;
    width: 20px !important;
    min-width: 20px !important;
    height: 20px !important;
    min-height: 20px !important;
    padding: 0 !important;
    background: #2563eb !important;
    color: #ffffff !important;
    border: 0 !important;
    outline: none !important;
    box-shadow: none !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    line-height: 20px !important;
    font-weight: 700 !important;
    font-size: 12px !important;
}

/* ipywidgets 8 (Lumino) Accordion headers. The default renderer paints these
   header bars (Basic Parameters / Advanced / Result / Visualization ...) a cool
   flat gray that clashes with the earthy/sage palette. We retint them to a sage
   header so the native widgets match the themed cards. The clickable header
   element is '.lm-AccordionPanel-title' (titleClassName in the bundle); the
   container is '.jupyter-widget-Accordion'. Lumino adds '.lm-mod-expanded' to
   the title when its panel is open. If a future ipywidgets/renderer renames
   these classes, the rules simply no-op (no harm). */
.stac2cube-root .lm-AccordionPanel-title {
    background: #e4e8d6 !important;
    color: #3f4a36 !important;
    border: 1px solid #d2d8bf !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stac2cube-root .lm-AccordionPanel-title:hover {
    background: #dde2cd !important;
}
.stac2cube-root .lm-AccordionPanel-title.lm-mod-expanded {
    background: #d9dfc6 !important;
    border-bottom-left-radius: 0 !important;
    border-bottom-right-radius: 0 !important;
}
</style>
"""


def gui_css_widget():
    """One shared stylesheet for all three GUIs.

    Unifies the card look, the help (?) button (size + colour), the horizontal-
    overflow fix (removes the stray scrollbar) and output text wrapping (so
    Status / Result text stays readable on small screens).
    """
    return widgets.HTML(GUI_CSS)


def _label_text(label_text):
    """Normalise a field label: strip, and append ':' if not already present."""
    lbl = (label_text or "").strip()
    if lbl and not lbl.endswith(":"):
        lbl = f"{lbl}:"
    return lbl


def _clear_description(widget):
    """Drop a widget's built-in (display-only) description so only the stacked
    label shows. Safe on widgets without a description/style."""
    try:
        widget.description = ""
    except Exception:
        pass
    try:
        widget.style.description_width = "0px"
    except Exception:
        pass


def help_button(tooltip="Show help"):
    """The circular '?' button. Appearance comes from the shared stylesheet
    (.stac2cube-help-btn), so all three GUIs look identical."""
    btn = widgets.Button(
        description="?",
        tooltip=tooltip,
        layout=widgets.Layout(width="20px", min_width="20px", height="20px", padding="0px"),
    )
    btn.add_class("stac2cube-help-btn")
    return btn


def field_group(title, children, subtitle=None, help_html=None, collapsible=False, open=True):
    """Wrap related fields in a subtly bordered sub-panel with a small heading,
    an optional subtitle and an optional '?' help toggle, to set them apart from
    neighbouring fields inside a card.

    When ``collapsible`` is True the panel is rendered as an Accordion titled
    ``title`` (open by default unless ``open`` is False), so the section can be
    folded individually while its parent panel stays expanded."""
    title_html = widgets.HTML(
        f"<div style='font-weight:600; font-size:13px; color:#374151; margin:0;'>{title}</div>"
    )

    btn = None
    help_box = None
    if help_html:
        btn = help_button()
        help_box = widgets.HTML(
            value=help_html,
            layout=widgets.Layout(
                display="none",
                border="1px solid #dbeafe",
                padding="8px",
                border_radius="8px",
                margin="2px 0 2px 0",
                width="100%",
            ),
        )

        def _toggle(_):
            help_box.layout.display = "" if help_box.layout.display == "none" else "none"

        btn.on_click(_toggle)

    subtitle_html = (
        widgets.HTML(
            f"<div style='font-size:12px; color:#6b7280; margin:0 0 2px 0;'>{subtitle}</div>"
        )
        if subtitle else None
    )

    if collapsible:
        # A custom collapse (NOT ipywidgets Accordion, which renders a few px wider
        # and causes the stray horizontal scrollbar). The whole thing is just the
        # normal .stac2cube-group box — which already lays out cleanly — with a
        # clickable header button on top that shows/hides the body.
        toggle = widgets.Button(
            description=f"{'▾' if open else '▸'}  {title}",
            layout=widgets.Layout(width="100%"),
        )
        toggle.add_class("stac2cube-group-toggle")

        body_children = []
        if subtitle_html is not None or btn is not None:
            row = []
            if subtitle_html is not None:
                row.append(subtitle_html)
            if btn is not None:
                row.append(btn)
            body_children.append(
                widgets.HBox(row, layout=widgets.Layout(align_items="center", gap="6px"))
            )
        if help_box is not None:
            body_children.append(help_box)
        body_children += list(children)
        body = widgets.VBox(
            body_children,
            layout=widgets.Layout(
                width="100%", gap="6px", display=("" if open else "none")
            ),
        )

        def _toggle_body(_):
            showing = body.layout.display != "none"
            body.layout.display = "none" if showing else ""
            toggle.description = f"{'▸' if showing else '▾'}  {title}"

        toggle.on_click(_toggle_body)

        box = widgets.VBox(
            [toggle, body], layout=widgets.Layout(width="100%", gap="6px")
        )
        box.add_class("stac2cube-group")
        return box

    if help_box is not None:
        header = [widgets.HBox([title_html, btn], layout=widgets.Layout(align_items="center", gap="6px"))]
    else:
        header = [title_html]

    if subtitle_html is not None:
        header.append(subtitle_html)
    if help_box is not None:
        header.append(help_box)

    box = widgets.VBox(
        header + list(children),
        layout=widgets.Layout(width="100%", gap="6px"),
    )
    box.add_class("stac2cube-group")
    return box


def boxed_control(widget):
    """Clear a widget's built-in description and wrap it in a width-100% box (the
    wrapper prevents the stray horizontal scrollbar). Use inside field_group when
    the group title already serves as the field's label."""
    _clear_description(widget)
    return widgets.Box([widget], layout=widgets.Layout(width="100%"))


def stacked_field(widget, label_text=None):
    """Label on its own row, widget directly below (no help icon)."""
    if label_text is None and hasattr(widget, "description"):
        label_text = widget.description or ""
    lbl = _label_text(label_text)
    _clear_description(widget)
    label = widgets.HTML(
        f"<div style='font-weight:500; line-height:1.2; margin:0; padding:0;'>{lbl}</div>"
    )
    # Wrap the control in a width-100% Box. Without this, controls with long
    # option text / values (Dropdown, SelectMultiple, Text) can force a stray
    # horizontal scrollbar on the field inside an accordion.
    widget_box = widgets.Box([widget], layout=widgets.Layout(width="100%"))
    return widgets.VBox([label, widget_box], layout=widgets.Layout(width="100%", gap="4px"))


def stacked_field_with_help(widget, label_text, help_html):
    """Label + circular '?' toggle on one row, a collapsible help box, then the
    widget. ``help_html`` is the resolved help string (the caller looks it up in
    its own per-GUI help dict, since the help text differs by tool)."""
    lbl = _label_text(label_text)
    _clear_description(widget)

    label = widgets.HTML(
        f"<div style='font-weight:500; line-height:1.2; margin:0; padding:0;'>{lbl}</div>"
    )
    btn = help_button()
    help_box = widgets.HTML(
        value=help_html or "",
        layout=widgets.Layout(
            display="none",
            border="1px solid #dbeafe",
            padding="8px",
            border_radius="8px",
            margin="2px 0 2px 0",
            width="100%",
        ),
    )

    def _toggle(_):
        help_box.layout.display = "" if help_box.layout.display == "none" else "none"

    btn.on_click(_toggle)
    header_row = widgets.HBox(
        [label, btn], layout=widgets.Layout(align_items="center", gap="6px")
    )
    # See stacked_field(): the width-100% Box prevents a stray horizontal
    # scrollbar on controls with long content.
    widget_box = widgets.Box([widget], layout=widgets.Layout(width="100%"))
    return widgets.VBox(
        [header_row, help_box, widget_box],
        layout=widgets.Layout(width="100%", gap="4px"),
    )
