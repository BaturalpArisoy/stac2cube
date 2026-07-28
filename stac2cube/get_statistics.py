import re

import numpy as np
import xarray as xr

from .stac_processing import (
    expand_season_windows,
    is_iso_date,
    is_mmdd,
    season_crosses_year,
)

_VALID_OPS = {"mean", "median", "min", "max", "std"}
_VALID_PERIODS = {"timeseries", "monthly", "annual"}

# A custom composite name becomes a NetCDF / Zarr variable name: letters, digits
# and underscores, never starting with a digit.
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# The only keys a custom composite dict may carry. Anything else is a typo and
# is rejected rather than silently ignored.
_CUSTOM_KEYS = {"op", "season", "window", "years", "name"}


def _as_list(stats):
    if stats is None:
        return []
    if isinstance(stats, (list, tuple)):
        return list(stats)
    return [stats]


def _reduce_timeseries(stac: xr.DataArray, op: str) -> xr.DataArray:
    func = getattr(stac, op)
    try:
        return func(dim="time", skipna=True)
    except TypeError:
        # Some xarray versions/ops don't accept skipna
        return func(dim="time")


def _groupby_reduce(stac: xr.DataArray, group: xr.DataArray, op: str) -> xr.DataArray:
    gb = stac.groupby(group)
    func = getattr(gb, op)
    try:
        return func(dim="time", skipna=True)
    except TypeError:
        return func(dim="time")


def _parse_custom(spec: dict) -> dict:
    """Validate one custom composite dict and return it normalised.

    Accepted forms:
      {"op": "mean", "season": ["04-01", "06-21"], "name": "spring_mean"}
      {"op": "mean", "season": [...], "years": [2024, 2025], "name": "..."}
      {"op": "mean", "window": ["2024-04-01", "2024-06-21"], "name": "spring24"}

    Every problem is reported here, before anything is computed, so a bad entry
    in a SLURM config fails immediately instead of writing a mislabelled layer.
    """
    unknown = sorted(set(spec) - _CUSTOM_KEYS)
    if unknown:
        raise ValueError(
            f"Custom composite has unsupported key(s) {unknown}. "
            f"Allowed keys: {sorted(_CUSTOM_KEYS)}."
        )

    op = str(spec.get("op", "") or "").strip().lower()
    if op not in _VALID_OPS:
        raise ValueError(
            f"Custom composite 'op' must be one of {sorted(_VALID_OPS)}, got "
            f"{spec.get('op')!r}."
        )

    has_season = spec.get("season") is not None
    has_window = spec.get("window") is not None
    if has_season == has_window:
        raise ValueError(
            "Custom composite needs exactly one of 'season' (repeats every "
            "year, e.g. ['04-01', '06-21']) or 'window' (a single period, "
            "e.g. ['2024-04-01', '2024-06-21'])."
        )

    pair = spec["season"] if has_season else spec["window"]
    key = "season" if has_season else "window"
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError(
            f"Custom composite '{key}' must be a two-element list [start, end]."
        )
    start, end = (str(p).strip() for p in pair)

    if has_season:
        if not (is_mmdd(start) and is_mmdd(end)):
            raise ValueError(
                f"Custom composite 'season' must be ['MM-DD', 'MM-DD'] "
                f"(e.g. ['04-01', '06-21']), got [{start!r}, {end!r}]."
            )
    else:
        if not (is_iso_date(start) and is_iso_date(end)):
            raise ValueError(
                f"Custom composite 'window' must be ['YYYY-MM-DD', "
                f"'YYYY-MM-DD'], got [{start!r}, {end!r}]."
            )
        if start > end:  # ISO dates sort lexicographically
            raise ValueError(
                f"Custom composite 'window' starts after it ends: "
                f"[{start}, {end}]."
            )

    years = spec.get("years")
    years_list = None
    if years is not None:
        if not has_season:
            raise ValueError(
                "'years' only applies to a seasonal composite ('season'); a "
                "'window' composite already names its year."
            )
        raw_years = years if isinstance(years, (list, tuple, set)) else [years]
        try:
            years_list = sorted({int(y) for y in raw_years})
        except (TypeError, ValueError):
            raise ValueError(
                f"Custom composite 'years' must be a list of years, got {years!r}."
            )
        if not years_list:
            raise ValueError("Custom composite 'years' is empty.")

    name = str(spec.get("name", "") or "").strip()
    if not name:
        raise ValueError(
            "Custom composite needs a 'name' - it becomes the variable name in "
            "the cube (e.g. 'spring_mean')."
        )
    if not _NAME_RE.match(name):
        raise ValueError(
            f"Custom composite name {name!r} cannot be used as a variable name. "
            "Use letters, digits and underscores only, not starting with a "
            "digit (e.g. 'spring_mean')."
        )

    return {
        "op": op,
        "mode": "season" if has_season else "window",
        "start": start,
        "end": end,
        "years": years_list,
        "name": name,
    }


def _parse_requests(stats):
    """Split the stats list into preset tokens and custom composite dicts.

    Returns ``(token_requests, custom_specs)`` where token_requests is the
    sorted list of (op, period) pairs and custom_specs preserves the order the
    user listed them in.
    """
    tokens: list[str] = []
    customs: list[dict] = []

    for raw in _as_list(stats):
        if raw is None:
            continue

        if isinstance(raw, dict):
            customs.append(_parse_custom(raw))
            continue

        t = str(raw).strip().lower()
        if not t:
            continue

        # Legacy: "mean" -> "mean_timeseries" (same for other ops)
        if t in _VALID_OPS:
            t = f"{t}_timeseries"

        # Alias: "{op}_full" -> "{op}_all" (pack)
        if t.endswith("_full"):
            op = t[: -len("_full")]
            if op not in _VALID_OPS:
                raise ValueError(
                    f"Statistic '{t}' is not supported. Allowed ops: {sorted(_VALID_OPS)}"
                )
            t = f"{op}_all"

        tokens.append(t)

    req: set[tuple[str, str]] = set()

    for t in tokens:
        # Pack: "{op}_all" expands to timeseries + monthly + annual
        if t.endswith("_all"):
            op = t[: -len("_all")]
            if op not in _VALID_OPS:
                raise ValueError(
                    f"Statistic op '{op}' is not supported. Allowed: {sorted(_VALID_OPS)}"
                )
            req.add((op, "timeseries"))
            req.add((op, "monthly"))
            req.add((op, "annual"))
            continue

        if "_" not in t:
            raise ValueError(
                f"Statistic '{t}' is not supported. Use tokens like "
                f"mean_timeseries, mean_monthly, mean_annual, mean_all (ops: {sorted(_VALID_OPS)})."
            )

        op, period = t.split("_", 1)

        if op not in _VALID_OPS:
            raise ValueError(
                f"Statistic op '{op}' is not supported. Allowed: {sorted(_VALID_OPS)}"
            )
        if period not in _VALID_PERIODS:
            raise ValueError(
                f"Statistic period '{period}' is not supported. Allowed: {sorted(_VALID_PERIODS)}"
            )

        req.add((op, period))

    order = {"timeseries": 0, "monthly": 1, "annual": 2}
    return sorted(req, key=lambda x: (x[0], order[x[1]])), customs


def _reserved_names(stac: xr.DataArray) -> set[str]:
    """Variable names a custom composite must not take: the time series itself
    and the band names, which would make the cube ambiguous to read."""
    reserved = {"Time_Series"}
    if "band" in stac.coords:
        reserved |= {str(b) for b in np.atleast_1d(stac["band"].values)}
    return reserved


def _candidate_years(time_da: xr.DataArray, start_md: str, end_md: str) -> list[int]:
    """Years a season could have scenes in, taken from the cube itself.

    A season that runs over New Year is labelled by its start year, so a scene
    in January of the cube's first year belongs to the season of the year
    BEFORE it - hence the extra candidate at the front.
    """
    years = np.unique(time_da.dt.year.values).astype(int)
    y0, y1 = int(years.min()), int(years.max())
    if season_crosses_year(start_md, end_md):
        y0 -= 1
    return list(range(y0, y1 + 1))


def _window_selection(stac: xr.DataArray, start_iso: str, end_iso: str):
    """The scenes inside [start, end], both ends inclusive of the WHOLE day.

    Uses a boolean mask rather than .sel(time=slice(...)) so it does not depend
    on the time axis being sorted (tile_handling="separate" and update mode can
    both leave it unsorted).
    """
    time_da = stac["time"]
    start = np.datetime64(start_iso)
    end_exclusive = np.datetime64(end_iso) + np.timedelta64(1, "D")
    mask = (time_da >= start) & (time_da < end_exclusive)
    idx = np.flatnonzero(np.asarray(mask.values))
    if idx.size == 0:
        return None
    return stac.isel(time=idx)


def _iso_day(value) -> str:
    """'2024-04-03' from a numpy datetime64 scene timestamp."""
    return str(np.datetime_as_string(np.datetime64(value), unit="D"))


def _custom_composites(stac: xr.DataArray, spec: dict, taken: set[str]) -> dict:
    """Compute one custom composite: one variable for a single window, or one
    per year for a season.

    Years in which the cube holds no scene inside the window are skipped with a
    note - they could only ever produce an all-NaN layer. A composite that
    matches no scene at all is an error, not an empty result.
    """
    if spec["mode"] == "window":
        windows = [(spec["name"], spec["start"], spec["end"])]
    else:
        years = spec["years"] or _candidate_years(
            stac["time"], spec["start"], spec["end"]
        )
        windows = [
            (f"{spec['name']}_{int(y)}", win[0], win[1])
            for y, win in zip(
                years, expand_season_windows(spec["start"], spec["end"], years)
            )
        ]

    out: dict[str, xr.DataArray] = {}
    skipped: list[str] = []
    # Only windows that overlap the cube's own date span are worth reporting as
    # skipped. A season derived from the cube's years can produce windows lying
    # entirely outside it (the year before a New-Year-crossing season, the year
    # after the last one); those are bookkeeping, not a data gap.
    cube_first = np.datetime64(stac["time"].values.min())
    cube_last = np.datetime64(stac["time"].values.max())

    for varname, start_iso, end_iso in windows:
        if varname in taken:
            raise ValueError(
                f"Custom composite would create the variable '{varname}', which "
                "already exists in this cube (a band, the time series, or "
                "another composite). Choose a different name."
            )

        sub = _window_selection(stac, start_iso, end_iso)
        if sub is None:
            overlaps = (
                np.datetime64(start_iso) <= cube_last
                and np.datetime64(end_iso) + np.timedelta64(1, "D") > cube_first
            )
            if overlaps:
                skipped.append(f"{start_iso}..{end_iso}")
            continue

        times = sub["time"].values
        composite = _reduce_timeseries(sub, spec["op"])
        # Written as plain str/int so they survive the Zarr JSON attribute
        # encoding as well as NetCDF. n_dates / first / last describe the scenes
        # that ACTUALLY contributed, which is how a window only partly covered
        # by the cube stays visible instead of silently passing as a full one.
        attrs = {
            "composite_op": spec["op"],
            "composite_mode": spec["mode"],
            "composite_start": start_iso,
            "composite_end": end_iso,
            "composite_n_dates": int(times.size),
            "composite_first_date": _iso_day(times.min()),
            "composite_last_date": _iso_day(times.max()),
        }
        if spec["mode"] == "season":
            attrs["composite_season"] = f"{spec['start']} - {spec['end']}"
        # Merge, never replace: whatever xarray propagated through the reduction
        # (grid_mapping and friends under keep_attrs) has to survive.
        composite.attrs = {**composite.attrs, **attrs}

        out[varname] = composite
        taken.add(varname)

    if not out:
        attempted = ", ".join(f"{s}..{e}" for _, s, e in windows)
        raise ValueError(
            f"Custom composite '{spec['name']}' matches no scene: none of the "
            f"windows {attempted} contain a date of this cube "
            f"({_iso_day(cube_first)} to {_iso_day(cube_last)}). Composites are "
            "reduced from the dates kept after the date, cloud and coverage "
            "filters - they never fetch new scenes."
        )

    if skipped:
        print(
            f"  composite '{spec['name']}': no scenes in "
            f"{', '.join(skipped)} - skipped."
        )

    return out


def calculate_statistics(stac: xr.DataArray, stats):
    """Create temporal composites as extra variables in an xr.Dataset.

    Preset tokens (case-insensitive):
      - {op}_timeseries   -> one variable over full selected time range: {op}_timeseries
      - {op}_monthly      -> one variable per month present: {op}_MM_YYYY
      - {op}_annual       -> one variable per year present:  {op}_YYYY
      - {op}_all          -> expands to timeseries + monthly + annual
      - {op}_full         -> alias for {op}_all

    ops: mean, median, min, max, std

    Custom composites are dicts in the same list, either a season that repeats
    every year the cube covers, or a single window:

      {"op": "mean", "season": ["04-01", "06-21"], "name": "spring_mean"}
        -> spring_mean_2024, spring_mean_2025, ...
      {"op": "mean", "season": [...], "years": [2024], "name": "spring_mean"}
        -> only the listed years
      {"op": "mean", "window": ["2024-04-01", "2024-06-21"], "name": "spring24"}
        -> spring24

    A season whose start is later than its end (e.g. ["12-01", "02-28"]) runs
    over New Year and is labelled by its START year, matching the seasonal
    `daterange` of get_stac_layers. Both ends are inclusive of the whole day.

    Every composite is reduced from the dates the cube actually holds; nothing
    is fetched. Each custom variable carries composite_* attributes recording
    the requested window and the scenes that contributed to it.

    Legacy:
      - "{op}" is treated as "{op}_timeseries"

    Returns:
      xr.Dataset with variable "Time_Series" plus composite variables.
    """
    if "time" not in stac.dims:
        raise ValueError(
            "Temporal composites require a 'time' dimension. "
            "Disable aggregator or provide an un-aggregated time series."
        )

    requests, customs = _parse_requests(stats)
    computed: dict[str, xr.DataArray] = {}

    need_monthly = any(period == "monthly" for _, period in requests)
    need_annual = any(period == "annual" for _, period in requests)

    ym = None
    year = None
    if need_monthly:
        # numeric YYYYMM for stable ordering; final variable names are MM_YYYY
        ym = (stac["time"].dt.year * 100 + stac["time"].dt.month).rename("ym")
    if need_annual:
        year = stac["time"].dt.year.rename("year")

    for op in sorted({op for op, _ in requests}):
        if (op, "timeseries") in requests:
            computed[f"{op}_timeseries"] = _reduce_timeseries(stac, op)

        if (op, "monthly") in requests:
            monthly = _groupby_reduce(stac, ym, op)  # dims: (ym, band, y, x)
            for label in monthly["ym"].values:
                label_int = int(label)
                yy = label_int // 100
                mm = label_int % 100
                varname = f"{op}_{mm:02d}_{yy}"
                # drop scalar coord 'ym' to avoid Dataset merge conflicts
                computed[varname] = monthly.sel(ym=label).reset_coords(drop=True)

        if (op, "annual") in requests:
            annual = _groupby_reduce(stac, year, op)  # dims: (year, band, y, x)
            for yy in annual["year"].values:
                yy_int = int(yy)
                varname = f"{op}_{yy_int}"
                # drop scalar coord 'year' to avoid Dataset merge conflicts
                computed[varname] = annual.sel(year=yy).reset_coords(drop=True)

    if customs:
        if stac.sizes.get("time", 0) == 0:
            raise ValueError(
                "Custom composites need at least one date, but this cube's time "
                "dimension is empty."
            )
        # Names already spoken for: the time series, the band names, and every
        # preset composite computed above.
        taken = _reserved_names(stac) | set(computed)
        for spec in customs:
            computed.update(_custom_composites(stac, spec, taken))

    return xr.Dataset({"Time_Series": stac, **computed})
