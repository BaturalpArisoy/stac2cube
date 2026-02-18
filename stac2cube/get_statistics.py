import xarray as xr

_VALID_OPS = {"mean", "median", "min", "max", "std"}
_VALID_PERIODS = {"timeseries", "monthly", "annual"}


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


def _parse_requests(stats) -> list[tuple[str, str]]:
    tokens: list[str] = []

    for raw in _as_list(stats):
        if raw is None:
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
    return sorted(req, key=lambda x: (x[0], order[x[1]]))


def calculate_statistics(stac: xr.DataArray, stats):
    """Create custom temporal composites as extra variables in an xr.Dataset.

    Supported tokens (case-insensitive):
      - {op}_timeseries   -> one variable over full selected time range: {op}_timeseries
      - {op}_monthly      -> one variable per month present: {op}_MM_YYYY
      - {op}_annual       -> one variable per year present:  {op}_YYYY
      - {op}_all          -> expands to timeseries + monthly + annual
      - {op}_full         -> alias for {op}_all

    ops: mean, median, min, max, std

    Legacy:
      - "{op}" is treated as "{op}_timeseries"

    Returns:
      xr.Dataset with variable "Spectral_Temporal_Stack" plus composite variables.
    """
    if "time" not in stac.dims:
        raise ValueError(
            "Temporal composites require a 'time' dimension. "
            "Disable aggregator or provide an un-aggregated time series."
        )

    requests = _parse_requests(stats)
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

    return xr.Dataset({"Spectral_Temporal_Stack": stac, **computed})
