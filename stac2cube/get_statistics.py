import xarray as xr


def calculate_statistics(stac, stats):

    computed_stats = {}
    for stat in stats:
        stat_key = f"{stat}_imagery"
        try:
            # Retrieve the statistic function from stac using getattr
            stat_func = getattr(stac, stat)
        except AttributeError:
            raise ValueError(f"Statistic '{stat}' is not supported.")
        computed_stats[stat_key] = stat_func(dim="time")

    dataset = xr.Dataset({"Spectral_Temporal_Stack": stac, **computed_stats})
    return dataset
