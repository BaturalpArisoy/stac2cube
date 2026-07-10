from .main import get_stac_layers
from .get_update import get_stac_parameters
from .get_update import find_missing_times
from s2cloudless import S2PixelCloudDetector
import numpy as np
import xarray as xr
import sys
from tqdm.auto import tqdm
from .export_cfg import export_stac, open_cube
from .clip import compute_cloud_percentage
import rioxarray as rio
import cv2
import os
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def _parse_slurm_mem(value):
    """SLURM memory string -> bytes. Bare numbers are MB (SLURM convention)."""
    value = value.strip().upper()
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    try:
        if value and value[-1] in multipliers:
            return float(value[:-1]) * multipliers[value[-1]]
        return float(value) * 1024**2
    except ValueError:
        return None


def _cgroup_available_bytes():
    """Available bytes under the current Linux cgroup memory limit, or None.

    On SLURM clusters and in containers the enforced limit is the job's
    cgroup, NOT the node's RAM - system-level probes over-report there and
    the job would be OOM-killed. Checks cgroup v2 (walking up from the
    process's own cgroup until a concrete limit is found), then cgroup v1.
    """
    try:
        with open("/proc/self/cgroup", "rt") as fh:
            entries = fh.read().splitlines()
    except OSError:
        return None

    def _read_int(path):
        try:
            with open(path, "rt") as fh:
                raw = fh.read().strip()
            return None if raw == "max" else int(raw)
        except (OSError, ValueError):
            return None

    for entry in entries:
        parts = entry.split(":", 2)
        if len(parts) != 3:
            continue
        if parts[0] == "0" and parts[1] == "":  # cgroup v2
            root, rel = "/sys/fs/cgroup", parts[2]
        elif "memory" in parts[1].split(","):  # cgroup v1
            root, rel = "/sys/fs/cgroup/memory", parts[2]
        else:
            continue
        names = (
            ("memory.max", "memory.current")
            if root == "/sys/fs/cgroup"
            else ("memory.limit_in_bytes", "memory.usage_in_bytes")
        )
        path = os.path.join(root, rel.lstrip("/")).rstrip("/")
        while len(path) >= len(root):
            limit = _read_int(os.path.join(path, names[0]))
            # v1 uses a huge sentinel value for "unlimited"
            if limit is not None and limit < 1 << 60:
                usage = _read_int(os.path.join(path, names[1])) or 0
                return max(limit - usage, 0)
            if path == root:
                break
            path = os.path.dirname(path)
    return None


def _memory_budget_bytes():
    """Byte budget for in-flight scene batches, adapted to the machine/job.

    Priority:
      1) STAC2CUBE_BATCH_MEMORY_MB env var - explicit override, meant for
         HPC job scripts (used directly as the budget).
      2) 20% of the memory actually available to THIS job, probed in order:
         Linux cgroup limit (SLURM/containers), SLURM_MEM_PER_* variables,
         psutil available system RAM.
      3) Conservative fixed fallback: 800 MB.
    """
    override = os.environ.get("STAC2CUBE_BATCH_MEMORY_MB")
    if override:
        parsed = _parse_slurm_mem(override)
        if parsed:
            return parsed
        warnings.warn(
            f"Could not parse STAC2CUBE_BATCH_MEMORY_MB={override!r}; "
            "falling back to automatic detection."
        )

    available = _cgroup_available_bytes()
    if available is None:
        mem = os.environ.get("SLURM_MEM_PER_NODE")
        if mem:
            available = _parse_slurm_mem(mem)
        else:
            per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
            cpus = os.environ.get("SLURM_CPUS_ON_NODE") or os.environ.get(
                "SLURM_CPUS_PER_TASK"
            )
            if per_cpu and cpus:
                mem_bytes = _parse_slurm_mem(per_cpu)
                if mem_bytes:
                    try:
                        available = mem_bytes * float(cpus)
                    except ValueError:
                        available = None
    if available is None:
        try:
            import psutil

            available = psutil.virtual_memory().available
        except Exception:
            available = None

    if available is None:
        return 800_000_000
    # 20%: the probability stack accumulated across ALL scenes (plus the rest
    # of the user's session) needs the remaining headroom.
    return max(0.20 * available, 256_000_000)


def _plan_scene_batches(values_per_scene):
    """Pick (batch_size, prefetch) for the scene loop from the memory budget.

    One batch of k scenes holds k * values_per_scene float32 values (4 B) and
    is in flight up to ~3x at peak: the batch being predicted, the prefetched
    next batch, and LightGBM's per-pixel temporaries (~0.6x a batch). Batch
    size is capped at 8 scenes - measured download parallelism plateaus there.
    If not even one scene fits the budget, process one scene at a time with
    prefetch disabled: peak memory then equals the plain per-scene loop, so
    large AOIs degrade in speed, never in memory.
    """
    budget_values = int(_memory_budget_bytes() / 3 / 4)
    batch_size = int(max(1, min(8, budget_values // max(values_per_scene, 1))))
    prefetch = values_per_scene <= budget_values
    return batch_size, prefetch


def _iter_scene_batches(stac, batch_size, num_workers=32, prefetch=True):
    """Yield (times, ndarray) batches of scenes, prefetching the next batch.

    The next batch is downloaded in a background thread (dask threaded
    scheduler, many workers - the per-scene cost of remote JP2/COG reads is
    network round-trip latency, so wide concurrency is what cuts it) while the
    caller runs s2cloudless on the current batch. Peak memory is bounded by
    ~2 batches (current + prefetched); with prefetch=False only ONE batch is
    ever held, matching the memory profile of a plain per-scene loop.

    Yields arrays of shape (k, y, x, band) ready for
    S2PixelCloudDetector.get_cloud_probability_maps.
    """
    from concurrent.futures import ThreadPoolExecutor

    n = stac.sizes["time"]
    slices = [slice(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]

    def fetch(sl):
        sub = stac.isel(time=sl).transpose("time", "y", "x", "band")
        data = sub.data
        if hasattr(data, "compute"):
            arr = data.compute(scheduler="threads", num_workers=num_workers)
        else:
            arr = np.asarray(data)
        return sub.time.values, arr

    if not prefetch:
        for sl in slices:
            yield fetch(sl)
        return

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fetch, slices[0])
        for next_slice in slices[1:]:
            times, arr = future.result()
            future = pool.submit(fetch, next_slice)
            yield times, arr
        times, arr = future.result()
        yield times, arr


def get_cloud_layers(
    polygon=None,
    daterange=None,
    output_clouds=None,
    output_masked=None,
    output=None,
    threshold=None,
    clip_raster=None,
    masking=None,
    update=None,
    input_cube=None,
    slurm_timer=None,
    compress=False,
):
    if output_clouds is None and output is not None:
        output_clouds = output

    # If we are called from an existing cube, we must use its exact time list.
    # This is what makes seasonal cubes correct: probability is computed on the
    # cube's exact dates, not on a continuous min..max range.
    #
    # A source cube can be supplied two ways:
    #   - masking=<cube>    -> derive dates AND mask the cube at the end
    #   - input_cube=<cube> -> derive dates only (probability / masks, no masking)
    reference_times = None

    source_cube = masking or input_cube
    if source_cube:
        stac_parameters = get_stac_parameters(source_cube)
        polygon = stac_parameters["polygon"]
        daterange = stac_parameters["daterange"]

        # Use the exact timestamps from the initial cube
        with open_cube(source_cube) as ds:
            if "Spectral_Temporal_Stack" in ds:
                reference_times = ds["Spectral_Temporal_Stack"].time.values
            else:
                reference_times = ds["time"].values

    if update:
        stac_parameters = get_stac_parameters(update)
        polygon = stac_parameters["polygon"]
        if daterange is None:
            daterange = stac_parameters.get("daterange")

    if not daterange:
        raise ValueError("Error: Please select a daterange.")
    if not polygon:
        raise ValueError("Error: Please select a polygon or bbox list with geographic coordinates.")

    # --- STAC Retrieval ---
    max_cc = 100
    mission = "sentinel_2_l1c"
    bands = [
        "coastal", "blue", "red", "rededge1", "nir", "nir08", "nir09",
        "cirrus", "swir16", "swir22",
    ]

    def _filter_to_reference_times(stac_da: xr.DataArray, ref_times) -> xr.DataArray:
        st = np.asarray(stac_da.time.values).astype("datetime64[ns]")
        rt = np.asarray(ref_times).astype("datetime64[ns]")

        # 1) exact timestamp match
        if np.all(np.isin(rt, st)):
            order = [int(np.where(st == t)[0][0]) for t in rt]
            out = stac_da.isel(time=order).assign_coords(time=ref_times)
            return out

        # 2) fallback: day-level matching (handles duplicates in the reference list)
        st_d = st.astype("datetime64[D]")
        rt_d = rt.astype("datetime64[D]")

        from collections import defaultdict
        pos = defaultdict(list)
        for i, d in enumerate(st_d):
            pos[d].append(i)

        used = defaultdict(int)
        order = []
        missing = []
        for d in rt_d:
            k = used[d]
            if d not in pos or k >= len(pos[d]):
                missing.append(d)
            else:
                order.append(pos[d][k])
                used[d] += 1

        if missing:
            ex = ", ".join(np.datetime_as_string(m, unit="D") for m in missing[:5])
            raise ValueError(
                "Cloud STAC retrieval is missing some reference dates. "
                f"Missing (first up to 5): {ex}"
            )

        out = stac_da.isel(time=order)
        # Keep the reference time coordinate for alignment
        if out.sizes["time"] == len(ref_times):
            out = out.assign_coords(time=ref_times)
        return out

    stac = get_stac_layers(
        mission=mission,
        polygon=polygon,
        daterange=daterange,
        bands=bands,
        max_cc=max_cc,
        clip_raster=clip_raster,
        # s2cloudless is documented and calibrated for bilinearly-resampled
        # reflectance. Pin bilinear explicitly so this does NOT inherit the
        # global get_stac_layers default (now "nearest") - nearest-resampled
        # bands would feed the detector off-spec inputs.
        resampling_method="bilinear",
        q=True,
    )

    if reference_times is not None:
        stac = _filter_to_reference_times(stac, reference_times)

    crs = stac.crs
    transform = stac.transform
    bbox = stac.bbox

    if update:
        with open_cube(update) as ds:
            stac_existing = ds["Cloud_Stack"].load()
        stac, missing_times = find_missing_times(stac_existing, stac)
        if not missing_times:
            raise ValueError("The probability map is up to date. Nothing to update!")

    # --- Cloud Probability Calculation ---
    # Set the parameters for the cloud detector.
    # Default threshold (0.7) for computing cloud probability.
    average_over = 4
    dilation_size = 2
    default_threshold = 0.7
    cloud_detector = S2PixelCloudDetector(
        threshold=default_threshold,
        average_over=average_over,
        dilation_size=dilation_size,
        all_bands=False,
    )

    cloud_prob_results = []
    times = []  # To store the time coordinate for each processed slice
    total = len(stac.time)

    if slurm_timer:
        import time

        slurm_timer = slurm_timer * 3600
        start_time = time.time()

    # Scenes are processed in batches: while s2cloudless predicts the current
    # batch, the next one is already downloading in the background (the
    # download is round-trip latency bound, ~constant per scene regardless of
    # AOI size, and dominates the runtime for small/medium AOIs). Batch size
    # and prefetch adapt to the memory actually available to this job (cgroup
    # / SLURM limit / free RAM; override: STAC2CUBE_BATCH_MEMORY_MB). When not
    # even one scene fits the budget, this degrades to one scene at a time
    # without prefetch - the plain per-scene loop's memory profile.
    values_per_scene = stac.sizes["y"] * stac.sizes["x"] * stac.sizes["band"]
    batch_size, prefetch = _plan_scene_batches(values_per_scene)
    scene_mb = values_per_scene * 4 / 1e6
    print(
        f"{total} scenes ({scene_mb:.0f} MB each), processed {batch_size} at a "
        f"time, prefetch {'on' if prefetch else 'off'} "
        f"(memory budget {_memory_budget_bytes() / 1e9:.1f} GB)",
        flush=True,
    )

    progress = tqdm(
        total=total,
        desc="Computing",
        unit="scene",
        file=sys.stdout,
        dynamic_ncols=False,
    )
    stop = False
    for batch_times, batch_np in _iter_scene_batches(stac, batch_size, prefetch=prefetch):
        progress.set_description(
            f"Computing {np.datetime_as_string(batch_times[-1], unit='D')}"
        )

        # Guard against silent read failures: with fail_on_error=False a
        # scene whose assets could not be read arrives filled with nodata
        # (all zeros) and would yield a plausible-looking ~0% cloud map.
        for i, bt in enumerate(batch_times):
            if not batch_np[i].any():
                warnings.warn(
                    f"Scene {np.datetime_as_string(bt, unit='D')} is entirely "
                    "empty (all asset reads failed or returned nodata); its "
                    "cloud probability map is NOT valid."
                )

        # Cloud probability maps for the whole batch, shape: (k, y, x).
        cp_batch = cloud_detector.get_cloud_probability_maps(batch_np)
        cloud_prob_results.append(cp_batch)
        times.extend(batch_times)
        progress.update(len(batch_times))
        del batch_np

        if slurm_timer:
            # Check if the elapsed time has reached or exceeded the threshold
            elapsed = time.time() - start_time
            if elapsed >= slurm_timer:
                progress.close()
                print(
                    "Time threshold reached! Exiting loop and exporting the collected cloud maps..."
                )
                stop = True
                break
    if not stop:
        progress.close()

    # Assemble the cloud probability DataArray.
    cp_stack = np.concatenate(cloud_prob_results, axis=0)  # shape: (time, y, x)
    cp_da = xr.DataArray(
        cp_stack,
        dims=["time", "y", "x"],
        coords={"time": times, "y": stac.y, "x": stac.x},
    )
    cp_da = cp_da.expand_dims(dim={"band": ["cloud_prob"]})

    cp_da.name = "Cloud_Stack"

    def update_prob_maps(stac_existing, cloud_only_stack):
        # keep band dimension with correct label
        stac_existing = stac_existing.sel(band=["cloud_prob"])
        cloud_only_stack = cloud_only_stack.sel(band=["cloud_prob"])

        out = xr.concat([stac_existing, cloud_only_stack], dim="time")
        out = out.sortby("time")
        return out

    # --- Determine Output Based on 'threshold' Parameter ---
    # If no threshold(s) are provided, return only the probability layer.

    # Always build probability layer first (uint8 0-100)
    cloud_prob_uint8 = (cp_da.sel(band="cloud_prob") * 100).astype(np.uint8)

    # Create a proper 4D stack: (time, band, y, x) with band label "cloud_prob"
    cloud_only_stack = cloud_prob_uint8.expand_dims(band=["cloud_prob"]).transpose("time", "band", "y", "x")
    cloud_only_stack.name = "Cloud_Stack"

    # If update: merge new probability dates into existing stack
    if update:
        cloud_only_stack = update_prob_maps(stac_existing, cloud_only_stack)

    # If threshold(s) are provided (NON-update only): compute masks and concat
    if threshold is not None:
        mask_da = mask_from_probability(
            cloud_only_stack.sel(band="cloud_prob"),
            threshold=threshold,
            average_over=average_over,
            dilation_size=dilation_size,
        )
        cloud_only_stack = xr.concat([cloud_only_stack, mask_da], dim="band").transpose("time", "band", "y", "x")

    # ---- attrs: set ALWAYS (update or not) ----
    cloud_only_stack.attrs["bbox"] = bbox
    cloud_only_stack.attrs["crs"] = crs
    cloud_only_stack.attrs["transform"] = transform

    # ---- export: do NOT hide behind `if not update` ----
    if output_clouds is not None:
        export_stac(cloud_only_stack, output_clouds, crs, transform, compress=compress)
    

    # ---- Masking (kept as before; typically not combined with update) ----
    if masking:
        if threshold is None:
            raise ValueError("Error: 'threshold' must be set when 'masking' is used.")
        if isinstance(threshold, list):
            raise ValueError("Error: 'masking' supports only a single threshold (not a list).")

        thr = int(threshold)
        mask_layer = f"cloud_mask_{thr}"

        if output_masked is None:
            dirname, filename = os.path.split(masking)
            name, ext = os.path.splitext(filename)
            output_masked = os.path.join(dirname, f"{name}_masked_{thr}{ext}")

        return mask_stac_clouds(
            masking, cloud_only_stack, mask_layer, output_masked, compress=compress
        )

    # Always return in-memory stack
    return cloud_only_stack


def mask_stac_clouds(stac, cloud, mask_layer, output=None, compress=False):
    # Track datasets we open here so we can close them before returning.
    # Leaving these handles open makes each repeated call stack another open
    # handle onto the same netCDF/HDF5 files, which on Windows can crash the
    # kernel (native segfault, not a catchable Python exception).
    _opened = []
    try:
        if isinstance(stac, (str, os.PathLike)):
            _ds = open_cube(stac)
            _opened.append(_ds)
            stac = _ds.Spectral_Temporal_Stack

        if isinstance(cloud, (str, os.PathLike)):
            _ds = open_cube(cloud)
            _opened.append(_ds)
            cloud = _ds.Cloud_Stack

        if isinstance(stac, xr.Dataset):
            stac = stac.Spectral_Temporal_Stack

        if isinstance(cloud, xr.Dataset):
            cloud = cloud.Cloud_Stack

        cloud_mask = cloud.sel(band=mask_layer)
        masked_stac = stac.where(cloud_mask == 0)

        # Cloud percentage per time slice, measured against the observable AOI
        # footprint: pixels missing in every scene (incl. anything outside a
        # non-rectangular clip) are excluded from numerator and denominator.
        pct = compute_cloud_percentage(masked_stac)
        if pct is not None:
            masked_stac = masked_stac.assign_coords(
                cloud_percentage=("time", np.asarray(pct.data))
            )

        if output is not None:
            # export_stac reads through the still-open source handles here,
            # then we close them below. Nothing is force-loaded into memory:
            # to_netcdf streams straight from the source file to the output.
            export_stac(masked_stac, output, compress=compress)
            for _ds in _opened:
                try:
                    _ds.close()
                except Exception:
                    pass
            _opened.clear()
            return output  # return path (old code returned None anyway)
        else:
            # In-memory return path (NOT used by the GUI, which always passes an
            # output). The returned array is lazy and still reads from the source
            # handles, so we must NOT close them here; clear _opened so the
            # finally below leaves them open (matches the original behavior).
            _opened.clear()
            return masked_stac
    finally:
        # Safety net: if export raised, don't leak the handles we opened.
        for _ds in _opened:
            try:
                _ds.close()
            except Exception:
                pass


def mask_from_probability(
    cloud_probability, threshold=0.7, average_over=4, dilation_size=2
):

    if not isinstance(threshold, list):
        thresholds = [threshold]
    else:
        thresholds = threshold

    # Normalize probabilities to [0, 1] if necessary.
    if cloud_probability.max() > 1:
        prob_da = cloud_probability / 100.0
    else:
        prob_da = cloud_probability

    band_dataarrays = []

    # One numpy stack (time, y, x), computed once and shared by all thresholds.
    # get_mask_from_prob operates on the full stack (cv2 runs per slice
    # internally), so no per-time Python loop / concat is needed.
    prob_np = prob_da.transpose("time", "y", "x").to_numpy()

    for t_val in thresholds:
        # Scale the threshold from 0-100 to 0-1.
        scaled_threshold = t_val / 100.0
        cloud_detector = S2PixelCloudDetector(
            threshold=scaled_threshold,
            average_over=average_over,
            dilation_size=dilation_size,
        )
        cm = cloud_detector.get_mask_from_prob(prob_np, threshold=scaled_threshold)
        threshold_mask_da = xr.DataArray(
            cm,
            dims=["time", "y", "x"],
            coords={"time": prob_da.time, "y": prob_da.y, "x": prob_da.x},
        )
        band_label = f"cloud_mask_{int(t_val)}"
        threshold_mask_da = threshold_mask_da.expand_dims(dim={"band": [band_label]})
        band_dataarrays.append(threshold_mask_da)

    final_mask_da = xr.concat(band_dataarrays, dim="band")
    final_mask_da = final_mask_da.transpose("time", "band", "y", "x")
    final_mask_da.name = "Cloud_Stack"

    return final_mask_da


def cloud_filter(inp, max_cloud):
    """
    Keep only time steps where cloud_percentage <= max_cloud.

    - if inp is a netcdf path (str): open it, take ds["Spectral_Temporal_Stack"], filter
    - if inp is an xr.Dataset: take ds["Spectral_Temporal_Stack"], filter
    - if inp is an xr.DataArray: filter directly
    """
    if isinstance(inp, str):
        da = open_cube(inp)["Spectral_Temporal_Stack"]
    elif isinstance(inp, xr.Dataset):
        da = inp["Spectral_Temporal_Stack"]
    else:  # assume xr.DataArray
        da = inp

    # da = da.where(da["cloud_percentage"] <= int(max_cloud), drop=True)
    # da = da.rio.write_crs(da.crs)

    return da.where(da["cloud_percentage"] <= int(max_cloud), drop=True)
