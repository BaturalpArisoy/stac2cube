"""Cloud shadow detection for Sentinel-2 L2A cubes.

Port of the cloud/shadow masking approach from the Google Earth Engine
s2cloudless tutorial
(https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless),
adapted to stac2cube data cubes. Per scene:

  1. Binary cloud mask, from one of:
       * the s2cloudless pipeline (probability thresholded at ``threshold``
         with the package's standard postprocessing - IDENTICAL to the
         ``cloud_mask_<thr>`` bands get_cloud_layers exports), or
       * the SCL cloud classes (8, 9, 10) - much cheaper, no L1C download, or
       * a precomputed Cloud_Stack the user already has on disk / in memory
         (an existing ``cloud_mask_<threshold>`` band is used byte-for-byte).
     The cloud mask is NEVER reshaped here. The GEE tutorial erodes + dilates
     it (60 m / 320 m) before projecting, which inflates clouds into
     unrealistic blobs; we deliberately deviate and keep the pipeline's
     realistic cloud outlines.
  2. Potential shadow pixels: NIR reflectance < ``nir_dark_threshold`` (0.18)
     that are not water (SCL 6) and not nodata (SCL 0 / NIR <= 0 / NaN).
  3. Cloud projection: each cloud is projected along the anti-solar direction
     (the per-scene MEAN solar azimuth read from the STAC item metadata, never
     guessed) up to ``proj_distance`` km in ``proj_step`` m increments - the
     numpy equivalent of GEE's directionalDistanceTransform at 100 m scale.
  4. raw shadow = dark pixels AND cloud projection; the raw (speckly) shadow
     mask is then smoothed with s2cloudless' OWN postprocessing (disk-mean
     convolution + majority threshold, then small disk dilation - the same
     operation that gives the package's cloud masks their realistic outlines),
     and finally cloud pixels are excluded.

Returned as a Cloud_Stack-convention DataArray with bands
``cloud_mask`` (the cleaned mask used for projection), ``shadow_mask`` and
``cloudshadow_mask`` (union - what you mask a cube with), on exactly the
input cube's grid/CRS, so it plugs into mask_stac_clouds unchanged.

Limitations (stated, not hidden):
  * Shadows cast by clouds OUTSIDE the cube extent cannot be projected -
    near the AOI edge on the sun-facing side, shadows can be missed.
  * The projection assumes a flat scene; no terrain correction.
  * Dark non-water surfaces inside the projection cone (e.g. shaded urban
    canyons, dark asphalt) can be false positives; the dark-pixel threshold
    is a heuristic from the GEE tutorial, not a physical retrieval.
"""

import os
import warnings

import numpy as np
import xarray as xr
# cv2 is imported inside the two helpers that use it - see the note on lazy
# imports; it is not needed to import the module.

from .get_data import get_stac, get_solar_geometry
from .get_update import get_stac_parameters
from .export_cfg import export_stac, open_cube, normalize_stack_name
from .cloud_masking import get_cloud_layers, mask_stac_clouds, mask_from_probability

# SCL classes treated as cloud by the package's SCL masking (see
# stac_processing._mission_cfg): 8 = cloud medium prob., 9 = cloud high
# prob., 10 = thin cirrus. Class 3 (ESA's own cloud-shadow class) is
# deliberately NOT used - detecting shadows independently is the point.
_SCL_CLOUD_CLASSES = (8, 9, 10)
_SCL_WATER = 6
_SCL_NODATA = 0


def _floor_days(times):
    return np.asarray(times).astype("datetime64[D]").astype("datetime64[ns]")


def _resolve_cube(input_cube):
    """Accept a path / Dataset / DataArray, return (DataArray, open handles)."""
    opened = []
    if isinstance(input_cube, (str, os.PathLike)):
        ds = open_cube(input_cube)
        opened.append(ds)
        input_cube = ds
    if isinstance(input_cube, xr.Dataset):
        # A Dataset handed in directly bypassed open_cube's migration, so a
        # legacy time-series name is normalised here.
        input_cube = normalize_stack_name(input_cube)["Time_Series"]
    if not isinstance(input_cube, xr.DataArray):
        raise TypeError(
            f"input_cube must be a cube path, Dataset or DataArray, got {type(input_cube)}"
        )
    return input_cube, opened


def _as_affine(transform):
    """attrs['transform'] round-trips NetCDF/Zarr as a 9-element array/list;
    export_stac's ``transform or ...`` needs a scalar-truthy Affine."""
    from affine import Affine

    if transform is None or isinstance(transform, Affine):
        return transform
    vals = np.asarray(transform, dtype=float).ravel()
    return Affine(*vals[:6].tolist())


def _grid_resolution(da):
    ry = float(abs(da.y.values[1] - da.y.values[0]))
    rx = float(abs(da.x.values[1] - da.x.values[0]))
    if abs(ry - rx) > 1e-6:
        raise ValueError(f"Non-square pixels are not supported (dy={ry}, dx={rx}).")
    return ry


def _align_to_cube_grid(da, cube, res, what):
    """Label-align an auxiliary layer to the cube grid, pixel-exact.

    Nearest-label selection with half-pixel tolerance, then the cube's own
    coordinates are stamped on so downstream xarray ops align by identity.
    Raises (never silently reprojects) if the grids don't overlap.
    """
    try:
        out = da.sel(y=cube.y, x=cube.x, method="nearest", tolerance=res / 2)
    except KeyError as e:
        raise ValueError(
            f"The {what} grid does not match the cube grid (offset larger than "
            "half a pixel). The layers were probably built with different "
            "bbox/resolution/CRS parameters."
        ) from e
    return out.assign_coords(y=cube.y, x=cube.x)


def _match_days(da, cube_days, what):
    """Select cube days out of an auxiliary time series (day-level matching)."""
    da = da.assign_coords(time=_floor_days(da.time.values))
    have = set(np.asarray(da.time.values).tolist())
    missing = [d for d in cube_days.tolist() if d not in have]
    if missing:
        ex = ", ".join(
            np.datetime_as_string(np.datetime64(m, "ns"), unit="D") for m in missing[:5]
        )
        raise ValueError(
            f"The {what} is missing cube dates (first up to 5): {ex}. "
            "Cannot detect shadows on dates without it."
        )
    return da.sel(time=cube_days)


def _cv2_disk(radius):
    """Disk structuring element, built exactly like s2cloudless.utils.cv2_disk
    (a filled cv2.circle), so the smoothing below matches the cloud pipeline's
    kernels bit-for-bit."""
    import cv2

    return cv2.circle(
        np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8),
        (radius, radius), radius, color=1, thickness=-1,
    )


def _smooth_binary_mask(mask, average_over, dilation_size):
    """s2cloudless' get_mask_from_prob postprocessing, applied to a binary mask.

    The cloud pipeline convolves the probability map with a normalized disk
    (cv2.filter2D, BORDER_REFLECT), thresholds, then dilates with a small disk
    - that is what gives the package's cloud masks their realistic outlines.
    A binary mask is its own 0/1 probability map, so the same convolution with
    a majority threshold (0.5) followed by the same dilation reproduces that
    smoothing for shadows: speckle is removed, outlines follow the detected
    shape instead of ballooning it.
    """
    import cv2

    out = mask.astype(np.float32)
    if average_over:
        disk = _cv2_disk(average_over).astype(np.float32)
        out = cv2.filter2D(
            out, -1, disk / disk.sum(), borderType=cv2.BORDER_REFLECT
        ) > 0.5
    else:
        out = out > 0.5
    if dilation_size:
        out = cv2.dilate(out.astype(np.uint8), _cv2_disk(dilation_size))
    return out.astype(bool)


def _shift_mask(mask, dr, dc):
    """Shift a 2-D boolean mask by (dr rows, dc cols), zero-filling edges."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    if abs(dr) >= h or abs(dc) >= w:
        return out
    src_r = slice(max(0, -dr), h - max(0, dr))
    dst_r = slice(max(0, dr), h - max(0, -dr))
    src_c = slice(max(0, -dc), w - max(0, dc))
    dst_c = slice(max(0, dc), w - max(0, -dc))
    out[dst_r, dst_c] = mask[src_r, src_c]
    return out


def _project_cloud_shadow_zone(cloud, sun_azimuth_deg, res, proj_distance_m, proj_step_m):
    """Union of the cloud mask shifted step-by-step along the anti-solar
    direction - the numpy equivalent of GEE's
    ``directionalDistanceTransform(90 - azimuth, dist).mask()``.

    Geometry (north-up grid, row+ = south, col+ = east): the sun sits at
    compass azimuth ``az``; shadows fall the opposite way, so a cloud pixel
    displaced by distance d lands at
        dE = -d * sin(az),  dN = -d * cos(az)
    which in array indices is  drow = +d*cos(az)/res, dcol = -d*sin(az)/res.
    """
    az = np.deg2rad(sun_azimuth_deg)
    zone = np.zeros_like(cloud)
    nsteps = max(1, int(round(proj_distance_m / proj_step_m)))
    for k in range(1, nsteps + 1):
        d = k * proj_step_m
        dr = int(round(d * np.cos(az) / res))
        dc = int(round(-d * np.sin(az) / res))
        zone |= _shift_mask(cloud, dr, dc)
    return zone


def solar_azimuths_for_days(polygon, cube_days, source):
    """Per-day mean solar azimuth (degrees) for the given cube days.

    Metadata-only STAC search over [min(day), max(day)]; raises when any cube
    day has no azimuth metadata - the projection direction is never guessed.
    """
    cube_days = np.asarray(cube_days)
    daterange = [
        np.datetime_as_string(cube_days.min(), unit="D"),
        np.datetime_as_string(cube_days.max(), unit="D"),
    ]
    geometry = get_solar_geometry(
        "sentinel_2_l2a", polygon, daterange, source=source
    )
    day_keys = [np.datetime_as_string(d, unit="D") for d in cube_days]
    missing = [d for d in day_keys if d not in geometry]
    if missing:
        raise ValueError(
            "No solar azimuth metadata found on the STAC items for: "
            f"{', '.join(missing[:5])}. Shadows cannot be projected for "
            "these dates - the projection direction would be a guess."
        )
    return np.array([geometry[d]["sun_azimuth"] for d in day_keys])


def detect_shadow_stack(
    cloud_np,
    nir_np,
    scl_np,
    sun_azimuths,
    res,
    nir_dark_threshold=0.18,
    proj_distance=1.0,
    proj_step=100.0,
    average_over=4,
    dilation_size=2,
):
    """Core per-scene shadow detection on aligned (time, y, x) numpy stacks.

    ``cloud_np`` boolean cloud mask, ``nir_np`` float reflectance, ``scl_np``
    integer class codes, ``sun_azimuths`` degrees per scene. Returns a uint8
    (time, y, x) shadow stack (1 = shadow; cloud pixels excluded). This is
    the shared GEE port used by both get_shadow_layers and
    get_stac_layers(shadow_masking=True).
    """
    shadow_out = np.zeros(cloud_np.shape, dtype=np.uint8)
    for t in range(cloud_np.shape[0]):
        c = cloud_np[t].astype(bool)
        dark = (
            (nir_np[t] < nir_dark_threshold)
            & (nir_np[t] > 0)
            & np.isfinite(nir_np[t])
            & (scl_np[t] != _SCL_WATER)
            & (scl_np[t] != _SCL_NODATA)
        )
        zone = _project_cloud_shadow_zone(
            c, sun_azimuths[t], res, proj_distance * 1000.0, proj_step
        )
        # Raw dark∩zone shadows are speckly; smooth them the same way the
        # cloud pipeline smooths its masks, then keep them off the clouds.
        shadow = _smooth_binary_mask(zone & dark, average_over, dilation_size)
        shadow &= ~c
        shadow_out[t] = shadow
    return shadow_out


def _cloud_mask_from_stack(cloud, cube, cube_days, res, threshold):
    """Binary cloud mask (time, y, x) out of a Cloud_Stack (path or array).

    The mask must be EXACTLY what the s2cloudless pipeline produces - it is
    never reshaped here:
      1. an existing ``cloud_mask_<threshold>`` band is used byte-for-byte;
      2. otherwise the ``cloud_prob`` band is turned into a mask with
         mask_from_probability - the same function (same postprocessing
         parameters) get_cloud_layers uses to export its mask bands;
      3. otherwise any premade ``cloud_mask_*`` band is used as-is.
    """
    opened = []
    try:
        if isinstance(cloud, (str, os.PathLike)):
            ds = open_cube(cloud)
            opened.append(ds)
            cloud = ds
        if isinstance(cloud, xr.Dataset):
            cloud = cloud["Cloud_Stack"]

        bands = [str(b) for b in np.atleast_1d(cloud.band.values)]
        mask_bands = [b for b in bands if b.startswith("cloud_mask")]
        exact = f"cloud_mask_{int(threshold)}"

        if exact in bands:
            binary = cloud.sel(band=exact)
            binary = _match_days(binary, cube_days, "cloud stack")
            binary = _align_to_cube_grid(binary, cube, res, "cloud stack")
            return (binary.transpose("time", "y", "x").values > 0)

        if "cloud_prob" in bands:
            prob = cloud.sel(band="cloud_prob")
            prob = _match_days(prob, cube_days, "cloud stack")
            prob = _align_to_cube_grid(prob, cube, res, "cloud stack")
            mask_da = mask_from_probability(prob, threshold=threshold)
            binary = mask_da.sel(band=exact)
            return (binary.transpose("time", "y", "x").values > 0)

        if mask_bands:
            binary = cloud.sel(band=mask_bands[0])
            binary = _match_days(binary, cube_days, "cloud stack")
            binary = _align_to_cube_grid(binary, cube, res, "cloud stack")
            return (binary.transpose("time", "y", "x").values > 0)

        raise ValueError(
            f"Cloud stack has no 'cloud_mask_*' or 'cloud_prob' band (bands: {bands})."
        )
    finally:
        for ds in opened:
            try:
                ds.close()
            except Exception:
                pass


def add_shadow_masks_to_cloud_stack(
    input_cube,
    cloud,
    mask_band,
    nir_dark_threshold=0.18,
    proj_distance=1.0,
    proj_step=100.0,
    average_over=4,
    dilation_size=2,
    output=None,
    compress=False,
):
    """Detect cloud shadows from ONE existing binary mask band of a
    Cloud_Stack and append them to that stack as two new bands:
    ``shadow_mask_<sfx>`` and ``cloudshadow_mask_<sfx>``, where ``<sfx>`` is
    the mask band's suffix ('cloud_mask_70' -> '70', 'cloud_mask_scl' ->
    'scl'). Existing bands with the same names are replaced; everything else
    (cloud_prob, other masks) is kept.

    ``input_cube`` supplies nir/scl/solar geometry (the data cube the masks
    belong to); ``cloud`` is the Cloud_Stack path or array. The stack is
    loaded fully into memory and its handle closed BEFORE exporting, so the
    default behavior - overwriting the same file when ``cloud`` is a path
    and ``output`` is None - is safe.

    Returns the combined Cloud_Stack (also exported when an output path
    applies).
    """
    cloud_path = None
    if isinstance(cloud, (str, os.PathLike)):
        cloud_path = str(cloud)
        # Loaded in full on the next line - see the open_cube note on
        # chunks="eager" (a dask-backed read peaks at ~1.9x, this at ~1.2x).
        with open_cube(cloud_path, chunks="eager") as ds:
            stack_full = ds["Cloud_Stack"].load()
    else:
        stack_full = cloud["Cloud_Stack"] if isinstance(cloud, xr.Dataset) else cloud
        stack_full = stack_full.load()

    bands = [str(b) for b in np.atleast_1d(stack_full.band.values)]
    if mask_band not in bands:
        raise ValueError(
            f"Band '{mask_band}' not found in the cloud stack (bands: {bands})."
        )
    if mask_band == "cloud_prob":
        raise ValueError(
            "Select a binary cloud mask band (not 'cloud_prob') to project "
            "shadows from."
        )

    shadow_stack = get_shadow_layers(
        input_cube,
        cloud=stack_full.sel(band=[mask_band]),
        nir_dark_threshold=nir_dark_threshold,
        proj_distance=proj_distance,
        proj_step=proj_step,
        average_over=average_over,
        dilation_size=dilation_size,
    )

    # The shadow bands exist only for the data cube's dates; appending them
    # to a stack with different dates would produce NaN-filled float bands.
    stack_days = _floor_days(stack_full.time.values)
    if not np.array_equal(np.sort(stack_days), np.sort(shadow_stack.time.values)):
        raise ValueError(
            "The cloud stack's dates differ from the data cube's dates - "
            "shadow bands can only be appended to a cloud stack built for "
            "the same cube."
        )
    stack_full = stack_full.assign_coords(time=stack_days)

    sfx = (
        mask_band[len("cloud_mask_"):]
        if mask_band.startswith("cloud_mask_")
        else mask_band
    )
    new_names = [f"shadow_mask_{sfx}", f"cloudshadow_mask_{sfx}"]
    new_bands = shadow_stack.sel(band=["shadow_mask", "cloudshadow_mask"]).assign_coords(
        band=new_names
    )

    keep = [b for b in bands if b not in new_names]
    combined = xr.concat(
        [stack_full.sel(band=keep), new_bands.sel(time=stack_full.time)],
        dim="band",
        coords="minimal",
    ).transpose("time", "band", "y", "x")
    combined.name = "Cloud_Stack"
    combined = combined.assign_coords(
        band=np.array([str(b) for b in combined.band.values], dtype=object)
    )
    combined.attrs.update(stack_full.attrs)
    combined.attrs["shadow_params"] = shadow_stack.attrs.get("shadow_params", "")

    if output is None and cloud_path is not None:
        output = cloud_path
    if output is not None:
        # Stale encodings from the loaded file (chunk shapes, band dtype) can
        # break re-export after the band concat - clear them all.
        combined.encoding = {}
        for coord in combined.coords:
            combined[coord].encoding = {}
        crs = combined.attrs.get("crs") or shadow_stack.attrs.get("crs")
        transform = combined.attrs.get("transform")
        if transform is None:
            transform = shadow_stack.attrs.get("transform")
        export_stac(
            combined, output, crs, _as_affine(transform),
            var_name="Cloud_Stack", compress=compress,
        )
    return combined


def get_shadow_layers(
    input_cube,
    cloud=None,
    cloud_source="s2cloudless",
    threshold=45,
    nir_dark_threshold=0.18,
    proj_distance=1.0,
    proj_step=100.0,
    average_over=4,
    dilation_size=2,
    masking=False,
    output_shadows=None,
    output_masked=None,
    compress=False,
    vrt=False,
):
    """Detect cloud shadows in a Sentinel-2 L2A cube (GEE s2cloudless approach).

    Parameters
    ----------
    input_cube : str | xr.Dataset | xr.DataArray
        A stac2cube Sentinel-2 L2A cube (must contain the ``nir`` band).
        If it also contains the ``scl`` band (built with 'scl' among the
        requested bands), that layer is reused and no SCL download happens.
    cloud : optional
        Precomputed Cloud_Stack from get_cloud_layers (path or array). An
        existing ``cloud_mask_<threshold>`` band is used byte-for-byte;
        otherwise the mask is derived from its ``cloud_prob`` band with the
        pipeline's own mask_from_probability. When given, ``cloud_source`` is
        ignored and no L1C download happens.
    cloud_source : "s2cloudless" | "scl"
        Where the cloud mask comes from when ``cloud`` is not given.
        "s2cloudless" computes the probability and masks it exactly like
        get_cloud_layers would; "scl" uses classes 8/9/10 and needs no L1C
        download. The cloud mask is never reshaped by the shadow step.
    threshold : int
        Cloud probability threshold, 0-100 (GEE tutorial default: 45).
        Single value only (no lists here).
    nir_dark_threshold : float
        NIR reflectance below which a non-water pixel is a shadow candidate
        (GEE tutorial: 0.18).
    proj_distance : float
        Maximum cloud-to-shadow projection distance in km (GEE tutorial
        default: 1). Larger values catch shadows of higher clouds but flag
        more dark surfaces as shadow - on an urban test scene 3 km roughly
        halved the fraction of genuinely darkened pixels in the mask.
    proj_step : float
        Projection step in meters (GEE works at 100 m scale).
    average_over, dilation_size : int
        Postprocessing of the SHADOW mask, in pixels: disk-mean convolution
        with majority threshold, then disk dilation - s2cloudless'
        get_mask_from_prob smoothing, with the same defaults (4 / 2) the
        cloud pipeline uses. Set to 0/None to disable.
    masking : bool
        Also produce the input cube with cloud AND shadow pixels set to NaN.
    output_shadows, output_masked : str, optional
        Export paths (.nc / .zarr) for the mask stack / the masked cube.
    compress : bool
        Passed through to export_stac.

    Returns
    -------
    xr.DataArray (Cloud_Stack) with bands ``cloud_mask``, ``shadow_mask``,
    ``cloudshadow_mask`` (uint8, 1 = flagged), plus a ``sun_azimuth`` time
    coordinate. With ``masking=True`` returns ``(mask_stack, masked_cube)``.
    """
    if isinstance(threshold, (list, tuple)):
        raise ValueError("Shadow detection supports a single threshold, not a list.")

    cube, opened = _resolve_cube(input_cube)
    try:
        if "band" not in cube.dims or "nir" not in [str(b) for b in cube.band.values]:
            raise ValueError(
                "Shadow detection needs the 'nir' band in the cube "
                f"(bands: {[str(b) for b in np.atleast_1d(cube.band.values)] if 'band' in cube.dims else 'none'})."
            )
        mission = cube.attrs.get("mission")
        if mission != "sentinel_2_l2a":
            raise ValueError(
                f"Shadow detection is implemented for sentinel_2_l2a cubes, got '{mission}'."
            )

        params = get_stac_parameters(cube)
        bbox = params["polygon"]
        daterange = params["daterange"]
        source = params.get("stac_api", "element84")
        res = _grid_resolution(cube)
        crs = cube.attrs.get("crs")
        transform = _as_affine(cube.attrs.get("transform"))
        cube_days = _floor_days(cube.time.values)

        nir = cube.sel(band="nir").transpose("time", "y", "x").astype("float32")
        nir_np = np.asarray(
            nir.data.compute() if hasattr(nir.data, "compute") else nir.data
        )
        finite = nir_np[np.isfinite(nir_np)]
        # Robust check (p99, not max: a handful of specular/bright-cloud pixels
        # legitimately exceed 1). DN-scale cubes sit around 10^3-10^4.
        if finite.size and float(np.percentile(finite, 99)) > 1.5:
            warnings.warn(
                "NIR 99th percentile exceeds 1.5 - the cube does not look like "
                f"scaled reflectance, but nir_dark_threshold={nir_dark_threshold} "
                "assumes reflectance in [0, 1]. The shadow mask will be unreliable."
            )

        # --- SCL (water/nodata screening; cloud source in "scl" mode) --------
        # A cube built with 'scl' among its bands carries the layer already
        # (pinned to nearest resampling and exempt from reflectance scaling by
        # the builder) - reuse it and skip the download entirely.
        band_labels = [str(b) for b in np.atleast_1d(cube.band.values)]
        scl_np = None
        used_cube_scl = False
        if "scl" in band_labels:
            raw = cube.sel(band="scl").transpose("time", "y", "x")
            raw = np.asarray(
                raw.data.compute() if hasattr(raw.data, "compute") else raw.data
            )
            finite_scl = raw[np.isfinite(raw)]
            # Integer class codes prove the band is usable; fractional values
            # mean it was interpolated (legacy cube) - refetch instead of
            # trusting it.
            if finite_scl.size and float(
                np.abs(finite_scl - np.round(finite_scl)).max()
            ) < 1e-3:
                print(
                    "Using the cube's own 'scl' band (no SCL download needed).",
                    flush=True,
                )
                scl_np = np.nan_to_num(raw, nan=_SCL_NODATA).astype(np.int16)
                used_cube_scl = True
            else:
                warnings.warn(
                    "The cube's 'scl' band contains fractional class codes "
                    "(interpolated resampling) - ignoring it and downloading "
                    "SCL with nearest resampling instead."
                )
        if scl_np is None:
            print("Fetching SCL layer (water / nodata screening)...", flush=True)
            scl_ds, _, _ = get_stac(
                "sentinel_2_l2a", bbox, res, daterange, ["scl"], None, None,
                source=source, resampling="nearest",
                # Load straight onto the cube's grid. _align_to_cube_grid below
                # still runs, but it now has nothing to fix: a grid re-derived
                # from the lon/lat bbox does not reproduce an AOI-pinned one.
                geobox=params.get("geobox"),
            )
            scl = _match_days(scl_ds["scl"], cube_days, "SCL layer")
            scl = _align_to_cube_grid(scl, cube, res, "SCL layer")
            scl_np = scl.transpose("time", "y", "x").values
            scl_np = np.nan_to_num(scl_np, nan=_SCL_NODATA).astype(np.int16)

        # --- per-scene mean solar azimuth from STAC item metadata -------------
        azimuths = solar_azimuths_for_days(bbox, cube_days, source)

        # --- binary cloud mask -------------------------------------------------
        if cloud is not None:
            cloud_np = _cloud_mask_from_stack(cloud, cube, cube_days, res, threshold)
        elif cloud_source == "s2cloudless":
            prob_stack = get_cloud_layers(
                polygon=bbox,
                daterange=daterange,
                input_cube=input_cube if isinstance(input_cube, (str, os.PathLike)) else None,
                # An IN-MEMORY cube is passed as input_cube=None (the reference
                # timestamps come from _match_days below instead), so
                # get_cloud_layers has no cube to recover the grid from and
                # would derive one from `bbox` - which does not reproduce an
                # AOI-pinned grid. The probability stack then misses
                # _align_to_cube_grid's half-pixel tolerance and the whole
                # shadow run dies. Update mode reaches this with every s2cloudless
                # + shadow cube (main._mask_new_scenes_s2cloudless).
                geobox=params.get("geobox"),
            )
            prob = prob_stack.sel(band="cloud_prob")
            prob = _match_days(prob, cube_days, "s2cloudless probability stack")
            prob = _align_to_cube_grid(prob, cube, res, "s2cloudless probability stack")
            # Same mask the cloud pipeline would export as cloud_mask_<thr>:
            # identical thresholding + postprocessing, realistic outlines.
            mask_da = mask_from_probability(prob, threshold=threshold)
            cloud_np = (
                mask_da.sel(band=f"cloud_mask_{int(threshold)}")
                .transpose("time", "y", "x").values > 0
            )
        elif cloud_source == "scl":
            cloud_np = np.isin(scl_np, _SCL_CLOUD_CLASSES)
            # A cloud-REMOVED cube has NaN holes in its scl band too, so the
            # cloud classes are gone and the projection would see no clouds.
            # Detect the contradiction and say it, instead of silently
            # returning an empty shadow mask.
            if used_cube_scl and "cloud_percentage" in cube.coords:
                pct = np.asarray(cube["cloud_percentage"].values)
                for t in range(len(cube_days)):
                    if pct[t] >= 5 and not cloud_np[t].any():
                        warnings.warn(
                            f"{np.datetime_as_string(cube_days[t], unit='D')}: the "
                            f"cube reports {int(pct[t])}% cloud but its scl band "
                            "contains no cloud classes - the cube was probably "
                            "exported with clouds removed (NaN holes erase SCL "
                            "too). Use cloud_source='s2cloudless' or pass cloud= "
                            "for a valid shadow projection on this cube."
                        )
        else:
            raise ValueError(
                f"cloud_source must be 's2cloudless' or 'scl', got '{cloud_source}'."
            )

        # --- per-scene shadow detection -----------------------------------------
        cloud_out = cloud_np.astype(np.uint8)
        shadow_out = detect_shadow_stack(
            cloud_np, nir_np, scl_np, azimuths, res,
            nir_dark_threshold=nir_dark_threshold,
            proj_distance=proj_distance,
            proj_step=proj_step,
            average_over=average_over,
            dilation_size=dilation_size,
        )

        # --- assemble Cloud_Stack-convention output ------------------------------
        coords = {"time": cube_days, "y": cube.y.values, "x": cube.x.values}
        stack = xr.DataArray(
            np.stack(
                [cloud_out, shadow_out, (cloud_out | shadow_out).astype(np.uint8)],
                axis=1,
            ),
            dims=["time", "band", "y", "x"],
            coords={**coords, "band": ["cloud_mask", "shadow_mask", "cloudshadow_mask"]},
            name="Cloud_Stack",
        )
        stack = stack.assign_coords(sun_azimuth=("time", azimuths))
        stack.attrs["bbox"] = bbox
        stack.attrs["crs"] = crs
        stack.attrs["transform"] = transform
        stack.attrs["shadow_params"] = (
            f"cloud_source={'precomputed' if cloud is not None else cloud_source}, "
            f"threshold={threshold}, nir_dark_threshold={nir_dark_threshold}, "
            f"proj_distance_km={proj_distance}, proj_step_m={proj_step}, "
            f"shadow_average_over={average_over}, shadow_dilation_size={dilation_size}"
        )

        if output_shadows is not None:
            export_stac(stack, output_shadows, crs, transform,
                        compress=compress, vrt=vrt)

        if masking or output_masked is not None:
            # Record the masking recipe so update mode can reproduce it. SCL
            # clouds + shadow reuse the scl_shadow_masked status; s2cloudless (or
            # a precomputed s2cloudless stack) + shadow use cloud_mask_<thr> plus
            # the shadow params, mirroring how get_stac_layers stores them.
            if cloud_source == "scl" and cloud is None:
                _status = "scl_shadow_masked"
            else:
                _status = f"cloud_mask_{int(threshold)}"
            _shadow_attrs = {
                "nir_dark_threshold": nir_dark_threshold,
                "shadow_proj_distance": proj_distance,
            }
            masked = mask_stac_clouds(
                cube, stack, "cloudshadow_mask", output_masked, compress=compress,
                vrt=vrt, cloud_status=_status, shadow_attrs=_shadow_attrs,
            )
            if output_masked is None:
                # The in-memory masked cube is lazy and still reads from the
                # handle we opened for input_cube - leave it open (same
                # contract as mask_stac_clouds' in-memory return path).
                opened.clear()
            return stack, masked

        return stack
    finally:
        for ds in opened:
            try:
                ds.close()
            except Exception:
                pass
