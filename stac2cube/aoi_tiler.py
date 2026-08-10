"""Build cube-shaped AOI rectangles: split an area, or grow one from a point.

Two producers, both emitting the same thing - a file of rectangles sized to a
whole number of pixels in the CRS the cube will be built in:

* :func:`tile_aoi` cuts an existing AOI into a regular grid of pieces.
* :func:`aoi_from_point` grows a rectangle of a given pixel size around one or
  more coordinates, the way ``cubo`` does.

Neither touches the build. :func:`stac2cube.get_stac_layers` already makes one
cube per feature of a multi-feature polygon, so the output goes straight back in
as ``polygon=``; the only requirements are that ``crs=`` matches what these
functions drew in and that ``clip_raster=False``, since the features are
rectangles.

tile_aoi: two jobs, one lattice

* **Mode 1, by number** (``n_pieces``). Split the AOI bounding box into
  ``rows x cols`` equal rectangles. This is the "make it fit" mode: it knows
  nothing about chips and only cares that the pieces are the same size and land
  on whole pixels.
* **Mode 2, by chip size** (``chip_size``). Size the grid so the AOI is a whole
  number of ``chip_size`` x ``chip_size`` training chips, then group those chips
  into pieces. Every chip comes out exactly ``chip_size`` px; only the pieces at
  the edge hold fewer chips.

The output is a multi-feature file of rectangles.
:func:`stac2cube.get_stac_layers` already builds one cube per feature of a
multi-feature polygon, so the result can be fed straight back in - nothing in
the build path had to change for this.

Why one cube per PIECE and not one per chip
-------------------------------------------
The build path is network-latency bound: measured 0.34 s of wall time per date
and essentially nothing per pixel (a 13.6x bigger AOI cost 21% more time). Every
cube re-pays the full per-date cost over roughly the same scenes, so 660 cubes
of 128x128 px cost ~100x more wall time than 6 cubes of 2048x2048 px and yield
exactly the same 660 chips. Chips are cut out of the finished cubes afterwards,
locally, with no STAC involved.

That is also why ``chips_per_piece`` defaults to ``None``, i.e. ONE cube: the
export writes as a stream and peak memory was measured flat (375-446 MB) while
the output grew 41x, so a big cube is not itself a memory problem. Splitting
costs roughly Nx serial wall time and buys four things, none of them automatic:
independent SLURM jobs, restartability on long builds, intermediate files small
enough to move, and the option of keeping each piece in the UTM zone its scenes
are native to. Ask for it when you want one of those.

The lattice is ABSOLUTE
-----------------------
Chip edges sit at exact multiples of ``chip_size * resolution`` measured from
the target CRS origin, never from the AOI's own corner. This is what makes chips
usable as machine-learning samples:

* two different AOIs that overlap produce byte-identical chips in the overlap,
  so training pairs drawn from separate runs line up;
* the same AOI at 10 m and at 20 m produces NESTED chips (one 128-px chip at
  20 m is exactly four 128-px chips at 10 m), because the anchor is a multiple
  of the coarsest chip size in metres;
* nudging the input polygon by a few hundred metres does not shift every chip.

An AOI-relative lattice (centring the expansion on the bbox, say) would break
all three, silently, by a fraction of a pixel.

The CRS has to match the build
------------------------------
The pieces are rectangles in ONE projection. :func:`stac2cube.get_stac_layers`
pins the output grid by reprojecting the polygon into the cube's target CRS and
taking its bounds, and that target CRS is chosen from the matched STAC items -
which has been verified to differ between sources for the same area (element84
gave EPSG:32632 where terrabyte and Planetary Computer gave EPSG:32633). Draw
the lattice in one zone and build in another and the rectangles arrive rotated,
their bounds grow, and 128 px becomes 131 px.

So these functions report the CRS they drew in, and that same value must be
passed to the build as ``crs=``. Pass ``crs=`` here too whenever you already
know it.

aoi_from_point: a rectangle around a coordinate
-----------------------------------------------
``edge_size`` is in PIXELS, which is the unit that matters when the cube is the
product; the metre size is derived and reported. The square is then SNAPPED to
the pixel grid rather than centred exactly on the coordinate:

    half = width_px * resolution / 2
    x0   = round((x - half) / resolution) * resolution
    x1   = x0 + width_px * resolution        # exactly width_px pixels

The centre therefore lands within half a pixel of the requested point (at most
5 m at 10 m, reported per point as ``offset_m``). That half pixel buys two
things exact centring cannot: the cube is read without resampling, because its
grid coincides with the scenes', and two cubes from nearby points share one
grid and can be compared pixel to pixel.

Deliberately NOT snapped to a coarser lattice. Snapping to multiples of
``edge_size * resolution`` would move the centre by up to half an edge - 640 m
for a 128 px square at 10 m - which is no longer the place the user picked.

Usage
-----
::

    # one site
    aoi = aoi_from_point(41.0421, 29.0173, edge_size=128, resolution=10,
                         out="site.gpkg")
    get_stac_layers(polygon="site.gpkg", crs=aoi.attrs["crs"],
                    resolution=10, clip_raster=False, ...)

    # many sites -> one cube each, via the normal batch path
    aoi_from_point([41.04, 41.11], [29.01, 29.22], edge_size=(256, 512),
                   resolution=10, out="sites.gpkg")

    # 3 rows x 4 cols of equal pieces out of an existing AOI
    tile_aoi("aoi.gpkg", n_pieces=(3, 4), resolution=10)
"""

import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

from .get_data import crs_attr_string, validate_target_crs
from .vector_refiner import read_polygon_file

__all__ = ["tile_aoi", "aoi_from_point", "read_point_file", "estimate_piece_bytes"]

# Lattice arithmetic runs on index = coordinate / cell_size, so the tolerance is
# in units of CELLS, not metres. At a 1280 m chip this is ~1 micrometre: far
# below any real coordinate error, far above the ~1e-9 m noise a polygon picks
# up round-tripping through EPSG:4326 on its way through the build. Without it,
# an AOI edge sitting exactly on a lattice line lands on 234.9999999999 and
# floor() hands back a spurious extra chip.
_LATTICE_TOL = 1e-9

_FIT_MODES = ("expand", "shrink")


# ------------------------------------------------------------------ lattice
def _floor_idx(value, cell):
    """Lattice index of the cell containing `value`, tolerant of float noise."""
    return int(math.floor(value / cell + _LATTICE_TOL))


def _ceil_idx(value, cell):
    """Lattice index of the first cell edge at or after `value`."""
    return int(math.ceil(value / cell - _LATTICE_TOL))


def _snap_span(lo, hi, cell, fit, axis):
    """Snap [lo, hi] outward ("expand") or inward ("shrink") to the lattice.

    Returns (i0, i1) as lattice indices, with i1 exclusive, so the snapped
    extent is [i0 * cell, i1 * cell] and holds i1 - i0 whole cells.
    """
    if fit == "expand":
        i0, i1 = _floor_idx(lo, cell), _ceil_idx(hi, cell)
    else:
        i0, i1 = _ceil_idx(lo, cell), _floor_idx(hi, cell)

    if i1 <= i0:
        raise ValueError(
            f'fit="shrink" leaves nothing on the {axis} axis: the AOI spans '
            f"{hi - lo:.1f} m there, which does not contain a whole "
            f"{cell:.1f} m cell. Use fit=\"expand\", or a smaller chip_size."
        )
    return i0, i1


def _split_counts(total, parts):
    """Split `total` whole units into `parts` groups differing by at most one.

    The remainder goes to the LEADING groups, so the sizes are non-increasing
    and the split is a deterministic function of (total, parts) alone.
    """
    if parts < 1:
        raise ValueError(f"cannot split into {parts} parts")
    if parts > total:
        raise ValueError(
            f"asked for {parts} pieces but the extent is only {total} units "
            "wide; every piece must hold at least one."
        )
    base, rem = divmod(total, parts)
    return [base + 1] * rem + [base] * (parts - rem)


def _group_counts(total, per_group):
    """Chop `total` units into groups of `per_group`, last group taking the rest."""
    if per_group < 1:
        raise ValueError(f"chips_per_piece must be >= 1, got {per_group}")
    n_full, rem = divmod(total, per_group)
    out = [per_group] * n_full
    if rem:
        out.append(rem)
    return out


def _edges(start_idx, counts):
    """Cumulative lattice indices bounding consecutive groups."""
    edges = [start_idx]
    for c in counts:
        edges.append(edges[-1] + c)
    return edges


# -------------------------------------------------------------- factorising
def _factor_grid(n, width, height):
    """Factor `n` into (rows, cols) giving pieces as close to square as possible.

    Ties break towards more columns, so a square AOI split into 2 comes out
    1x2 rather than 2x1 - the reading order people expect. A prime `n` has only
    1 x n available and will produce slivers; the caller prints what was chosen
    so that is visible rather than silent.
    """
    if n < 1:
        raise ValueError(f"n_pieces must be >= 1, got {n}")
    best = None
    for rows in range(1, n + 1):
        if n % rows:
            continue
        cols = n // rows
        # Aspect ratio of one piece, always >= 1 so it can be minimised directly.
        pw, ph = width / cols, height / rows
        aspect = max(pw, ph) / min(pw, ph)
        key = (round(aspect, 9), -cols)
        if best is None or key < best[0]:
            best = (key, rows, cols)
    return best[1], best[2]


def _as_grid(n_pieces, width, height):
    """Normalise n_pieces (int or (rows, cols)) into (rows, cols, was_explicit)."""
    if isinstance(n_pieces, (tuple, list)):
        if len(n_pieces) != 2:
            raise ValueError(
                f"n_pieces as a pair must be (rows, cols), got {n_pieces!r}"
            )
        rows, cols = (int(v) for v in n_pieces)
        if rows < 1 or cols < 1:
            raise ValueError(f"n_pieces=(rows, cols) must both be >= 1, got {n_pieces!r}")
        return rows, cols, True
    rows, cols = _factor_grid(int(n_pieces), width, height)
    return rows, cols, False


# ------------------------------------------------------------------- input
def _read_aoi(aoi, crs, q):
    """AOI geometry in a metric target CRS, plus the CRS string used.

    A bbox list/tuple is taken as EPSG:4326, matching every other bbox in
    stac2cube.
    """
    if isinstance(aoi, (list, tuple)):
        if len(aoi) != 4:
            raise ValueError(
                f"a bbox AOI must be [minx, miny, maxx, maxy] in EPSG:4326, got {aoi!r}"
            )
        src = gpd.GeoDataFrame(
            geometry=[box(*(float(v) for v in aoi))], crs="EPSG:4326"
        )
    else:
        src = read_polygon_file(aoi)
        if src is None:
            raise ValueError(f"could not read an AOI from {aoi!r}")
        if src.crs is None:
            raise ValueError(
                "the AOI has no CRS, so its extent cannot be measured in metres."
            )
        # A multi-feature file already means something to the rest of stac2cube:
        # one cube per feature. Tiling MERGES those features into a single
        # outline and cuts its bounding box, which quietly replaces the user's
        # batch with an unrelated grid - and covers the gaps between the
        # features, which were never part of the AOI. Loud, because the result
        # still looks perfectly sensible.
        if len(src) > 1 and not q:
            print(
                f"Note: this file holds {len(src)} features. They are merged "
                "into ONE outline and its bounding box is tiled, so the "
                "tiles do not correspond to the features and the gaps between "
                "the features are included. Tile each feature separately if "
                "that is not what you want.",
                flush=True,
            )

    if crs is not None:
        target = validate_target_crs(crs)
    else:
        # geopandas' estimate_utm_crs takes the bbox CENTRE and follows the
        # nominal 6-degree zone rule. That is a geometric preference only: it
        # knows nothing about which MGRS tiles actually image the AOI, and it
        # disagrees with delivery in the Norway/Svalbard exception zones. Hence
        # the warning below - this guess is exactly what crs= is for.
        target = crs_attr_string(src.to_crs("EPSG:4326").estimate_utm_crs())
        if not q:
            print(
                f"No crs= given; drawing the lattice in {target}, guessed from "
                "the AOI's position. The build chooses its CRS from the matched "
                "STAC items instead, and the two can differ - pass this value "
                "to get_stac_layers(crs=...) so the pieces are not reprojected.",
                flush=True,
            )

    return unary_union(src.to_crs(target).geometry.values), target


# ------------------------------------------------------------------ driver
def tile_aoi(
    aoi,
    n_pieces=None,
    chip_size=None,
    resolution=None,
    chips_per_piece=None,
    fit="expand",
    overlap=0,
    drop_empty=True,
    min_fill=0.0,
    crs=None,
    out=None,
    q=False,
):
    """Cut an AOI into a regular grid of bbox pieces, one per cube.

    Exactly one of ``n_pieces`` (mode 1) or ``chip_size`` (mode 2) is required.

    Parameters
    ----------
    aoi:
        Polygon file, GeoDataFrame, or ``[minx, miny, maxx, maxy]`` in
        EPSG:4326. Only the outline is used - pieces are rectangles, never
        clipped to the polygon, so build them with ``clip_raster=False``.
    n_pieces:
        MODE 1. ``12`` factors into the ``rows x cols`` grid whose pieces are
        closest to square; ``(3, 4)`` fixes the grid explicitly. Pieces are
        equal to within one pixel.
    chip_size:
        MODE 2. Training chip side in PIXELS, e.g. 128. Every chip in the
        output is exactly this size.
    resolution:
        Pixel size in the target CRS's metres. Required in both modes: piece
        edges have to land on whole pixels or neighbouring cubes end up on
        different pixel grids and cannot be mosaicked without resampling.
    chips_per_piece:
        MODE 2 only. Cap on the cube size, in chips: ``16`` gives pieces of at
        most 16x16 chips (2048 px at ``chip_size=128``), ``(12, 16)`` caps the
        two axes separately as (rows, cols). ``None``, the default, puts the
        whole AOI in ONE cube. See the module docstring for why splitting is
        opt-in.
    fit:
        What to do when the AOI is not a whole number of chips.
        ``"expand"`` (default) grows the extent outward to the enclosing
        lattice lines, so the whole AOI is covered and a few percent extra
        comes with it. ``"shrink"`` cuts inward to the largest whole-chip
        extent inside the AOI bbox, which downloads nothing extra but drops a
        strip of up to one chip per side, and fails outright on an AOI smaller
        than one chip. Mode 1 uses this at the pixel level, where it is a
        sub-pixel difference either way.
    overlap:
        Pixels of overlap added to every internal edge, 0 by default. Pieces
        stop being disjoint, and their pixel dimensions stop being whole chip
        multiples - the ``chip_off_*`` columns say where the first whole chip
        of the absolute lattice starts inside each piece, so chipping stays
        exact regardless.
    drop_empty:
        Drop pieces that do not touch the AOI polygon. A regular grid over a
        sprawling AOI's bbox is mostly empty - a river corridor filling 6% of
        its own bbox leaves most of the grid over nothing at all - so this is
        on by default. It means the piece count can come back lower than the
        grid implies.
    min_fill:
        Raise the bar from "touches the AOI at all" to "at least this fraction
        of the piece is inside it", 0..1. Only used when ``drop_empty``.
    crs:
        Target CRS to draw the lattice in. Guessed from the AOI when omitted,
        with a warning - see the module docstring, this must match the build.
    out:
        Optional path to write the pieces to (``.gpkg`` recommended).

    Returns
    -------
    GeoDataFrame of rectangles in the target CRS, in row-major order from the
    north-west so the batch's ``_01.._NN`` naming reads like the grid. Columns:

    ``piece``           label, ``p01``-style, matching the batch suffix
    ``row``, ``col``    position in the piece grid, 0-based from the north-west
    ``width_px``, ``height_px``   the cube's expected pixel dimensions
    ``n_chips_x``, ``n_chips_y``  whole chips fully inside the piece (mode 2)
    ``chip_i0``, ``chip_j0``      ABSOLUTE lattice index of the piece's
                                  south-west chip; x rises east, y rises north
    ``chip_off_x``, ``chip_off_y``  pixels from the piece's west/north edge to
                                  the first whole chip boundary (0 unless
                                  ``overlap`` is set)
    ``bbox_km2``, ``aoi_km2``, ``fill_pct``   how much AOI the piece holds

    ``gdf.attrs`` carries the lattice itself: ``crs``, ``resolution``,
    ``chip_size``, ``chip_m``, ``fit``, ``overlap``, ``mode``, ``grid``,
    ``n_chips_total``.
    """
    # --- validate the mode ----------------------------------------------------
    if (n_pieces is None) == (chip_size is None):
        raise ValueError(
            "give exactly one of n_pieces (split into a number of equal pieces) "
            "or chip_size (split into whole training chips)."
        )
    if fit not in _FIT_MODES:
        raise ValueError(f'fit must be one of {_FIT_MODES}, got {fit!r}')
    if resolution is None:
        raise ValueError(
            "resolution is required: piece edges must land on whole pixels, "
            "or neighbouring cubes sit on different pixel grids and can only "
            "be mosaicked by resampling."
        )
    resolution = float(resolution)
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    overlap = int(overlap)
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0 pixels, got {overlap}")
    if chip_size is None and chips_per_piece is not None:
        raise ValueError(
            "chips_per_piece belongs to the chip_size mode; with n_pieces the "
            "piece count is already what you asked for."
        )

    geom, target_crs = _read_aoi(aoi, crs, q)
    minx, miny, maxx, maxy = geom.bounds

    # --- lay out the lattice --------------------------------------------------
    # `cell` is the unit both modes count in: one chip in mode 2, one pixel in
    # mode 1. Everything downstream works in whole cells, so the two modes share
    # all the grid arithmetic and only differ in how pieces are grouped.
    if chip_size is not None:
        chip_size = int(chip_size)
        if chip_size < 1:
            raise ValueError(f"chip_size must be >= 1 pixel, got {chip_size}")
        cell = chip_size * resolution
        mode = "chip"
    else:
        chip_size = None
        cell = resolution
        mode = "number"

    i0, i1 = _snap_span(minx, maxx, cell, fit, "x")
    j0, j1 = _snap_span(miny, maxy, cell, fit, "y")
    n_cells_x, n_cells_y = i1 - i0, j1 - j0

    # --- group cells into pieces ---------------------------------------------
    if mode == "chip":
        if chips_per_piece is None:
            cpp_y, cpp_x = n_cells_y, n_cells_x       # one piece, the whole AOI
        elif isinstance(chips_per_piece, (tuple, list)):
            if len(chips_per_piece) != 2:
                raise ValueError(
                    "chips_per_piece as a pair must be (rows, cols), got "
                    f"{chips_per_piece!r}"
                )
            cpp_y, cpp_x = (int(v) for v in chips_per_piece)
        else:
            cpp_y = cpp_x = int(chips_per_piece)
        cols_cells = _group_counts(n_cells_x, cpp_x)
        rows_cells = _group_counts(n_cells_y, cpp_y)
        grid = (len(rows_cells), len(cols_cells))
    else:
        rows, cols, explicit = _as_grid(
            n_pieces, n_cells_x * resolution, n_cells_y * resolution
        )
        cols_cells = _split_counts(n_cells_x, cols)
        rows_cells = _split_counts(n_cells_y, rows)
        grid = (rows, cols)
        if not q and not explicit:
            print(
                f"n_pieces={n_pieces} -> a {rows} x {cols} grid "
                f"({rows * cols} pieces).",
                flush=True,
            )

    x_edges = _edges(i0, cols_cells)
    y_edges = _edges(j0, rows_cells)

    # --- build the rectangles -------------------------------------------------
    # Row 0 is the NORTH-most, so the y edges are walked in reverse: a raster's
    # first row is its top one, and the batch naming should read the same way.
    pad = overlap * resolution
    full = (i0 * cell, j0 * cell, i1 * cell, j1 * cell)
    records, geoms = [], []
    for r in range(len(rows_cells)):
        jb, jt = y_edges[len(rows_cells) - 1 - r], y_edges[len(rows_cells) - r]
        for c in range(len(cols_cells)):
            il, ir = x_edges[c], x_edges[c + 1]

            # Grow by the overlap, then clamp to the tiled extent so the outer
            # boundary of the mosaic is unchanged and only internal edges gain.
            x0 = max(il * cell - pad, full[0])
            x1 = min(ir * cell + pad, full[2])
            y0 = max(jb * cell - pad, full[1])
            y1 = min(jt * cell + pad, full[3])

            # Offsets of the first whole lattice cell inside the piece. Zero
            # without overlap; with it, the pad is what the chipper must skip.
            off_x = int(round((il * cell - x0) / resolution))
            off_y = int(round((y1 - jt * cell) / resolution))

            records.append(
                {
                    "row": r,
                    "col": c,
                    "width_px": int(round((x1 - x0) / resolution)),
                    "height_px": int(round((y1 - y0) / resolution)),
                }
            )
            if mode == "chip":
                # Only the chip mode has a chip lattice to describe. Mode 1
                # leaves these out entirely rather than filling them with
                # sentinels a reader would have to know to ignore.
                records[-1].update(
                    n_chips_x=ir - il,
                    n_chips_y=jt - jb,
                    chip_i0=il,
                    chip_j0=jb,
                    chip_off_x=off_x,
                    chip_off_y=off_y,
                )
            geoms.append(box(x0, y0, x1, y1))

    gdf = gpd.GeoDataFrame(pd.DataFrame(records), geometry=geoms, crs=target_crs)
    gdf["bbox_km2"] = gdf.geometry.area / 1e6
    gdf["aoi_km2"] = [geom.intersection(g).area / 1e6 for g in gdf.geometry]
    gdf["fill_pct"] = 100.0 * gdf.aoi_km2 / gdf.bbox_km2

    # --- drop the pieces that hold no AOI ------------------------------------
    n_before = len(gdf)
    if drop_empty:
        keep = gdf.fill_pct > (100.0 * float(min_fill)) if min_fill > 0 else gdf.aoi_km2 > 0
        gdf = gdf[keep].reset_index(drop=True)
        if gdf.empty:
            raise ValueError(
                "every piece was dropped: none of the grid holds any AOI"
                + (f" above min_fill={min_fill}." if min_fill > 0 else ".")
            )

    # Labels are assigned AFTER the drop, so p01..pNN matches the batch's
    # _01.._NN suffixes one-for-one. row/col still carry the original grid
    # position, which is what a later mosaic needs.
    pad_w = max(2, len(str(len(gdf))))
    gdf.insert(0, "piece", [f"p{i + 1:0{pad_w}d}" for i in range(len(gdf))])

    gdf.attrs = {
        "crs": target_crs,
        "resolution": resolution,
        "chip_size": chip_size,
        "chip_m": cell if mode == "chip" else None,
        "fit": fit,
        "overlap": overlap,
        "mode": mode,
        "grid": grid,
        "n_chips_total": int(n_cells_x * n_cells_y) if mode == "chip" else 0,
    }

    if not q:
        _report(gdf, geom, mode, n_cells_x, n_cells_y, chip_size, target_crs,
                fit, n_before, drop_empty)

    if out:
        gdf.to_file(out)
        if not q:
            print(f"wrote {out}", flush=True)
    return gdf


def _report(gdf, geom, mode, n_cells_x, n_cells_y, chip_size, target_crs,
            fit, n_before, drop_empty):
    b = geom.bounds
    aoi_bbox_km2 = (b[2] - b[0]) * (b[3] - b[1]) / 1e6
    print(f"AOI            : {geom.area / 1e6:.0f} km2, bbox {aoi_bbox_km2:.0f} km2, "
          f"in {target_crs}")
    if mode == "chip":
        print(f"chip lattice   : {n_cells_x} x {n_cells_y} chips of {chip_size} px "
              f'({n_cells_x * n_cells_y} chips, fit="{fit}")')
    print(f"pieces         : {len(gdf)} cube(s) "
          f"(grid {gdf.attrs['grid'][0]} x {gdf.attrs['grid'][1]}"
          + (f", {n_before - len(gdf)} dropped as empty" if drop_empty and
             n_before != len(gdf) else "") + ")")
    # chip_size sizes the extent; it never splits. Without this line the one-row
    # result reads as "the tool did nothing", especially at small chip sizes
    # where snapping the extent moves the edges by almost nothing.
    if mode == "chip" and len(gdf) == 1:
        print("                 one cube holding every chip - chip_size only "
              "sizes the extent.\n                 Pass chips_per_piece=N to "
              "split it into several cubes.")
    print(f"piece size     : {gdf.width_px.min()} x {gdf.height_px.min()} to "
          f"{gdf.width_px.max()} x {gdf.height_px.max()} px")
    covered = unary_union(gdf.geometry.values)
    missed = geom.difference(covered).area / 1e6
    print(f"AOI uncovered  : {missed:.4f} km2"
          + ("  (fit=\"shrink\" trims the edges)" if missed > 0 else ""))
    # The downloaded area against the AOI's own bbox is the honest cost line:
    # on a small AOI, expanding to whole chips can more than double it.
    dl = gdf.bbox_km2.sum()
    print(f"total pixels   : {int(gdf.width_px.astype('int64').mul(gdf.height_px).sum()):,}"
          f"  ({dl:.1f} km2 downloaded for a {aoi_bbox_km2:.1f} km2 AOI bbox, "
          f"{dl / aoi_bbox_km2:.2f}x)")


# ------------------------------------------------------- rectangle from point
def _edge_pixels(edge_size):
    """Normalise edge_size into (height_px, width_px).

    A pair is (rows, cols) - the same order as ``n_pieces`` and as a raster's
    own shape, so the two functions cannot disagree about which number is which.
    """
    if isinstance(edge_size, (tuple, list)):
        if len(edge_size) != 2:
            raise ValueError(
                f"edge_size as a pair must be (height, width) in pixels, "
                f"got {edge_size!r}"
            )
        h, w = (int(v) for v in edge_size)
    else:
        h = w = int(edge_size)
    if h < 1 or w < 1:
        raise ValueError(f"edge_size must be >= 1 pixel, got {edge_size!r}")
    return h, w


def read_point_file(points):
    """Read a point file (or GeoDataFrame) into latitudes, longitudes, source CRS.

    Whatever the file is projected in, the coordinates come back as degrees:
    ``to_crs("EPSG:4326")`` is applied unconditionally, so a UTM or national-grid
    point file needs no preparation. A file with NO CRS is refused rather than
    assumed to be lon/lat, because assuming would silently put the AOI somewhere
    else on the planet.

    Returns ``(lats, lons, source_crs)``; ``source_crs`` is there so a caller can
    tell the user their coordinates were converted.
    """
    src = points if isinstance(points, gpd.GeoDataFrame) else gpd.read_file(points)
    if src.crs is None:
        raise ValueError(
            "the point file has no CRS, so its coordinates cannot be placed on "
            "the globe. Assign one in your GIS and try again."
        )
    kinds = set(src.geom_type)
    if kinds - {"Point"}:
        raise ValueError(
            "this needs POINT geometries; the file holds "
            f"{sorted(kinds)}. Use tile_aoi for polygons."
        )
    if len(src) == 0:
        raise ValueError("the point file is empty.")

    pts = src.to_crs("EPSG:4326").geometry
    return ([float(p.y) for p in pts], [float(p.x) for p in pts],
            crs_attr_string(src.crs))


def _read_points(lat, lon):
    """Normalise the point input into parallel lists of lat/lon in degrees.

    Accepts scalars, equal-length sequences, or - with ``lon`` left out - a
    point file or GeoDataFrame, which is how a GIS user already stores sample
    locations. A CSV is one pandas line away from the sequence form, so it is
    deliberately not special-cased here.
    """
    if lon is None:
        lats, lons, _ = read_point_file(lat)
        return lats, lons

    lats = [float(lat)] if np.isscalar(lat) else [float(v) for v in lat]
    lons = [float(lon)] if np.isscalar(lon) else [float(v) for v in lon]
    if len(lats) != len(lons):
        raise ValueError(
            f"lat and lon must be the same length, got {len(lats)} and {len(lons)}."
        )
    return lats, lons


def aoi_from_point(lat, lon=None, edge_size=128, resolution=None, names=None,
                   crs=None, out=None, q=False):
    """A rectangle of a given PIXEL size around one or more coordinates.

    The ``cubo``-style entry point: say where and how many pixels, get an AOI
    the build turns into a cube of exactly that size. With several points it
    emits one feature each, which :func:`stac2cube.get_stac_layers` builds as a
    batch - one cube per point.

    Parameters
    ----------
    lat, lon:
        Decimal degrees (EPSG:4326). Scalars for one site, equal-length
        sequences for many. Alternatively pass a point file or GeoDataFrame as
        ``lat`` and leave ``lon`` out.
    edge_size:
        Cube side in PIXELS: ``128`` for a square, or ``(height, width)`` for a
        rectangle. The metre size is ``edge_size * resolution`` and is reported.
    resolution:
        Pixel size in the target CRS's metres. Required - it is what turns
        pixels into ground distance.
    names:
        Optional label per point, kept in the ``point`` column. The build's own
        batch naming stays positional (``_01.._NN``), so these are descriptive
        only.
    crs:
        Target CRS. Guessed from the points when omitted, with a warning. It
        must match the build's - see the module docstring.
    out:
        Optional path to write to (``.gpkg`` recommended).

    Returns
    -------
    GeoDataFrame of rectangles in the target CRS, one row per point, in the
    order given. Columns: ``point``, ``lat``, ``lon`` (as requested),
    ``width_px``, ``height_px``, ``width_m``, ``height_m``, ``offset_m``
    (straight-line distance from the requested coordinate to the rectangle's
    centre - half a pixel per axis, so up to half a pixel times root two),
    ``bbox_km2``.
    """
    if resolution is None:
        raise ValueError(
            "resolution is required: edge_size is in pixels, and turning that "
            "into a ground rectangle needs the pixel size."
        )
    resolution = float(resolution)
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")

    height_px, width_px = _edge_pixels(edge_size)
    lats, lons = _read_points(lat, lon)
    if not lats:
        raise ValueError("no points given - there is nothing to build an AOI around.")
    for la, lo in zip(lats, lons):
        if not (-90.0 <= la <= 90.0) or not (-180.0 <= lo <= 180.0):
            raise ValueError(
                f"({la}, {lo}) is not a valid (lat, lon) in degrees. The order "
                "is LATITUDE first, longitude second."
            )
    if names is not None and len(list(names)) != len(lats):
        raise ValueError(
            f"names has {len(list(names))} entries but there are {len(lats)} points."
        )

    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326"
    )
    if crs is not None:
        target = validate_target_crs(crs)
    else:
        target = crs_attr_string(pts.estimate_utm_crs())
        if not q:
            print(
                f"No crs= given; drawing the rectangles in {target}, guessed "
                "from the points. The build chooses its CRS from the matched "
                "STAC items instead, and the two can differ - pass this value "
                "to get_stac_layers(crs=...) so the rectangles are not "
                "reprojected.",
                flush=True,
            )
        # One CRS for the whole batch, so points scattered across UTM zones all
        # get drawn in one of them. Metres stay metres, but the further a point
        # is from its own zone the more its scenes have to be warped at build
        # time. Worth saying out loud rather than leaving in the timings.
        zones = {crs_attr_string(pts.iloc[[i]].estimate_utm_crs())
                 for i in range(len(pts))}
        if len(zones) > 1 and not q:
            print(
                f"Note: these points fall in {len(zones)} different UTM zones "
                f"({', '.join(sorted(zones))}). They are all drawn in {target}, "
                "so scenes for the outlying ones will be reprojected during the "
                "build. Consider one run per zone.",
                flush=True,
            )

    proj = pts.to_crs(target)
    half_w = width_px * resolution / 2.0
    half_h = height_px * resolution / 2.0

    records, geoms = [], []
    for i, (p, la, lo) in enumerate(zip(proj.geometry, lats, lons)):
        # Snap the LOWER-LEFT corner to the pixel grid, then step a whole number
        # of pixels. Snapping the corner rather than the centre is what keeps
        # the width exactly width_px: rounding both edges independently could
        # land a pixel apart.
        x0 = round((p.x - half_w) / resolution) * resolution
        y0 = round((p.y - half_h) / resolution) * resolution
        x1 = x0 + width_px * resolution
        y1 = y0 + height_px * resolution

        records.append({
            "point": (list(names)[i] if names is not None
                      else f"p{i + 1:0{max(2, len(str(len(lats))))}d}"),
            "lat": la,
            "lon": lo,
            "width_px": width_px,
            "height_px": height_px,
            "width_m": width_px * resolution,
            "height_m": height_px * resolution,
            "offset_m": float(math.hypot((x0 + x1) / 2.0 - p.x,
                                         (y0 + y1) / 2.0 - p.y)),
        })
        geoms.append(box(x0, y0, x1, y1))

    gdf = gpd.GeoDataFrame(pd.DataFrame(records), geometry=geoms, crs=target)
    gdf["bbox_km2"] = gdf.geometry.area / 1e6
    gdf.attrs = {
        "crs": target,
        "resolution": resolution,
        "edge_size": (height_px, width_px),
        "mode": "point",
        "n_points": len(gdf),
    }

    if not q:
        print(f"points         : {len(gdf)} in {target}")
        print(f"cube size      : {width_px} x {height_px} px at {resolution:g} m "
              f"= {width_px * resolution:,.0f} x {height_px * resolution:,.0f} m "
              f"({width_px * resolution / 1000:.2f} x "
              f"{height_px * resolution / 1000:.2f} km)")
        # Half a pixel PER AXIS, so the straight-line offset reported here can
        # reach half a pixel times root two - 7.1 m at 10 m, not 5.
        print(f"centre offset  : up to {gdf.offset_m.max():.1f} m from the "
              f"requested point (pixel-grid snap: at most half a pixel per axis)")
        if len(gdf) > 1:
            print(f"batch          : {len(gdf)} data cubes, one per point")
    if out:
        gdf.to_file(out)
        if not q:
            print(f"wrote {out}", flush=True)
    return gdf


def estimate_piece_bytes(gdf, n_dates, n_bands, itemsize=4):
    """Rough on-disk size of the cubes a piece grid implies.

    ``width x height x dates x bands x itemsize`` per piece, uncompressed and
    ignoring coordinates and metadata, which is what the GUI needs to show the
    consequence of a ``chips_per_piece`` choice before anything is downloaded.
    ``itemsize=4`` is float32, the dtype a scaled cube carries in memory; an
    int16-packed export is half that.

    Returns a dict with ``per_piece`` (list of bytes), ``largest``, ``total``.
    """
    per = [
        int(w) * int(h) * int(n_dates) * int(n_bands) * int(itemsize)
        for w, h in zip(gdf.width_px, gdf.height_px)
    ]
    return {"per_piece": per, "largest": max(per), "total": sum(per)}
