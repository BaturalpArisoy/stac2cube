import os
import geopandas as gpd
import fiona
import pyproj


def polygon_2_gdf(polygon) -> gpd.GeoDataFrame:
    """
    Reads a polygon file and returns a processed GeoDataFrame containing only the first geometry,
    reprojected to WGS84.
    """
    gdf = read_polygon_file(polygon)
    if gdf is None:
        return None
    return _process_gdf(gdf)


def polygon_2_bbox(polygon) -> list:
    """
    Reads a polygon file and returns the bounding box of the first geometry as a list:
    [minx, miny, maxx, maxy]. The geometry is reprojected to WGS84.
    """
    gdf = polygon_2_gdf(polygon)
    if gdf is None:
        return None
    bbox = gdf.total_bounds  # returns [minx, miny, maxx, maxy]
    return [float(coord) for coord in bbox]


# This function does not work as intended. Will be reworked or deleted :(
def proj_2_geo(polygon, source_epsg=None):

    if isinstance(polygon, list):  # DELETE
        bbox = polygon  # DELETE
        source_epsg = source_epsg  # DELETE
    else:
        gdf = polygon_2_gdf(polygon)
        crs_str = str(gdf.crs)
        # source_epsg = int(crs_str.split(":")[1])
        bbox = gdf.total_bounds.tolist()

    """
    dest_epsg = 4326
    transformer = pyproj.Transformer.from_crs(f"EPSG:{source_epsg}", f"EPSG:{dest_epsg}", always_xy=True)
    lon_min, lat_min = transformer.transform(bbox[0], bbox[1])
    lon_max, lat_max = transformer.transform(bbox[2], bbox[3])
    new_bbox = [lon_min, lat_min, lon_max, lat_max]
    print(new_bbox)
    """
    return bbox


def proj_check(polygon):

    gdf = polygon_2_gdf(polygon)
    if str(gdf.crs) != "EPSG:4326":
        polygon = proj_2_geo(polygon)
    return polygon


def read_polygon_file(polygon) -> gpd.GeoDataFrame:
    """
    Reads a polygon file from various geospatial formats and returns a GeoDataFrame.
    """
    split_up = os.path.splitext(polygon)
    geo_format = split_up[1].lower()  # use lowercase to handle extensions like '.KML'

    if geo_format == ".kml":
        fiona.drvsupport.supported_drivers["KML"] = "rw"
        gdf = gpd.read_file(polygon, driver="KML")

    elif geo_format == ".gpkg":
        gdf = gpd.read_file(polygon)

    elif geo_format == ".geojson":
        gdf = gpd.read_file(polygon, driver="GeoJSON")

    elif geo_format == ".gml":
        gdf = gpd.read_file(polygon, driver="GML")

    elif geo_format == ".kmz":
        fiona.drvsupport.supported_drivers["KML"] = "rw"
        from zipfile import ZipFile

        kmz = ZipFile(polygon, "r")
        # Extract the KML file; adjust the name and folder as needed
        kmz.extract("doc.kml", "data/")
        with kmz.open("doc.kml", "r") as kml:
            gdf = gpd.read_file(kml, driver="KML")
        # Remove any Z-dimension from geometries (if present)
        from shapely.geometry import Polygon

        gdf.geometry = gdf.geometry.apply(
            lambda geom: Polygon([(x, y) for x, y, *_ in geom.exterior.coords])
        )

    elif geo_format == ".shp":
        gdf = gpd.read_file(polygon, driver="shapefile")
        _shp_warning()

    else:
        try:
            gdf = gpd.read_file(polygon)
        except UserWarning:
            print("Not a supported format. Please import one of the supported formats.")
            return None

    return gdf


def _process_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Processes the GeoDataFrame by:
      - Selecting only the first row (geometry)
      - Reprojecting to WGS84 (EPSG:4326) if needed
    """
    # Select only the first geometry
    # gdf = gdf.iloc[[0]]

    # Define target CRS
    target_crs = "EPSG:4326"
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    return gdf


def _shp_warning():

    print("##################")
    print("##################")
    print(
        "##################",
        "Mmm, yes.",
        "Shapefile, old it is.",
        "New ways, better ways, there are.",
        "Learn you must, young geospatial padawan.",
        "Efficiency and versatility, modern formats offer.",
        "Transition you should, hmm? Yes, hmmm.",
        "Improve your data management, you will.",
        "From Shapefile, switch you must.",
        "To better practices, embrace the future, you shall.",
        "May the geospatial force be with you.",
        "Mmm.",
        "##################",
        sep="\n",
    )

    print("\n#shapefilemustdie, http://switchfromshapefile.org/\n")
    print("##################")
    print("##################")
    print("##################")
