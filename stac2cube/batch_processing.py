import os
from .vector_refiner import polygon_2_gdf
from .main import get_stac_layers


def multi_stac_layers(
    output_path,
    polygon,
    mission=None,
    resolution=None,
    daterange=None,
    bands=None,
    max_cc=None,
    clip_raster=None,
    cloud_masking=None,
    indices=None,
    aggregator=None,
    stats=None,
    topographic_features=None,
    animation=None,
    q=None,
):

    gdf = polygon_2_gdf(polygon)

    for i in range(len(gdf)):
        print(f"{i+1}/{len(gdf)} is being computed.", flush=True)
        # Access the geometry of the current feature
        geom = gdf.iloc[i].geometry
        # Compute the bounding box (minx, miny, maxx, maxy) for this feature
        bbox = list(geom.bounds)

        # Set dynamic output
        output = os.path.join(output_path, f"stack_{i}.nc")

        get_stac_layers(
            mission=mission,
            polygon=bbox,
            resolution=resolution,
            daterange=daterange,
            bands=bands,
            max_cc=max_cc,
            clip_raster=clip_raster,
            cloud_masking=cloud_masking,
            indices=indices,
            output=output,
            aggregator=aggregator,
            stats=stats,
            animation=animation,
            q=q,
        )


def multi_cloud_layers():
    pass


def multi_stac_update():

    pass
