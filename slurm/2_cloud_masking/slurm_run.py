from stac2cube import cloud_masking
from stac2cube.get_update import get_stac_parameters
import json
import os


def load_config_json(config_file):

    with open(config_file, 'r') as file:
        config = json.load(file)

    return config['parameters']


if __name__ == "__main__":

    config_file = os.path.expanduser('get_cloud_layers.json')
    config = load_config_json(config_file)

    # Optional convenience: derive polygon + daterange from an existing data cube,
    # exactly like the "Manually Build Cloud Masking Data Cube" GUI step
    # (it calls stac2cube.get_update.get_stac_parameters under the hood).
    # 'input_cube' is consumed here and is NOT passed to get_cloud_layers.
    input_cube = config.pop("input_cube", None)
    if input_cube:
        params = get_stac_parameters(input_cube)
        config.setdefault("polygon", params["polygon"])
        config.setdefault("daterange", params["daterange"])

    cloud_masking.get_cloud_layers(**config)
