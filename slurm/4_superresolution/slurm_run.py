from stac2cube import super_resolution
import json
import os


def load_config_json(config_file):

    with open(config_file, 'r') as file:
        config = json.load(file)

    return config['parameters']


if __name__ == "__main__":

    config_file = os.path.expanduser('superresolution.json')
    config = load_config_json(config_file)

    super_resolution.super_resolve_cube(**config)
