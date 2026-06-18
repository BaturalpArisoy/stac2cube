from stac2cube import cloud_masking
import json
import os


def load_config_json(config_file):

    with open(config_file, 'r') as file:
        config = json.load(file)

    return config['parameters']


if __name__ == "__main__":

    config_file = os.path.expanduser('get_cloud_layers.json')
    config = load_config_json(config_file)

    cloud_masking.get_cloud_layers(**config)
