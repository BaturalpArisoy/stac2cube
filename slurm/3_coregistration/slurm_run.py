from stac2cube import coregistration
import json
import os


def load_config_json(config_file):

    with open(config_file, 'r') as file:
        config = json.load(file)

    return config['parameters']


if __name__ == "__main__":

    config_file = os.path.expanduser('coregistration.json')
    config = load_config_json(config_file)

    coregistration.coregister_cube(**config)
