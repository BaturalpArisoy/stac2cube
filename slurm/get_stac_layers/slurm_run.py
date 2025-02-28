from stac2ardcube import main
import json
import os


def load_config_json(config_file):
    
    with open(config_file, 'r') as file:
        config = json.load(file)
        
    return config['parameters']
    

if __name__ == "__main__":
    
    config_file = os.path.expanduser('get_stac_layers.json')
    config = load_config_json(config_file)
    
    main.get_stac_layers(**config)