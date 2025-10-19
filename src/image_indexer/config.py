import logging
import yaml
from argparse import Namespace, ArgumentParser
from typing import Any, Dict

DEFAULT_CONFIG = {
    "input_dirs": [],
    "output_file": "index.jsonl",
    "model_name": "google/siglip-base-patch16-naflex",
    "batch_size": 32,
    "device": "auto",
    "output_format": "jsonl",
    "log_level": "INFO",
    "config_file": None,
    "reindex": False,
    "source_index": None,
    "create_thumbnails": False,
    "thumbnail_dir": "thumbnails",
    "thumbnail_size": [128, 128],
}

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file"""
    if not config_path:
        return {}
    try:
        with open(config_path, 'r') as f:
            logging.info(f"Loading configuration from: {config_path}")
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logging.warning(f"Configuration file not found at: {config_path}. Using default values.")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file'{config_path}': {e}")
        return {}

def merge_configs(cli_args: Namespace, parser: ArgumentParser) -> Dict[str, Any]:
    """
    Combine the default values, YAML file, and CLI arguments.
    The order of priority is: CLI (if explicitly set) > YAML > Default Values.
    """
    config = DEFAULT_CONFIG.copy()
    cli_dict = vars(cli_args)
    yaml_config_path = cli_dict.get('config_file')

    if yaml_config_path:
        yaml_config = load_yaml_config(yaml_config_path)
        config.update(yaml_config)

    for key, value in cli_dict.items():
        if value is not None and value != parser.get_default(key):
            config[key] = value

    return config