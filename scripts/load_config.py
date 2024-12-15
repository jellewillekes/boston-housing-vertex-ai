# scripts/load_config.py
import os
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config

config = load_config()
PROJECT_ID = config["project_id"]
REGION = config["region"]
BUCKET = config["bucket"]
REPO = config["repo"]
