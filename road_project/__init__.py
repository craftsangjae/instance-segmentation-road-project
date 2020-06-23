"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from road_project.setup import process_dataset
import os

PROJECT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_DIR, 'datasets/')
IMAGE_DIR = os.path.join(DATA_DIR, "images/")
LABEL_DIR = os.path.join(DATA_DIR, 'labels/')
PROCESS_DIR = os.path.join(DATA_DIR, "processed/")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(PROCESS_DIR, exist_ok=True)

ROOT_DIR = os.path.dirname(os.path.dirname(PROJECT_DIR))
LOG_DIR = os.path.join(ROOT_DIR, "logs/")
os.makedirs(LOG_DIR, exist_ok=True)

HYUNDAI_ZIP_FILE = os.path.join(DATA_DIR, 'images_labels.zip')
