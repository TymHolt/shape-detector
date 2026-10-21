#!/bin/bash

sudo apt-get update
sudo apt-get install python3 python3-venv python3-pip

python3 -m venv .venv

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install opencv-python

python --version
python -c "import cv2; print(cv2.__version__)"

deactivate