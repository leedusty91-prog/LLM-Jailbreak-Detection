#!/bin/bash
# This script takes care of all the computational SETUP required to run the data
# processing and experiments.
#
# We use a python virtual environment based on a Python 3.9 Installation
# - We used homebrew to install the base python and built the environment
# - from that point

virtualenv venv --python=/usr/local/bin/python3.9
source venv/bin/activate

pip3 install -r requirements.txt

pip install transformers[torch]

python3 scripts/install_nltk_data.py

ipython kernel install --user --name=venv
