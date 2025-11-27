#!/bin/bash
# This script takes care of all the computational SETUP required to run the data
# processing and experiments.
#
# We use a python virtual environment based on a Python 3.9 Installation
# - We used homebrew to install the base python and built the environment
# - from that point

/usr/local/bin/python3.10  -m venv pyenv3.10
virtualenv pyenv3.10 --python=/usr/local/bin/python3.10
source pyenv3.10/bin/activate

pip3 install -r requirements.txt

pip install transformers[torch]

python3 scripts/install_nltk_data.py

ipython kernel install --user --name=venv
