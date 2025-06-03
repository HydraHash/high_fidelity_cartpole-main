#!/bin/bash
# Requires Python 3.10.9 and ffmpeg
# brew install ffmpeg
# brew install pyenv
# pyenv install 3.10.9
# alias python=/.pyenv/versions/3.10.9/bin/python

# Create & Activate python environment
ENV_NAME=cartpole-realistic
if [ ! -d "$ENV_NAME" ]; then
	python3 -m venv $ENV_NAME
fi
source $ENV_NAME/bin/activate

# Install requirements if not already installed
if ! python3 -c "import tensorflow"; then
	pip install --upgrade pip
	pip install --no-cache-dir -r requirements.txt
fi
pip install -e ./cartpole_realistic
# generate video
python3 gen_video.py