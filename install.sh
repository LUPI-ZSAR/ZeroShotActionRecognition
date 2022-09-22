#!/usr/bin/env bash

# Install FAISS. This usually mess up numpy, so I reinstall it
pip install faiss-gpu
export PYTHONPATH=/usr/local/lib/python3.5/dist-packages/faiss
sudo apt-get install libopenblas-dev
pip install -U numpy

# Useful tools
pip install -U gpustat tensorboardx joblib

# Jpeg to read Kinetics and Something-Something. Not needed for UCF and HMDB
pip install -U simplejson
pip install -U jpeg4py
sudo apt-get install libturbojpeg

# The last version of Torchvision is needed for r2plus1d_18 network
# sudo pip3 install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

# Download Word2Vec google model
pip install gensim

# wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz -O /workplace/GoogleNews-vectors-negative300.bin.gz
# gunzip -c /workplace/GoogleNews-vectors-negative300.bin.gz > /workplace/GoogleNews-vectors-negative300.bin

# Natural Language Processing Tool
pip install nltk
python3 -c "import nltk; nltk.download('wordnet')"