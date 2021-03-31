#!/bin/bash
set -e
set -x

DATA_DIR=/tmp/data

if [ ! -d $DATA_DIR ]; then
    mkdir $DATA_DIR
fi

cd $DATA_DIR

if [ ! -f "$DATA_DIR/CLEVR_v1.0.zip.*" ]; then
    wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
    echo "CLEVR_v1 downloaded to $DATA_DIR/CLEVR_v1.0.zip"
else
    echo "$DATA_DIR/CLEVR_v1.0.zip already exists, skipping download"
fi

echo "unzipping CLEVR_v1 to $DATA_DIR/CLEVR_v1.0"
rm -rf CLEVR_v1.0
unzip -q CLEVR_v1.0.zip





