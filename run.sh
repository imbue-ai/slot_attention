#!/bin/bash
set -e
set -x

poetry install

chmod +x download_clevr.sh
./download_clevr.sh

python slot_attention/train.py
