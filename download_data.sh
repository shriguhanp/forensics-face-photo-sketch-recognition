#!/bin/bash
# Downloads the CUHK Face Sketch Database (CUFS) from Kaggle.
# Requires a Kaggle API token at ~/.kaggle/kaggle.json (see README section 2).

set -e

mkdir -p data
cd data

if [ -d "cufs" ] && [ "$(ls -A cufs 2>/dev/null)" ]; then
  echo "Dataset already present at data/cufs — skipping download."
  exit 0
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "ERROR: ~/.kaggle/kaggle.json not found."
  echo "Get an API token from https://www.kaggle.com/settings -> API -> Create New Token"
  echo "then: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
  exit 1
fi

echo "Downloading CUFS from Kaggle..."
kaggle datasets download -d arbazkhan971/cuhk-face-sketch-database-cufs -p cufs --unzip

echo "Done. Contents of data/cufs:"
find cufs -maxdepth 3 -type d
echo ""
echo "Next: run 'python3 explore_data.py' to inspect the actual structure."
