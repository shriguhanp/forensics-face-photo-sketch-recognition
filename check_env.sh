#!/bin/bash
PYBIN="/Users/manju/Documents/Forensics/venv/bin/python3"

echo "=== Python version ==="
$PYBIN --version

echo "=== Checking packages ==="
$PYBIN -c "
packages = ['torch', 'torchvision', 'facenet_pytorch', 'matplotlib', 'PIL', 'numpy', 'sklearn', 'scipy', 'h5py', 'skimage']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'  OK: {pkg}')
    except ImportError as e:
        print(f'  MISSING: {pkg} ({e})')
        missing.append(pkg)
if missing:
    print(f'\nMISSING PACKAGES: {missing}')
else:
    print('\nALL PACKAGES OK')
"

echo "=== MPS check ==="
$PYBIN -c "import torch; print('MPS:', torch.backends.mps.is_available())"

echo "=== Model files ==="
ls -lh /Users/manju/Documents/Forensics/stage1_model.pth /Users/manju/Documents/Forensics/stage2_model.pth 2>&1

echo "=== Data check ==="
ls /Users/manju/Documents/Forensics/data/cufs/sketches | wc -l
ls /Users/manju/Documents/Forensics/data/cufs/photos | wc -l

echo "=== JSON files ==="
ls /Users/manju/Documents/Forensics/train_pairs.json /Users/manju/Documents/Forensics/test_pairs.json 2>&1

echo "=== DONE ==="
