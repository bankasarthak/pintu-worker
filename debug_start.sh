#!/usr/bin/env bash
set -e

echo "=== PINTU WORKER STARTUP ==="
echo "$(date) Step 1: Checking environment..."
echo "  Python: $(python --version 2>&1)"
echo "  Torch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>&1)"
echo "  Network volume: $(ls /runpod-volume/ 2>&1 | head -20)"
echo "  Models dir: $(ls /runpod-volume/models/ 2>&1 | head -20)"

echo ""
echo "$(date) Step 2: Checking ComfyUI installation..."
echo "  ComfyUI dir: $(ls /comfyui/ 2>&1 | head -10)"
echo "  Custom nodes: $(ls /comfyui/custom_nodes/ 2>&1)"

echo ""
echo "$(date) Step 3: Checking custom node imports..."
python -c "
import sys
sys.path.insert(0, '/comfyui')
print('  Testing gguf import...')
try:
    import gguf
    print(f'    gguf OK: {gguf.__version__}')
except Exception as e:
    print(f'    gguf FAILED: {e}')

print('  Testing opencv import...')
try:
    import cv2
    print(f'    cv2 OK: {cv2.__version__}')
except Exception as e:
    print(f'    cv2 FAILED: {e}')

print('  Testing imageio_ffmpeg import...')
try:
    import imageio_ffmpeg
    print(f'    imageio_ffmpeg OK: {imageio_ffmpeg.get_ffmpeg_exe()}')
except Exception as e:
    print(f'    imageio_ffmpeg FAILED: {e}')
" 2>&1

echo ""
echo "$(date) Step 4: Testing ComfyUI import (this is where hangs usually happen)..."
timeout 60 python -c "
import sys, os
sys.path.insert(0, '/comfyui')
os.chdir('/comfyui')
print('  Importing comfy...')
try:
    import main
    print('  ComfyUI main module imported OK')
except SystemExit:
    print('  ComfyUI main triggered SystemExit (normal for import test)')
except Exception as e:
    print(f'  ComfyUI import FAILED: {e}')
" 2>&1 || echo "  TIMEOUT: ComfyUI import hung for 60s!"

echo ""
echo "$(date) Step 5: Starting actual worker (calling original /start.sh)..."
exec /start.sh
