#!/usr/bin/env python3
"""
Download and convert pretrained segmentation models to ONNX format.
Automatically saves to edge_perception/models/ directory.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import torch
    import torchvision
except ImportError:
    print("ERROR: PyTorch not installed. Install with:")
    print("  pip install torch torchvision")
    sys.exit(1)

try:
    import onnx
    import onnxscript
except ImportError:
    print("ERROR: ONNX dependencies not installed. Install with:")
    print("  pip install onnx onnxscript")
    sys.exit(1)


MODELS = {
    'lraspp_mobilenet': {
        'loader': lambda: torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights='DEFAULT'),
        'input_size': (640, 360),
        'description': 'Lightweight (3MB), very fast, 21 classes (PASCAL VOC)'
    },
    'deeplabv3_mobilenet': {
        'loader': lambda: torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights='DEFAULT'),
        'input_size': (640, 360),
        'description': 'Medium weight (12MB), good accuracy, 21 classes (PASCAL VOC)'
    },
    'deeplabv3_resnet50': {
        'loader': lambda: torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT'),
        'input_size': (640, 360),
        'description': 'Heavier (160MB), high accuracy, 21 classes (PASCAL VOC)'
    },
    'fcn_resnet50': {
        'loader': lambda: torchvision.models.segmentation.fcn_resnet50(weights='DEFAULT'),
        'input_size': (640, 360),
        'description': 'Classic FCN (135MB), 21 classes (PASCAL VOC)'
    }
}


def download_and_convert(model_name: str, output_path: str, width: int = 640, height: int = 360):
    """Download pretrained model and convert to ONNX."""
    if model_name not in MODELS:
        print(f"ERROR: Unknown model '{model_name}'. Available models:")
        for name, info in MODELS.items():
            print(f"  {name}: {info['description']}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Downloading model: {model_name}")
    print(f"Description: {MODELS[model_name]['description']}")
    print(f"{'='*70}\n")

    # Load pretrained model
    print("Loading pretrained weights from torchvision...")
    model = MODELS[model_name]['loader']()
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, height, width)

    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    print(f"Input size: {width}x{height}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'},
                              'output': {0: 'batch', 2: 'height', 3: 'width'}},
                opset_version=11,
                do_constant_folding=True
            )
        print(f"\n✓ Successfully exported to: {output_path}")
        
        # Show file size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Model size: {size_mb:.1f} MB")
        
        print("\nModel is ready! You can now run:")
        print("  ros2 launch edge_perception semantic_seg.launch.py")
        
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Download and convert segmentation models to ONNX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  lraspp_mobilenet      - Lightweight, very fast (recommended for Jetson)
  deeplabv3_mobilenet   - Good balance of speed/accuracy
  deeplabv3_resnet50    - Higher accuracy, slower
  fcn_resnet50          - Classic FCN architecture

Example:
  python3 download_model.py --model lraspp_mobilenet
  python3 download_model.py --model deeplabv3_mobilenet --width 512 --height 288
        """
    )
    
    parser.add_argument('--model', type=str, default='lraspp_mobilenet',
                        help='Model to download (default: lraspp_mobilenet)')
    parser.add_argument('--output', type=str, default='',
                        help='Output path (default: auto-detect edge_perception/models/segmentation.onnx)')
    parser.add_argument('--width', type=int, default=1280,
                        help='Input width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='Input height (default: 720)')
    parser.add_argument('--list', action='store_true',
                        help='List available models and exit')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable models:\n")
        for name, info in MODELS.items():
            print(f"  {name:25s} - {info['description']}")
        print()
        return
    
    # Auto-detect output path if not specified
    if not args.output:
        script_dir = Path(__file__).parent
        pkg_dir = script_dir.parent
        output_path = pkg_dir / 'models' / f'{args.model}_w{args.width}_h{args.height}_segmentation.onnx'
    else:
        output_path = Path(args.output)
    
    download_and_convert(args.model, str(output_path), args.width, args.height)


if __name__ == '__main__':
    main()
