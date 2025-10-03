#!/usr/bin/env python3
"""
YOLOv11 Validation Script
This script validates a trained YOLOv11 model on your dataset
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def validate_model(
    model_path,
    dataset_config="./configs/dataset.yaml",
    save_json=True,
    save_hybrid=False,
    conf=0.001,
    iou=0.6,
    max_det=300,
    half=False,
    device=None,
    dnn=False,
    plots=True,
    rect=False,
    split='val'
):
    """
    Validate YOLOv11 model
    
    Args:
        model_path (str): Path to trained model weights
        dataset_config (str): Path to dataset configuration YAML
        save_json (bool): Save results to JSON file
        save_hybrid (bool): Save hybrid version of labels
        conf (float): Confidence threshold
        iou (float): IoU threshold for NMS
        max_det (int): Maximum detections per image
        half (bool): Use half precision
        device (str/int): Device to run on
        dnn (bool): Use OpenCV DNN for inference
        plots (bool): Save plots
        rect (bool): Rectangular inference
        split (str): Dataset split to validate on
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Check if dataset config exists
    if not os.path.exists(dataset_config):
        raise FileNotFoundError(f"Dataset config not found: {dataset_config}")
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Set device
    if device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'
    
    print(f"Validation device: {device}")
    print(f"Dataset config: {dataset_config}")
    print(f"Confidence threshold: {conf}")
    print(f"IoU threshold: {iou}")
    print(f"Max detections: {max_det}")
    
    # Validate the model
    results = model.val(
        data=dataset_config,
        split=split,
        save_json=save_json,
        save_hybrid=save_hybrid,
        conf=conf,
        iou=iou,
        max_det=max_det,
        half=half,
        device=device,
        dnn=dnn,
        plots=plots,
        rect=rect
    )
    
    print("\n" + "="*50)
    print("Validation completed!")
    print("="*50)
    
    # Print metrics
    if hasattr(results, 'results_dict'):
        print("\nValidation metrics:")
        for key, value in results.results_dict.items():
            print(f"{key}: {value}")
    
    # Print class-wise metrics if available
    if hasattr(results, 'ap_class_index') and hasattr(results, 'ap'):
        print("\nClass-wise Average Precision (AP):")
        for i, ap in enumerate(results.ap[0]):  # AP@0.5
            class_idx = results.ap_class_index[i] if i < len(results.ap_class_index) else i
            print(f"Class {class_idx}: {ap:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Validate YOLOv11 model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights (.pt file)')
    parser.add_argument('--dataset', type=str, default='./configs/dataset.yaml',
                       help='Path to dataset configuration YAML')
    parser.add_argument('--conf', type=float, default=0.001,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.6,
                       help='IoU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=300,
                       help='Maximum detections per image')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run on (0, 1, 2, ... or cpu)')
    parser.add_argument('--half', action='store_true',
                       help='Use half precision')
    parser.add_argument('--dnn', action='store_true',
                       help='Use OpenCV DNN for inference')
    parser.add_argument('--save-json', action='store_true', default=True,
                       help='Save results to JSON file')
    parser.add_argument('--save-hybrid', action='store_true',
                       help='Save hybrid version of labels')
    parser.add_argument('--plots', action='store_true', default=True,
                       help='Save plots')
    parser.add_argument('--rect', action='store_true',
                       help='Rectangular inference')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to validate on')
    
    args = parser.parse_args()
    
    try:
        results = validate_model(
            model_path=args.model,
            dataset_config=args.dataset,
            save_json=args.save_json,
            save_hybrid=args.save_hybrid,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            half=args.half,
            device=args.device,
            dnn=args.dnn,
            plots=args.plots,
            rect=args.rect,
            split=args.split
        )
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()