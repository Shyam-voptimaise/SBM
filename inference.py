#!/usr/bin/env python3
"""
YOLOv11 Inference Script
This script runs inference using a trained YOLOv11 model
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def run_inference(
    model_path,
    source,
    conf=0.25,
    iou=0.45,
    max_det=1000,
    device=None,
    save=True,
    save_txt=False,
    save_conf=False,
    save_crop=False,
    show=False,
    vid_stride=1,
    line_width=None,
    visualize=False,
    augment=False,
    agnostic_nms=False,
    classes=None,
    retina_masks=False,
    boxes=True
):
    """
    Run inference on images, videos, or camera stream
    
    Args:
        model_path (str): Path to trained model weights
        source (str): Source for inference (image, video, directory, URL, webcam)
        conf (float): Confidence threshold
        iou (float): IoU threshold for NMS
        max_det (int): Maximum detections per image
        device (str/int): Device to run on
        save (bool): Save results
        save_txt (bool): Save results to txt files
        save_conf (bool): Save confidences in txt files
        save_crop (bool): Save cropped prediction boxes
        show (bool): Show results
        vid_stride (int): Video frame-rate stride
        line_width (int): Bounding box thickness
        visualize (bool): Visualize features
        augment (bool): Apply test time augmentation
        agnostic_nms (bool): Class-agnostic NMS
        classes (list): Filter by class
        retina_masks (bool): Use high resolution masks
        boxes (bool): Show boxes
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Set device
    if device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'
    
    print(f"Inference device: {device}")
    print(f"Source: {source}")
    print(f"Confidence threshold: {conf}")
    print(f"IoU threshold: {iou}")
    
    # Run inference
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        save=save,
        save_txt=save_txt,
        save_conf=save_conf,
        save_crop=save_crop,
        show=show,
        vid_stride=vid_stride,
        line_width=line_width,
        visualize=visualize,
        augment=augment,
        agnostic_nms=agnostic_nms,
        classes=classes,
        retina_masks=retina_masks,
        boxes=boxes,
        project="./runs/predict",
        name="exp"
    )
    
    print("\n" + "="*50)
    print("Inference completed!")
    if save:
        print("Results saved to: ./runs/predict/exp")
    print("="*50)
    
    return results


def process_single_image(model_path, image_path, output_dir="./outputs"):
    """
    Process a single image and return results
    
    Args:
        model_path (str): Path to trained model weights
        image_path (str): Path to input image
        output_dir (str): Directory to save results
    
    Returns:
        dict: Detection results
    """
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path)
    
    # Process results
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                # Extract box coordinates, confidence, and class
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                
                detections.append({
                    'bbox': xyxy.tolist(),
                    'confidence': float(conf),
                    'class': int(cls),
                    'class_name': model.names[int(cls)]
                })
    
    # Save annotated image
    if detections:
        os.makedirs(output_dir, exist_ok=True)
        image_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{image_name}_detected.jpg")
        
        # Save annotated image
        annotated_frame = results[0].plot()
        cv2.imwrite(output_path, annotated_frame)
        print(f"Annotated image saved: {output_path}")
    
    return {
        'image_path': image_path,
        'detections': detections,
        'total_detections': len(detections)
    }


def main():
    parser = argparse.ArgumentParser(description='Run YOLOv11 inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                       help='Source for inference (image, video, directory, URL, webcam)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000,
                       help='Maximum detections per image')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run on (0, 1, 2, ... or cpu)')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save results')
    parser.add_argument('--save-txt', action='store_true',
                       help='Save results to txt files')
    parser.add_argument('--save-conf', action='store_true',
                       help='Save confidences in txt files')
    parser.add_argument('--save-crop', action='store_true',
                       help='Save cropped prediction boxes')
    parser.add_argument('--show', action='store_true',
                       help='Show results')
    parser.add_argument('--vid-stride', type=int, default=1,
                       help='Video frame-rate stride')
    parser.add_argument('--line-width', type=int, default=None,
                       help='Bounding box thickness')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize features')
    parser.add_argument('--augment', action='store_true',
                       help='Apply test time augmentation')
    parser.add_argument('--agnostic-nms', action='store_true',
                       help='Class-agnostic NMS')
    parser.add_argument('--classes', nargs='+', type=int,
                       help='Filter by class: --classes 0 2 3')
    
    args = parser.parse_args()
    
    try:
        results = run_inference(
            model_path=args.model,
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            save=args.save,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save_crop=args.save_crop,
            show=args.show,
            vid_stride=args.vid_stride,
            line_width=args.line_width,
            visualize=args.visualize,
            augment=args.augment,
            agnostic_nms=args.agnostic_nms,
            classes=args.classes
        )
        
        # Print detection summary
        print(f"\nProcessed {len(results)} image(s)/frame(s)")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()