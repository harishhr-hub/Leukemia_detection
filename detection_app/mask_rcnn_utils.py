"""
Mask R-CNN (ResNet-50-FPN) for Blood Cell Detection

This module provides utilities for cell detection in blood smear images using
Mask R-CNN with ResNet-50-FPN backbone.
"""

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np
import cv2
from PIL import Image
import os
from django.conf import settings

# Global model cache
MASK_RCNN_MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MaskRCNNDetector:
    """Mask R-CNN detector for blood cell segmentation and detection."""
    
    def __init__(self, model_path=None, pretrained=True, num_classes=2):
        """
        Initialize Mask R-CNN detector.
        
        Args:
            model_path: Path to fine-tuned model weights
            pretrained: Use pretrained COCO weights
            num_classes: Number of classes (2 for blood cells: background + cell)
        """
        self.device = DEVICE
        self.num_classes = num_classes
        self.model = self._build_model(pretrained, num_classes)
        
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"Loaded fine-tuned model from {model_path}")
            except Exception as e:
                print(f"Could not load fine-tuned model: {e}. Using pretrained weights.")
        
        self.model.to(self.device)
        self.model.eval()
    
    def _build_model(self, pretrained=True, num_classes=2):
        """Build Mask R-CNN ResNet-50-FPN model."""
        if pretrained:
            model = maskrcnn_resnet50_fpn(pretrained=True)
        else:
            model = maskrcnn_resnet50_fpn(pretrained=False)
        
        # Replace classifier for custom number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, num_classes)
        model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features, num_classes * 4)
        
        # Mask head
        in_features_mask = model.roi_heads.mask_head.fc1.in_channels
        hidden_dim = 256
        model.roi_heads.mask_head.fc1 = torch.nn.Linear(in_features_mask, hidden_dim)
        model.roi_heads.mask_head.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        
        return model
    
    def detect(self, image_path, confidence_threshold=0.5, nms_threshold=0.5):
        """
        Detect cells in image.
        
        Args:
            image_path: Path to blood smear image
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-maximum suppression threshold
            
        Returns:
            Dictionary with detections:
            {
                'boxes': List of [x1, y1, x2, y2] coordinates,
                'scores': Detection confidence scores,
                'masks': Binary segmentation masks,
                'num_cells': Total number of detected cells,
                'image': Original image array,
                'annotated_image': Image with bounding boxes
            }
        """
        # Load and preprocess image
        pil_image = Image.open(image_path).convert('RGB')
        image_array = np.array(pil_image)
        
        # Convert to tensor
        tensor_image = F.to_tensor(pil_image).to(self.device)
        
        # Run detection
        with torch.no_grad():
            predictions = self.model([tensor_image])
        
        pred = predictions[0]
        
        # Filter by confidence threshold
        keep_idx = pred['scores'] >= confidence_threshold
        boxes = pred['boxes'][keep_idx].cpu().numpy()
        scores = pred['scores'][keep_idx].cpu().numpy()
        masks = pred['masks'][keep_idx].cpu().numpy()
        
        # Apply NMS if multiple detections
        if len(boxes) > 0:
            keep = self._nms(boxes, scores, nms_threshold)
            boxes = boxes[keep]
            scores = scores[keep]
            masks = masks[keep]
        
        # Create annotated image
        annotated_image = self._annotate_image(image_array, boxes, scores)
        
        # Extract cell statistics
        cell_stats = self._extract_cell_statistics(image_array, masks, boxes)
        
        return {
            'boxes': boxes.tolist(),
            'scores': scores.tolist(),
            'masks': masks.astype(np.uint8),
            'num_cells': len(boxes),
            'image': image_array,
            'annotated_image': annotated_image,
            'cell_statistics': cell_stats,
            'cell_count': len(boxes),
            'cell_density': self._calculate_cell_density(image_array, masks)
        }
    
    def _nms(self, boxes, scores, threshold):
        """Non-maximum suppression."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def _annotate_image(self, image, boxes, scores, color=(0, 255, 0)):
        """Add bounding boxes to image."""
        annotated = image.copy()
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f'{score:.2f}', (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return annotated
    
    def _extract_cell_statistics(self, image, masks, boxes):
        """Extract statistics from detected cells."""
        stats = {
            'cell_areas': [],
            'cell_intensities': [],
            'cell_circularity': []
        }
        
        for mask, (x1, y1, x2, y2) in zip(masks, boxes):
            # Clip mask to valid region
            mask = mask[0] if len(mask.shape) > 2 else mask
            mask = (mask > 0.5).astype(np.uint8)
            
            # Calculate area
            area = np.sum(mask)
            stats['cell_areas'].append(float(area))
            
            # Calculate mean intensity in masked region
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            roi = image[y1:y2, x1:x2]
            if roi.size > 0:
                intensity = np.mean(roi)
                stats['cell_intensities'].append(float(intensity))
            
            # Calculate circularity
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    stats['cell_circularity'].append(float(circularity))
        
        return stats
    
    def _calculate_cell_density(self, image, masks):
        """Calculate cell density (cells per unit area)."""
        h, w = image.shape[:2]
        image_area = h * w
        cell_count = len(masks)
        return cell_count / (image_area / 1000000.0)  # cells per million pixels


def get_mask_rcnn_detector(fine_tuned_path=None):
    """
    Get or create cached Mask R-CNN detector.
    
    Args:
        fine_tuned_path: Path to fine-tuned model weights
        
    Returns:
        MaskRCNNDetector instance
    """
    global MASK_RCNN_MODEL
    
    if MASK_RCNN_MODEL is None:
        if fine_tuned_path is None:
            fine_tuned_path = os.path.join(
                str(settings.BASE_DIR), 'model', 'mask_rcnn_finetuned.pth'
            )
        
        MASK_RCNN_MODEL = MaskRCNNDetector(
            model_path=fine_tuned_path,
            pretrained=True,
            num_classes=2
        )
    
    return MASK_RCNN_MODEL


def detect_cells_in_image(image_path, confidence_threshold=0.5):
    """
    Convenience function to detect cells in image.
    
    Args:
        image_path: Path to blood smear image
        confidence_threshold: Minimum detection confidence
        
    Returns:
        Detection results dictionary
    """
    detector = get_mask_rcnn_detector()
    return detector.detect(image_path, confidence_threshold=confidence_threshold)
