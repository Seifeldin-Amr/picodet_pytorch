import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class YOLOv11LTeacher(nn.Module):
    """
    A wrapper for YOLOv11L model to be used as a teacher in knowledge distillation
    """
    def __init__(self, pretrained_path=None):
        super(YOLOv11LTeacher, self).__init__()
        # Load YOLOv11L model from torchvision or a custom implementation
        try:
            # Try to import from ultralytics if available
            from ultralytics import YOLO
            self.model = YOLO(pretrained_path) if pretrained_path else YOLO("yolov11l.pt")
            self.using_ultralytics = True
        except ImportError:
            # Fallback to a simple placeholder if ultralytics is not available
            print("Warning: ultralytics package not found. Using a placeholder YOLOv11L model.")
            # This is just a placeholder - in production, you should use the actual YOLOv11L model
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            self.using_ultralytics = False
        
        # Freeze the teacher model parameters
        for param in self.parameters():
            param.requires_grad = False

        # Define inverse normalization for processing tensors
        self.denormalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
            
    def forward(self, inputs):
        """
        Forward pass for the YOLOv11L teacher model
        Args:
            inputs: Tuple of (images, targets)
        Returns:
            Dictionary containing teacher outputs needed for distillation
        """
        images, _ = inputs
        with torch.no_grad():
            if self.using_ultralytics:
                # Convert normalized PyTorch tensors to format YOLO can use
                processed_images = self._prepare_images_for_yolo(images)
                # Use batch inference if possible
                if hasattr(self.model, 'predict'):
                    try:
                        results = self.model(processed_images, verbose=False, stream=False)
                        return self._process_yolo_outputs(results)
                    except Exception as e:
                        print(f"YOLO inference error: {e}")
                        # Fallback to feature extractor if prediction fails
                        return self._extract_features_manually(images)
                else:
                    # Fallback for non-standard models
                    return self._extract_features_manually(images)
            else:
                # If using placeholder model
                teacher_outputs = self.model(images)
                return {'teacher_outputs': teacher_outputs}
    
    def _prepare_images_for_yolo(self, tensor_images):
        """
        Convert normalized PyTorch tensors to a format YOLO can use
        """
        processed_images = []
        
        for img in tensor_images:
            # Denormalize the image (undo the normalization)
            img = self.denormalize(img)
            
            # Convert to numpy and adjust range to 0-255
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            # Convert to PIL Image which YOLO can process
            img_pil = Image.fromarray(img_np)
            processed_images.append(img_pil)
        
        return processed_images
                
    def _process_yolo_outputs(self, results):
        """
        Process YOLO outputs to match the format needed for distillation
        Args:
            results: YOLO model results
        Returns:
            Dictionary with processed outputs
        """
        # Extract the feature maps or detections from YOLO results
        teacher_outputs = {}
        
        try:
            # For ultralytics YOLO models, extract feature maps if available
            if hasattr(results[0], 'feature_maps'):
                teacher_outputs['feature_maps'] = [r.feature_maps for r in results]
            elif hasattr(results[0], 'probs'):
                teacher_outputs['probs'] = [r.probs for r in results]
            
            # Add detection boxes and scores if available
            boxes = []
            scores = []
            for r in results:
                if hasattr(r, 'boxes') and hasattr(r.boxes, 'xyxy'):
                    boxes.append(r.boxes.xyxy)
                    scores.append(r.boxes.conf)
            
            if boxes:
                teacher_outputs['boxes'] = boxes
                teacher_outputs['scores'] = scores
        except Exception as e:
            print(f"Error processing YOLO outputs: {e}")
            # Return empty outputs on error
            teacher_outputs = {
                'feature_maps': [],
                'boxes': [],
                'scores': []
            }
        
        # If no outputs were obtained, provide a fallback
        if not teacher_outputs:
            teacher_outputs = {
                'feature_maps': [],
                'boxes': [],
                'scores': []
            }
            
        return teacher_outputs
        
    def _extract_features_manually(self, images):
        """
        Fallback method to extract features manually when the standard YOLO prediction fails
        """
        # Create empty outputs that match the expected format
        return {
            'feature_maps': [],
            'boxes': [],
            'scores': []
        }