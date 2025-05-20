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
            print(f"Successfully loaded YOLO model from {pretrained_path}")
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
                        # Execute YOLO model with feature extraction enabled
                        results = self.model(processed_images, verbose=False, stream=False)
                        outputs = self._process_yolo_outputs(results)
                        
                        # Debug: check what's in the outputs
                        print("\nYOLO teacher outputs:")
                        for key, value in outputs.items():
                            if isinstance(value, list):
                                print(f"  - {key}: List with {len(value)} elements")
                                if value and hasattr(value[0], 'shape'):
                                    print(f"    First element shape: {value[0].shape}")
                            else:
                                print(f"  - {key}: {type(value)}")
                        
                        # If there are no feature maps, try to extract them directly
                        if 'feature_maps' not in outputs or not outputs['feature_maps']:
                            print("No feature maps found in YOLO output, attempting to extract features from backbone")
                            # Try to get features directly from the model
                            return self._extract_features_manually(images)
                        
                        return outputs
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
            # For ultralytics YOLO models, try to extract features
            # Check if we can access features directly from the model
            # YOLOv8 models store feature maps in different places depending on version
            if hasattr(results[0], 'feature_maps'):
                teacher_outputs['feature_maps'] = [r.feature_maps for r in results]
                print(f"Found feature_maps in results object")
            elif hasattr(results[0], 'features'):
                teacher_outputs['feature_maps'] = [r.features for r in results]
                print(f"Found features in results object")
            elif hasattr(results[0], 'probs'):
                teacher_outputs['probs'] = [r.probs for r in results]
                print(f"Found probs in results object")
                
            # Try to access the model's feature extractor
            if not teacher_outputs.get('feature_maps'):
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'backbone'):
                    print("Trying to extract features from YOLO backbone directly")
                    # This is a placeholder - actual implementation would depend on YOLO version
                    # In a real scenario, you'd run images through the backbone manually
            
            # Add detection boxes and scores if available
            boxes = []
            scores = []
            class_scores = []
            
            for r in results:
                if hasattr(r, 'boxes'):
                    if hasattr(r.boxes, 'xyxy') and len(r.boxes.xyxy) > 0:
                        boxes.append(r.boxes.xyxy)
                        print(f"Found boxes: {r.boxes.xyxy.shape}")
                    
                    if hasattr(r.boxes, 'conf') and len(r.boxes.conf) > 0:
                        scores.append(r.boxes.conf)
                        print(f"Found confidence scores: {r.boxes.conf.shape}")
                    
                    if hasattr(r.boxes, 'cls') and len(r.boxes.cls) > 0:
                        class_scores.append(r.boxes.cls)
                        print(f"Found class scores: {r.boxes.cls.shape}")
                    
                # Try to get raw predictions if available
                if hasattr(r, 'probs'):
                    teacher_outputs['probs'] = r.probs
                    print(f"Found probability outputs: {r.probs.shape if hasattr(r.probs, 'shape') else 'unknown shape'}")
            
            if boxes:
                teacher_outputs['boxes'] = boxes
                teacher_outputs['scores'] = scores
                if class_scores:
                    teacher_outputs['class_scores'] = class_scores
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
            print("No outputs extracted from YOLO model, using empty fallback")
            teacher_outputs = {
                'feature_maps': [],
                'boxes': [],
                'scores': []
            }
            
        return teacher_outputs
        
    def _extract_features_manually(self, images):
        """
        Extract features manually from the YOLO backbone
        """
        try:
            device = next(self.model.parameters()).device
            # For YOLOv8/v11, we need to access the backbone
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'backbone'):
                print("Extracting features from backbone")
                # Convert tensor images to format expected by YOLO backbone
                processed_images = []
                for img in images:
                    img = img.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
                    img = (img * 255).astype(np.uint8)  # Back to 0-255 range
                    processed_images.append(img)
                
                # Try to get features - this is just a placeholder
                # In production you'd need to properly run the backbone with the right preprocessing
                features = []
                return {'feature_maps': features, 'boxes': [], 'scores': []}
            else:
                print("Cannot extract features: backbone not accessible")
        except Exception as e:
            print(f"Error extracting features manually: {e}")
        
        # Return empty outputs if feature extraction fails
        return {
            'feature_maps': [],
            'boxes': [],
            'scores': []
        }