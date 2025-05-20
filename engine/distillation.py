import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for object detection
    Combines:
    1. Feature-level distillation (KL divergence between feature maps)
    2. Box-level distillation (L2 loss between predicted boxes)
    3. Classification-level distillation (KL divergence between class predictions)
    """
    def __init__(self, temperature=2.0, alpha=0.5, beta=0.5, gamma=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for feature distillation
        self.beta = beta    # Weight for box distillation
        self.gamma = gamma  # Weight for classification distillation
        
    def kl_loss(self, student_logits, teacher_logits):
        """
        Compute KL divergence loss between student and teacher logits
        """
        student_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_prob = F.softmax(teacher_logits / self.temperature, dim=-1)
        loss = F.kl_div(student_prob, teacher_prob, reduction='batchmean') * (self.temperature ** 2)
        return loss
        
    def feature_distillation_loss(self, student_features, teacher_features):
        """
        Distill knowledge from teacher feature maps to student feature maps
        """
        total_loss = 0.0
        
        # If features are already matched in dimension
        if isinstance(student_features, torch.Tensor) and isinstance(teacher_features, torch.Tensor):
            # Simple MSE loss if dimensions match
            if student_features.shape == teacher_features.shape:
                return F.mse_loss(student_features, teacher_features)
            else:
                # Resize student features to match teacher if needed
                student_resized = F.interpolate(student_features, size=teacher_features.shape[-2:])
                return F.mse_loss(student_resized, teacher_features)
                
        # Handle case where features come as lists or dictionaries
        if isinstance(student_features, list) and isinstance(teacher_features, list):
            # Match as many feature maps as possible
            min_len = min(len(student_features), len(teacher_features))
            for i in range(min_len):
                s_feat = student_features[i]
                t_feat = teacher_features[i]
                if s_feat.shape[-2:] != t_feat.shape[-2:]:
                    s_feat = F.interpolate(s_feat, size=t_feat.shape[-2:])
                total_loss += F.mse_loss(s_feat, t_feat)
            return total_loss / min_len
            
        return torch.tensor(0.0, device=student_features[0].device if isinstance(student_features, list) else student_features.device)
        
    def box_distillation_loss(self, student_boxes, teacher_boxes):
        """
        Distill knowledge from teacher box predictions to student box predictions
        """
        if student_boxes is None or teacher_boxes is None:
            return torch.tensor(0.0, device=student_boxes[0].device if isinstance(student_boxes, list) else student_boxes.device)
            
        # Simple implementation - L2 loss between boxes
        if isinstance(student_boxes, torch.Tensor) and isinstance(teacher_boxes, torch.Tensor):
            return F.mse_loss(student_boxes, teacher_boxes)
            
        # Handle case where boxes come as lists
        total_loss = 0.0
        count = 0
        if isinstance(student_boxes, list) and isinstance(teacher_boxes, list):
            min_len = min(len(student_boxes), len(teacher_boxes))
            for i in range(min_len):
                s_box = student_boxes[i]
                t_box = teacher_boxes[i]
                if s_box.shape == t_box.shape:
                    total_loss += F.mse_loss(s_box, t_box)
                    count += 1
            return total_loss / max(count, 1)
            
        return torch.tensor(0.0, device=student_boxes[0].device if isinstance(student_boxes, list) else student_boxes.device)
        
    def forward(self, student_outputs, teacher_outputs):
        """
        Compute distillation loss between student and teacher outputs
        Args:
            student_outputs: Dictionary containing student model outputs
            teacher_outputs: Dictionary containing teacher model outputs
        Returns:
            Total distillation loss
        """
        total_loss = 0.0
        
        # Feature distillation
        if 'feature_maps' in student_outputs and 'feature_maps' in teacher_outputs:
            feat_loss = self.feature_distillation_loss(student_outputs['feature_maps'], 
                                                     teacher_outputs['feature_maps'])
            total_loss += self.alpha * feat_loss
            
        # Box distillation
        if 'boxes' in student_outputs and 'boxes' in teacher_outputs:
            box_loss = self.box_distillation_loss(student_outputs['boxes'], 
                                                teacher_outputs['boxes'])
            total_loss += self.beta * box_loss
            
        # Classification distillation
        if 'scores' in student_outputs and 'scores' in teacher_outputs:
            if isinstance(student_outputs['scores'], torch.Tensor) and isinstance(teacher_outputs['scores'], torch.Tensor):
                cls_loss = self.kl_loss(student_outputs['scores'], teacher_outputs['scores'])
                total_loss += self.gamma * cls_loss
        
        return total_loss