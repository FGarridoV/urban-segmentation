import torch
import numpy as np
from PIL import Image
from typing import List, Dict
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from .base_model import SemanticSegmenter

class Mask2Former(SemanticSegmenter):
    """
    Implementation of Mask2Former (Swin-Large) for Semantic Segmentation.
    """
    
    def load_model(self):
        self.model_id = "facebook/mask2former-swin-large-ade-semantic"
        print(f"Loading processor and model: {self.model_id} on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        
    def process_batch(self, image_paths: List[str]) -> List[Dict[str, float]]:
        images = []
        target_sizes = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            images.append(img)
            target_sizes.append(img.size[::-1]) # (height, width)
            
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Post-process semantic segmentation
        results = self.processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        
        batch_results = []
        for result in results:
            pred_seg = result.cpu().numpy()
            total_area = pred_seg.shape[0] * pred_seg.shape[1]
            
            # Count pixels per class
            unique_labels, counts = np.unique(pred_seg, return_counts=True)
            
            class_areas = {}
            for label_idx, count in zip(unique_labels, counts):
                label_name = self.model.config.id2label[label_idx]
                area_percent = (count / total_area) * 100.0
                class_areas[label_name] = area_percent
                
            batch_results.append(class_areas)
            
        return batch_results
