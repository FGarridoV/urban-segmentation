from abc import ABC, abstractmethod
from typing import List, Dict

class SemanticSegmenter(ABC):
    """
    Abstract Base Class for semantic segmentation models.
    """
    
    def __init__(self, device: str):
        self.device = device
        self.load_model()
        
    @abstractmethod
    def load_model(self):
        """Loads the model and processor onto the specified device."""
        pass
        
    @abstractmethod
    def process_batch(self, image_paths: List[str]) -> List[Dict[str, float]]:
        """
        Processes a batch of image paths and returns a list of dictionaries.
        Each dictionary maps the class name to its area percentage (0-100%).
        If a class is not present, it doesn't need to be in the dict (handled by main).
        """
        pass
