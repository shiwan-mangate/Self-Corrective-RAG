from typing import List, Tuple
from ingestion.models import EnrichedElement

class OverlapCalculator:
    """
    Dynamically calculates token-based overlap for chunk transitions.
    Prevents massive elements from accidentally starving the token budget 
    of the subsequent chunk.
    """

    def __init__(self, max_tokens: int, overlap_percent: float = 0.1):
        """
        :param max_tokens: The maximum token budget for a single chunk.
        :param overlap_percent: The ideal percentage of max_tokens to carry over (e.g., 0.10).
        """
        self.max_tokens = max_tokens
        self.target_overlap = int(max_tokens * overlap_percent)
        
        self.max_single_overlap = int(max_tokens * 0.3)

    def get_overlap(self, elements: List[EnrichedElement]) -> Tuple[List[EnrichedElement], int]:
        """
        Works backward through the elements, accumulating them until the 
        target token overlap is reached.
        
        :param elements: The list of elements from the just-finalized chunk.
        :return: A tuple of (overlapping_elements, total_overlapping_tokens).
        """
        if self.target_overlap <= 0 or not elements:
            return [], 0

        overlap_elements: List[EnrichedElement] = []
        overlap_tokens = 0
        
        for el in reversed(elements):
            tokens = el.metadata["statistics"]["token_count"] 

            if overlap_tokens + tokens > self.target_overlap:
                
                if overlap_elements:
                    break
                
                if tokens <= self.max_single_overlap:
                    overlap_elements.insert(0, el)
                    overlap_tokens += tokens
                    
                break  
                
            overlap_elements.insert(0, el)
            overlap_tokens += tokens
            
        return overlap_elements, overlap_tokens