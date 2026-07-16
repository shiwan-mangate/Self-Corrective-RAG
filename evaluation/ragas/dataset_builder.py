import logging
from typing import Optional, List, Dict, Any
from datasets import Dataset
from evaluation.models import EvaluationRequest
logger = logging.getLogger(__name__)

class RagasDatasetBuilder:
    """
    Translates our internal domain models into a Hugging Face Dataset.
    
    Responsibility:
    Acts as the data transformation boundary between our platform and 
    third-party evaluation frameworks that expect tabular/columnar formats.
    """
    
    def build(self, request: EvaluationRequest, ground_truth: Optional[str] = None) -> Dataset:
        
        if not request.original_query.strip():
            raise ValueError("EvaluationRequest missing required field: 'original_query' cannot be empty.")
            
        # Safety fallback in case generation failed but we still want to evaluate
        answer_text = request.answer.strip() if request.answer and request.answer.strip() else "NO_ANSWER"
        
        # If retrieved_chunks is empty (e.g., Out of Domain), fallback to the string context
        contexts_list: List[str] = request.retrieved_chunks if request.retrieved_chunks else [request.context]
        
        # HuggingFace Datasets requires the context list to have at least one string
        if not contexts_list:
            contexts_list = [""]

        dataset_columns: Dict[str, List[Any]] = {
            "question": [request.original_query],
            "answer": [answer_text],
            "contexts": [contexts_list]
        }

        if ground_truth is not None:
            dataset_columns["ground_truth"] = [ground_truth]

        try:
            dataset = Dataset.from_dict(dataset_columns)
            
            mode = "BENCHMARK" if ground_truth is not None else "LIVE"
            logger.info(
                f"RAGAS Dataset Built | "
                f"Mode={mode} | "
                f"Contexts={len(contexts_list)} | "
                f"GroundTruth={ground_truth is not None}"
            )
            return dataset 
        except Exception:
            logger.exception("Hugging Face Dataset construction failed.")
            raise