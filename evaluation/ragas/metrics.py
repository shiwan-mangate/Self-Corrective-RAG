from enum import Enum
from typing import List, Dict

class RagasMetric(str, Enum):
    """
    The core assessment metrics supported by our platform.
    Abstracted from the underlying third-party library to prevent vendor lock-in.
    """
  
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"

    
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"


class RagasEvaluationMode(str, Enum):
    """
    Defines the execution context for the assessment engine.
    """
    LIVE = "live"          
    BENCHMARK = "benchmark" 



EVALUATION_METRICS_MAP: Dict[RagasEvaluationMode, List[RagasMetric]] = {
    RagasEvaluationMode.LIVE: [
        RagasMetric.FAITHFULNESS,
        RagasMetric.ANSWER_RELEVANCY
    ],
    RagasEvaluationMode.BENCHMARK: [
        RagasMetric.FAITHFULNESS,
        RagasMetric.ANSWER_RELEVANCY,
        RagasMetric.CONTEXT_PRECISION,
        RagasMetric.CONTEXT_RECALL
    ]
}