import time
import logging
from typing import List, Dict
from datasets import Dataset
import sys
from unittest.mock import MagicMock
from evaluation.models import RagasResult
from evaluation.ragas.metrics import RagasMetric, RagasEvaluationMode, EVALUATION_METRICS_MAP
from ai.base_ragas_service import BaseRagasLLM, BaseRagasEmbeddings
from shared.exceptions import EvaluationExecutionError
logger = logging.getLogger(__name__)

# --- START BUGFIX FOR RAGAS ---
# Ragas eagerly imports VertexAI which breaks on newer Langchain versions.
# We mock it here so the import succeeds, as we only use Groq anyway.
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
# --- END BUGFIX ---


class RagasEvaluator:
    """
    The execution engine for third-party RAGAS evaluation.
    
    Responsibility:
    Accept a prepared Hugging Face Dataset, execute the evaluation using the 
    appropriate metrics for the selected mode, translate the library's output 
    into our domain RagasResult, and hide all RAGAS details from the pipeline.
    """

    def __init__(self, llm: BaseRagasLLM, embeddings: BaseRagasEmbeddings):
        self.llm = llm
        self.embeddings = embeddings

    def evaluate(self, dataset: Dataset, mode: RagasEvaluationMode) -> RagasResult:
        """
        Executes the RAGAS framework against a validated dataset.
        """
        if len(dataset) == 0:
            raise ValueError("RagasEvaluator cannot evaluate an empty dataset.")

        start_time = time.perf_counter()

        try:
          
            domain_metrics = self._get_metrics(mode)
            ragas_metrics = self._translate_metrics(domain_metrics)
            
            raw_result = self._execute_ragas(dataset, ragas_metrics)
            
            scores = self._extract_scores(raw_result)
            
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            result = self._build_result(scores, latency_ms)
            
            self._log_success(mode, result, len(ragas_metrics))
            
            return result

        except Exception as e:
            logger.exception(f"RAGAS evaluation failed during {mode.value.upper()} mode.")
            raise EvaluationExecutionError(f"RAGAS framework execution failed: {str(e)}") from e

    def _get_metrics(self, mode: RagasEvaluationMode) -> List[RagasMetric]:
        """Determines which metrics to execute based on the evaluation profile."""
        return EVALUATION_METRICS_MAP[mode]

    def _translate_metrics(self, domain_metrics: List[RagasMetric]) -> List:
        """
        Translates our domain enums into RAGAS Python objects.
        Uses Lazy Loading to prevent heavy ML library initialization at module load.
        """
       
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        mapping = {
            RagasMetric.FAITHFULNESS: faithfulness,
            RagasMetric.ANSWER_RELEVANCY: answer_relevancy,
            RagasMetric.CONTEXT_PRECISION: context_precision,
            RagasMetric.CONTEXT_RECALL: context_recall,
        }
        return [mapping[m] for m in domain_metrics]

    def _execute_ragas(self, dataset: Dataset, metrics: List):
        """Invokes the external RAGAS library."""
        from ragas import evaluate
        
        return evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            raise_exceptions=True, 
        )

    def _extract_scores(self, raw_result) -> Dict[str, float]:
        """
        Version-agnostic extractor for RAGAS output.
        RAGAS objects change frequently; this quarantines the parsing logic.
        """
        
        if hasattr(raw_result, "to_pandas"):
            df = raw_result.to_pandas()
            if not df.empty:
                return df.iloc[0].to_dict()
                
        
        if hasattr(raw_result, "keys"):
            return {k: raw_result[k] for k in raw_result.keys()}
            
        raise RuntimeError(f"Unrecognized RAGAS result format: {type(raw_result)}")

    def _build_result(self, scores: Dict[str, float], latency_ms: float) -> RagasResult:
        """Hydrates the strict Pydantic domain model."""
        return RagasResult(
            faithfulness_score=scores.get("faithfulness", 0.0),
            answer_relevancy_score=scores.get("answer_relevancy", 0.0),
            context_precision_score=scores.get("context_precision", None),
            context_recall_score=scores.get("context_recall", None),
            latency_ms=latency_ms
        )

    def _log_success(self, mode: RagasEvaluationMode, result: RagasResult, metric_count: int):
        """Rich observability logging, adapting output based on available metrics."""
        log_msg = (
            f"RAGAS Complete | Mode={mode.value.upper()} | "
            f"Metrics={metric_count} | Latency={result.latency_ms}ms | "
            f"Faith={result.faithfulness_score:.2f} | "
            f"Relevancy={result.answer_relevancy_score:.2f}"
        )
        
       
        if mode == RagasEvaluationMode.BENCHMARK:
            cp = result.context_precision_score or 0.0
            cr = result.context_recall_score or 0.0
            log_msg += f" | Precision={cp:.2f} | Recall={cr:.2f}"
            
        logger.info(log_msg)