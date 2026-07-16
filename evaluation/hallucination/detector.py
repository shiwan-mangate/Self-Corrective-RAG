import time
import logging
from enum import Enum


from evaluation.models import EvaluationRequest, HallucinationResult, HallucinationRisk
from evaluation.constants import MAX_EVALUATION_TIME_MS
from evaluation.hallucination.base import BaseHallucinationDetector
from evaluation.hallucination.prompts import HALLUCINATION_SYSTEM_PROMPT, HALLUCINATION_USER_PROMPT


from ai.base_judge_service import BaseAIJudgeService

logger = logging.getLogger(__name__)


class HallucinationFailureReason(str, Enum):
    """Standard failure reasons for Hallucination Detection."""
    API_ERROR = "API_ERROR"
    INVALID_JSON = "INVALID_JSON"
    TIMEOUT = "TIMEOUT"
    EMPTY_ANSWER = "EMPTY_ANSWER"
    EMPTY_CONTEXT = "EMPTY_CONTEXT"


class LLMHallucinationDetector(BaseHallucinationDetector):
    """
    Executes Hallucination Detection using a dedicated LLM Judge.
    """

    def __init__(self, judge_service: BaseAIJudgeService):
        self.judge_service = judge_service

    def detect(self, request: EvaluationRequest) -> HallucinationResult:
        start_time = time.perf_counter()

        context = request.context.strip()
        answer = request.answer.strip()

        if not answer:
            logger.warning("Hallucination detection skipped: answer is empty.")
            return self._build_fallback_result(
                reason="Generated answer is empty.",
                failure_code=HallucinationFailureReason.EMPTY_ANSWER,
            )
            
        if not context:
            logger.warning("Hallucination detection skipped: context is empty.")
            return self._build_fallback_result(
                reason="Retrieved context is empty.",
                failure_code=HallucinationFailureReason.EMPTY_CONTEXT,
            )

       
        user_prompt = HALLUCINATION_USER_PROMPT.format(
            original_question=request.original_query,
            optimized_question=request.optimized_query,
            context=context,
            answer=answer,
        )

        try:
            
            result: HallucinationResult = self.judge_service.evaluate(
                system_prompt=HALLUCINATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                query_id=request.query_id,
                timeout_ms=MAX_EVALUATION_TIME_MS,
                response_model=HallucinationResult,
            )

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

          
            logger.info(
                "Hallucination Detection Complete | "
                f"Has Hallucination={result.has_hallucination} | "
                f"Latency={latency_ms}ms | "
                f"Answer Length={len(answer)} chars | "
                f"Hallucinated Claims={len(result.hallucinated_claims)}"
            )

            return result

        except TimeoutError:
            logger.exception("Hallucination Judge timed out.")
            return self._build_fallback_result(
                reason="Judge execution exceeded timeout.",
                failure_code=HallucinationFailureReason.TIMEOUT,
            )

        except Exception:
            logger.exception("Hallucination detection failed.")
            return self._build_fallback_result(
                reason="Judge service returned an invalid response or encountered an API failure.",
                failure_code=HallucinationFailureReason.API_ERROR,
            )

    def _build_fallback_result(
        self,
        reason: str,
        failure_code: HallucinationFailureReason,
    ) -> HallucinationResult:
        """
        Fail-Closed: Always assumes a hallucination exists if the evaluator crashes.
        """
        return HallucinationResult(
            has_hallucination=True,
            hallucinated_claims=[
                f"System Verification Interrupted: [{failure_code.value}]"
            ],
            reasoning=f"Fail-Closed Triggered: {reason}",
            risk=HallucinationRisk.HIGH,
        )