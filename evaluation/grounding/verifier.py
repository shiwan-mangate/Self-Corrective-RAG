import time
import logging
from enum import Enum

from evaluation.models import EvaluationRequest, GroundingResult

from evaluation.constants import MAX_EVALUATION_TIME_MS
from evaluation.grounding.base import BaseGroundingVerifier
from evaluation.grounding.prompts import (
    GROUNDING_SYSTEM_PROMPT,
    GROUNDING_USER_PROMPT,
)

from ai.base_judge_service import BaseAIJudgeService

logger = logging.getLogger(__name__)


class GroundingFailureReason(str, Enum):
    """Standard failure reasons for Grounding Verification."""
    API_ERROR = "API_ERROR"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    TIMEOUT = "TIMEOUT"
    EMPTY_CONTEXT = "EMPTY_CONTEXT"
    APOLOGY_DETECTED = "APOLOGY_DETECTED"


class LLMGroundingVerifier(BaseGroundingVerifier):
    """
    Executes Grounding Verification using a dedicated LLM Judge.

    Responsibility:
    - Build evaluation prompts.
    - Invoke the Judge service.
    - Measure latency.
    - Return a valid GroundingResult.
    - Fail closed if anything goes wrong.
    """

    def __init__(self, judge_service: BaseAIJudgeService):
        self.judge_service = judge_service

    def verify(self, request: EvaluationRequest) -> GroundingResult:
        start_time = time.perf_counter()

        context = request.context.strip()
        answer = request.answer.strip()

        # ---> THE FIX: INTERCEPT APOLOGIES O(1) <---
        # Prevents the judge from giving a "passing" grade to an apology.
        if "I apologize" in answer or "I could not generate" in answer:
            logger.info("Grounding Verification intercepted an apology. Failing fast.")
            return self._build_fallback_result(
                reason="The generated answer was a refusal to answer due to missing context.",
                failure_code=GroundingFailureReason.APOLOGY_DETECTED,
            )

        if not context:
            logger.warning("Grounding verification skipped because context is empty.")
            return self._build_fallback_result(
                reason="Retrieved context is empty.",
                failure_code=GroundingFailureReason.EMPTY_CONTEXT,
            )

        user_prompt = GROUNDING_USER_PROMPT.format(
            context=context,
            answer=answer,
        )

        try:
            result: GroundingResult = self.judge_service.evaluate(
                system_prompt=GROUNDING_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                query_id=request.query_id,
                timeout_ms=MAX_EVALUATION_TIME_MS,
                response_model=GroundingResult, 
            )

            latency_ms = round(
                (time.perf_counter() - start_time) * 1000,
                2,
            )

            logger.info(
                "Grounding Verification Complete | "
                f"Grounded={result.is_grounded} | "
                f"Latency={latency_ms}ms | "
                f"Unsupported Claims={len(result.unsupported_claims)} | "
                f"Confidence={result.confidence:.2f}"
            )

            return result

        except TimeoutError:
            logger.exception("Grounding Judge timed out.")
            return self._build_fallback_result(
                reason="Judge execution exceeded timeout.",
                failure_code=GroundingFailureReason.TIMEOUT,
            )

        except Exception:
            logger.exception("Grounding verification failed.")
            return self._build_fallback_result(
                reason="Judge service returned an invalid response or encountered an API failure.",
                failure_code=GroundingFailureReason.API_ERROR,
            )

    def _build_fallback_result(
        self,
        reason: str,
        failure_code: GroundingFailureReason,
    ) -> GroundingResult:
        """
        Fail-closed helper.

        If evaluation cannot be completed, the answer is treated as
        NOT grounded so that downstream Self-Healing can decide what
        corrective action to take.
        """
        return GroundingResult(
            is_grounded=False,
            supported_claims=[],
            unsupported_claims=[
                f"Grounding verification failed ({failure_code.value})."
            ],
            confidence=0.0,
            explanation=reason,
        )