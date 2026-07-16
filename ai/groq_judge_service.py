import json
import logging
import re
from typing import TypeVar, Type

from groq import Groq
from pydantic import BaseModel

from ai.base_judge_service import BaseAIJudgeService
from config.settings import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class GroqJudgeService(BaseAIJudgeService):
    """
    Infrastructure implementation for high-capability LLM judges.

    Responsibilities
    ----------------
    - Execute the API call
    - Enforce JSON mode
    - Remove accidental markdown fences
    - Parse JSON
    - Validate against the supplied Pydantic model
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.0,
    ):
        self.api_key = api_key or settings.GROQ_API_KEY

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please configure it in your Render Environment Variables."
            )

        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.temperature = temperature

    def evaluate(
        self,
        system_prompt: str,
        user_prompt: str,
        query_id: str,
        timeout_ms: int,
        response_model: Type[T],
    ) -> T:
        """
        Execute the Judge model and return a validated Pydantic object.
        """
        logger.info(f"Groq Judge evaluating Query ID: {query_id}")

        timeout_seconds = timeout_ms / 1000.0

        schema = response_model.model_json_schema()

        adapted_system_prompt = (
            f"{system_prompt}\n\n"
            "Return ONLY valid JSON.\n"
            "The JSON MUST conform to the following schema:\n"
            f"{json.dumps(schema, separators=(',', ':'))}"
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            timeout=timeout_seconds,
            messages=[
                {
                    "role": "system",
                    "content": adapted_system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        raw_text = completion.choices[0].message.content

        if not raw_text:
            raise RuntimeError("Judge returned an empty response.")

        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw_text.strip(),
            flags=re.MULTILINE,
        )

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error("Judge returned invalid JSON:\n%s", cleaned)
            raise RuntimeError("Judge returned malformed JSON.") from e

        try:
            return response_model.model_validate(parsed)
        except Exception as e:
            logger.error("Judge response failed schema validation.")
            raise RuntimeError(
                f"Judge response does not match {response_model.__name__}."
            ) from e