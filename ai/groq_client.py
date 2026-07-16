# ai/groq_client.py

import logging

from groq import Groq

from ai.text_generation_service import AITextGenerationService
from config.settings import settings

logger = logging.getLogger(__name__)


class GroqClient(AITextGenerationService):
    """
    Production Groq implementation of the generic AITextGenerationService.

    Responsibilities:
    - Manage Groq client
    - Perform deterministic text generation
    - Return plain generated text

    No business logic belongs here.
    """

    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        query_id: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """
        Generate text using Groq.

        Args:
            system_prompt: System instruction.
            user_prompt: User message.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text.

        Raises:
            Exception if the Groq API request fails.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.exception(f"Groq generation failed for Query ID: {query_id}")
            raise RuntimeError(f"Groq API Error: {str(e)}") from e