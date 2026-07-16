import logging
from typing import List

from generation.models import (
    GenerationRequest,
    PromptPackage,
    GeneratedAnswer,
    GenerationCitation,
    GenerationMetadata,
    GenerationResponse
)
from generation.constants import DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """
    The final assembly boundary for the Generation Subsystem.
    
    Responsibility:
    Takes the scattered domain objects (Request, Prompt, Answer, Citations, Formatted Text) 
    and compiles them into a single, cohesive GenerationResponse API contract.
    Does not perform logic, search, or generation. Purely structural packaging.
    """

    def build(
        self,
        request: GenerationRequest,
        prompt_package: PromptPackage,
        generated_answer: GeneratedAnswer,
        resolved_citations: List[GenerationCitation],
        formatted_text: str
    ) -> GenerationResponse:
        """
        Compiles the final API payload by bridging Synthesis data with Read-side telemetry.
        """
        
        documents_used = self._build_documents_used(resolved_citations)
        
        
        generation_metadata = self._build_metadata(generated_answer, prompt_package)
        
      
        logger.info(
            f"Response Assembly Complete | "
            f"Citations: {len(resolved_citations)} | "
            f"Docs Used: {len(documents_used)} | "
            f"Answer Length: {len(formatted_text)} chars | "
            f"Tokens: {generated_answer.token_usage.total_tokens} | "
            f"Latency: {generation_metadata.latency_ms}ms"
        )
        
        
        return GenerationResponse(
            answer=formatted_text,
            citations=resolved_citations,
            documents_used=documents_used,
            generation_metadata=generation_metadata,
            retrieval_metadata=request.retrieval_metadata  
        )

    def _build_documents_used(self, citations: List[GenerationCitation]) -> List[str]:
        """
        Builds a clean, deduplicated list of document names/sources for the UI.
        Matches the 'documents_used' field requirement in GenerationResponse.
        
        Example Output: ["Deep Learning.pdf (Page 4)", "LangGraph Guide.md"]
        """
        docs = []
        seen = set()
        
        for cit in citations:
           
            doc_name = cit.title or cit.source or cit.document_id
            
           
            if cit.page:
                doc_name = f"{doc_name} (Page {cit.page})"
                
            if doc_name not in seen:
                seen.add(doc_name)
                docs.append(doc_name)
                
        return docs

    def _build_metadata(self, generated_answer: GeneratedAnswer, prompt_package: PromptPackage) -> GenerationMetadata:
        """
        Wraps the execution telemetry into the final GenerationMetadata model.
        """
        return GenerationMetadata(
            latency_ms=generated_answer.generation_time_ms,
            token_usage=generated_answer.token_usage,
            finish_reason=generated_answer.finish_reason,
            model_name=DEFAULT_MODEL_NAME,     
            temperature=DEFAULT_TEMPERATURE,   
            template_name=prompt_package.template_name  
        )