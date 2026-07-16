import os
import tempfile
import logging
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool

# Schemas
from schemas.document import DocumentIngestionRequest, DocumentIngestionResponse

# Dependencies & Domain Models
from api.dependencies import get_ingestion_pipeline
from ingestion.pipeline import IngestionPipeline
from ingestion.models import IngestionResult

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"]
)


ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".txt", ".log", 
    ".csv", ".md", ".markdown", ".html", ".htm"
}


@router.post("/url", response_model=DocumentIngestionResponse)
def ingest_url(
    request: DocumentIngestionRequest,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline)
) -> DocumentIngestionResponse:
    """
    Ingests a document from a remote URL.
    The underlying LoaderFactory automatically detects the 'http/https' schema
    and routes it to the URLLoader.
    """
    logger.info(f"Received URL ingestion request: {request.source}")
    
    # The pipeline is fully synchronous, so a standard `def` allows FastAPI 
    # to safely execute this in its background thread pool.
    result: IngestionResult = pipeline.ingest(request.source)
    
    return _map_to_response(result)


@router.post("/upload", response_model=DocumentIngestionResponse)
async def ingest_upload(
    file: UploadFile = File(...),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline)
) -> DocumentIngestionResponse:
    """
    Ingests a document from a direct file upload.
    Safely bridges the HTTP UploadFile streaming abstraction into a physical 
    file path so the domain LoaderFactory can process it natively.
    """
    logger.info(f"Received File upload ingestion request: {file.filename}")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
        

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415, 
            detail=f"Unsupported file extension: {file_ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )


    temp_fd, temp_path = tempfile.mkstemp(suffix=file_ext)
    
    try:
 
        with os.fdopen(temp_fd, "wb") as f:
            content = await file.read()
            f.write(content)
            
        logger.debug(f"File temporarily saved to {temp_path} for ingestion processing.")

  
        result: IngestionResult = await run_in_threadpool(pipeline.ingest, temp_path)
        
        return _map_to_response(result)

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.debug(f"Temporary file {temp_path} cleaned up successfully.")


def _map_to_response(result: IngestionResult) -> DocumentIngestionResponse:
    """
    Explicit schema mapping.
    Prevents internal domain modifications from accidentally leaking into or breaking the public API.
    """
    return DocumentIngestionResponse(
        documents_processed=result.documents_processed,
        chunks_generated=result.chunks_generated,
        chunks_persisted=result.chunks_persisted,
        warnings=result.warnings,
        elapsed_time_sec=result.elapsed_time_sec
    )