"""initial schema

Revision ID: 7651905d3dac
Revises: 
Create Date: 2026-07-07 11:07:05.683320

"""
from pgvector.sqlalchemy import Vector

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '7651905d3dac'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ======================================================
    # document_chunks
    # ======================================================

    op.create_table(
        "document_chunks",

        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),

        sa.Column(
            "chunk_id",
            sa.String(),
            nullable=False,
        ),

        sa.Column(
            "document_id",
            sa.String(),
            nullable=False,
        ),

        sa.Column(
            "text",
            sa.Text(),
            nullable=False,
        ),

        sa.Column(
            "embedding",
            Vector(384),
            nullable=False,
        ),

        sa.Column(
            "checksum",
            sa.String(),
            nullable=False,
        ),

        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),

        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),

        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_document_chunks_chunk_id",
        "document_chunks",
        ["chunk_id"],
        unique=True,
    )

    op.create_index(
        "ix_document_chunks_document_id",
        "document_chunks",
        ["document_id"],
    )

    op.create_index(
        "ix_document_chunks_checksum",
        "document_chunks",
        ["checksum"],
    )

    # ======================================================
    # evaluation_runs
    # ======================================================

    op.create_table(
        "evaluation_runs",

        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("query_id", sa.String(length=100), nullable=False),
        sa.Column(
            "evaluation_timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),

        sa.Column("original_query", sa.Text(), nullable=False),
        sa.Column("optimized_query", sa.Text(), nullable=False),
        sa.Column("generated_answer", sa.Text(), nullable=False),

        sa.Column("is_grounded", sa.Boolean(), nullable=False),
        sa.Column(
            "supported_claims",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "unsupported_claims",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("grounding_reason", sa.Text(), nullable=False),

        sa.Column("has_hallucination", sa.Boolean(), nullable=False),
        sa.Column(
            "hallucinated_claims",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("hallucination_reason", sa.Text(), nullable=False),

        sa.Column("overall_confidence", sa.Float(), nullable=False),
        sa.Column("retrieval_confidence", sa.Float(), nullable=False),
        sa.Column("grounding_confidence", sa.Float(), nullable=False),
        sa.Column("hallucination_risk", sa.String(20), nullable=False),
        sa.Column("confidence_reason", sa.Text(), nullable=False),

        sa.Column("faithfulness", sa.Float(), nullable=False),
        sa.Column("answer_relevancy", sa.Float(), nullable=False),
        sa.Column("context_precision", sa.Float(), nullable=True),
        sa.Column("context_recall", sa.Float(), nullable=True),
        sa.Column("ragas_latency_ms", sa.Float(), nullable=True),

        sa.Column("decision", sa.String(30), nullable=False),
        sa.Column("retry_recommendation", sa.String(50), nullable=True),
        sa.Column(
            "warnings",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),

        sa.Column("retrieval_latency_ms", sa.Float(), nullable=False),
        sa.Column("generation_latency_ms", sa.Float(), nullable=False),
        sa.Column("evaluation_latency_ms", sa.Float(), nullable=False),

        sa.Column("total_tokens", sa.Integer(), nullable=False),

        sa.Column("generation_model_name", sa.String(100), nullable=False),
        sa.Column("judge_model_name", sa.String(100), nullable=False),

        sa.Column("evaluation_mode", sa.String(20), nullable=False),
        sa.Column("evaluation_version", sa.String(20), nullable=False),

        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "idx_evaluation_runs_query_id",
        "evaluation_runs",
        ["query_id"],
    )

    op.create_index(
        "idx_evaluation_runs_timestamp",
        "evaluation_runs",
        ["evaluation_timestamp"],
    )

    op.create_index(
        "idx_evaluation_runs_decision",
        "evaluation_runs",
        ["decision"],
    )

    op.create_index(
        "idx_evaluation_runs_retry",
        "evaluation_runs",
        ["retry_recommendation"],
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_index(
        "idx_evaluation_runs_retry",
        table_name="evaluation_runs",
    )

    op.drop_index(
        "idx_evaluation_runs_decision",
        table_name="evaluation_runs",
    )

    op.drop_index(
        "idx_evaluation_runs_timestamp",
        table_name="evaluation_runs",
    )

    op.drop_index(
        "idx_evaluation_runs_query_id",
        table_name="evaluation_runs",
    )

    op.drop_table("evaluation_runs")

    op.drop_index(
        "ix_document_chunks_checksum",
        table_name="document_chunks",
    )

    op.drop_index(
        "ix_document_chunks_document_id",
        table_name="document_chunks",
    )

    op.drop_index(
        "ix_document_chunks_chunk_id",
        table_name="document_chunks",
    )

    op.drop_table("document_chunks")
