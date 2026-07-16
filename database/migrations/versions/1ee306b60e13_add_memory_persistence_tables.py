"""add memory persistence tables

Revision ID: 1ee306b60e13
Revises: 51f278e7250c
Create Date: 2026-07-13 10:06:30.847446
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "1ee306b60e13"
down_revision: Union[str, Sequence[str], None] = "51f278e7250c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create Memory persistence tables."""

    # ==========================================
    # Chat Sessions
    # ==========================================

    op.create_table(
        "chat_sessions",
        sa.Column(
            "session_id",
            sa.String(length=100),
            nullable=False,
            comment="Unique identifier for the active connection/session.",
        ),
        sa.Column(
            "user_id",
            sa.String(length=100),
            nullable=True,
            comment="Optional identifier to link sessions to a specific user profile.",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_activity",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="Automatically updates via SQL whenever the row is modified.",
        ),
        sa.Column(
            "message_count",
            sa.Integer(),
            nullable=False,
            comment="Telemetry metric tracking the volume of activity in this session.",
        ),
        sa.Column(
            "active",
            sa.Boolean(),
            nullable=False,
            comment="Lifecycle flag. False indicates the session is expired or archived.",
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
            comment="Flexible storage for browser info, client version, locale, etc.",
        ),
        sa.PrimaryKeyConstraint("session_id"),
    )

    op.create_index(
        "ix_chat_sessions_active",
        "chat_sessions",
        ["active"],
        unique=False,
    )

    op.create_index(
        "ix_chat_sessions_last_activity",
        "chat_sessions",
        ["last_activity"],
        unique=False,
    )

    op.create_index(
        op.f("ix_chat_sessions_user_id"),
        "chat_sessions",
        ["user_id"],
        unique=False,
    )

    # ==========================================
    # Conversation Messages
    # ==========================================

    op.create_table(
        "conversation_messages",
        sa.Column(
            "message_id",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "session_id",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "role",
            sa.String(length=20),
            nullable=False,
        ),
        sa.Column(
            "content",
            sa.Text(),
            nullable=False,
            comment="Conversation message content.",
        ),
        sa.Column(
            "tokens",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["chat_sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("message_id"),
    )

    op.create_index(
        op.f("ix_conversation_messages_session_id"),
        "conversation_messages",
        ["session_id"],
        unique=False,
    )

    op.create_index(
        "ix_conversation_messages_session_time",
        "conversation_messages",
        ["session_id", "timestamp"],
        unique=False,
    )

    # ==========================================
    # Conversation Summaries
    # ==========================================

    op.create_table(
        "conversation_summaries",
        sa.Column(
            "session_id",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "summary_text",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "covered_messages",
            sa.Integer(),
            nullable=False,
            comment="Running total of messages compressed into the active summary.",
        ),
        sa.Column(
            "summary_version",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "model_name",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "last_query_id",
            sa.String(length=100),
            nullable=True,
            comment="Optional distributed trace identifier for summary generation.",
        ),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["chat_sessions.session_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("session_id"),
    )


def downgrade() -> None:
    """Remove Memory persistence tables."""

    # Child table first
    op.drop_table("conversation_summaries")

    op.drop_index(
        "ix_conversation_messages_session_time",
        table_name="conversation_messages",
    )

    op.drop_index(
        op.f("ix_conversation_messages_session_id"),
        table_name="conversation_messages",
    )

    op.drop_table("conversation_messages")

    # Parent table last
    op.drop_index(
        op.f("ix_chat_sessions_user_id"),
        table_name="chat_sessions",
    )

    op.drop_index(
        "ix_chat_sessions_last_activity",
        table_name="chat_sessions",
    )

    op.drop_index(
        "ix_chat_sessions_active",
        table_name="chat_sessions",
    )

    op.drop_table("chat_sessions")