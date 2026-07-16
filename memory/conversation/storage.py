# memory/conversation/storage.py

import logging
from abc import ABC, abstractmethod
from typing import Optional, Set
from sqlalchemy.orm import Session as DBSession
from sqlalchemy.exc import SQLAlchemyError

from memory.models import ConversationHistory, ConversationMessage, ConversationSummary, MessageRole
from database.models.conversation import ConversationMessageModel, ConversationSummaryModel

logger = logging.getLogger(__name__)


class BaseConversationStorage(ABC):
    """
    Abstract contract for persisting and retrieving Conversation state.
    Ensures the Memory subsystem relies on abstractions, not databases.
    """
    @abstractmethod
    def load_history(self, session_id: str, query_id: Optional[str] = None) -> ConversationHistory:
        pass

    @abstractmethod
    def load_summary(self, session_id: str, query_id: Optional[str] = None) -> Optional[ConversationSummary]:
        pass

    @abstractmethod
    def commit_summary(
        self, 
        session_id: str, 
        summary: ConversationSummary, 
        message_ids_to_delete: Set[str],
        query_id: Optional[str] = None
    ) -> None:
        pass

    @abstractmethod
    def save_turn(
        self, 
        session_id: str, 
        user_message: ConversationMessage, 
        assistant_message: ConversationMessage,
        query_id: Optional[str] = None
    ) -> int:
        pass


class PostgresConversationStorage(BaseConversationStorage):
    """
    SQLAlchemy implementation for Conversation Storage.
    Safely executes multi-table atomic transactions and acts as a strict Anti-Corruption Layer.
    """
    def __init__(self, db: DBSession):
        self.db = db

    def load_history(self, session_id: str, query_id: Optional[str] = None) -> ConversationHistory:
        """
        Loads all un-summarized, active messages for a session chronologically.
        Does NOT calculate token/message totals. Leaves that to the HistoryService.
        """
        try:
            records = self.db.query(ConversationMessageModel)\
                .filter_by(session_id=session_id)\
                .order_by(ConversationMessageModel.timestamp.asc())\
                .all()

            messages = [
                ConversationMessage(
                    message_id=rec.message_id,
                    role=MessageRole(rec.role),
                    content=rec.content,
                    tokens=rec.tokens,
                    timestamp=rec.timestamp,
                    metadata=rec.message_metadata
                ) for rec in records
            ]

            # Return raw list wrapped in Pydantic. Totals are handled by HistoryService.
            return ConversationHistory(messages=messages)
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to load history for session {session_id} | QueryID={query_id} | Error: {str(e)}")
            raise

    def load_summary(self, session_id: str, query_id: Optional[str] = None) -> Optional[ConversationSummary]:
        """Loads the active compression state."""
        try:
            record = self.db.query(ConversationSummaryModel).filter_by(session_id=session_id).first()
            if not record:
                return None

            return ConversationSummary(
                summary=record.summary_text,
                covered_messages=record.covered_messages,
                summary_version=record.summary_version,
                generated_at=record.generated_at,
                model_name=record.model_name
            )
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to load summary for session {session_id} | QueryID={query_id} | Error: {str(e)}")
            raise

    def commit_summary(
        self, 
        session_id: str, 
        summary: ConversationSummary, 
        message_ids_to_delete: Set[str],
        query_id: Optional[str] = None
    ) -> None:
        """
        ATOMIC MULTI-TABLE TRANSACTION:
        Upserts the new summary AND deletes the compressed messages.
        The Domain model is the absolute source of truth for counts and versions.
        """
        try:
            # 1. Upsert Summary (Domain Model is the source of truth)
            record = self.db.query(ConversationSummaryModel).filter_by(session_id=session_id).first()
            if not record:
                record = ConversationSummaryModel(
                    session_id=session_id,
                    summary_text=summary.summary,
                    covered_messages=summary.covered_messages,
                    summary_version=summary.summary_version,
                    model_name=summary.model_name,
                    last_query_id=query_id
                )
                self.db.add(record)
            else:
                record.summary_text = summary.summary
                record.covered_messages = summary.covered_messages
                record.summary_version = summary.summary_version
                record.model_name = summary.model_name
                record.last_query_id = query_id

            # 2. Prune old messages
            if message_ids_to_delete:
                self.db.query(ConversationMessageModel)\
                    .filter(ConversationMessageModel.message_id.in_(message_ids_to_delete))\
                    .delete(synchronize_session=False)

            # 3. Single Commit
            self.db.commit()
            logger.debug(
                f"Atomically committed summary v{summary.summary_version} and pruned "
                f"{len(message_ids_to_delete)} messages for Session {session_id} | QueryID={query_id}"
            )

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Transaction failed during commit_summary for session {session_id} | QueryID={query_id} | Error: {str(e)}")
            raise

    def save_turn(
        self, 
        session_id: str, 
        user_message: ConversationMessage, 
        assistant_message: ConversationMessage,
        query_id: Optional[str] = None
    ) -> int:
        """
        ATOMIC TRANSACTION: Saves both the User and Assistant messages.
        Returns the number of messages successfully inserted.
        """
        try:
            user_rec = ConversationMessageModel(
                message_id=user_message.message_id,
                session_id=session_id,
                role=user_message.role.value,
                content=user_message.content,
                tokens=user_message.tokens,
                message_metadata=user_message.metadata,
                timestamp=user_message.timestamp
            )
            
            asst_rec = ConversationMessageModel(
                message_id=assistant_message.message_id,
                session_id=session_id,
                role=assistant_message.role.value,
                content=assistant_message.content,
                tokens=assistant_message.tokens,
                message_metadata=assistant_message.metadata,
                timestamp=assistant_message.timestamp
            )

            self.db.add_all([user_rec, asst_rec])
            self.db.commit()
            
            logger.debug(f"Successfully saved conversation turn for Session {session_id} | QueryID={query_id}")
            return 2

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Transaction failed while saving turn for session {session_id} | QueryID={query_id} | Error: {str(e)}")
            raise