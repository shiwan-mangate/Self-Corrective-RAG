# memory/user/preferences.py

import logging
from typing import Optional
from memory.models import UserPreferences

logger = logging.getLogger(__name__)

class UserPreferenceService:
    """
    Manages long-term behavioral instructions for the AI assistant.
    """

    def load_preferences(self, user_id: str, query_id: Optional[str] = None) -> UserPreferences:
        """
        Retrieves the persistent prompt-formatting preferences for the user.
        Currently returns a safe default until User Storage is implemented.
        """
        logger.debug(f"Loading user preferences for {user_id} | QueryID={query_id}")
        
        return UserPreferences(user_id=user_id)

    def update_preferences(self, user_id: str, updates: dict, query_id: Optional[str] = None) -> UserPreferences:
        """
        Future API for mutating long-term settings.
        """
        logger.error(f"Attempted to update preferences for {user_id} but storage is not implemented.")
        raise NotImplementedError("Preference persistence has not been implemented yet.")