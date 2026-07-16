# memory/user/profile.py

import logging
from typing import Optional
from memory.models import UserProfile

logger = logging.getLogger(__name__)

class UserProfileService:
    """
    Manages long-term identity and demographic information for the user.
    """

    def load_profile(self, user_id: str, query_id: Optional[str] = None) -> UserProfile:
        """
        Retrieves the persistent profile for a given user.
        Currently returns a safe default until User Storage is implemented.
        """
        logger.debug(f"Loading user profile for {user_id} | QueryID={query_id}")
        
        return UserProfile(
            user_id=user_id,
            timezone="UTC",
            locale="en-US"
        )