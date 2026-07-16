from abc import ABC, abstractmethod
from self_healing.models import SelfHealingRequest, RecoveryDecision

class BaseRecoveryValidator(ABC):
    """
    Abstract contract for diagnosing system state and determining a recovery path.
    Belongs strictly to the Self-Healing subsystem.
    """
    @abstractmethod
    def validate(self, request: SelfHealingRequest) -> RecoveryDecision:
        raise NotImplementedError