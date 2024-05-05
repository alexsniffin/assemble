import enum


class SystemStates(str, enum.Enum):
    """
    Internal system states for the Agent's state machine.
    """
    IDLE: str = "idle"
    EXIT: str = "exit"
