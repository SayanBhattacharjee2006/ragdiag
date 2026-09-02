"""Exception hierarchy for LLM judge failures."""


class JudgeError(Exception):
    """Base exception for all judge-related errors."""


class JudgeAuthenticationError(JudgeError):
    """Raised when authentication with the LLM provider fails or API keys are missing."""


class JudgeProviderError(JudgeError):
    """Raised when the LLM provider encounters an API error, rate limit, or timeout."""


class JudgeParseError(JudgeError):
    """Raised when the LLM response cannot be parsed into a valid JudgeResult."""
