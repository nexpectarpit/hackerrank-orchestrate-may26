"""
Custom exception hierarchy for the support triage agent.

Every module raises a specific subclass so that callers can catch
exactly the errors they care about while the top-level batch
processor catches the broad AgentError base class for graceful
degradation.
"""


class AgentError(Exception):
    """Base error for all agent-related failures."""


# ---- Corpus errors --------------------------------------------------------

class CorpusParsingError(AgentError):
    """Raised when a markdown file in data/ cannot be parsed."""


class ChunkingError(AgentError):
    """Raised when a parsed document cannot be chunked."""


# ---- Index / retrieval errors ---------------------------------------------

class IndexBuildError(AgentError):
    """Raised when the FAISS index cannot be built or loaded."""


class RetrievalError(AgentError):
    """Raised when similarity search fails."""


# ---- LLM errors -----------------------------------------------------------

class LLMError(AgentError):
    """Base error for Groq / LLM communication issues."""


class RateLimitError(LLMError):
    """HTTP 429 from the LLM provider -- retry with backoff."""


class TokenLimitError(LLMError):
    """Request exceeds the model context window."""


class InvalidResponseError(LLMError):
    """LLM returned a response that could not be parsed as JSON."""


# ---- Safety errors --------------------------------------------------------

class SafetyError(AgentError):
    """Raised when a ticket triggers a safety guardrail."""


# ---- Schema / validation errors -------------------------------------------

class SchemaValidationError(AgentError):
    """Output row does not conform to the expected CSV schema."""
