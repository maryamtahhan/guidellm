"""
Quality validation and benchmarking tools for embeddings.

This module provides comprehensive quality validation capabilities for embeddings
including cosine similarity validation against baseline models and MTEB (Massive
Text Embedding Benchmark) integration for standardized quality evaluation.

Supports both local model evaluation and remote endpoint evaluation (OpenAI-compatible).
"""

from __future__ import annotations

from .mteb_integration import DEFAULT_MTEB_TASKS, MTEBValidator, RemoteMTEBValidator
from .validators import EmbeddingsQualityValidator, compute_cosine_similarity

__all__ = [
    "DEFAULT_MTEB_TASKS",
    "EmbeddingsQualityValidator",
    "MTEBValidator",
    "RemoteMTEBValidator",
    "compute_cosine_similarity",
]
