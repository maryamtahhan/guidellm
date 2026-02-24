"""
MTEB (Massive Text Embedding Benchmark) integration for embeddings quality evaluation.

Provides standardized benchmark evaluation using MTEB tasks like STS (Semantic Textual
Similarity) to measure embedding quality across multiple standardized datasets. Follows
vLLM patterns for MTEB evaluation with configurable task selection and lightweight
defaults suitable for CI/CD environments.

Supports both local model evaluation (via SentenceTransformers) and remote endpoint
evaluation (via OpenAI-compatible API) following vLLM's testing patterns.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_MTEB_TASKS",
    "MTEBValidator",
    "RemoteMTEBValidator",
]

DEFAULT_MTEB_TASKS = ["STS12", "STS13", "STSBenchmark"]
"""Default MTEB tasks for lightweight evaluation (Semantic Textual Similarity)."""


class MTEBValidator:
    """
    MTEB benchmark integration for standardized quality evaluation.

    Runs MTEB evaluation tasks on embedding models to produce standardized quality
    scores. Supports configurable task selection with defaults focused on lightweight
    STS (Semantic Textual Similarity) tasks suitable for regular benchmarking.

    Example:
    ::
        validator = MTEBValidator(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            task_names=["STS12", "STS13"]
        )

        results = validator.run_evaluation()
        print(f"MTEB Main Score: {results['mteb_main_score']:.4f}")
        for task, score in results['mteb_task_scores'].items():
            print(f"{task}: {score:.4f}")
    """

    def __init__(
        self,
        model_name: str,
        task_names: list[str] | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ):
        """
        Initialize MTEB validator with model and task configuration.

        :param model_name: HuggingFace model name or path for evaluation
        :param task_names: List of MTEB tasks to evaluate (uses
            DEFAULT_MTEB_TASKS if None)
        :param device: Device for model inference ("cpu", "cuda", "mps", or
            None for auto)
        :param batch_size: Batch size for encoding during evaluation
        :raises ImportError: If mteb or sentence-transformers is not
            installed
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for MTEB evaluation. "
                "Install with: pip install sentence-transformers"
            ) from e

        try:
            import mteb
        except ImportError as e:
            raise ImportError(
                "mteb is required for MTEB evaluation. Install with: pip install mteb"
            ) from e

        self.model_name = model_name
        self.task_names = task_names if task_names is not None else DEFAULT_MTEB_TASKS
        self.device = device
        self.batch_size = batch_size

        # Load model
        self.model = SentenceTransformer(model_name, device=device)

        # Store mteb module reference
        self.mteb = mteb

    def run_evaluation(
        self,
        output_folder: str | None = None,  # noqa: ARG002
        verbosity: int = 1,
    ) -> dict[str, Any]:
        """
        Run MTEB evaluation on configured tasks.

        Executes MTEB benchmark tasks and computes standardized quality scores.
        Returns both individual task scores and an aggregated main score.

        :param output_folder: Optional folder to save detailed results (unused,
            kept for API compatibility)
        :param verbosity: Verbosity level (0=silent, 1=progress, 2=detailed)
        :return: Dictionary with 'mteb_main_score' and 'mteb_task_scores'

        Example:
        ::
            results = validator.run_evaluation()

            # Access main score (average across tasks)
            main_score = results['mteb_main_score']

            # Access individual task scores
            for task, score in results['mteb_task_scores'].items():
                print(f"{task}: {score:.4f}")
        """
        # Get MTEB task objects
        tasks = self.mteb.get_tasks(tasks=self.task_names)

        # Run evaluation using modern mteb.evaluate() API
        # Following vLLM's pattern from pooling_mteb_test/mteb_embed_utils.py
        # Suppress sklearn FutureWarnings from MTEB's internal classification models
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=".*n_jobs.*has no effect.*",
            )
            results = self.mteb.evaluate(
                self.model,
                tasks,
                cache=None,
                show_progress_bar=(verbosity > 0),
            )

        # Extract scores from results
        # mteb.evaluate() returns a list of TaskResult objects
        task_scores = {}
        for task_result in list(results):
            task_name = task_result.task_name
            # Get main score from the test split
            # Following vLLM's pattern: results[0].scores["test"][0]["main_score"]
            if "test" in task_result.scores and task_result.scores["test"]:
                test_scores = task_result.scores["test"][0]
                if "main_score" in test_scores:
                    task_scores[task_name] = float(test_scores["main_score"])

        # Compute main score as average across tasks
        main_score = float(np.mean(list(task_scores.values()))) if task_scores else 0.0

        return {
            "mteb_main_score": main_score,
            "mteb_task_scores": task_scores,
        }

    def get_available_tasks(self) -> list[str]:
        """
        Get list of all available MTEB tasks.

        :return: List of available task names

        Example:
        ::
            validator = MTEBValidator(model_name="...")
            tasks = validator.get_available_tasks()
            print(f"Available tasks: {tasks}")
        """
        all_tasks = self.mteb.get_tasks()
        return [task.metadata.name for task in all_tasks]

    def get_task_info(self, task_name: str) -> dict[str, Any]:
        """
        Get metadata information about a specific MTEB task.

        :param task_name: Name of the MTEB task
        :return: Dictionary with task metadata
        :raises ValueError: If task is not found

        Example:
        ::
            info = validator.get_task_info("STS12")
            print(f"Task: {info['name']}")
            print(f"Description: {info['description']}")
        """
        tasks = self.mteb.get_tasks(tasks=[task_name])

        if not tasks:
            raise ValueError(f"MTEB task '{task_name}' not found")

        task = tasks[0]
        metadata = task.metadata

        return {
            "name": metadata.name,
            "description": getattr(metadata, "description", ""),
            "type": getattr(metadata, "type", ""),
            "category": getattr(metadata, "category", ""),
            "eval_splits": getattr(metadata, "eval_splits", []),
            "main_score": getattr(metadata, "main_score", ""),
        }

    @staticmethod
    def get_recommended_tasks(category: str = "sts") -> list[str]:
        """
        Get recommended MTEB tasks for specific evaluation categories.

        :param category: Evaluation category ("sts", "classification",
            "retrieval", etc.)
        :return: List of recommended task names

        Example:
        ::
            sts_tasks = MTEBValidator.get_recommended_tasks("sts")
            # Returns: ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark"]
        """
        recommendations = {
            "sts": [
                "STS12",
                "STS13",
                "STS14",
                "STS15",
                "STS16",
                "STSBenchmark",
                "SICK-R",
            ],
            "classification": [
                "AmazonCounterfactualClassification",
                "AmazonPolarityClassification",
                "AmazonReviewsClassification",
                "Banking77Classification",
                "EmotionClassification",
            ],
            "clustering": [
                "ArxivClusteringP2P",
                "ArxivClusteringS2S",
                "BiorxivClusteringP2P",
                "BiorxivClusteringS2S",
                "MedrxivClusteringP2P",
            ],
            "retrieval": [
                "ArguAna",
                "ClimateFEVER",
                "CQADupstackRetrieval",
                "DBPedia",
                "FEVER",
            ],
            "lightweight": DEFAULT_MTEB_TASKS,  # Fastest tasks for CI/CD
        }

        return recommendations.get(category.lower(), DEFAULT_MTEB_TASKS)


# Following vLLM's pattern from
# tests/models/language/pooling_mteb_test/mteb_embed_utils.py
# for MTEB integration with OpenAI-compatible endpoints


class _MtebEmbedMixin:
    """
    Mixin implementing MTEB EncoderProtocol for remote endpoints.

    Provides similarity computation methods required by MTEB. Based on vLLM's
    MtebEmbedMixin pattern for testing remote embedding endpoints.
    """

    mteb_model_meta: Any  # Will be set by subclass

    def similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings.

        :param embeddings1: First set of embeddings (shape: [n, dim])
        :param embeddings2: Second set of embeddings (shape: [m, dim])
        :return: Similarity matrix (shape: [n, m])
        """
        # Cosine similarity
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        return np.dot(embeddings1, embeddings2.T) / (norm1 * norm2.T)

    def similarity_pairwise(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity between embeddings.

        :param embeddings1: First set of embeddings (shape: [n, dim])
        :param embeddings2: Second set of embeddings (shape: [n, dim])
        :return: Pairwise similarities (shape: [n])
        """
        # Cosine similarity
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        return np.sum(embeddings1 * embeddings2, axis=1) / (
            norm1.flatten() * norm2.flatten()
        )


class _OpenAIClientMtebEncoder(_MtebEmbedMixin):
    """
    MTEB encoder for OpenAI-compatible remote endpoints.

    Wraps an OpenAI client to provide MTEB-compatible encoding interface for
    benchmarking remote embedding endpoints. Based on vLLM's OpenAIClientMtebEncoder.

    Example:
    ::
        from openai import OpenAI

        client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
        encoder = _OpenAIClientMtebEncoder("model-name", client)

        # Use with MTEB
        import mteb
        tasks = mteb.get_tasks(tasks=["STS12"])
        results = mteb.evaluate(encoder, tasks)
    """

    def __init__(self, model_name: str, client: Any):
        """
        Initialize MTEB encoder with OpenAI client.

        :param model_name: Name of the embedding model
        :param client: OpenAI client instance configured for the endpoint
        """
        try:
            from mteb.models import ModelMeta
        except ImportError as e:
            msg = "mteb is required. Install with: pip install mteb"
            raise ImportError(msg) from e

        self.model_name = model_name
        self.client = client
        self.rng = np.random.default_rng(seed=42)

        # Create model metadata following vLLM pattern
        self.mteb_model_meta = ModelMeta(
            loader=None,
            name=f"remote/{model_name}",
            revision="1",
            release_date=None,
            languages=None,
            framework=[],
            similarity_fn_name=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            open_weights=None,
            public_training_code=None,
            public_training_data=None,
            use_instructions=None,
            training_datasets=None,
            modalities=["text"],
        )

    def encode(
        self,
        inputs: Any,  # DataLoader[BatchedInput]
        *args,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> np.ndarray:
        """
        Encode texts using remote OpenAI-compatible endpoint.

        Extracts texts from MTEB DataLoader and calls the remote embeddings API.
        Randomizes order to test scheduling robustness (following vLLM pattern).

        :param inputs: MTEB DataLoader with batched input texts
        :return: Embeddings as numpy array (shape: [n_texts, embedding_dim])
        """
        # Extract sentences from DataLoader
        sentences = [text for batch in inputs for text in batch["text"]]

        # Randomize order to discover potential scheduling issues
        r = self.rng.permutation(len(sentences))
        sentences = [sentences[i] for i in r]

        # Call remote endpoint
        embeddings = self.client.embeddings.create(
            model=self.model_name, input=sentences
        )
        outputs = [d.embedding for d in embeddings.data]
        embeds = np.array(outputs)

        # Restore original order and return
        return embeds[np.argsort(r)]


class RemoteMTEBValidator:
    """
    MTEB benchmark integration for remote OpenAI-compatible endpoints.

    Evaluates remote embedding endpoints using standardized MTEB tasks. Supports
    vLLM, OpenAI, and any OpenAI-compatible embedding API.

    Example:
    ::
        validator = RemoteMTEBValidator(
            base_url="http://localhost:8000",
            model_name="ibm-granite/granite-embedding-english-r2",
            task_names=["STS12", "STS13"]
        )

        results = validator.run_evaluation()
        print(f"MTEB Main Score: {results['mteb_main_score']:.4f}")
        for task, score in results['mteb_task_scores'].items():
            print(f"{task}: {score:.4f}")
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        task_names: list[str] | None = None,
        api_key: str = "EMPTY",
    ):
        """
        Initialize remote MTEB validator.

        :param base_url: Base URL of the OpenAI-compatible endpoint
            (e.g., "http://localhost:8000")
        :param model_name: Name of the embedding model to evaluate
        :param task_names: List of MTEB tasks to evaluate (uses
            DEFAULT_MTEB_TASKS if None)
        :param api_key: API key for authentication (defaults to "EMPTY" for
            vLLM/local endpoints)
        :raises ImportError: If mteb or openai is not installed
        """
        try:
            import mteb
        except ImportError as e:
            raise ImportError(
                "mteb is required for MTEB evaluation. Install with: pip install mteb"
            ) from e

        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "openai is required for remote MTEB evaluation. "
                "Install with: pip install openai"
            ) from e

        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.task_names = task_names if task_names is not None else DEFAULT_MTEB_TASKS
        self.api_key = api_key

        # Create OpenAI client for remote endpoint
        self.client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=api_key,
        )

        # Create MTEB encoder wrapper
        self.encoder = _OpenAIClientMtebEncoder(model_name, self.client)

        # Store mteb module reference
        self.mteb = mteb

    def run_evaluation(
        self,
        output_folder: str | None = None,  # noqa: ARG002
        verbosity: int = 1,
    ) -> dict[str, Any]:
        """
        Run MTEB evaluation on remote endpoint.

        Executes MTEB benchmark tasks against the remote endpoint and computes
        standardized quality scores. Returns both individual task scores and an
        aggregated main score.

        :param output_folder: Optional folder to save detailed results (unused,
            kept for API compatibility)
        :param verbosity: Verbosity level (0=silent, 1=progress, 2=detailed)
        :return: Dictionary with 'mteb_main_score' and 'mteb_task_scores'

        Example:
        ::
            results = validator.run_evaluation()

            # Access main score (average across tasks)
            main_score = results['mteb_main_score']

            # Access individual task scores
            for task, score in results['mteb_task_scores'].items():
                print(f"{task}: {score:.4f}")
        """
        # Get MTEB task objects
        tasks = self.mteb.get_tasks(tasks=self.task_names)

        # Run evaluation using modern mteb.evaluate() API
        # Following vLLM's pattern from pooling_mteb_test/mteb_embed_utils.py
        # Suppress sklearn FutureWarnings from MTEB's internal classification models
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=".*n_jobs.*has no effect.*",
            )
            results = self.mteb.evaluate(
                self.encoder,  # type: ignore[arg-type]
                tasks,
                cache=None,
                show_progress_bar=(verbosity > 0),
            )

        # Extract scores from results
        # mteb.evaluate() returns a list of TaskResult objects
        task_scores = {}
        for task_result in list(results):
            task_name = task_result.task_name
            # Get main score from the test split
            # Following vLLM's pattern: results[0].scores["test"][0]["main_score"]
            if "test" in task_result.scores and task_result.scores["test"]:
                test_scores = task_result.scores["test"][0]
                if "main_score" in test_scores:
                    task_scores[task_name] = float(test_scores["main_score"])

        # Compute main score as average across tasks
        main_score = float(np.mean(list(task_scores.values()))) if task_scores else 0.0

        return {
            "mteb_main_score": main_score,
            "mteb_task_scores": task_scores,
        }

    @staticmethod
    def get_recommended_tasks(category: str = "sts") -> list[str]:
        """
        Get recommended MTEB tasks for specific evaluation categories.

        :param category: Evaluation category ("sts", "classification",
            "retrieval", etc.)
        :return: List of recommended task names

        Example:
        ::
            sts_tasks = RemoteMTEBValidator.get_recommended_tasks("sts")
            # Returns: ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark"]
        """
        return MTEBValidator.get_recommended_tasks(category)
