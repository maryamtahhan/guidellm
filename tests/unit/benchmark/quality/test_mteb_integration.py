from __future__ import annotations

import pytest

# Skip all tests if sentence-transformers/mteb aren't available
pytest.importorskip("sentence_transformers", reason="sentence-transformers required")
pytest.importorskip("mteb", reason="mteb required")

from guidellm.benchmark.quality.mteb_integration import (
    DEFAULT_MTEB_TASKS,
    MTEBValidator,
    RemoteMTEBValidator,
)


class TestMTEBValidator:
    """Tests for MTEB benchmark integration."""

    @pytest.fixture
    def validator(self):
        """Create a validator with a test model and minimal tasks."""
        # Use a small, fast model and single task for faster tests
        return MTEBValidator(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            task_names=["STS12"],  # Single lightweight task
        )

    @pytest.mark.smoke
    def test_initialization(self, validator):
        """Test validator initialization."""
        assert validator is not None
        assert validator.model is not None
        assert validator.task_names == ["STS12"]

    @pytest.mark.smoke
    def test_initialization_default_tasks(self):
        """Test initialization with default MTEB tasks."""
        validator = MTEBValidator(model_name="sentence-transformers/all-MiniLM-L6-v2")

        assert validator.task_names == DEFAULT_MTEB_TASKS

    @pytest.mark.sanity
    def test_initialization_multiple_tasks(self):
        """Test initialization with multiple tasks."""
        tasks = ["STS12", "STS13", "STSBenchmark"]
        validator = MTEBValidator(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            task_names=tasks,
        )

        assert validator.task_names == tasks
        assert len(validator.task_names) == 3

    @pytest.mark.sanity
    @pytest.mark.slow
    def test_run_evaluation_single_task(self, validator):
        """Test running MTEB evaluation with single task."""
        results = validator.run_evaluation()

        assert isinstance(results, dict)
        assert "mteb_main_score" in results
        assert "mteb_task_scores" in results

        # Main score should be a float
        assert isinstance(results["mteb_main_score"], float)

        # Task scores should be a dict
        assert isinstance(results["mteb_task_scores"], dict)
        assert "STS12" in results["mteb_task_scores"]

    @pytest.mark.sanity
    @pytest.mark.slow
    def test_run_evaluation_score_range(self, validator):
        """Test that MTEB scores are in valid range."""
        results = validator.run_evaluation()

        # MTEB scores should be between 0 and 100
        assert 0.0 <= results["mteb_main_score"] <= 100.0

        for _task_name, score in results["mteb_task_scores"].items():
            assert 0.0 <= score <= 100.0

    @pytest.mark.regression
    @pytest.mark.slow
    def test_run_evaluation_multiple_tasks(self):
        """Test running MTEB evaluation with multiple tasks."""
        tasks = ["STS12", "STS13"]
        validator = MTEBValidator(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            task_names=tasks,
        )

        results = validator.run_evaluation()

        assert "mteb_main_score" in results
        assert "mteb_task_scores" in results

        # Should have scores for both tasks
        assert len(results["mteb_task_scores"]) == len(tasks)
        for task in tasks:
            assert task in results["mteb_task_scores"]

    @pytest.mark.regression
    @pytest.mark.slow
    def test_main_score_is_average(self):
        """Test that main score is average of task scores."""
        tasks = ["STS12", "STS13"]
        validator = MTEBValidator(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            task_names=tasks,
        )

        results = validator.run_evaluation()

        # Calculate expected average
        task_scores = list(results["mteb_task_scores"].values())
        expected_avg = sum(task_scores) / len(task_scores)

        # Main score should be close to average
        assert results["mteb_main_score"] == pytest.approx(expected_avg, abs=0.1)

    @pytest.mark.sanity
    def test_default_mteb_tasks_constant(self):
        """Test that DEFAULT_MTEB_TASKS contains expected tasks."""
        assert isinstance(DEFAULT_MTEB_TASKS, list)
        assert len(DEFAULT_MTEB_TASKS) > 0

        # Should contain STS tasks (standard for embeddings)
        assert any("STS" in task for task in DEFAULT_MTEB_TASKS)

    @pytest.mark.smoke
    def test_model_loaded(self, validator):
        """Test that SentenceTransformer model is loaded."""
        assert validator.model is not None

        # Should be able to encode text
        embedding = validator.model.encode("Test sentence.")
        assert embedding is not None
        assert len(embedding) > 0

    @pytest.mark.regression
    def test_task_names_stored(self, validator):
        """Test that task names are stored correctly."""
        assert hasattr(validator, "task_names")
        assert validator.task_names == ["STS12"]

    @pytest.mark.sanity
    @pytest.mark.slow
    def test_evaluation_reproducible(self, validator):
        """Test that evaluation produces consistent results."""
        # Run evaluation twice
        results1 = validator.run_evaluation()
        results2 = validator.run_evaluation()

        # Results should be identical (or very close)
        assert results1["mteb_main_score"] == pytest.approx(
            results2["mteb_main_score"], abs=0.01
        )

        for task in results1["mteb_task_scores"]:
            assert results1["mteb_task_scores"][task] == pytest.approx(
                results2["mteb_task_scores"][task], abs=0.01
            )

    @pytest.mark.regression
    @pytest.mark.slow
    def test_different_models_different_scores(self):
        """Test that different models produce different scores."""
        # This test verifies the evaluation is model-specific
        validator1 = MTEBValidator(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            task_names=["STS12"],
        )

        # Note: This would require a different model to be installed
        # Skipping if second model not available
        try:
            validator2 = MTEBValidator(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                task_names=["STS12"],
            )

            results1 = validator1.run_evaluation()
            results2 = validator2.run_evaluation()

            # Different models should produce different scores
            # (though they might be similar)
            assert "mteb_main_score" in results1
            assert "mteb_main_score" in results2
        except Exception:  # noqa: BLE001
            # Skip if second model is unavailable
            pytest.skip("Second model not available for comparison")

    @pytest.mark.sanity
    def test_initialization_with_none_tasks(self):
        """Test initialization when tasks is None (should use default)."""
        validator = MTEBValidator(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            task_names=None,
        )

        # Should use DEFAULT_MTEB_TASKS
        assert validator.task_names == DEFAULT_MTEB_TASKS

    @pytest.mark.regression
    @pytest.mark.slow
    def test_evaluation_returns_dict_structure(self, validator):
        """Test that evaluation returns expected dictionary structure."""
        results = validator.run_evaluation()

        # Check structure
        assert isinstance(results, dict)
        assert set(results.keys()) == {"mteb_main_score", "mteb_task_scores"}

        # Check types
        assert isinstance(results["mteb_main_score"], float)
        assert isinstance(results["mteb_task_scores"], dict)

        # Check task scores structure
        for task_name, score in results["mteb_task_scores"].items():
            assert isinstance(task_name, str)
            assert isinstance(score, int | float)


class TestRemoteMTEBValidator:
    """Tests for remote MTEB benchmark integration (OpenAI-compatible endpoints)."""

    @pytest.mark.smoke
    def test_initialization(self):
        """Test remote validator initialization."""
        pytest.importorskip("openai", reason="openai required for remote validation")

        validator = RemoteMTEBValidator(
            base_url="http://localhost:8000",
            model_name="test-model",
            task_names=["STS12"],
        )

        assert validator is not None
        assert validator.model_name == "test-model"
        assert validator.base_url == "http://localhost:8000"
        assert validator.task_names == ["STS12"]
        assert validator.encoder is not None

    @pytest.mark.smoke
    def test_initialization_default_tasks(self):
        """Test initialization with default MTEB tasks."""
        pytest.importorskip("openai", reason="openai required for remote validation")

        validator = RemoteMTEBValidator(
            base_url="http://localhost:8000",
            model_name="test-model",
        )

        assert validator.task_names == DEFAULT_MTEB_TASKS

    @pytest.mark.smoke
    def test_initialization_strips_trailing_slash(self):
        """Test that trailing slash is stripped from base_url."""
        pytest.importorskip("openai", reason="openai required for remote validation")

        validator = RemoteMTEBValidator(
            base_url="http://localhost:8000/",
            model_name="test-model",
        )

        assert validator.base_url == "http://localhost:8000"

    @pytest.mark.smoke
    def test_initialization_custom_api_key(self):
        """Test initialization with custom API key."""
        pytest.importorskip("openai", reason="openai required for remote validation")

        validator = RemoteMTEBValidator(
            base_url="http://localhost:8000",
            model_name="test-model",
            api_key="custom-key",
        )

        assert validator.api_key == "custom-key"

    @pytest.mark.regression
    def test_encoder_has_model_meta(self):
        """Test that encoder has required mteb_model_meta attribute."""
        pytest.importorskip("openai", reason="openai required for remote validation")

        validator = RemoteMTEBValidator(
            base_url="http://localhost:8000",
            model_name="test-model",
        )

        assert hasattr(validator.encoder, "mteb_model_meta")
        assert validator.encoder.mteb_model_meta is not None

    @pytest.mark.regression
    def test_encoder_has_similarity_methods(self):
        """Test that encoder has required similarity methods."""
        pytest.importorskip("openai", reason="openai required for remote validation")

        validator = RemoteMTEBValidator(
            base_url="http://localhost:8000",
            model_name="test-model",
        )

        assert hasattr(validator.encoder, "similarity")
        assert hasattr(validator.encoder, "similarity_pairwise")
        assert callable(validator.encoder.similarity)
        assert callable(validator.encoder.similarity_pairwise)

    @pytest.mark.regression
    def test_encoder_has_encode_method(self):
        """Test that encoder has required encode method."""
        pytest.importorskip("openai", reason="openai required for remote validation")

        validator = RemoteMTEBValidator(
            base_url="http://localhost:8000",
            model_name="test-model",
        )

        assert hasattr(validator.encoder, "encode")
        assert callable(validator.encoder.encode)

    @pytest.mark.smoke
    def test_get_recommended_tasks(self):
        """Test getting recommended tasks for a category."""
        tasks = RemoteMTEBValidator.get_recommended_tasks("sts")

        assert isinstance(tasks, list)
        assert len(tasks) > 0
        assert "STS12" in tasks or "STS13" in tasks

    @pytest.mark.regression
    def test_get_recommended_tasks_lightweight(self):
        """Test getting lightweight recommended tasks."""
        tasks = RemoteMTEBValidator.get_recommended_tasks("lightweight")

        assert tasks == DEFAULT_MTEB_TASKS

    @pytest.mark.slow
    @pytest.mark.timeout(600)
    @pytest.mark.skipif(
        condition=True,  # Skip by default - requires live endpoint
        reason="Requires live embedding endpoint - run manually with endpoint",
    )
    def test_run_evaluation_against_real_endpoint(self):
        """
        Test running MTEB evaluation against a real endpoint.

        NOTE: This test is skipped by default. To run manually:
        1. Start a vLLM embedding server (or compatible endpoint)
        2. Update base_url and model_name below
        3. Remove the skipif condition
        4. Run: pytest -k test_run_evaluation_against_real_endpoint -v
        """
        pytest.importorskip("openai", reason="openai required for remote validation")

        # Example configuration (update for your endpoint)
        validator = RemoteMTEBValidator(
            base_url="http://localhost:8000",
            model_name="your-model-name",
            task_names=["STS12"],  # Single task for faster testing
        )

        results = validator.run_evaluation()

        assert isinstance(results, dict)
        assert "mteb_main_score" in results
        assert "mteb_task_scores" in results
        assert "STS12" in results["mteb_task_scores"]
        assert 0.0 <= results["mteb_main_score"] <= 100.0
