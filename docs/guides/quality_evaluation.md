# Quality Evaluation for Embeddings

guidellm provides comprehensive quality evaluation tools for embedding models, including both local model validation and remote endpoint evaluation using standardized MTEB (Massive Text Embedding Benchmark) tasks.

## Overview

Quality evaluation helps you:

- **Validate embedding quality** using standardized benchmarks
- **Compare models** objectively with industry-standard metrics
- **Test remote endpoints** without downloading models locally
- **Ensure production readiness** with quality thresholds

## MTEB Evaluation

### What is MTEB?

MTEB (Massive Text Embedding Benchmark) is the industry-standard benchmark for evaluating embedding models. It provides:

- Standardized tasks (semantic similarity, classification, retrieval, etc.)
- Reproducible quality scores
- Leaderboard comparisons with state-of-the-art models

### Remote Endpoint Evaluation

Evaluate OpenAI-compatible embedding endpoints (vLLM, OpenAI, or any compatible API) without downloading models locally.

#### Quick Start

```python
from guidellm.benchmark.quality import RemoteMTEBValidator

# Create validator for your endpoint
validator = RemoteMTEBValidator(
    base_url="http://localhost:8000",
    model_name="ibm-granite/granite-embedding-english-r2",
    task_names=["STS12", "STS13"]
)

# Run evaluation
results = validator.run_evaluation()

# Display results
print(f"MTEB Main Score: {results['mteb_main_score']:.4f}")
for task, score in results['mteb_task_scores'].items():
    print(f"  {task}: {score:.4f}")
```

#### Example Output

```
MTEB Main Score: 75.56%
  STS12: 67.02%
  STS13: 84.09%
```

### Local Model Evaluation

Evaluate locally downloaded models using SentenceTransformers:

```python
from guidellm.benchmark.quality import MTEBValidator

# Create validator with local model
validator = MTEBValidator(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    task_names=["STS12", "STS13"]
)

# Run evaluation
results = validator.run_evaluation()
```

## MTEB Task Selection

### Default Tasks (Lightweight)

The default tasks are optimized for quick evaluation:

- `STS12` - Semantic Textual Similarity 2012
- `STS13` - Semantic Textual Similarity 2013
- `STSBenchmark` - STS Benchmark dataset

### Category-Based Results

guidellm automatically groups MTEB tasks by category and displays average scores for each category, matching the MTEB leaderboard structure:

```
ℹ Quality Metrics - MTEB Evaluation
|=========|=======|================|============|===========|======|
| Bench   | MTEB  | MTEB Tasks                                     |
| Strat   | Main  | Classification | Clustering | Retrieval | STS  |
|         | %     | %              | %          | %         | %    |
|---------|-------|----------------|------------|-----------|------|
| const   | 75.2  | 68.5           | 72.1       | 78.9      | 76.3 |
|=========|=======|================|============|===========|======|
```

**Categories**:

- **Classification** - Text classification tasks
- **Clustering** - Document clustering tasks
- **Pair Classification** - Pairwise classification
- **Reranking** - Search result reranking
- **Instruction Reranking** - Instruction-following reranking
- **Retrieval** - Information retrieval tasks
- **STS** - Semantic Textual Similarity
- **Bitext Mining** - Parallel sentence mining
- **Multilabel Classification** - Multi-label classification

### Comprehensive English Task List

#### Quick Evaluation (6 tasks, ~10 minutes)

```bash
guidellm benchmark embeddings \
  --target http://localhost:8000 \
  --model your-embedding-model \
  --data "prompt_tokens=100" \
  --max-requests 50 \
  --rate 10 \
  --enable-mteb \
  --mteb-tasks "AmazonCounterfactualClassification,ArxivClusteringS2S,NFCorpus,STS12,STS13,STS14" \
  --outputs html,json
```

#### Full Evaluation (13 tasks, ~30 minutes)

```bash
guidellm benchmark embeddings \
  --target http://localhost:8000 \
  --model your-embedding-model \
  --data "prompt_tokens=100" \
  --max-requests 100 \
  --rate 10 \
  --enable-mteb \
  --mteb-tasks "AmazonCounterfactualClassification,DBpediaClassification,ArxivClusteringS2S,BigPatentClustering,NFCorpus,ArguAna,STS12,STS13,STS14,STS15,STS16,STSBenchmark,SICK-R" \
  --outputs html,csv,json
```

### Recommended Tasks by Category

#### Classification

- `AmazonCounterfactualClassification` - Product review sentiment (English, German, Japanese)
- `DBpediaClassification` - Wikipedia article classification
- `FinancialPhrasebankClassification` - Financial news sentiment

#### Clustering

- `ArxivClusteringS2S` - Academic paper clustering
- `BigPatentClustering` - Patent document clustering
- `BiorxivClusteringP2P` - Biomedical paper clustering

#### Retrieval

- `ATLAStatutes` - Legal statute retrieval
- `ArguAna` - Argument retrieval
- `NFCorpus` - Medical information retrieval
- `TRECCOVID` - COVID-19 research retrieval
- `SciFact` - Scientific fact verification

#### STS (Semantic Textual Similarity)

- `STS12` - SemEval 2012 (news, europarl, video descriptions)
- `STS13` - SemEval 2013 (news, headlines, FNWN)
- `STS14` - SemEval 2014 (news, forum, Twitter)
- `STS15` - SemEval 2015 (forum, news, student answers)
- `STS16` - SemEval 2016 (news, Q&A, plagiarism)
- `STSBenchmark` - STS Benchmark (diverse sources)
- `SICK-R` - Sentences Involving Compositional Knowledge

#### Reranking

- `AskUbuntuDupQuestions` - Duplicate question detection

## Understanding MTEB Scores

**Important**: MTEB returns scores in the 0-1 range (e.g., 0.7556), which represent percentages (75.56%). guidellm automatically converts these to percentages for display.

### Score Scale

- **API Returns**: Values from 0.0 to 1.0 (e.g., `0.7556`)
- **Display Format**: Percentages from 0% to 100% (e.g., `75.56%`)
- **Meaning**: Percentage of similarity between predicted and ground truth pairs

### Score Interpretation

| Score Range | Quality Level | Description                             |
| ----------- | ------------- | --------------------------------------- |
| 90-100%     | Excellent     | State-of-the-art performance            |
| 80-90%      | Very Good     | Production-ready for most applications  |
| 70-80%      | Good          | Suitable for general use cases          |
| 60-70%      | Acceptable    | May need improvement for critical tasks |
| \<60%       | Poor          | Investigate model or configuration      |

### Example Analysis

```python
# Sample results from API (0-1 scale)
results = {
    'mteb_main_score': 0.7556,  # Returned as 0-1
    'mteb_task_scores': {
        'STS12': 0.6702,  # Returned as 0-1
        'STS13': 0.8409,  # Returned as 0-1
    }
}

# Interpretation (convert to percentages):
# - Overall good quality (75.56%) suitable for production
# - Strong on STS13 (84.09% - semantic similarity)
# - Moderate on STS12 (67.02% - may vary by dataset characteristics)
```

**Note**: guidellm automatically displays scores as percentages in console/CSV/HTML outputs for better readability, but the underlying API values remain in 0-1 range.

## Integration with vLLM

This implementation follows vLLM's MTEB testing pattern for consistency with industry practices.

### Start vLLM Embedding Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ibm-granite/granite-embedding-english-r2 \
    --port 8000
```

### Evaluate Quality

```python
from guidellm.benchmark.quality import RemoteMTEBValidator

validator = RemoteMTEBValidator(
    base_url="http://localhost:8000",
    model_name="ibm-granite/granite-embedding-english-r2",
    task_names=["STS12"]
)

results = validator.run_evaluation()
print(f"MTEB Score: {results['mteb_main_score']:.4f}")
```

## Baseline Comparison

Compare your model's embeddings against a reference baseline:

```python
from guidellm.benchmark.quality import EmbeddingsQualityValidator

# Create validator with baseline model
validator = EmbeddingsQualityValidator(
    baseline_model="sentence-transformers/all-MiniLM-L6-v2",
    tolerance=1e-2  # 1% tolerance
)

# Validate target embeddings
text = "Example sentence"
target_embedding = your_model.encode(text)

similarity = validator.validate_against_baseline(text, target_embedding)
is_valid = validator.check_tolerance(similarity)

print(f"Similarity to baseline: {similarity:.4f}")
print(f"Within tolerance: {is_valid}")
```

### Tolerance Levels

Following vLLM patterns:

- **Standard validation**: `tolerance=1e-2` (1%)
- **MTEB-level validation**: `tolerance=5e-4` (0.05%)

## Advanced Usage

### Custom Task Selection

```python
# Run specific MTEB tasks
validator = RemoteMTEBValidator(
    base_url="http://localhost:8000",
    model_name="your-model",
    task_names=[
        "STS12",
        "STS13",
        "STSBenchmark",
        "SICK-R"
    ]
)

results = validator.run_evaluation(verbosity=2)  # Detailed output
```

### Quiet Mode

```python
# Silent execution (no progress bars)
results = validator.run_evaluation(verbosity=0)
```

### Batch Validation

```python
# Validate multiple embeddings
texts = ["Text 1", "Text 2", "Text 3"]
target_embeddings = model.encode(texts)

similarities = validator.validate_batch(texts, target_embeddings)
mean_similarity = np.mean(similarities)
```

### Self-Consistency Check

```python
# Verify model produces consistent embeddings
text = "Consistency test"
embeddings = [model.encode(text) for _ in range(5)]

mean_sim, is_consistent = validator.check_self_consistency(text, embeddings)
print(f"Self-consistency: {mean_sim:.4f} (consistent: {is_consistent})")
```

## Performance Considerations

- **Evaluation Time**: Each MTEB task takes 30-60 seconds depending on endpoint speed
- **Network Latency**: Remote endpoints add network overhead vs local evaluation
- **Batch Processing**: The validator randomizes request order to test scheduler robustness
- **Caching**: MTEB results are not cached by default

## Remote vs Local Evaluation

### Remote Evaluation (RemoteMTEBValidator)

✅ No model download required ✅ Test production endpoints directly ✅ Works with any OpenAI-compatible API ❌ Slower due to network latency ❌ Requires running server

### Local Evaluation (MTEBValidator)

✅ Faster (no network overhead) ✅ Works offline ❌ Requires downloading model (~GB) ❌ Requires local compute resources

## Dependencies

For quality evaluation:

```bash
pip install guidellm[quality]
```

For remote MTEB evaluation:

```bash
pip install guidellm[quality] openai
```

Required packages:

- `mteb` - MTEB benchmark framework
- `sentence-transformers` - For baseline comparisons
- `openai` - OpenAI Python client (for remote evaluation)

## Troubleshooting

### "Module 'openai' not found"

```bash
pip install openai
```

### "Module 'mteb' not found"

```bash
pip install mteb
```

### Connection Errors

- Verify endpoint URL is correct
- Check that the server is running
- Ensure the model name matches what the server expects

### Low Scores

- Verify the model is suitable for embedding tasks
- Check that the endpoint is configured correctly
- Try different MTEB tasks to identify specific weaknesses

## References

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [vLLM MTEB Testing](https://github.com/vllm-project/vllm/tree/main/tests/models/language/pooling_mteb_test)
- [MTEB Paper](https://arxiv.org/abs/2210.07316)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
