# Backends

GuideLLM is designed to work with OpenAI-compatible HTTP servers, enabling seamless integration with a variety of generative AI backends. This compatibility ensures that users can evaluate and optimize their large language model (LLM) deployments efficiently. While the current focus is on OpenAI-compatible servers, we welcome contributions to expand support for other backends, including additional server implementations and Python interfaces.

## Supported Backends

### OpenAI-Compatible HTTP Servers

GuideLLM supports OpenAI-compatible HTTP servers, which provide a standardized API for interacting with LLMs. This includes popular implementations such as [vLLM](https://github.com/vllm-project/vllm) and [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference). These servers allow GuideLLM to perform evaluations, benchmarks, and optimizations with minimal setup.

### Supported Endpoints

GuideLLM supports the following OpenAI-compatible API endpoints:

- **Text Completions** (`/v1/completions`): Legacy text completion endpoint for prompt-based generation
- **Chat Completions** (`/v1/chat/completions`): Chat-based completion endpoint for conversational interactions
- **Embeddings** (`/v1/embeddings`): Text embedding endpoint for converting text into vector representations
- **Audio Transcriptions** (`/v1/audio/transcriptions`): Audio-to-text transcription endpoint
- **Audio Translations** (`/v1/audio/translations`): Audio translation endpoint

Each endpoint can be benchmarked using the `--request-type` parameter. For example, to benchmark embeddings with concurrent requests:

```bash
guidellm benchmark \
    --target "http://localhost:8000" \
    --request-type embeddings \
    --profile concurrent \
    --rate 32 \
    --max-requests 500 \
    --data "prompt_tokens=256,output_tokens=1" \
    --processor "BAAI/bge-small-en-v1.5"
```

## Examples for Spinning Up Compatible Servers

### 1. vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-performance OpenAI-compatible server designed for efficient LLM inference. It supports a variety of models and provides a simple interface for deployment.

First ensure you have vLLM installed (`pip install vllm`), and then run the following command to start a vLLM server with a Llama 3.1 8B quantized model:

```bash
vllm serve "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
```

For more information on starting a vLLM server, see the [vLLM Documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).

#### vLLM with Embedding Models

vLLM also supports embedding models. To start a vLLM server with an embedding model:

```bash
vllm serve "BAAI/bge-small-en-v1.5"
```

You can then benchmark the embeddings endpoint with concurrent requests:

```bash
guidellm benchmark \
    --target "http://localhost:8000" \
    --request-type embeddings \
    --profile concurrent \
    --rate 32 \
    --max-requests 500 \
    --data "prompt_tokens=256,output_tokens=1" \
    --processor "BAAI/bge-small-en-v1.5"
```

For more details on embeddings benchmarking, see the [Embeddings Guide](./embeddings.md).

### 2. Text Generation Inference (TGI)

[Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) is another OpenAI-compatible server that supports a wide range of models, including those hosted on Hugging Face. TGI is optimized for high-throughput and low-latency inference.

To start a TGI server with a Llama 3.1 8B model using Docker, run the following command:

```bash
docker run --gpus 1 -ti --shm-size 1g --ipc=host --rm -p 8080:80 \
  -e MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct \
  -e NUM_SHARD=1 \
  -e MAX_INPUT_TOKENS=4096 \
  -e MAX_TOTAL_TOKENS=6000 \
  -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  ghcr.io/huggingface/text-generation-inference:2.2.0
```

For more information on starting a TGI server, see the [TGI Documentation](https://huggingface.co/docs/text-generation-inference/index).

## Expanding Backend Support

GuideLLM is an open platform, and we encourage contributions to extend its backend support. Whether it's adding new server implementations, integrating with Python-based backends, or enhancing existing capabilities, your contributions are welcome. For more details on how to contribute, see the [CONTRIBUTING.md](https://github.com/vllm-project/guidellm/blob/main/CONTRIBUTING.md) file.
