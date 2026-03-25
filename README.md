# Mistral Embeddings — Text Similarity

Compare two pieces of text by computing their embeddings with the [Mistral API](https://docs.mistral.ai/capabilities/embeddings/) and measuring the distance between them.

## Prerequisites

- Python 3.10+
- A [Mistral API key](https://console.mistral.ai/)

## Setup

**1. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure the `.env` file**

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
MISTRAL_API_KEY=your_api_key_here
MISTRAL_EMBED_MODEL=mistral-embed
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `MISTRAL_API_KEY` | Yes | — | Your [Mistral API key](https://console.mistral.ai/) |
| `MISTRAL_EMBED_MODEL` | No | `mistral-embed` | Embedding model to use |

> You can also pass the API key via `--api-key` at runtime.

## Usage

```bash
python compare_embeddings.py "TEXT 1" "TEXT 2"
```

You can also pass the API key directly instead of using the environment variable:

```bash
python compare_embeddings.py "TEXT 1" "TEXT 2" --api-key "your_api_key_here"
```

### Example

```bash
python compare_embeddings.py \
  "The cat is sleeping on the couch" \
  "A feline is resting on the sofa"
```

```
Text 1 : 'The cat is sleeping on the couch'
Text 2 : 'A feline is resting on the sofa'

Generating embeddings via Mistral…

─────────────────────────────────────────────
  Model              : mistral-embed
  Dimension          : 1024
─────────────────────────────────────────────
  Cosine similarity  : +0.942371  (1 = identical)
  Cosine distance    : 0.057629  (0 = identical)
  Euclidean distance : 0.339218
  Dot product        : +0.942371
─────────────────────────────────────────────

  Verdict : Very similar (cosine = 0.9424)
```

## Metrics

| Metric | Range | Identical texts |
|---|---|---|
| Cosine similarity | [-1, 1] | 1 |
| Cosine distance | [0, 2] | 0 |
| Euclidean distance | [0, ∞] | 0 |
| Dot product | (-∞, +∞) | maximum value |

### Verdict thresholds

| Cosine similarity | Verdict |
|---|---|
| ≥ 0.95 | Very similar |
| ≥ 0.80 | Similar |
| ≥ 0.50 | Moderately similar |
| < 0.50 | Not similar |
