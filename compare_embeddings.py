"""
Compare two text embeddings using the Mistral API.

Metrics computed:
  - Cosine similarity  (1 = identical direction, 0 = orthogonal, -1 = opposite)
  - Cosine distance    (1 - cosine_similarity)
  - Euclidean distance (L2 norm of the difference vector)
  - Dot product        (unnormalised similarity)
"""

import os
import sys
import math
import argparse
from dotenv import load_dotenv
from mistralai.client import Mistral
from mistralai.client.errors import SDKError

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

MODEL = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")


def get_embedding(client: Mistral, text: str) -> list[float]:
    try:
        response = client.embeddings.create(model=MODEL, inputs=[text])
    except SDKError as e:
        if e.status_code == 401:
            sys.exit(
                "\nError: Invalid or missing API key (401 Unauthorized).\n"
                "  → Check that MISTRAL_API_KEY is correctly set in your .env file.\n"
                "  → Get a key at https://console.mistral.ai/\n"
            )
        if e.status_code == 429:
            sys.exit("\nError: Rate limit exceeded (429). Wait a moment and try again.\n")
        sys.exit(f"\nError: Mistral API returned status {e.status_code}.\n  {e}\n")
    return response.data[0].embedding


def dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def magnitude(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    mag_a, mag_b = magnitude(a), magnitude(b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot_product(a, b) / (mag_a * mag_b)


def euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare(text1: str, text2: str, api_key: str | None = None) -> None:
    key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not key:
        sys.exit(
            "\nError: MISTRAL_API_KEY is not set.\n"
            "  → Add your key to the .env file: MISTRAL_API_KEY=<your_key>\n"
            "  → Get a key at https://console.mistral.ai/\n"
        )

    client = Mistral(api_key=key)

    print(f"\nText 1 : {text1!r}")
    print(f"Text 2 : {text2!r}\n")
    print("Generating embeddings via Mistral…")

    emb1 = get_embedding(client, text1)
    emb2 = get_embedding(client, text2)

    dim = len(emb1)
    cos_sim = cosine_similarity(emb1, emb2)
    cos_dist = 1.0 - cos_sim
    euc_dist = euclidean_distance(emb1, emb2)
    dot = dot_product(emb1, emb2)

    print(f"\n{'─' * 45}")
    print(f"  Model              : {MODEL}")
    print(f"  Dimension          : {dim}")
    print(f"{'─' * 45}")
    print(f"  Cosine similarity  : {cos_sim:+.6f}  (1 = identical)")
    print(f"  Cosine distance    : {cos_dist:.6f}  (0 = identical)")
    print(f"  Euclidean distance : {euc_dist:.6f}")
    print(f"  Dot product        : {dot:+.6f}")
    print(f"{'─' * 45}\n")

    if cos_sim >= 0.95:
        verdict = "Very similar"
    elif cos_sim >= 0.80:
        verdict = "Similar"
    elif cos_sim >= 0.50:
        verdict = "Moderately similar"
    else:
        verdict = "Not similar"

    print(f"  Verdict : {verdict} (cosine = {cos_sim:.4f})\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two texts using their Mistral embeddings."
    )
    parser.add_argument("text1", help="First text")
    parser.add_argument("text2", help="Second text")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Mistral API key (defaults to MISTRAL_API_KEY environment variable)",
    )
    args = parser.parse_args()

    compare(args.text1, args.text2, api_key=args.api_key)
