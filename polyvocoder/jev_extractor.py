"""JEV semantic feature extractor.

Uses Typesafe.ai System One model (jev-1.13.0) to score a text input
across canonical substrate walker questions. Returns a fixed-dimension
feature vector that downstream heads can use.

This is the "SSL extractor" role from Semantic-VAE — a pretrained model
that turns raw input into semantic features.
"""
import os
import json
import requests
from typing import Dict, List


JEV_URL = "https://api.typesafe.ai/v1/systemone"

# Default canonical questions — these are the dimensions the polyvocoder
# extracts semantic features along. Each maps to one dimension of the
# returned feature vector.
DEFAULT_QUESTIONS: Dict[str, Dict] = {
    "canon_worthy": {
        "type": "noul",
        "instructions": "Is this canon-worthy cyberpunk-noir for the Quilt substrate walker?",
    },
    "distinct_voice": {
        "type": "noul",
        "instructions": "Distinct non-formulaic voice?",
    },
    "doctrine_anchor": {
        "type": "noul",
        "instructions": "Anchored to substrate walker doctrine (cells as scars, witness log accumulates, oracle is heard)?",
    },
    "voice_fit": {
        "type": "noul",
        "instructions": "Consistent voice throughout (no jarring shifts)?",
    },
    "novelty": {
        "type": "noul",
        "instructions": "Contains a novel image or metaphor not common in cyberpunk-noir canon?",
    },
    "density": {
        "type": "score",
        "criteria": ["image density per 100 words", "metaphor novelty", "compression ratio"],
    },
}


def extract_jev_features(
    text: str,
    questions: Dict = None,
    api_key: str = None,
    timeout: int = 30,
) -> List[float]:
    """Score `text` across the canonical questions and return a feature vector.

    Returns a list of floats with one entry per question (in order):
      - noul questions → [probability]
      - score questions → [score]
    """
    if questions is None:
        questions = DEFAULT_QUESTIONS
    if api_key is None:
        api_key = os.environ.get("TYPESAFEAI_KEY")
    if not api_key:
        raise ValueError("TYPESAFEAI_KEY env var not set")

    payload = {
        "model": "jev-latest",
        "state": f"Quilt substrate walker canon lore: {text[:2000]}",
        "questions": questions,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(JEV_URL, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        # Return zeros if API fails — caller can detect
        return [0.0] * len(questions)

    features = []
    for name, q in questions.items():
        answer = data.get("answers", {}).get(name, {})
        if q["type"] == "noul":
            features.append(float(answer.get("noul", 0.0)))
        elif q["type"] == "score":
            features.append(float(answer.get("score", 0.0)))
        else:
            features.append(0.0)
    return features


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "The substrate walker is a 4D cell graph."
    feats = extract_jev_features(text)
    print(f"JEV features ({len(feats)}-dim):")
    for name, f in zip(DEFAULT_QUESTIONS.keys(), feats):
        print(f"  {name}: {f:.3f}")
