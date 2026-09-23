"""Canon tagger — adds JEV-derived canon metadata to lore files.

Reads lore files, extracts canon-related terms, and tags them with
doctrine anchors.

Usage:
    python3 canon_tagger.py lore.md
    python3 canon_tagger.py lore_inbox/*.md
"""
import sys
import json
import re
from pathlib import Path


DOCTRINE_TERMS = {
    "cells_are_scars": ["scar", "scars", "scarred", "scarring", "wound", "wounds", "hurt", "heal", "remembering"],
    "oracle_is_heard": ["canon gate", "gate", "oracle", "hear", "heard", "audible", "frequency", "chord"],
    "witness_log_is_prediction": ["witness log", "log", "accumulate", "predict", "future", "past"],
    "canon_gate_is_chord": ["chord", "notes", "frequencies", "tone", "lock", "sustained"],
    "substrate_quantum": ["substrate", "quantum", "amplitudes", "interference", "collapse"],
}


def tag_lore(text):
    """Tag a lore with doctrine anchors based on term matches."""
    text_lower = text.lower()
    tags = {}
    for doctrine, terms in DOCTRINE_TERMS.items():
        # Count occurrences
        count = sum(text_lower.count(t) for t in terms)
        if count > 0:
            tags[doctrine] = count
    return tags


def tag_file(path):
    """Tag a lore file."""
    content = Path(path).read_text()
    tags = tag_lore(content)
    return {
        "path": str(path),
        "size": len(content),
        "doctrine_tags": tags,
        "primary_doctrine": max(tags, key=tags.get) if tags else None,
        "doctrine_count": len(tags),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 canon_tagger.py <lore.md> [lore2.md ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        result = tag_file(path)
        print(json.dumps(result, indent=2))
