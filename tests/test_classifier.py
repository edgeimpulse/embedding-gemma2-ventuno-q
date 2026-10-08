# SPDX-License-Identifier: MPL-2.0
"""Offline tests for ZeroShotClassifier: a fake embedder stands in for the llama.cpp runner.

    python -m pytest tests
"""

import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app/bricks"))

from embedding_gemma import ZeroShotClassifier, l2_normalize  # noqa: E402

DIM = 8


class FakeEmbedder:
    """Maps each text to a fixed one-hot vector (slots 0-2), so scores are predictable."""

    dim = DIM

    def embed_texts(self, texts, task="query"):
        out = np.zeros((len(texts), DIM), np.float32)
        for i, t in enumerate(texts):
            out[i, zlib.crc32(t.encode()) % 3] = 1.0
        return out


def unit(i):
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v


def test_labels_are_deduplicated_and_stripped():
    clf = ZeroShotClassifier(FakeEmbedder(), [" cat ", "dog", "cat", ""])
    assert clf.labels == ["cat", "dog"]


def test_probabilities_sum_to_one_and_sort_best_first():
    clf = ZeroShotClassifier(FakeEmbedder(), ["cat", "dog"])
    res = clf.classify(l2_normalize(np.ones(DIM, np.float32)))
    assert abs(sum(p for _, p, _ in res) - 1) < 1e-5
    assert [s for _, _, s in res] == sorted((s for _, _, s in res), reverse=True)


def test_examples_override_text_prior():
    clf = ZeroShotClassifier(FakeEmbedder(), ["a", "b"])
    img = unit(5)  # slot no label's text vector uses
    for _ in range(3):
        clf.add_example("b", img)
    assert clf.classify(img)[0][0] == "b"
    assert clf.example_counts() == {"a": 0, "b": 3}
    clf.clear_examples()
    assert clf.example_counts() == {"a": 0, "b": 0}


def test_add_example_creates_unknown_label():
    clf = ZeroShotClassifier(FakeEmbedder(), ["a"])
    clf.add_example("new", unit(1))
    assert "new" in clf.labels


def test_empty_classifier_returns_nothing():
    assert ZeroShotClassifier(FakeEmbedder()).classify(unit(0)) == []
