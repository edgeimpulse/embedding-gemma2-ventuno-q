# SPDX-License-Identifier: MPL-2.0
"""EmbeddingGemma 2 brick: multimodal embeddings from a llama.cpp runner container.

Text and images are embedded into one 768-d space. Vectors are L2-normalized, so the
dot product of two of them is their cosine similarity.

    from embedding_gemma import EmbeddingGemma

    eg = EmbeddingGemma()
    labels = eg.embed_texts(["a cat", "a dog"], task="query")
    image = eg.embed_image(frame)              # BGR ndarray from Camera, or JPEG/PNG bytes
    scores = labels @ image                    # cosine similarities
"""

from __future__ import annotations

import base64
import os
import threading
import time
from collections.abc import Sequence

import numpy as np
import requests

try:  # Inside App Lab: register with the app lifecycle and use its logger.
    from arduino.app_utils import Logger, brick
except ImportError:  # Plain Python (tests on a laptop).
    import logging

    def brick(cls):  # type: ignore[no-redef]
        return cls

    Logger = logging.getLogger  # type: ignore[assignment,misc]

logger = Logger("EmbeddingGemma")

RUNNER_HOST = "embedding-gemma-runner"
RUNNER_PORT = 8080

# EmbeddingGemma 2 task prompts (model card). Images, audio and video take no prefix.
TASK_PROMPTS = {
    "query": "task: search result | query: {}",
    "document": "title: none | text: {}",
    "classification": "task: classification | query: {}",
    "clustering": "task: clustering | query: {}",
    "similarity": "task: sentence similarity | query: {}",
    "none": "{}",
}
MRL_DIMS = (768, 512, 256, 128)


def _default_base_url() -> str:
    if url := os.getenv("EMBEDDING_GEMMA_URL"):
        return url.rstrip("/")
    host = RUNNER_HOST
    if os.getenv("LOCAL_DEV", "false").lower() == "true":
        host = "127.0.0.1"
    elif remote := os.getenv("REMOTE_DEV"):
        host = remote
    return f"http://{host}:{RUNNER_PORT}"


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def to_image_bytes(image) -> bytes:
    """JPEG bytes from a BGR ndarray (Camera frames), a PIL image, a path, or bytes."""
    if isinstance(image, (bytes, bytearray, memoryview)):
        return bytes(image)
    if isinstance(image, str):
        with open(image, "rb") as f:
            return f.read()
    if isinstance(image, np.ndarray):
        import cv2

        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("Could not encode frame as JPEG")
        return buf.tobytes()
    if hasattr(image, "save"):  # PIL
        import io

        out = io.BytesIO()
        image.convert("RGB").save(out, format="JPEG", quality=90)
        return out.getvalue()
    raise TypeError(f"Unsupported image type: {type(image)}")


@brick
class EmbeddingGemma:
    """Client for the EmbeddingGemma 2 llama.cpp runner."""

    def __init__(self, base_url: str | None = None, dim: int | None = None, timeout: float = 60.0):
        self.base_url = (base_url or _default_base_url()).rstrip("/")
        self.dim = int(dim or os.getenv("EMBEDDING_GEMMA_DIM", "768"))
        if self.dim not in MRL_DIMS:
            raise ValueError(f"dim must be one of {MRL_DIMS}")
        self.timeout = timeout
        self._session = requests.Session()
        self._marker: str | None = None
        self._modalities: dict = {}
        self._lock = threading.Lock()

    # App lifecycle -------------------------------------------------------------------
    def start(self) -> None:
        logger.info(f"Using EmbeddingGemma runner at {self.base_url}")

    def stop(self) -> None:
        self._session.close()

    # Runner state --------------------------------------------------------------------
    def is_ready(self) -> bool:
        try:
            return self._session.get(f"{self.base_url}/health", timeout=3).status_code == 200
        except requests.RequestException:
            return False

    def wait_until_ready(self, timeout: float = 1800, poll: float = 2.0) -> None:
        """Block until the runner has the model loaded (the first boot downloads it)."""
        deadline = time.monotonic() + timeout
        while not self.is_ready():
            if time.monotonic() > deadline:
                raise TimeoutError(f"EmbeddingGemma runner not ready at {self.base_url}")
            time.sleep(poll)
        self._load_props()

    def _load_props(self) -> None:
        props = self._session.get(f"{self.base_url}/props", timeout=10).json()
        # The media marker is randomized per server start: it must be read, not hardcoded.
        self._marker = props.get("media_marker") or "<__media__>"
        self._modalities = props.get("modalities", {})
        logger.info(f"Runner ready, modalities: {self._modalities}")

    @property
    def supports_vision(self) -> bool:
        if self._marker is None:
            self._load_props()
        return bool(self._modalities.get("vision"))

    # Embeddings ----------------------------------------------------------------------
    def _finish(self, vecs: np.ndarray) -> np.ndarray:
        # Matryoshka: keep the leading dims, then re-normalize.
        return l2_normalize(vecs[..., : self.dim])

    def embed_texts(self, texts: Sequence[str], task: str = "query") -> np.ndarray:
        """Embed texts with an EmbeddingGemma task prompt. Returns (n, dim)."""
        if isinstance(texts, str):
            texts = [texts]
        template = TASK_PROMPTS[task]
        payload = {"input": [template.format(t) for t in texts]}
        r = self._session.post(f"{self.base_url}/v1/embeddings", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return self._finish(np.array([d["embedding"] for d in data], dtype=np.float32))

    def embed_text(self, text: str, task: str = "query") -> np.ndarray:
        return self.embed_texts([text], task)[0]

    def embed_image(self, image) -> np.ndarray:
        """Embed one image (BGR ndarray, PIL image, path, or encoded bytes). Returns (dim,)."""
        if self._marker is None:
            self._load_props()
        b64 = base64.b64encode(to_image_bytes(image)).decode()
        payload = {"content": {"prompt_string": self._marker, "multimodal_data": [b64]}}
        r = self._session.post(f"{self.base_url}/embeddings", json=payload, timeout=self.timeout)
        if r.status_code == 500 and "media marker" in r.text.lower():
            self._load_props()  # The runner restarted with a new marker.
            payload["content"]["prompt_string"] = self._marker
            r = self._session.post(f"{self.base_url}/embeddings", json=payload, timeout=self.timeout)
        r.raise_for_status()
        emb = np.asarray(r.json()[0]["embedding"], dtype=np.float32).reshape(-1)
        return self._finish(emb)

    def embed_images(self, images) -> np.ndarray:
        return np.stack([self.embed_image(i) for i in images]) if images else np.zeros((0, self.dim), np.float32)


class ZeroShotClassifier:
    """Zero-shot labels from text prompts, optionally refined with few-shot image examples.

    Text prototypes come from the label names. When labelled images are added, each label
    also gets an image prototype (mean of its examples), which is usually much stronger
    than the text alone. The score blends the two.
    """

    def __init__(self, eg: EmbeddingGemma, labels: Sequence[str] = (), prompt: str = "{}", task: str = "query"):
        self.eg = eg
        self.prompt = prompt
        self.task = task
        self.labels: list[str] = []
        self._text: np.ndarray = np.zeros((0, eg.dim), np.float32)
        self._examples: dict[str, list[np.ndarray]] = {}
        if labels:
            self.set_labels(labels)

    def set_labels(self, labels: Sequence[str]) -> None:
        labels = [l.strip() for l in labels if l and l.strip()]
        self.labels = list(dict.fromkeys(labels))
        if self.labels:
            self._text = self.eg.embed_texts([self.prompt.format(l) for l in self.labels], self.task)
        else:
            self._text = np.zeros((0, self.eg.dim), np.float32)

    def add_example(self, label: str, embedding: np.ndarray) -> None:
        if label not in self.labels:
            self.set_labels([*self.labels, label])
        self._examples.setdefault(label, []).append(l2_normalize(embedding))

    def clear_examples(self) -> None:
        self._examples.clear()

    def example_counts(self) -> dict[str, int]:
        return {l: len(self._examples.get(l, [])) for l in self.labels}

    def scores(self, image_embedding: np.ndarray, text_weight: float = 0.3) -> np.ndarray:
        """Cosine similarity per label, blending text and image prototypes."""
        if not self.labels:
            return np.zeros(0, np.float32)
        s = self._text @ image_embedding
        for i, label in enumerate(self.labels):
            ex = self._examples.get(label)
            if ex:
                proto = l2_normalize(np.mean(ex, axis=0))
                s[i] = text_weight * s[i] + (1 - text_weight) * float(proto @ image_embedding)
        return s

    def classify(self, image_embedding: np.ndarray, temperature: float = 0.01) -> list[tuple[str, float, float]]:
        """[(label, softmax probability, cosine score)] sorted best first."""
        s = self.scores(image_embedding)
        if s.size == 0:
            return []
        z = (s - s.max()) / temperature
        p = np.exp(z) / np.exp(z).sum()
        order = np.argsort(-s)
        return [(self.labels[i], float(p[i]), float(s[i])) for i in order]


__all__ = ["EmbeddingGemma", "ZeroShotClassifier", "TASK_PROMPTS", "l2_normalize", "to_image_bytes"]
