# SPDX-License-Identifier: MPL-2.0
"""EmbeddingGemma 2 on VENTUNO Q, with Edge Impulse.

Camera frames are embedded on the board with EmbeddingGemma 2 (via the local
embedding_gemma brick) and used to:

  * Classify:  zero-shot labels from text, refined by few-shot examples.
  * Collect:   send the current frame to an Edge Impulse project with the predicted
               (or a chosen) label: a labelled dataset straight from the camera.
  * Curate:    embed every image in an Edge Impulse project, then suggest labels for
               unlabelled samples and flag samples whose label looks wrong.
  * Search:    find the project images closest to a text query or to the camera frame.

The web UI is served by the web_ui brick on port 7000 (App Lab reserves it for the UI).
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from arduino.app_bricks.web_ui import WebUI
from arduino.app_peripherals.camera import Camera
from arduino.app_utils import App, Logger
from fastapi import Body, HTTPException

from edge_impulse import EdgeImpulse, EdgeImpulseError
from embedding_gemma import EmbeddingGemma, ZeroShotClassifier, l2_normalize, to_image_bytes

logger = Logger("EmbeddingGemmaApp")

DATA_DIR = Path(os.getenv("APP_DATA_DIR", "/app/data"))
DEFAULT_LABELS = os.getenv("DEFAULT_LABELS", "a person,a cup,a phone,an Arduino board,a plant")
CLASSIFY_INTERVAL_S = float(os.getenv("CLASSIFY_INTERVAL_S", "0.5"))
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE")  # None: first available camera

eg = EmbeddingGemma()
ui = WebUI(assets_dir_path=os.getenv("APP_ASSETS_DIR", "/app/assets"))


def open_camera():
    if CAMERA_SOURCE == "none":
        return None
    try:
        return Camera(CAMERA_SOURCE, resolution=(640, 480), fps=15)
    except Exception as e:  # noqa: BLE001 - no camera: the UI falls back to image upload
        logger.warning(f"No camera ({e}). Classify uploaded images instead.")
        return None


camera = open_camera()


class State:
    """Everything the UI can read or change, guarded by one lock.

    last_frame is the image the last result came from: a BGR ndarray from the camera,
    or the encoded bytes of an uploaded image.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.ready = False
        self.status = "Starting the EmbeddingGemma runner (the first boot downloads ~0.9 GB)"
        self.classifier: ZeroShotClassifier | None = None
        self.prompt = "a photo of {}"
        self.labels = [l.strip() for l in DEFAULT_LABELS.split(",") if l.strip()]
        self.live = True
        self.last_frame: np.ndarray | None = None
        self.last_embedding: np.ndarray | None = None
        self.last_result: list = []
        self.embed_ms = deque(maxlen=20)
        self.ei: EdgeImpulse | None = None
        self.ei_info: dict | None = None
        # Project index: one row per sample.
        self.index_meta: list[dict] = []
        self.index_emb = np.zeros((0, eg.dim), np.float32)
        self.index_job = {"running": False, "done": 0, "total": 0, "error": None}


S = State()


# Edge Impulse connection --------------------------------------------------------------
def connect_ei(api_key: str, project_id: str | None = None) -> dict:
    ei = EdgeImpulse(api_key.strip(), project_id or None)
    info = ei.project_info()
    with S.lock:
        S.ei, S.ei_info = ei, info
        S.index_meta, S.index_emb = [], np.zeros((0, eg.dim), np.float32)
        _load_index_cache()
    logger.info(f"Connected to Edge Impulse project {info['id']} ({info['name']})")
    return info


def _index_cache() -> Path:
    return DATA_DIR / f"ei-index-{S.ei.project_id}-{eg.dim}.npz"


def _load_index_cache() -> None:
    p = _index_cache()
    if p.exists():
        z = np.load(p, allow_pickle=True)
        S.index_meta, S.index_emb = list(z["meta"]), z["emb"]
        logger.info(f"Loaded {len(S.index_meta)} cached embeddings from {p}")


def _save_index_cache() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(_index_cache(), meta=np.array(S.index_meta, dtype=object), emb=S.index_emb)


def build_index() -> None:
    """Embed every image sample of the project (training and testing). Resumable."""
    ei = S.ei
    job = S.index_job
    try:
        samples = [s for c in ("training", "testing") for s in ei.iter_samples(c)]
        with S.lock:
            known = {m["id"]: i for i, m in enumerate(S.index_meta)}
            job.update(total=len(samples), done=0, error=None)
        live_ids = {s["id"] for s in samples}
        metas, embs = [], []
        for s in samples:
            if not job["running"]:
                break
            if s["id"] in known:  # Re-use the cached vector, refresh the label.
                embs.append(S.index_emb[known[s["id"]]])
            else:
                embs.append(eg.embed_image(ei.sample_image(s["id"])))
            metas.append(s)
            job["done"] += 1
        with S.lock:
            # Keep cached rows not reached yet (stopped early) if they still exist.
            seen = {m["id"] for m in metas}
            for m, e in zip(S.index_meta, S.index_emb):
                if m["id"] not in seen and m["id"] in live_ids:
                    metas.append(m)
                    embs.append(e)
            S.index_meta = metas
            S.index_emb = np.stack(embs) if embs else np.zeros((0, eg.dim), np.float32)
            _save_index_cache()
    except Exception as e:  # noqa: BLE001
        logger.exception("Indexing failed")
        job["error"] = str(e)
    finally:
        job["running"] = False
        ui.send_message("index", index_status())


def index_status() -> dict:
    with S.lock:
        labels = [m["label"] for m in S.index_meta]
        return {
            **S.index_job,
            "indexed": len(S.index_meta),
            "labels": {l: labels.count(l) for l in sorted(set(labels))},
        }


def project_prototypes() -> tuple[list[str], np.ndarray]:
    """Mean embedding of each label's training images."""
    with S.lock:
        meta, emb = S.index_meta, S.index_emb
    labels = sorted({m["label"] for m in meta if m["label"] and m["category"] == "training"})
    protos = []
    for l in labels:
        rows = [i for i, m in enumerate(meta) if m["label"] == l and m["category"] == "training"]
        protos.append(l2_normalize(emb[rows].mean(axis=0)))
    return labels, np.stack(protos) if protos else np.zeros((0, eg.dim), np.float32)


# Classifier ---------------------------------------------------------------------------
def rebuild_classifier() -> None:
    with S.lock:
        old = S.classifier
        clf = ZeroShotClassifier(eg, S.labels, prompt=S.prompt)
        if old:  # Keep few-shot examples of labels that survive.
            for l, ex in old._examples.items():
                if l in clf.labels:
                    clf._examples[l] = list(ex)
        S.classifier = clf


def classify_loop() -> None:
    """Background: wait for the runner, then embed and classify frames."""
    try:
        eg.wait_until_ready()
    except TimeoutError as e:
        S.status = str(e)
        ui.send_message("status", status())
        return
    rebuild_classifier()
    S.ready = True
    S.status = "Ready"
    ui.send_message("status", status())
    if (key := os.getenv("EI_API_KEY")):
        try:
            connect_ei(key, os.getenv("EI_PROJECT_ID"))
            ui.send_message("status", status())
        except EdgeImpulseError as e:
            logger.warning(f"Edge Impulse connection from EI_API_KEY failed: {e}")

    while True:
        if camera is None or not S.live:
            time.sleep(0.2)
            continue
        t0 = time.monotonic()
        try:
            frame = camera.capture()
            if frame is None:
                continue
            t1 = time.perf_counter()
            emb = eg.embed_image(frame)
            S.embed_ms.append(1000 * (time.perf_counter() - t1))
            with S.lock:
                result = S.classifier.classify(emb) if S.classifier else []
                S.last_frame, S.last_embedding, S.last_result = frame, emb, result
            ui.send_message("classification", classification_payload())
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Classification failed: {e}")
            time.sleep(1)
        time.sleep(max(0.0, CLASSIFY_INTERVAL_S - (time.monotonic() - t0)))


def classification_payload() -> dict:
    with S.lock:
        return {
            "results": [{"label": l, "probability": p, "score": s} for l, p, s in S.last_result],
            "embed_ms": round(float(np.mean(S.embed_ms)), 1) if S.embed_ms else None,
            "examples": S.classifier.example_counts() if S.classifier else {},
        }


def current_embedding() -> tuple[np.ndarray | bytes, np.ndarray]:
    with S.lock:
        if S.last_frame is None:
            raise HTTPException(409, "No camera frame or uploaded image yet")
        return S.last_frame, S.last_embedding


def require_ei() -> EdgeImpulse:
    if S.ei is None:
        raise HTTPException(400, "Connect an Edge Impulse project first")
    return S.ei


# API ----------------------------------------------------------------------------------
def status() -> dict:
    return {
        "ready": S.ready,
        "status": S.status,
        "camera": camera is not None,
        "dim": eg.dim,
        "labels": S.labels,
        "prompt": S.prompt,
        "live": S.live,
        "project": S.ei_info,
        "index": index_status(),
    }


def api_set_labels(body: dict = Body(...)) -> dict:
    labels = body.get("labels", [])
    if isinstance(labels, str):
        labels = labels.split(",")
    with S.lock:
        S.labels = [l.strip() for l in labels if l.strip()]
        S.prompt = body.get("prompt") or "{}"
        if "{}" not in S.prompt:
            S.prompt += " {}"
    rebuild_classifier()
    return status()


def api_live(body: dict = Body(...)) -> dict:
    S.live = bool(body.get("live", True))
    return status()


def api_add_example(body: dict = Body(...)) -> dict:
    label = (body.get("label") or "").strip()
    if not label:
        raise HTTPException(400, "label is required")
    _, emb = current_embedding()
    with S.lock:
        if label not in S.labels:
            S.labels.append(label)
        S.classifier.add_example(label, emb)
    return classification_payload()


def api_clear_examples() -> dict:
    with S.lock:
        if S.classifier:
            S.classifier.clear_examples()
    return classification_payload()


def api_learn_from_project() -> dict:
    """Use the project's labelled training images as few-shot examples."""
    labels, protos = project_prototypes()
    if not labels:
        raise HTTPException(409, "Index the project first")
    with S.lock:
        S.labels = labels
    rebuild_classifier()
    with S.lock:
        S.classifier.clear_examples()
        for i, m in enumerate(S.index_meta):
            if m["category"] == "training" and m["label"] in labels:
                S.classifier.add_example(m["label"], S.index_emb[i])
    return status()


def api_connect(body: dict = Body(...)) -> dict:
    try:
        connect_ei(body.get("api_key", ""), body.get("project_id"))
    except EdgeImpulseError as e:
        raise HTTPException(400, str(e)) from e
    return status()


def api_upload(body: dict = Body(...)) -> dict:
    ei = require_ei()
    frame, emb = current_embedding()
    label = (body.get("label") or "").strip()
    if not label and S.last_result:
        label = S.last_result[0][0]
    category = body.get("category", "training")
    try:
        res = ei.upload_image(to_image_bytes(frame), label, category)
    except EdgeImpulseError as e:
        raise HTTPException(502, str(e)) from e
    return {"label": label, "category": category, "files": res.get("files", [])}


def api_index_start() -> dict:
    require_ei()
    if not S.index_job["running"]:
        S.index_job["running"] = True
        threading.Thread(target=build_index, daemon=True, name="ei-index").start()
    return index_status()


def api_index_stop() -> dict:
    S.index_job["running"] = False
    return index_status()


def api_index_status() -> dict:
    return index_status()


def api_curate(body: dict = Body(default={})) -> dict:
    """Suggest labels for unlabelled samples and flag likely mislabelled ones.

    Suggestions use the labels of the training set. Each sample is scored against the
    leave-one-out prototype of every label, so a sample can't vote for its own label.
    """
    # Mismatches need a clear margin: below ~0.01 they were mostly noise on test data.
    min_margin = float(body.get("min_margin", 0.01))
    with S.lock:
        meta, emb = list(S.index_meta), S.index_emb.copy()
    labels = sorted({m["label"] for m in meta if m["label"] and m["category"] == "training"})
    if not labels:
        raise HTTPException(409, "No labelled training samples indexed")
    sums = {l: emb[[i for i, m in enumerate(meta) if m["label"] == l]].sum(axis=0) for l in labels}
    counts = {l: sum(m["label"] == l for m in meta) for l in labels}
    suggestions = []
    for i, m in enumerate(meta):
        scores = []
        for l in labels:
            s, n = sums[l], counts[l]
            if m["label"] == l:
                s, n = s - emb[i], n - 1
            scores.append(float(l2_normalize(s) @ emb[i]) if n > 0 else -1.0)
        order = np.argsort(scores)[::-1]
        best = labels[order[0]]
        margin = scores[order[0]] - (scores[order[1]] if len(order) > 1 else 0)
        kind = "unlabelled" if not m["label"] else "unknown label" if m["label"] not in labels else "mismatch"
        if best != m["label"] and (kind != "mismatch" or margin >= min_margin):
            suggestions.append({
                **m,
                "suggested": best,
                "margin": round(margin, 4),
                "scores": dict(zip(labels, [round(s, 4) for s in scores])),
                "kind": kind,
            })
    # Labels outside the training vocabulary first, then the most confident mismatches.
    suggestions.sort(key=lambda s: (s["kind"] == "mismatch", -s["margin"]))
    return {"labels": labels, "suggestions": suggestions, "checked": len(meta)}


def api_relabel(body: dict = Body(...)) -> dict:
    ei = require_ei()
    sid, label = int(body["id"]), str(body["label"]).strip()
    try:
        ei.set_label(sid, label)
    except EdgeImpulseError as e:
        raise HTTPException(502, str(e)) from e
    with S.lock:
        for m in S.index_meta:
            if m["id"] == sid:
                m["label"] = label
        _save_index_cache()
    return {"id": sid, "label": label}


def api_search(body: dict = Body(...)) -> dict:
    """Nearest project images to a text query, or to the current camera frame."""
    with S.lock:
        meta, emb = list(S.index_meta), S.index_emb
    if not meta:
        raise HTTPException(409, "Index the project first")
    q = body.get("query", "").strip()
    qv = eg.embed_text(q, "query") if q else current_embedding()[1]
    sims = emb @ qv
    top = np.argsort(-sims)[: int(body.get("k", 12))]
    pid = S.ei.project_id
    return {
        "query": q or "(camera frame)",
        "results": [
            {**meta[i], "score": round(float(sims[i]), 4),
             "studio_url": f"https://studio.edgeimpulse.com/studio/{pid}/acquisition/training?sampleId={meta[i]['id']}"}
            for i in top
        ],
    }


def api_thumbnail(sample_id: int):
    from fastapi.responses import Response

    ei = require_ei()
    try:
        return Response(ei.sample_image(sample_id), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})
    except EdgeImpulseError as e:
        raise HTTPException(404, str(e)) from e


def api_classify_image(body: dict = Body(...)) -> dict:
    """Classify an uploaded image (base64 JPEG/PNG). Works without a camera."""
    import base64

    try:
        img = base64.b64decode(body["image"].split(",")[-1])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, "image must be base64") from e
    t1 = time.perf_counter()
    emb = eg.embed_image(img)
    S.embed_ms.append(1000 * (time.perf_counter() - t1))
    with S.lock:
        result = S.classifier.classify(emb) if S.classifier else []
        S.last_frame, S.last_embedding, S.last_result = img, emb, result
    payload = classification_payload()
    ui.send_message("classification", payload)
    return payload


def api_embed_text(body: dict = Body(...)) -> dict:
    """Embedding as a service: other apps on the board can call this."""
    v = eg.embed_texts(body.get("texts", []), body.get("task", "query"))
    return {"dim": eg.dim, "embeddings": v.tolist()}


for method, path, fn in [
    ("GET", "/api/status", status),
    ("POST", "/api/labels", api_set_labels),
    ("POST", "/api/live", api_live),
    ("POST", "/api/examples", api_add_example),
    ("DELETE", "/api/examples", api_clear_examples),
    ("POST", "/api/learn-from-project", api_learn_from_project),
    ("POST", "/api/ei/connect", api_connect),
    ("POST", "/api/ei/upload", api_upload),
    ("POST", "/api/ei/index", api_index_start),
    ("DELETE", "/api/ei/index", api_index_stop),
    ("GET", "/api/ei/index", api_index_status),
    ("POST", "/api/ei/curate", api_curate),
    ("POST", "/api/ei/relabel", api_relabel),
    ("POST", "/api/search", api_search),
    ("GET", "/api/ei/thumb/{sample_id}", api_thumbnail),
    ("POST", "/api/classify", api_classify_image),
    ("POST", "/api/embed/text", api_embed_text),
]:
    ui.expose_api(method, path, fn)

if camera is not None:
    ui.expose_camera("/api/camera", camera)

ui.on_connect(lambda sid: ui.send_message("status", status(), room=sid))
threading.Thread(target=classify_loop, daemon=True, name="classify").start()

App.run()
