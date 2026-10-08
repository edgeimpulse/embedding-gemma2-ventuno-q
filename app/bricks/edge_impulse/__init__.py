"""Minimal Edge Impulse API client for EmbeddingGemma dataset workflows.

Uses a project API key (x-api-key). Only the endpoints this app needs:
  - list samples and download their images (Studio API)
  - change a sample label (Studio API)
  - upload new images with a label (Ingestion API)
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator

import requests

STUDIO = os.getenv("EI_STUDIO_HOST", "https://studio.edgeimpulse.com").rstrip("/")
INGESTION = os.getenv("EI_INGESTION_HOST", "https://ingestion.edgeimpulse.com").rstrip("/")


class EdgeImpulseError(RuntimeError):
    pass


class EdgeImpulse:
    def __init__(self, api_key: str, project_id: int | None = None, timeout: float = 30.0):
        if not api_key:
            raise EdgeImpulseError("An Edge Impulse project API key is required")
        self.s = requests.Session()
        self.s.headers["x-api-key"] = api_key
        self.timeout = timeout
        self.project_id = int(project_id) if project_id else self._resolve_project()

    # Plumbing ------------------------------------------------------------------------
    def _api(self, method: str, path: str, project: bool = True, **kw) -> dict:
        """Call the Studio API. path is relative to the project unless project=False."""
        url = f"{STUDIO}/v1/api/{self.project_id}{path}" if project else f"{STUDIO}/v1/api/{path}"
        for attempt in range(4):
            r = self.s.request(method, url, timeout=self.timeout, **kw)
            if r.status_code in (429, 502, 503, 504) and attempt < 3:
                time.sleep(2**attempt)
                continue
            break
        try:
            body = r.json()
        except ValueError:
            raise EdgeImpulseError(f"{method} {path}: HTTP {r.status_code}") from None
        if not body.get("success", r.ok):
            raise EdgeImpulseError(f"{method} {path}: {body.get('error', r.status_code)}")
        return body

    def _resolve_project(self) -> int:
        projects = self._api("GET", "projects", project=False).get("projects", [])
        if len(projects) != 1:
            raise EdgeImpulseError("Set EI_PROJECT_ID: the API key does not map to exactly one project")
        return int(projects[0]["id"])

    # Project -------------------------------------------------------------------------
    def project_info(self) -> dict:
        body = self._api("GET", "")
        p, d = body.get("project", {}), body.get("dataSummary", {})
        return {
            "id": self.project_id,
            "name": p.get("name"),
            "labels": d.get("labels", []),
            "sample_count": d.get("dataCount"),
            "studio_url": f"{STUDIO}/studio/{self.project_id}",
        }

    def count(self, category: str = "training") -> int:
        return int(self._api("GET", "/raw-data/count", params={"category": category})["count"])

    def iter_samples(self, category: str = "training", labels: list[str] | None = None, page: int = 100) -> Iterator[dict]:
        """Yield image samples (id, filename, label, category) of a category."""
        offset = 0
        while True:
            params = {"category": category, "limit": page, "offset": offset}
            if labels:
                params["labels"] = json.dumps(labels)
            samples = self._api("GET", "/raw-data", params=params).get("samples", [])
            for s in samples:
                if s.get("chartType") == "image" and not s.get("isDisabled"):
                    yield {"id": s["id"], "filename": s["filename"], "label": s.get("label", ""), "category": category}
            if len(samples) < page:
                return
            offset += page

    def sample_image(self, sample_id: int) -> bytes:
        url = f"{STUDIO}/v1/api/{self.project_id}/raw-data/{sample_id}/image"
        r = self.s.get(url, timeout=self.timeout)
        if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/"):
            raise EdgeImpulseError(f"Could not download image for sample {sample_id}: HTTP {r.status_code}")
        return r.content

    def set_label(self, sample_id: int, label: str) -> None:
        self._api("POST", f"/raw-data/{sample_id}/edit-label", json={"label": label})

    # Ingestion -----------------------------------------------------------------------
    def upload_image(self, jpeg: bytes, label: str, category: str = "training", filename: str | None = None) -> dict:
        if category not in ("training", "testing", "anomaly"):
            raise ValueError("category must be training, testing or anomaly")
        filename = filename or f"{label or 'unlabeled'}.jpg"
        headers = {"x-add-date-id": "1", "x-disallow-duplicates": "1"}
        if label:
            headers["x-label"] = label
        r = self.s.post(
            f"{INGESTION}/api/{category}/files",
            headers=headers,
            files={"data": (filename, jpeg, "image/jpeg")},
            timeout=self.timeout,
        )
        if r.status_code != 200:
            raise EdgeImpulseError(f"Upload failed: HTTP {r.status_code} {r.text[:200]}")
        body = r.json()
        if not body.get("success"):
            raise EdgeImpulseError(f"Upload failed: {body.get('error')}")
        return body
