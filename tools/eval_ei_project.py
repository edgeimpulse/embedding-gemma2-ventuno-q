"""Evaluate EmbeddingGemma 2 on an Edge Impulse image project, the way the app uses it.

Embeds the train and test images of a project, then reports:
  * zero-shot accuracy (text prompts from the label names only)
  * few-shot accuracy (class prototypes from the training images, scored on the test set)
  * k-NN accuracy (nearest training image)

    LOCAL_DEV=true EI_API_KEY=... python tools/eval_ei_project.py [--prompt "a photo of a {} egg"]

Embeddings are cached in out/emb-<project>.npz so re-runs only re-score.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app/bricks"), str(ROOT / "app/python")]

from embedding_gemma import EmbeddingGemma, ZeroShotClassifier, l2_normalize  # noqa: E402
from edge_impulse import EdgeImpulse  # noqa: E402


def embed_category(ei: EdgeImpulse, eg: EmbeddingGemma, category: str):
    ids, labels, vecs, t = [], [], [], 0.0
    for s in ei.iter_samples(category):
        img = ei.sample_image(s["id"])
        t0 = time.perf_counter()
        vecs.append(eg.embed_image(img))
        t += time.perf_counter() - t0
        ids.append(s["id"])
        labels.append(s["label"])
    n = max(len(ids), 1)
    print(f"  {category}: {len(ids)} images, {1000 * t / n:.0f} ms/image embed")
    return np.array(ids), np.array(labels), np.stack(vecs) if vecs else np.zeros((0, eg.dim))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="{}", help="Label prompt template, e.g. 'a photo of a {} egg'")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--refresh", action="store_true", help="Re-download and re-embed")
    a = ap.parse_args()

    ei = EdgeImpulse(os.environ["EI_API_KEY"], os.getenv("EI_PROJECT_ID"))
    info = ei.project_info()
    print(f"Project {info['id']} '{info['name']}', labels {info['labels']}")
    eg = EmbeddingGemma(dim=768)
    eg.wait_until_ready(timeout=60)

    cache = ROOT / f"out/emb-{ei.project_id}.npz"
    if cache.exists() and not a.refresh:
        z = np.load(cache)
        tr_l, tr_e, te_l, te_e = z["tr_l"], z["tr_e"], z["te_l"], z["te_e"]
    else:
        _, tr_l, tr_e = embed_category(ei, eg, "training")
        _, te_l, te_e = embed_category(ei, eg, "testing")
        cache.parent.mkdir(exist_ok=True)
        np.savez(cache, tr_l=tr_l, tr_e=tr_e, te_l=te_l, te_e=te_e)

    # Matryoshka truncation is done on cached 768-d vectors.
    eg.dim = a.dim
    tr_e, te_e = l2_normalize(tr_e[:, : a.dim]), l2_normalize(te_e[:, : a.dim])
    labels = sorted(set(tr_l))
    keep = np.isin(te_l, labels)
    if (~keep).any():
        print(f"  skipping {int((~keep).sum())} test samples with labels not in training: {sorted(set(te_l[~keep]))}")
    te_l, te_e = te_l[keep], te_e[keep]
    ev_l, ev_e = (te_l, te_e) if len(te_l) else (tr_l, tr_e)

    zs = ZeroShotClassifier(eg, labels, prompt=a.prompt)
    pred = [zs.classify(e)[0][0] for e in ev_e]
    print(f"zero-shot  ({a.prompt!r}, dim {a.dim}): {np.mean(np.array(pred) == ev_l):.1%} on {len(ev_l)}")

    if len(te_l):
        for l in labels:
            for e in tr_e[tr_l == l]:
                zs.add_example(l, e)
        pred = [zs.classify(e)[0][0] for e in te_e]
        print(f"few-shot   (train prototypes + text): {np.mean(np.array(pred) == te_l):.1%}")
        nn = tr_l[np.argmax(te_e @ tr_e.T, axis=1)]
        print(f"1-NN       (nearest training image):  {np.mean(nn == te_l):.1%}")
        for l in labels:
            m = te_l == l
            print(f"    {l:>12}: few-shot {np.mean(np.array(pred)[m] == l):.0%} of {m.sum()}")


if __name__ == "__main__":
    main()
