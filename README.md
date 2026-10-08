# EmbeddingGemma 2 Vision Lab for VENTUNO Q

An [Arduino App Lab](https://docs.arduino.cc/software/app-lab/) app that runs Google EmbeddingGemma 2 on the board and connects it to an Edge Impulse image project. Text and images map into one 768-d vector space, so you can classify camera frames from label names alone, then improve the result with examples.

> Internal repo. Not reviewed for public release.

![Zero-shot and few-shot classification of an egg image](docs/images/classify.png)

## Why

Classic edge AI needs a labelled dataset and a training run before a model can tell two things apart. That is slow when you are still deciding what to detect, and the dataset itself is often the weak point: unlabelled samples, wrong labels, and no easy way to find an example of a case.

A multimodal embedding model removes much of that. It needs no training run to start, so you can type labels and get a working classifier. The same vectors help clean up the dataset you build next. This app shows that loop on a small board: bootstrap with text, collect camera data into Edge Impulse, find the label problems, then train a compact production model in Studio. The embedding model is a data tool here. It does not replace the model you deploy.

## What EmbeddingGemma 2 does

EmbeddingGemma 2 turns text or an image into a 768-number vector. Similar meaning gives nearby vectors, and text and images share one space, so a photo can be compared directly with a phrase.

The app uses that in four ways:

1. **Embed.** A llama.cpp container (`llama-server --embedding`) serves the Q8_0 GGUF model on the board CPU. The brick sends text to `/v1/embeddings` and images to `/embeddings`, capped at 70 vision tokens per image. Vectors are L2-normalized, so a dot product is cosine similarity.
2. **Zero-shot.** Each label becomes a text vector (for example `a photo of a cracked egg`). A frame is scored against every label vector and the scores go through a softmax.
3. **Few-shot.** Labelled images are averaged into one image vector per label. The final score blends 30% text and 70% image vector, so a few examples correct a weak prompt.
4. **Dataset checks.** Every project image is embedded once and cached. Search ranks the cache by similarity to a query. Label checks compare each sample with the other samples of each label (leaving itself out) and flag any whose label disagrees with its nearest group by a clear margin.

Matryoshka output sizes (512, 256, 128) are supported by truncating and re-normalizing the vector.

## What it does

| Feature | How it works |
| --- | --- |
| **Live classification** | Embeds each camera frame (about 50 ms on the development machine) and scores it against your labels. No camera? Choose an image instead. |
| **Zero-shot labels** | Type labels and a prompt template such as `a photo of a {} egg`. No training step. |
| **Few-shot examples** | Add the current frame as an example of a label, or load every labelled training image from an Edge Impulse project. Each label gets an image prototype blended with its text prototype. |
| **Collect** | Upload the current frame to an Edge Impulse project as `training` or `testing`, with the predicted or a chosen label. |
| **Find label issues** | Embeds every project image, then flags unlabelled samples and samples whose label disagrees with their neighbours. Each sample is scored against leave-one-out label prototypes, so it can't vote for its own label. |
| **Relabel** | Applies a suggested label to the sample in Edge Impulse in one click. |
| **Search** | Finds project images closest to a text query or the camera frame. |
| **Embedding service** | `POST /api/embed/text` lets other apps on the board reuse the model. |

![Label issues found in an Edge Impulse project](docs/images/curate.png)

![Text search over project images](docs/images/search.png)

## Layout

```
app/
  app.yaml                    App Lab manifest (bricks and variables)
  python/main.py              FastAPI endpoints, camera loop, index and curate logic
  assets/                     Web UI (served by the web_ui brick on port 7000)
  bricks/embedding_gemma/     Brick: llama.cpp runner container + Python client
  bricks/edge_impulse/        Brick: Edge Impulse Studio and Ingestion API client
tools/eval_ei_project.py      Offline accuracy check against a project
tests/                        Offline unit tests (no board, model or network)
out/                          Cached embeddings for project 1132984 (Eggs AI)
docs/images/                  README screenshots
```

## Run on a board

1. Copy `app/` to the board as an App Lab app (or open it in App Lab).
2. Start it. The first boot downloads about 0.9 GB of GGUF weights from Hugging Face into the app's `data/` folder. Later starts work offline.
3. Open the app UI. The status pill reads **Ready · 768-d** when the model is loaded.
4. Connect Edge Impulse. Either paste a project API key in the UI (Studio > Dashboard > Keys) or set the `EI_API_KEY` secret on the Edge Impulse brick.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_GEMMA_HF_REPO` | `ggml-org/embeddinggemma-2-GGUF:Q8_0` | GGUF repo and quantization |
| `EMBEDDING_GEMMA_IMAGE_MAX_TOKENS` | `70` | Vision tokens per image. Lower is faster. |
| `EMBEDDING_GEMMA_THREADS` | `-1` | llama.cpp CPU threads (-1 lets it choose) |
| `EMBEDDING_GEMMA_DIM` | `768` | Output size: 768, or Matryoshka 512, 256, 128 |
| `EI_API_KEY` | empty | Project API key (secret) |
| `EI_PROJECT_ID` | empty | Resolved from the key when empty |
| `EI_STUDIO_HOST`, `EI_INGESTION_HOST` | production hosts | Override for internal deployments |
| `DEFAULT_LABELS` | `a person,a cup,...` | Labels shown before you set your own |
| `CLASSIFY_INTERVAL_S` | `0.5` | Minimum time between live classifications |
| `CAMERA_SOURCE` | first camera | Set to `none` to use image upload only |

The runner image is `ghcr.io/ggml-org/llama.cpp:server-b11459`, pinned by digest. It is the first build that includes the `gemma-embedding2` architecture.

## Run on a laptop

For development without a board, start any llama.cpp server build that supports EmbeddingGemma 2 on port 8080:

```bash
llama-server --embedding -hf ggml-org/embeddinggemma-2-GGUF:Q8_0 \
  --image-max-tokens 70 --ctx-size 8192 --batch-size 2048 --ubatch-size 2048 --port 8080
```

Then run the app against it. You need the `arduino-app-bricks` Python package (for `WebUI` and `Camera`), plus `fastapi`, `numpy` and `requests`.

```bash
LOCAL_DEV=true CAMERA_SOURCE=none APP_ASSETS_DIR=app/assets APP_DATA_DIR=out/data \
  python app/python/main.py
```

`LOCAL_DEV=true` points the brick at `127.0.0.1` instead of the runner container. Open `http://localhost:7000`.

## Results on the Eggs AI project

Measured by `tools/eval_ei_project.py` on the cached embeddings in `out/emb-1132984.npz`: 69 training images, 27 testing images with `cracked` or `intact` labels (12 more testing samples carry the label `testing` and are skipped).

| Method | Accuracy |
| --- | --- |
| Zero-shot, label names only | 51.9% |
| Zero-shot, prompt `a photo of a {} egg` | 88.9% |
| Few-shot (training prototypes + text) | 96.3% |
| 1-NN (nearest training image) | 100% |

This is a small, single-project test. Treat the numbers as indicative, not a benchmark. The prompt template matters a lot: the same labels gain 37 points with a one-line prompt.

Reproduce, or run it on another project:

```bash
LOCAL_DEV=true EI_API_KEY=ei_... python tools/eval_ei_project.py --prompt "a photo of a {} egg"
```

## Tests

```bash
pip install pytest numpy requests
python -m pytest tests
```

The tests cover the classifier with a fake embedder, so they run without a model or network.

## Notes

- **The API key is never stored by the app.** It lives in memory for the session. Use the brick secret for a persistent connection.
- **Upload and relabel write to your project.** Use a test project or a copy while trying them.
- **The index is cached** in `data/ei-index-<project>-<dim>.npz`, so re-indexing only embeds new samples.
- `tools/applab_compose_test.go.txt` is a Go test for `arduino-app-cli` that renders the brick's compose file. Copy it into the CLI's `orchestrator` package to run it.

## License

[MPL-2.0](LICENSE)
