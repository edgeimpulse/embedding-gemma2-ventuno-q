<!--
DRAFT. Not reviewed or approved. See the Review list at the end before publishing.
-->

## Metadata

| Field | Value |
| --- | --- |
| Headline options | 1. Zero-Shot Labels and Dataset Cleanup on Arduino VENTUNO Q With EmbeddingGemma 2 (recommended: names the technique, the board, and the model)<br>2. Start Classifying Before You Train: EmbeddingGemma 2 on Arduino VENTUNO Q<br>3. Use a Multimodal Embedding Model to Bootstrap and Clean an Edge Impulse Dataset |
| Dek (custom excerpt) | Type your labels, point the camera, and get a working image classifier on the Arduino VENTUNO Q. Then use the same embeddings to find label mistakes before you train the model you deploy. |
| Meta title | EmbeddingGemma 2 on Arduino VENTUNO Q With Edge Impulse |
| Meta description | Run EmbeddingGemma 2 in an Arduino App Lab app on VENTUNO Q to classify camera frames from label names, find label issues, and train a compact model. |
| Primary tag | edge ai |
| Secondary tags | Computer Vision, Projects, arduino |
| Author | Jenny Speelman |
| Publish date | [TBD] |
| Post type | Technical deep dive with a getting-started section |

**Feature image:** `images/embeddinggemma2-ventuno-q-feature.png` (1920 × 1080 PNG). Blue-to-turquoise brand gradient, the headline "Zero-Shot Labels on VENTUNO Q" in Montserrat ExtraBold, a tilted card cropped from the app's live classification screenshot, and the Edge Impulse primary logo at the bottom center. Alt text: "The app's live classification card showing a cracked egg scored 97% for cracked and 3% for intact."

---

# Zero-Shot Labels and Dataset Cleanup on Arduino VENTUNO Q with EmbeddingGemma 2

The slowest part of a new computer vision project is often the start. Before your model can tell two things apart, you need a labeled dataset, and you usually don't know yet which labels you need. Then the dataset turns out to have unlabeled samples and wrong labels, and you don't find out until after a training run.

An [Arduino App Lab](https://docs.edgeimpulse.com/hardware/deployments/run-arduino-app-lab) app can move some of that work to the front. [EmbeddingGemma 2 Vision Lab](https://github.com/edgeimpulse/embedding-gemma2-ventuno-q) runs Google's [EmbeddingGemma 2](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2) on the Arduino VENTUNO Q, so you can type a few labels, point a camera at something, and get a classifier with no training run. It connects to an Edge Impulse project, so the same embeddings also help you collect data, find label problems, and relabel samples. Edge Impulse Studio then turns that clean dataset into a compact model built for your target. EmbeddingGemma 2 gets you to good data faster, and the model you train in Studio is the one you deploy.

## What an embedding model gives you

EmbeddingGemma 2 turns text or an image into a vector of 768 numbers. Similar meaning gives nearby vectors. Text and images share one space, so you can compare a photo directly with a phrase such as "a photo of a cracked egg."

Comparing vectors like this powers every feature in the app:

1. **Zero-shot classification.** Each label becomes a text vector. A camera frame is embedded and scored against every label vector, and the scores go through a softmax.
2. **Few-shot classification.** You add labeled frames as examples. The app averages them into one image vector per label and blends it with the text vector (30% text, 70% image), so a few examples correct a weak prompt.
3. **Search.** Every image in the project is embedded once and cached. You can then rank the dataset by similarity to a text query or the current frame.
4. **Label checks.** Each sample is compared with the other samples of each label. A sample can't vote for its own label. If it sits clearly closer to a different label, the app flags it.

## How it runs on the VENTUNO Q

The [Arduino VENTUNO Q](https://docs.arduino.cc/hardware/ventuno-q/) pairs a Qualcomm Dragonwing IQ8 processor running Ubuntu with an STM32H5F5 microcontroller. [Arduino App Lab](https://docs.edgeimpulse.com/hardware/deployments/run-arduino-app-lab) combines sketches, Python, and AI models into apps that run on the board. App Lab also connects to Edge Impulse, so models you train in Studio show up in your App Lab bricks.

The app uses three bricks from `app.yaml`:

- `arduino:web_ui` serves the browser interface on port 7000.
- `embedding_gemma` runs a llama.cpp server container (`llama-server --embedding`) with the `ggml-org/embeddinggemma-2-GGUF:Q8_0` model. Images are capped at 70 vision tokens, and vectors are L2-normalized so a dot product is cosine similarity.
- `edge_impulse` is a small client for the Studio and Ingestion APIs. It reads samples, uploads frames, and relabels.

The embedding model runs on the board CPU. The first start downloads about 0.9 GB of weights from Hugging Face, and later starts work offline.

## The loop: bootstrap, collect, audit, train

### 1. Bootstrap with text

Type your labels in the app, set a prompt template such as `a photo of a {} egg`, and select **Apply**. The live card scores each camera frame against the labels. No camera? Choose an image instead.

![The app's live classification card showing a cracked egg scored at 97% for "cracked" and 3% for "intact"](../docs/images/classify.png)
Zero-shot and few-shot classification of an egg image. The label rows show the number of examples behind each label.

The template matters. A bare word like "cracked" says little about what an image looks like, but "a photo of a cracked egg" reads like a caption for one. On the egg test set, classifying with the bare label names was right 51.9% of the time. Adding the template `a photo of a {} egg` raised that to 88.9%. See the results below for the full comparison.

### 2. Collect into Edge Impulse

Connect a project by pasting a project API key (**Dashboard > Keys** in Studio) or by setting the `EI_API_KEY` secret on the `edge_impulse` brick. Then use **Upload current frame** to send frames to the project as `training` or `testing` samples. You can pass your own label or use the top prediction. The app doesn't store the API key. It stays in memory for the session.

### 3. Find label issues

**Update embeddings** embeds every image in the project. **Find label issues** then flags unlabeled samples and samples whose label disagrees with their neighbors.

On the [Eggs AI project](https://studio.edgeimpulse.com/public/20687/latest), the app embedded 108 images (47 cracked, 49 intact, and 12 that carried the label `testing`) and flagged those 12. Each one comes with a suggested label, a margin, and a **Relabel** button that writes the change back to the project.

![A grid of 12 egg images flagged as label issues, each with a suggested label and a Relabel button](../docs/images/curate.png)
Twelve of 108 samples flagged. The margin shows how clearly one label beats the next.

Treat the suggestions as a review queue, not an answer. In this run, the last thumbnail in the grid isn't an egg at all, and the suggestion for it ("intact") is wrong. Low margins are the ones to check by eye. The app ignores mismatches below a margin of 0.01, because below that they were mostly noise on test data.

The same index powers search, so you can type "a cracked egg" and see the closest project images.

![Search results for the query "a cracked egg" over project images](../docs/images/search.png)
Text search over the project's images.

### 4. Train the model you deploy

When the labels are fixed, go back to Studio and retrain your existing impulse on the corrected dataset. If the project is new and has no impulse yet, create one first. Then deploy the model. You can run the result on the VENTUNO Q through App Lab, or with `edge-impulse-linux-runner`, which uses hardware acceleration on the Hexagon NPU. See the [VENTUNO Q guide](https://docs.edgeimpulse.com/hardware/boards/arduino-ventuno-q) and [Run Arduino App Lab](https://docs.edgeimpulse.com/hardware/deployments/run-arduino-app-lab) in the docs.

## Results on the Eggs AI project

I checked the approach with `tools/eval_ei_project.py`. It scores cached embeddings for 69 training images and 27 testing images labeled `cracked` or `intact`. The 12 samples labeled `testing` are skipped.

| Method | Accuracy |
| --- | --- |
| Zero-shot, label names only | 51.9% |
| Zero-shot, prompt `a photo of a {} egg` | 88.9% |
| Few-shot (training prototypes plus text) | 96.3% |
| 1-NN (nearest training image) | 100% |

Wrapping the label names in a prompt (`a photo of a {} egg`) took zero-shot accuracy from 51.9% to 88.9%. Adding labeled training images as examples raised it to 96.3%.

## Limitations

- **Small test.** This is one project with 27 test images. Treat the numbers as indicative, not as a benchmark.
- **A data tool, not a deployment.** An embedding model is larger than the compact model you'd ship. Use it to get to a good dataset faster, then train a model sized for your target.
- **Prompts are fragile.** Label names alone scored 51.9%. Test your prompt template before you trust the scores.
- **Writes to your project.** Upload and relabel change the project in Studio. Clone the [public Eggs AI project](https://studio.edgeimpulse.com/public/20687/latest) into your account to try it safely.
- **Latency.** [TBD: measured per-frame embedding time on the VENTUNO Q. The repo's figure of about 50 ms was measured on a development machine, not on the board, so the figure is left out.]

## Try it yourself

You need an Arduino VENTUNO Q set up with Arduino App Lab, a camera (optional), and an Edge Impulse project with an image dataset. No dataset yet? Clone the [public Eggs AI project](https://studio.edgeimpulse.com/public/20687/latest), which has 108 images labeled `cracked` and `intact`.

1. Clone the [GitHub repo](https://github.com/edgeimpulse/embedding-gemma2-ventuno-q) and copy the `app/` folder to the board as an App Lab app, or open it in App Lab.
2. Start the app. Wait for the status pill to read **Ready · 768-d**.
3. Connect your Edge Impulse project with an API key.
4. Enter your labels, apply a prompt template, and watch the live card.
5. Run **Update embeddings**, then **Find label issues**, and review the suggestions.
6. Retrain your impulse in Studio on the corrected labels, then deploy the model.

The full source is on GitHub at [edgeimpulse/embedding-gemma2-ventuno-q](https://github.com/edgeimpulse/embedding-gemma2-ventuno-q).

## Get started

Two accounts get you started: Edge Impulse and Arduino.

- **Create a free Edge Impulse account.** [Sign up for a free Developer plan account](https://studio.edgeimpulse.com/signup?utm_medium=outbound&utm_source=blogpost&utm_campaign=54947365-new-user-acquisition_fall2026&utm_content=google-embedding-blogpost) and start building.
- **Create an Arduino account.** [Sign up at Arduino](https://login.arduino.cc/login) to use Arduino App Lab, and use the same email address as your Edge Impulse account so your models show up in App Lab.

## Build your own with an AI agent

You don't have to write an app like this by hand. Point your coding agent at the Edge Impulse agent skills and describe what you want: an Arduino App Lab app with a camera, a model, and a web UI that reads and writes your Edge Impulse project. The skills give the agent the context it needs for App Lab bricks, EIM model integration, and the Studio and Ingestion APIs, so it can scaffold the app and wire it to your project. You review the result and adjust it to your own labels and data.

Join us in the [Edge Impulse forum](https://forum.edgeimpulse.com/), our [Discord server](https://discord.gg/edgeimpulse), or tag us on social media. We'd love to see what you're building.

---

## Promotion kit

**LinkedIn** (under 700 characters, one link, two or three hashtags):
Starting a vision project usually means labeling data before you know which labels you need. This Arduino App Lab app runs EmbeddingGemma 2 on the Arduino VENTUNO Q so you can type labels, point a camera, and classify with no training run. The same embeddings help you find label mistakes in your Edge Impulse dataset before you train the model you deploy. Code, screenshots, and results on a small egg-inspection project: [link] #EdgeAI #Arduino #ComputerVision

**X or Bluesky** (under 280 characters):
Classify camera frames on the Arduino VENTUNO Q from label names alone, then use the same EmbeddingGemma 2 embeddings to find label issues in your Edge Impulse dataset. Walkthrough and code: [link]

**Newsletter blurb** (under 50 words):
Type your labels, point the camera, and classify on the Arduino VENTUNO Q with EmbeddingGemma 2. Then use the same embeddings to catch label mistakes in your Edge Impulse dataset before you train. Read the walkthrough.

## Review list

| Item | Source | Approver | Status |
| --- | --- | --- | --- |
| Repo is public by the publish date. Remove "Internal repo. Not reviewed for public release." from the README first, and finish the open source review | README.md | Open source review (ei-review-osr), legal | Pending |
| Accuracy table (51.9%, 88.9%, 96.3%, 100%) on 27 test images | README.md, `tools/eval_ei_project.py` | Product or applied research | Pending |
| Per-frame latency on the VENTUNO Q | Not measured on board | Engineering | Missing |
| Screenshots show the working copy "Eggs AI - Jenny (#1132984)". The post links the public project (ID 20687). Confirm the public project matches the data in the results table, and that the working copy's name and ID can show | docs/images/curate.png, classify.png | Jenny Speelman | Pending |
| VENTUNO Q description and Arduino/Qualcomm trademarks | docs.arduino.cc | Arduino, brand | Pending |
| Agent skills link: add the public link to the Edge Impulse agent skills in "Build your own with an AI agent" once it exists | Blog post | DevRel | Pending |
| Third-party model: Google EmbeddingGemma 2 (Apache 2.0 license; the model card also says deployments must follow the Gemma Prohibited Use Policy) | [Model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2), [Hugging Face](https://huggingface.co/google/embeddinggemma-2) | Legal | Pending |
| Tracked signup link (campaign 54947365-new-user-acquisition_fall2026) | Provided by author | Marketing | Pending |
| Feature image (built from the repo screenshot with the official Edge Impulse logo file) | `blog/images/` | Design | Pending |
