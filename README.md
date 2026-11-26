# AuraSense — Real-Time Multimodal Emotion Analysis

A privacy-preserving, live-only research dashboard built with **Python**, **Streamlit** and **`streamlit-webrtc`**, using a Material Design 3 pastel interface. It works only from a live webcam and microphone. There are no file uploads, no clip inspector and no saved recordings, and no model is trained.

## Highest-priority feature: dual-model facial gender presentation estimate

> These models estimate **perceived Male/Female presentation from facial appearance**. They **cannot** determine a person's gender identity.

| Stage | Implementation |
|---|---|
| Detection + 5 landmarks | InsightFace `buffalo_l` SCRFD (`det_10g.onnx`) |
| Quality gate (before inference) | Rejects faces that are small (<72 px or <2 % of the frame), blurred (Laplacian variance <28), too dark (luma <45) or overexposed (>235), clipped at the edge or occluded (detection score <0.65), turned too far (roll >28° or yaw asymmetry >0.42, or pitch outside range), or moving fast (>22 % of the diagonal) |
| Model 1 | InsightFace `buffalo_l` `genderage.onnx` (96×96 crop with roll alignment) |
| Model 2 | FairFace ResNet34 `res34_fair_align_multi_7_20190809.pt` (224×224 crop from a 5-point similarity alignment, ImageNet normalisation) |
| Decision | Both models run on the **same detected face**. The two models' outputs are **never averaged**. A label is shown only when the quality gate passes, **both models agree** and **each model's confidence is ≥65 %**. The confidence shown is the lower of the two scores. In every other case the app shows **"Uncertain – Please adjust camera position"** |
| Voice | `prithivMLmods/Common-Voice-Gender-Detection` appears in its own card and **never overrides** the facial estimate |

### Evaluation on held-out Indian faces (`piadev/fairface-indian` validation split)
Run with `PYTHONPATH=. .venv/bin/python aurasense/evaluation/indian_benchmark.py`. The full subgroup breakdown (presentation, age, skin-tone L*, lighting) is in [indian_faces_benchmark_results.json](aurasense/evaluation/indian_faces_benchmark_results.json) and on the **Diagnostics** page.

| Metric (n = 346 detected out of 360 stratified samples) | Value |
|---|---|
| Buffalo_L accuracy | 73.70 % |
| FairFace ResNet34 accuracy | 91.62 % |
| Inter-model agreement | 74.57 % |
| Agree and both ≥65 %: coverage / accuracy | 62.14 % / 95.81 % |
| Full live policy (quality gate + agreement): coverage / accuracy | 24.57 % / 96.47 % |
| Disagreements in which at least one model was wrong | 100 % (88 of 88) |

Interpretation: requiring agreement raises the accuracy of the labels that are shown, but the system abstains much more often. On these small, compressed 224 px dataset images the webcam blur and lighting gates reject many faces. The numbers measure this pipeline on this dataset only. They do **not** show better accuracy than any earlier version, because the earlier version was never evaluated.

### Real webcam test
Four live frames from the MacBook Air camera were processed in memory only, with nothing written to disk. Results: 1 face detected, quality gate passed (luma 65, sharpness 56, roll −10°), Buffalo_L Male 99.99 %, FairFace Male 99.98 %, **Models Agree**. This is a single test with one person and does not demonstrate general accuracy.

### Limitations
- Accuracy is lower for people under 20, darker skin tones and dim lighting (see the subgroup table). Buffalo_L is noticeably weaker than FairFace on Indian faces.
- Both models are binary and trained on annotator-perceived labels. They cannot represent non-binary or gender-nonconforming presentation, or identity.
- Makeup, beards, hair coverings, masks, glasses and cultural styling can shift predictions.
- Softmax scores are uncalibrated and are not probabilities of being correct.
- The conservative gates make the app frequently show "Uncertain". This is deliberate.

## Other capabilities
- **Facial emotion:** DeepFace CNN on the aligned crop, with CLAHE in low light.
- **Voice emotion:** `Dpngtm/wav2vec2-emotion-recognition` on a rolling 4 s window of 16 kHz mono audio, with speech detection. It shows "Waiting for speech" when no speech is detected.
- **Fusion:** configurable weights (default 0.5/0.5), confidence thresholds, EMA or rolling smoothing and hysteresis. The app says when only one modality contributed and shows "Insufficient data" when neither is reliable. Output is labelled as an *estimated expressed emotion*.
- **Privacy:** bounded RAM buffers only, with nothing written to disk. A Privacy Shield indicator is shown on screen. The Diagnostics page shows the models, device and latency.

## Installation
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
export AURASENSE_DEVICE=cpu   # optional: cpu | cuda | auto
```

## Run
```bash
.venv/bin/streamlit run app.py
```
Open http://localhost:8501 and click **Start Analysis**.

## Tests
```bash
PYTHONPATH=. .venv/bin/python tests/test_pipeline.py
```

---

## Deploying to Streamlit Community Cloud

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) using your GitHub account.
2. Click **New app** and connect the **`WHITeCOLLAR29/aura-sense-dissertation`** repository.
3. Select the **`main`** branch.
4. Set the main file path (entry point) to **`app.py`**.
5. Under **Advanced settings**, select **Python 3.11** (matching the pinned dependencies in `requirements.txt` and system packages in `packages.txt`).
6. Click **Deploy!**

### Known Streamlit Community Cloud Limitations
- **Memory & Cold-Start Time**: AuraSense loads four pretrained models (`InsightFace buffalo_l`, `FairFace ResNet34`, `DeepFace Emotion`, and `Wav2Vec2`). First launch downloads ~1.5–2 GB of weights and may exceed free-tier RAM limits (~1 GB) on Community Cloud.
- **WebRTC NAT / Firewall Traversal**: `app.py` uses Google's public STUN server (`stun:stun.l.google.com:19302`). Users behind strict symmetric NATs or corporate firewalls may need a TURN server for remote browser webcam/microphone streaming.
- **CPU-Only Execution**: Community Cloud runs on CPU, so periodic inference latency is higher than on GPU.

