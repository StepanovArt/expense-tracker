# DECISIONS.md

One line per decision: **problem → choice → why → trade-off**. Newest at the bottom.

| # | Date | Problem → Choice → Why → Trade-off |
|---|------|-------------------------------------|
| 1 | 2026-10-22 | Repo is an existing Spanish Telegram expense bot → build receipt-vlm as a self-contained subproject in `receipt-vlm/` → keeps the working bot untouched; the Telegram client example (§7) can later feed it → two pyprojects in one repo. |
| 2 | 2026-10-22 | Dev container has no GPU and its network policy blocks `huggingface.co`, `download.pytorch.org`, `download.openmmlab.com`, Google Drive, Kaggle → implement + smoke-test everything on CPU here, run real models on a GPU box via `notebooks/train_colab.ipynb` → CLAUDE.md §2 → no real-model numbers can be produced in this session; nothing is invented for them. |
| 3 | 2026-10-22 | SROIE needs a mirror → `zzzDavid/ICDAR-2019-SROIE` git repo (pinned commit `27be427`; its `data/{img,key}` has 626 labelled receipts with the organisers' label typos fixed; repo is MIT, the underlying data is ICDAR research-use) → reachable from here and from Colab, no HF needed → only the 626 trainval images have key labels in the mirror, so there is no official SROIE test split; we split 80/10/10 ourselves. |
| 4 | 2026-10-22 | Python 3.13 is the system default, but `rapidocr-onnxruntime`/`vllm` pin `<3.13`/`<3.15` → use Python 3.12 via `uv --python 3.12`, `requires-python >=3.11,<3.13` → matches Colab (3.12) → none. |
| 5 | 2026-10-22 | CPU torch wheels come from download.pytorch.org (blocked) → install the default PyPI torch (CUDA build, runs on CPU) → only option here → ~3 GB of unused CUDA libs in the dev env. |
| 6 | 2026-10-22 | OCR engine for B0 → Tesseract 5 (apt `tesseract-ocr` + `eng/ara/rus` packs) via `pytesseract` → installs cleanly and covers all three target languages offline; PaddleOCR needs model downloads from bcebos.com (blocked) → Tesseract is weaker than PaddleOCR on photos, which makes B0 a somewhat easier baseline to beat; stated in RESULTS.md. |
