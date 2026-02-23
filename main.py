import io
import os
import re
import zipfile
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

# Configuration dataclass for extraction parameters
@dataclass
class ExtractConfig:
    entropy_threshold: float = 3.2
    roi_top: int = 200
    roi_bottom_margin: int = 230
    bw_thresh: int = 235
    erode_ksize: int = 9
    dilate_ksize: int = 7
    min_area: int = 2000
    min_w: int = 90
    min_h: int = 70
    max_ar: float = 2.4
    min_ar: float = 0.45
    pad: int = 4
    jpg_quality: int = 95

def sanitize_prefix(s: str) -> str:
    """Sanitize prefix for filenames."""
    s = (s or "").strip()
    s = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", s).strip("_")
    return s[:60] if s else "screenshot"

def entropy_gray(bgr: np.ndarray) -> float:
    """Compute entropy of grayscale version of BGR image."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def natural_sort_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Sort bounding boxes top to bottom, then left to right."""
    return sorted(boxes, key=lambda b: (b[1] // 5, b[0]))

def extract_zip_from_image_bytes(img_bytes: bytes, prefix: str, cfg: ExtractConfig) -> bytes:
    """Extract photos from screenshot bytes and return ZIP bytes."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")

    h, w = img.shape[:2]
    y0 = max(0, min(cfg.roi_top, h - 1))
    y1 = max(y0 + 1, min(h - cfg.roi_bottom_margin, h))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = gray[y0:y1, :]

    # threshold to find content (non-white background)
    content = (roi < cfg.bw_thresh).astype(np.uint8) * 255

    er_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.erode_ksize, cfg.erode_ksize))
    di_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.dilate_ksize, cfg.dilate_ksize))

    proc = cv2.erode(content, er_ker, iterations=1)
    proc = cv2.dilate(proc, di_ker, iterations=1)
    proc = cv2.morphologyEx(
        proc, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1
    )

    num, labels, stats, _ = cv2.connectedComponentsWithStats(proc, connectivity=8)
    boxes = []
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < cfg.min_area:
            continue
        x2 = max(0, x - cfg.pad)
        y2 = max(0, y - cfg.pad)
        ww2 = min(w - x2, ww + 2 * cfg.pad)
        hh2 = min((y1 - y0) - y2, hh + 2 * cfg.pad)
        if ww2 < cfg.min_w or hh2 < cfg.min_h:
            continue
        ar = ww2 / max(1, hh2)
        if ar > cfg.max_ar or ar < cfg.min_ar:
            continue
        boxes.append((x2, y0 + y2, ww2, hh2))

    if not boxes:
        raise RuntimeError("No content blocks found; adjust ROI or threshold.")

    # filter by entropy to drop low-information blocks (banner)
    keep = []
    for (x, y, ww, hh) in boxes:
        crop = img[y:y+hh, x:x+ww]
        if entropy_gray(crop) >= cfg.entropy_threshold:
            keep.append((x, y, ww, hh))

    keep = natural_sort_boxes(keep)

    # generate zip of crops
    prefix = sanitize_prefix(prefix)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, (x, y, ww, hh) in enumerate(keep, start=1):
            crop = img[y:y+hh, x:x+ww]
            ok, jpg = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.jpg_quality])
            if ok:
                name = f"{prefix}_img-{idx:03d}.jpg"
                zf.writestr(name, jpg.tobytes())
    return buf.getvalue()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

APP_PASSWORD = os.getenv("APP_PASSWORD", "")

LOGIN_PAGE = """
<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Login</title></head>
<body style="font-family:Arial, sans-serif; margin:40px;">
<h2>Login</h2>
<form method="post" action="/login">
<input type="password" name="password" placeholder="Password">
<button type="submit">Login</button>
</form>
</body></html>
"""

UPLOAD_PAGE = """
<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Upload Screenshot</title></head>
<body style="font-family:Arial, sans-serif; margin:40px;">
<h2>Upload screenshot(s)</h2>
<form method="post" action="/upload" enctype="multipart/form-data">
<label>Prefix: <input type="text" name="prefix" value="istock_02"></label><br><br>
<input type="file" name="files" multiple accept=".png,.jpg,.jpeg"><br><br>
<button type="submit">Process</button>
</form>
</body></html>
"""

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if APP_PASSWORD:
        token = request.cookies.get("token")
        if token != APP_PASSWORD:
            return HTMLResponse(LOGIN_PAGE)
    return HTMLResponse(UPLOAD_PAGE)

@app.post("/login")
async def login(password: str = Form(...)):
    if APP_PASSWORD and password == APP_PASSWORD:
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie("token", password, httponly=True, max_age=3600*8)
        return response
    return HTMLResponse(LOGIN_PAGE, status_code=401)

@app.post("/upload")
async def upload(prefix: str = Form("istock_02"), files: List[UploadFile] = File(...)):
    cfg = ExtractConfig()
    if not files:
        return HTMLResponse("No files uploaded", status_code=400)

    out_zip = io.BytesIO()
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as master_zip:
        for f in files:
            data = await f.read()
            try:
                zbytes = extract_zip_from_image_bytes(data, prefix, cfg)
                with zipfile.ZipFile(io.BytesIO(zbytes)) as subzip:
                    for name in subzip.namelist():
                        content = subzip.read(name)
                        master_zip.writestr(name, content)
            except Exception:
                continue
    out_zip.seek(0)
    return StreamingResponse(out_zip, media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="{sanitize_prefix(prefix)}_photos.zip"'
    })
