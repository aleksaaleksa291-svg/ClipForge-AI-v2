import os, re, uuid, shutil, subprocess, requests
from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()
BASE = Path(__file__).parent
WORK = BASE / "work"
WORK.mkdir(exist_ok=True)
app = FastAPI(title="ClipForge AI")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2500:])
    return p

def fetch_direct_video(url: str, dest: Path):
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Use a valid HTTPS direct video URL, or upload a video file.")
    # Only accept common video file extensions; this avoids pretending that arbitrary
    # social-media page URLs are direct media files.
    if not re.search(r"\.(mp4|mov|m4v|webm)(?:$|[?#])", parsed.path + ("?" + parsed.query if parsed.query else ""), re.I):
        raise ValueError("That looks like a webpage, not a direct video file. Download a video you have permission to use and upload the file.")
    with requests.get(url, stream=True, timeout=(15, 120)) as r:
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "video" not in ctype and "octet-stream" not in ctype:
            raise ValueError("The URL did not return a video file.")
        total = 0
        with dest.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                total += len(chunk)
                if total > 800 * 1024 * 1024:
                    raise ValueError("Video exceeds the 800 MB upload limit.")
                f.write(chunk)

def make_srt(video: Path, srt: Path, language: str = "auto"):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("Speech recognition is not installed. Install requirements.txt.")
    model_name = os.getenv("WHISPER_MODEL", "small")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    kwargs = {} if language == "auto" else {"language": language}
    segments, _ = model.transcribe(str(video), vad_filter=True, **kwargs)
    def stamp(seconds):
        ms = int(seconds * 1000)
        h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    rows = []
    for i, seg in enumerate(segments, 1):
        text = (seg.text or "").strip()
        if text:
            rows.extend([str(i), f"{stamp(seg.start)} --> {stamp(seg.end)}", text, ""])
    srt.write_text("\n".join(rows), encoding="utf-8")
    if not rows:
        raise RuntimeError("No speech was detected; try another video.")

def render_video(src: Path, out: Path, srt: Path, aspect: str, style: str):
    # Scale/crop to requested aspect ratio; captions are burned in for portability.
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    if aspect == "16:9":
        vf = "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080"
    elif aspect == "1:1":
        vf = "scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080"
    escaped = str(srt.resolve()).replace("\\", "/").replace(":", r"\:")
    force = "FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=1,Alignment=2,MarginV=70"
    if style == "bold":
        force = "FontName=Arial,FontSize=22,Bold=1,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=1,Alignment=2,MarginV=80"
    vf += f",subtitles='{escaped}':force_style='{force}'"
    run(["ffmpeg", "-y", "-i", str(src), "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "23", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)])

@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE / "static" / "index.html").read_text(encoding="utf-8")

@app.post("/api/edit")
async def edit(background: BackgroundTasks, video: UploadFile = File(None),
               video_url: str = Form(""), aspect: str = Form("9:16"),
               caption_style: str = Form("clean"), language: str = Form("auto")):
    if not video and not video_url.strip():
        raise HTTPException(400, "Upload a video or enter a direct video-file URL.")
    job = uuid.uuid4().hex
    folder = WORK / job
    folder.mkdir()
    src = folder / "input.mp4"
    try:
        if video and video.filename:
            size = 0
            with src.open("wb") as f:
                while chunk := await video.read(1024 * 1024):
                    size += len(chunk)
                    if size > 800 * 1024 * 1024:
                        raise ValueError("Video exceeds the 800 MB limit.")
                    f.write(chunk)
        else:
            fetch_direct_video(video_url.strip(), src)
        srt, out = folder / "captions.srt", folder / "clipforge-edited.mp4"
        make_srt(src, srt, language)
        render_video(src, out, srt, aspect, caption_style)
        return {"status": "complete", "download_url": f"/api/download/{job}", "captions_url": f"/api/captions/{job}"}
    except Exception as e:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(500, str(e))

@app.get("/api/download/{job}")
def download(job: str):
    p = WORK / job / "clipforge-edited.mp4"
    if not p.exists(): raise HTTPException(404, "Video not found.")
    return FileResponse(p, media_type="video/mp4", filename="clipforge-edited.mp4")

@app.get("/api/captions/{job}")
def captions(job: str):
    p = WORK / job / "captions.srt"
    if not p.exists(): raise HTTPException(404, "Captions not found.")
    return FileResponse(p, media_type="text/plain", filename="captions.srt")

@app.post("/api/animate")
async def animate(prompt: str = Form(...), video: UploadFile = File(...)):
    # Video-to-video generation requires a paid/authorized provider. Do not silently
    # fake an animation; return a clear setup message until an adapter is configured.
    if not os.getenv("VIDEO_ANIMATION_API_URL") or not os.getenv("VIDEO_ANIMATION_API_KEY"):
        raise HTTPException(501, "Video-to-video animation is not configured yet. Add a provider API URL and key in .env; see README.")
    raise HTTPException(501, "Provider credentials are present, but a provider-specific adapter must be implemented for your chosen model. See README.")

