from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import os
import uuid
import subprocess
import time
import shutil
import tempfile
from typing import Optional

# Initialize FastAPI app
app = FastAPI(title="DiffRhythm API", description="Generate full-length songs with AI")

# Add CORS support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for storing uploads and outputs
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Map the static files directory for serving the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Data models
class GenerationRequest(BaseModel):
    style_prompt: str
    audio_length: int = 95  # 95 for base model, 285 for full model
    model_type: str = "base"  # "base" or "full"

class GenerationStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None
    audio_url: Optional[str] = None
    error: Optional[str] = None

# Store job statuses
jobs = {}

# Helper functions
def check_device():
    """Check if CUDA is available and return the appropriate device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def generate_song(
    job_id: str, 
    lrc_path: str, 
    style_prompt: str, 
    audio_length: int = 95, 
    model_type: str = "base"
):
    """Background task to generate a song."""
    try:
        # Update job status
        jobs[job_id]["status"] = "processing"
        
        # Determine repo_id based on model_type
        repo_id = f"ASLP-lab/DiffRhythm-{model_type}"
        output_dir = f"outputs/{job_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare command based on device
        device = check_device()
        cmd_env = os.environ.copy()
        
        if device == "mps":
            # macOS specific environment variables
            cmd_env["PHONEMIZER_ESPEAK_LIBRARY"] = "/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.dylib"
        
        # Build command
        cmd = [
            "python",
            "infer/infer.py",
            f"--lrc-path={lrc_path}",
            f"--ref-prompt={style_prompt}",
            f"--audio-length={audio_length}",
            f"--repo_id={repo_id}",
            f"--output-dir={output_dir}",
            "--chunked"
        ]
        
        # Execute command
        process = subprocess.Popen(cmd, env=cmd_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error: {stderr.decode()}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = stderr.decode()
            return
        
        # Check if output file exists
        output_path = os.path.join(output_dir, "output.wav")
        if os.path.exists(output_path):
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["audio_url"] = f"/api/download/{job_id}"
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "Output file not found"
            
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        print(f"Error during generation: {e}")

@app.get("/")
async def read_root():
    """Serve the index.html file."""
    return FileResponse("static/index.html")

@app.post("/api/generate", response_model=GenerationStatusResponse)
async def generate(
    background_tasks: BackgroundTasks,
    lrc_file: UploadFile = File(...),
    style_prompt: str = Form(...),
    audio_length: int = Form(95),
    model_type: str = Form("base")
):
    """Generate a song based on the uploaded LRC file and style prompt."""
    # Create a unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded LRC file
    lrc_content = await lrc_file.read()
    lrc_path = f"uploads/{job_id}.lrc"
    with open(lrc_path, "wb") as f:
        f.write(lrc_content)
    
    # Initialize job status
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "audio_url": None,
        "error": None
    }
    
    # Start generation in background
    background_tasks.add_task(
        generate_song,
        job_id=job_id,
        lrc_path=lrc_path,
        style_prompt=style_prompt,
        audio_length=audio_length,
        model_type=model_type
    )
    
    return GenerationStatusResponse(
        job_id=job_id,
        status="queued"
    )

@app.get("/api/status/{job_id}", response_model=GenerationStatusResponse)
async def get_status(job_id: str):
    """Get the status of a generation job."""
    if job_id not in jobs:
        return JSONResponse(
            status_code=404,
            content={"error": "Job not found"}
        )
    
    job = jobs[job_id]
    return GenerationStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        audio_url=job.get("audio_url"),
        error=job.get("error")
    )

@app.get("/api/download/{job_id}")
async def download_audio(job_id: str):
    """Download the generated audio file."""
    output_path = f"outputs/{job_id}/output.wav"
    if not os.path.exists(output_path):
        return JSONResponse(
            status_code=404,
            content={"error": "Audio file not found"}
        )
    
    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="diffrhythm_song.wav"
    )

@app.get("/api/examples")
async def get_examples():
    """Return example LRC files and style prompts."""
    examples = [
        {
            "name": "English Example",
            "lrc_path": "/api/examples/en",
            "style_prompts": [
                "Pop emotional vocals with piano",
                "Rock energetic with guitar",
                "Classical hopeful mood with piano",
                "Electronic upbeat dance"
            ]
        },
        {
            "name": "Chinese Example",
            "lrc_path": "/api/examples/zh",
            "style_prompts": [
                "中文流行歌曲，钢琴伴奏",
                "中文摇滚，吉他伴奏",
                "古风，竹笛伴奏",
                "电子舞曲，现代感"
            ]
        }
    ]
    return examples

@app.get("/api/examples/en")
async def get_english_example():
    """Return the English example LRC file."""
    return FileResponse("infer/example/eg_en.lrc")

@app.get("/api/examples/zh")
async def get_chinese_example():
    """Return the Chinese example LRC file."""
    return FileResponse("infer/example/eg_cn.lrc")

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    if job_id not in jobs:
        return JSONResponse(
            status_code=404,
            content={"error": "Job not found"}
        )
    
    # Remove job from jobs dictionary
    job = jobs.pop(job_id)
    
    # Remove uploaded LRC file
    lrc_path = f"uploads/{job_id}.lrc"
    if os.path.exists(lrc_path):
        os.remove(lrc_path)
    
    # Remove output directory
    output_dir = f"outputs/{job_id}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    return {"message": "Job deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
