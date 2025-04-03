from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import os
import cv2
from src.extract_metrics import Get_Metrics
from src.prompting import Get_Prompt
import ollama

app = FastAPI()

# Create directory for uploaded videos if it doesn't exist
UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve static files (HTML, CSS, etc.)
app.mount("/static", StaticFiles(directory=Path(__file__).parent.parent / "static"), name="static")

# Serve the frontend HTML
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = Path(__file__).parent.parent / "static" / "index.html"
    return HTMLResponse(content=index_path.read_text())

# Serve uploaded videos
app.mount("/videos", StaticFiles(directory=UPLOAD_DIR), name="videos")

# Upload and process video
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save the uploaded video
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run Pose Estimation
    YOLO_path = "src/models/yolo11l-pose.pt"
    swingnet_path = "src/models/swingnet_1800.pth"

    try:
        user_angles, user_lean = Get_Metrics(file_path, YOLO_path, swingnet_path)
        pro_angles, pro_lean = Get_Metrics("src/data/iron/face_on/homa_iron_face.mp4", YOLO_path, swingnet_path)
        prompt = Get_Prompt(user_angles, user_lean, pro_angles, pro_lean, "I feel like I am not getting enough distance and I am slicing the ball.")
        response = ollama.chat(
            model='golf_coach-gemma', 
            messages=[{'role': 'user', 'content': prompt}]
        )
        formatted_response = response['message']['content']
        formatted_response = formatted_response.replace("\n", "").replace("`", "").replace("html", "")


    except Exception as e:
        import traceback
        print("FULL EXCEPTION TRACEBACK:")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})

    return {
        "filename": file.filename,
        "video_url": f"http://127.0.0.1:8000/videos/{file.filename}",
        "user_angles": user_angles,
        "user_lean": user_lean,
        "pro_angles": pro_angles,
        "pro_lean": pro_lean,
        "prompt": prompt,
        "response": formatted_response
    }