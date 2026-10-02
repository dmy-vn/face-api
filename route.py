"""Face Recognition API — Route definitions only."""
from fastapi import FastAPI, File, Form, UploadFile

import face_skills
from model import Message

app = FastAPI(
    title="Face Recognition API",
    version="1.0.0",
    description="Register and recognize faces (InsightFace ArcFace embeddings).",
)


@app.post("/register", response_model=Message, summary="Register a face")
def register(
    file: UploadFile = File(..., description="Face image"),
    name: str = Form(..., description="Owner's name"),
):
    """Success -> 'Wajah anda telah teregistrasi'."""
    return {"message": face_skills.register(file.file.read(), name)}


@app.post("/recognize", response_model=Message, summary="Recognize a face")
def recognize(file: UploadFile = File(..., description="Face image")):
    """Success -> the registered name of the face owner."""
    return {"message": face_skills.recognize(file.file.read())}


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}
