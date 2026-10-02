"""Detection + embedding pipeline for /register and /recognize."""
import cv2
import numpy as np
from insightface.app import FaceAnalysis

import database

# Fail errors
NO_FACE = "No face detected"
MULTI_FACE = "More than one face is detected"

_fa: FaceAnalysis | None = None


def get_analyzer() -> FaceAnalysis:
    """models load once per process."""
    global _fa
    if _fa is None:
        _fa = FaceAnalysis(name="buffalo_l")
        _fa.prepare(ctx_id=-1)   # use ctx_id=0 if you have a CUDA GPU
    return _fa


def embed_single_face(image_bytes: bytes) -> np.ndarray | str:
    """Decode -> detect -> guard -> embed.

    Returns the 512-d embedding, or the spec's error string.
    """
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return NO_FACE

    faces = get_analyzer().get(img)
    if len(faces) == 0:
        return NO_FACE
    if len(faces) > 1:
        return MULTI_FACE
    return faces[0].embedding


def register(image_bytes: bytes, name: str) -> str:
    emb = embed_single_face(image_bytes)
    if isinstance(emb, str):
        return emb                      # error string passes through
    database.save(name, emb)
    return "Wajah anda telah teregistrasi"


def recognize(image_bytes: bytes) -> str:
    emb = embed_single_face(image_bytes)
    if isinstance(emb, str):
        return emb                      # error string passes through
    result = database.match(emb)
    if result is None:
        return "your face is not registered"
    return result[0]                    # return the owner's name
