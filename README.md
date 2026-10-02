# Face Recognition API

FastAPI service that stores a face embedding when you register a photo and returns the owner's name when you send a new photo.

## API Reference

| Method & path | Input | Success output | Error outputs |
|---|---|---|---|
| `POST /register` | multipart: `file` (image), `name` (text) | `Wajah anda telah teregistrasi` | `No face detected`, `More than one face is detected` |
| `POST /recognize` | multipart: `file` (image) | registered name of the face owner | `your face is not registered`, `No face detected`, `More than one face is detected` |
| `GET /health` | none | `{"status": "ok"}` | none |

All responses are JSON: `{"message": "..."}`. Interactive Swagger docs sit at `/docs` and are generated from the code, so they stay in sync with the routes.

## Pipeline

Both endpoints share one path through `face_skills.py`:

1. OpenCV decodes the upload into an image array.
2. SCRFD detects faces. Zero faces returns `No face detected`; two or more returns `More than one face is detected`. The check runs before embedding, so bad inputs never reach the expensive step.
3. ArcFace turns the single face into a 512-dimension vector.
4. `/register` writes the vector to SQLite. `/recognize` computes cosine similarity against every stored vector and returns the best name when its score clears 0.45, otherwise `your face is not registered`.

## Tech stack

| Piece | Choice |
|---|---|
| Language | Python 3.10+ |
| Web framework | FastAPI + uvicorn |
| Face detection | SCRFD (InsightFace `buffalo_l`) |
| Face embedding | ArcFace, 512-d |
| Database | SQLite, `data/faces.db` (standard library `sqlite3`) |
| Matching | Cosine similarity, threshold 0.45 |

## Install and run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows: .venv\Scripts\activate
                                # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn route:app --reload
```

Open http://127.0.0.1:8000/docs.

The first run downloads the InsightFace `buffalo_l` model pack (~280 MB) into `~/.insightface/models/buffalo_l`. A `FutureWarning` about `tform.estimate` from insightface 0.26 is harmless.

## Project structure

```
route.py         route definitions, no business logic
face_skills.py   decode, detect, guard, embed, register/recognize
database.py      SQLite save() and match()
model.py         Pydantic response schema
test.py          standalone check of the detect/guard/embed pipeline
data/            faces.db created here at runtime, git-ignored
sample.jpg       test image for test.py
```

## Database

`database.py` creates its single table on first use:

```sql
CREATE TABLE faces (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL,
    embedding BLOB NOT NULL      -- float32[512] = 2,048 bytes
);
```

Embeddings store as raw float32 bytes, which keeps each face at 2 KB and round-trips through `np.frombuffer` without drift. Queries use `?` placeholders.

## Test the pipeline

```bash
python test.py
```

With a single-face photo (like `sample.jpg`) it prints `embedding shape: (512,)` and the bounding box. A group photo prints `more than one face is detected`; a photo with no people prints `no face detected`.

## Limitations

- One face per upload. Group photos return the error string by design.
- The 0.45 threshold suits ArcFace scores. A different embedding model needs a different value, tuned on real query photos.
- Registered faces live in `data/faces.db`. Delete that file to start fresh.
- The 25-identity accuracy and F1 evaluation required by the test is not in this repository yet.

