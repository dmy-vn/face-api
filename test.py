import cv2
from insightface.app import FaceAnalysis

fa = FaceAnalysis(name="buffalo_l")
fa.prepare(ctx_id=-1)

img = cv2.imread("sample.jpg") # place in directory
faces = fa.get(img)

if len(faces) == 0:
    print("no face detected")
elif len(faces) > 1:
    print("more than one face is detected")
else:
    emb = faces[0].embedding
    print("embedding shape:", emb.shape)
    print("detected box:", faces[0].bbox)
