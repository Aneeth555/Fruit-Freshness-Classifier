from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

from app.utils import predict_image

app = FastAPI(title="Food Freshness Classifier")

# ✅ CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # OK for local dev
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Food Freshness API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image"):
        return JSONResponse(
            status_code=400,
            content={"error": "Upload a valid image"}
        )

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    result = predict_image(image)
    return result
