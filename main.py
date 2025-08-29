from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import base64

app = FastAPI()

# تحميل النموذج
model = load_model("plant_disease_model.h5")
print("✅ Model loaded successfully")

# أسماء الكلاسات
class_labels = ["leaf blight", "healthy", "ESCA", "black rot"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # قراءة الصورة وتحويلها لـ RGB
    img = Image.open(file.file).convert("RGB")
    
    # نسخة للتنبؤ
    img_for_model = img.resize((224, 224))
    img_array = np.array(img_for_model) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # التنبؤ
    pred = model.predict(img_array)
    pred_class = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred) * 100

    # تحويل الصورة الأصلية لبايتس
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")  # ترميز Base64

    return JSONResponse({
        "disease_name": class_labels[pred_class],
        "confidence": f"{confidence:.1f}%",
        "image_base64": img_base64
    })
