from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import cv2
import numpy as np

 
# Charger le modèle pré-entrainé
model = tf.keras.models.load_model('imageclassifier.h5')

# Créer une instance de l'application FastAPI
app = FastAPI()

# Configurer CORS
origins = [
    "http://localhost:8081",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoint pour la prédiction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
 
    try:
        contents = await file.read()
        # Lire l'image téléchargée
        nparr = np.frombuffer(contents, np.uint8)
        print(nparr)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Prétraitement de l'image
        resized_img = tf.image.resize(img, (256, 256))
        resized_img = resized_img / 255.0
        
        # Effectuer la prédiction
        prediction = model.predict(np.expand_dims(resized_img, 0))
        
        # Déterminer la classe prédite
        if prediction > 0.5:
            predicted_class = "violence"
        else:
            predicted_class = "non_violence"
        
        # Retourner le résultat de la prédiction
        return JSONResponse(content={"prediction": predicted_class})
    except Exception as e:
        return JSONResponse(content={"prediction": "violence"})

       # return JSONResponse(content={"error": str(e)})


