from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
from modelo import EnhancedTransMixNetDR
from flask_cors import CORS  # Importación para CORS

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ruta del modelo
model_path = "EnhancedTransMixNetDR_APTOS19_best.pth"

# Carga del modelo
try:
    model = EnhancedTransMixNetDR(num_classes=5, model_name="vit_small_patch16_224").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {str(e)}")
    model = None

# Preprocesamiento de imágenes
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error en preprocesamiento: {str(e)}")
        raise

# Endpoints
@app.route("/")
def home():
    return "✅ API de retinopatía diabética activa. Usa /predict para enviar una imagen."

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No se proporcionó imagen"}), 400

    if model is None:
        return jsonify({"error": "Modelo no disponible"}), 503

    try:
        image = request.files["image"]
        image_bytes = image.read()
        print(f"📁 Imagen recibida - Tamaño: {len(image_bytes)} bytes")

        input_tensor = preprocess_image(image_bytes)
        
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
        
        result = {
            "prediction": int(predicted.item()),
            "class_name": str(predicted.item()),
            "message": "Predicción exitosa"
        }
        
        print(f"🎯 Resultado: {result}")
        return jsonify(result)

    except Exception as e:
        error_msg = f"Error en la predicción: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

