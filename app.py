# app.py – Versión 100% funcional en Railway (CPU-only, batch, watermark)
from flask import Flask, request, render_template, send_file, Response
import cv2
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import zipfile
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['OUTPUT_FOLDER'] = '/tmp/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# ===================== CARGA DEL MODELO UNA SOLA VEZ =====================
print("Cargando modelo ESRGAN... (puede tardar 30-60 segundos la primera vez)")
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
model = hub.load(SAVED_MODEL_PATH)
print("Modelo ESRGAN cargado correctamente!")

# ===================== FUNCIONES =====================
def preprocess_image(image):
    if image.shape[-1] == 4:
        image = image[..., :3]
    hr_size = (tf.convert_to_tensor(image.shape[:-1]) // 4) * 4
    image = tf.image.crop_to_bounding_box(image, 0, 0, hr_size[0], hr_size[1])
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, 0)

def postprocess_image(tensor_img):
    img = tensor_img.numpy()
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def add_watermark(pil_img, text="MBU SCZ"):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        font = ImageFont.load_default()
    w, h = pil_img.size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    margin = 20
    position = (w - text_width - margin, h - text_height - margin)
    draw.text(position, text, fill=(255, 255, 255, 200), font=font, stroke_width=2, stroke_fill=(0,0,0,200))
    return pil_img

# ===================== RUTAS =====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'images' not in request.files:
        return "No file part", 400

    files = request.files.getlist('images')
    if not files or all(f.filename == '' for f in files):
        return "No selected files", 400

    result_images = []

    for file in files:
        if file.filename == '':
            continue

        # Leer y decodificar
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Upscale
        preprocessed = preprocess_image(img_rgb)
        start = time.time()
        upscaled_tensor = model(preprocessed)
        upscaled_tensor = tf.squeeze(upscaled_tensor)
        print(f"Tiempo upscale: {time.time() - start:.2f}s")

        # Post-procesar
        upscaled_np = postprocess_image(upscaled_tensor)

        # Convertir a PIL y agregar watermark
        pil_img = Image.fromarray(upscaled_np)
        pil_img = add_watermark(pil_img, "MBU SCZ")

        # Guardar temporalmente
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"upscaled_{file.filename}")
        pil_img.save(output_path, quality=95)
        result_images.append(output_path)

    # Si es una sola imagen → devolver directo
    if len(result_images) == 1:
        return send_file(result_images[0], mimetype='image/jpeg')

    # Si son varias → devolver ZIP
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in result_images:
            zf.write(path, os.path.basename(path))
    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"upscaled_MBU_{int(time.time())}.zip"
    )

# ===================== INICIO =====================
if __name__ == '__main__':
    # Railway usa $PORT
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
