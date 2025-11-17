# main.py – MBU Upscaler x4 – FUNCIONA PERFECTO EN RAILWAY (probado hoy)
import os
import zipfile
from flask import Flask, request, send_file, render_template_string
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf  # <--- ESTA LÍNEA ES LA QUE FALTABA
import tensorflow_hub as hub
from datetime import datetime

# Silencia warnings y fuerza CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Cargando modelo ESRGAN x4 desde TensorFlow Hub... (30-50 segundos)")
model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
print("¡MODELO CARGADO! MBU UPSCALER LISTO")

HTML = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MBU Upscaler x4 PRO</title>
    <style>
        body {font-family: Arial; background: linear-gradient(135deg, #000, #3d0b0b); color: white; text-align: center; padding: 40px;}
        h1 {font-size: 3.5em; color: #ff0000; text-shadow: 0 0 30px #ff0000;}
        .container {max-width: 900px; margin:  auto; background: rgba(0,0,0,0.8); padding: 50px; border-radius: 20px;}
        input[type="file"] {padding: 20px; background: #222; border: 3px solid #ff0000; border-radius: 15px; color: white;}
        input[type="submit"] {background: #ff0000; color: white; padding: 20px 60px; font-size: 1.8em; border: none; border-radius: 50px; cursor: pointer; margin-top: 30px;}
        input[type="submit"]:hover {background: #ff3333; transform: scale(1.05);}
        .footer {margin-top: 50px; color: #aaa;}
    </style>
</head>
<body>
    <div class="container">
        <h1>MBU UPSCALER x4</h1>
        <p style="font-size: 1.8em;">Sube hasta 20 fotos → calidad ULTRA HD + sello MBU SCZ</p>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="files" multiple accept="image/*" required>
            <br><br>
            <input type="submit" value="UPSCALAR AHORA">
        </form>
        <p class="footer">© 2025 MBU SCZ - Diego Márquez B.</p>
    </div>
</body>
</html>
'''

def add_watermark(img):
    text = "MBU SCZ"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 3
    thickness = 6
    color = (0, 0, 255)
    margin = 30
    size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = img.shape[1] - size[0] - margin
    y = img.shape[0] - margin
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('files')
        if not files or len(files) > 20:
            return "Máximo 20 imágenes", 400

        output_files = []

        for f in files:
            filename = secure_filename(f.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(input_path)

            img = cv2.imread(input_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = img_rgb.astype(np.float32) / 255.0
            img_rgb = np.expand_dims(img_rgb, axis=0)

            upscaled = model(img_rgb)
            upscaled = tf.squeeze(upscaled)
            upscaled = tf.cast(upscaled, tf.uint8).numpy()

            upscaled_bgr = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
            upscaled_bgr = cv2.detailEnhance(upscaled_bgr, sigma_s=15, sigma_r=0.25)
            upscaled_bgr = cv2.convertScaleAbs(upscaled_bgr, alpha=1.08, beta=6)
            upscaled_bgr = add_watermark(upscaled_bgr)

            new_name = f"MBU_UPSCALED_{filename.rsplit('.', 1)[0]}.png"
            output_path = os.path.join(UPLOAD_FOLDER, new_name)
            cv2.imwrite(output_path, upscaled_bgr)
            output_files.append(output_path)

        if len(output_files) == 1:
            return send_file(output_files[0], as_attachment=True)

        zip_name = f"MBU_UPSCALED_{datetime.now():%Y%m%d_%H%M%S}.zip"
        zip_path = os.path.join(UPLOAD_FOLDER, zip_name)
        with zipfile.ZipFile(zip_path, 'w') as z:
            for f in output_files:
                z.write(f, os.path.basename(f))

        return send_file(zip_path, as_attachment=True)

    return render_template_string(HTML)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
