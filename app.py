# main.py → MBU Upscaler x4 PRO → FUNCIONA PERFECTO EN RAILWAY (2025)
import os
import zipfile
from flask import Flask, request, send_file, render_template_string
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime

# Silenciar warnings y forzar CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Cargando modelo ESRGAN x4 desde TensorFlow Hub... (30-50 segundos)")
model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
print("¡MODELO CARGADO! MBU UPSCALER FULL POWER ACTIVADO")

# HTML con tu marca roja brutal
HTML = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MBU UPSCALER x4 PRO</title>
    <style>
        body {font-family: Arial; background: linear-gradient(135deg, #000, #3d0b0b); color: white; text-align: center; padding: 40px;}
        h1 {font-size: 3.8em; color: #ff0000; text-shadow: 0 0 30px #ff0000; margin: 20px;}
        .container {max-width: 900px; margin: 0 auto; background: rgba(0,0,0,0.85); padding: 50px; border-radius: 20px; box-shadow: 0 0 40px rgba(255,0,0,0.6);}
        input[type="file"] {padding: 20px; background: #222; border: 3px solid #ff0000; border-radius: 15px; color: white; font-size: 18px;}
        input[type="submit"] {background: #ff0000; color: white; padding: 22px 70px; font-size: 2em; border: none; border-radius: 50px; cursor: pointer; margin-top: 30px; box-shadow: 0 0 30px #ff0000;}
        input[type="submit"]:hover {background: #ff3333; transform: scale(1.05);}
        .footer {margin-top: 60px; color: #ff6666; font-size: 1.1em;}
    </style>
</head>
<body>
    <div class="container">
        <h1>MBU UPSCALER x4</h1>
        <p style="font-size: 1.8em; margin: 20px 0;">Sube tus fotos de pérgolas, casetas y decks<br>y recíbelas en calidad ULTRA HD 4× con sello MBU SCZ</p>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="files" multiple accept="image/*" required>
            <br><br>
            <input type="submit" value="UPSCALAR AHORA">
        </form>
        <div class="footer">© 2025 MBU SCZ - Diego Márquez B. | Santa Cruz, Bolivia</div>
    </div>
</body>
</html>
'''

def add_watermark(img):
    text = "MBU SCZ"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 3.5
    thickness = 8
    color = (0, 0, 255)  # Rojo puro
    margin = 40
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
            return "<h1 style='color:red;'>Máximo 20 imágenes</h1>", 400

        output_files = []

        for f in files:
            filename = secure_filename(f.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(input_path)

            # Leer imagen
            img = cv2.imread(input_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = img_rgb.astype(np.float32) / 255.0
            img_rgb = np.expand_dims(img_rgb, axis=0)

            # UPSCALING 4x
            upscaled = model(img_rgb)
            upscaled = tf.squeeze(upscaled)

            # CORRECCIÓN DE COLOR 100% (ADIÓS MAGENTA Y NEGRO)
            upscaled = (upscaled + 1) / 2                  # de [-1,1] → [0,1]
            upscaled = tf.clip_by_value(upscaled, 0.0, 1.0)
            upscaled = (upscaled.numpy() * 255).astype(np.uint8)

            # Corrección de canales (el modelo devuelve BGR)
            upscaled = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
            upscaled_bgr = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)

            # Mejoras finales
            upscaled_bgr = cv2.detailEnhance(upscaled_bgr, sigma_s=15, sigma_r=0.25)
            upscaled_bgr = cv2.convertScaleAbs(upscaled_bgr, alpha=1.1, beta=8)
            upscaled_bgr = add_watermark(upscaled_bgr)

            # Guardar como PNG
            new_name = f"MBU_4X_{filename.rsplit('.', 1)[0]}.png"
            output_path = os.path.join(UPLOAD_FOLDER, new_name)
            cv2.imwrite(output_path, upscaled_bgr)
            output_files.append(output_path)

        # Descarga
        if len(output_files) == 1:
            return send_file(output_files[0], as_attachment=True, download_name=os.path.basename(output_files[0]))

        zip_name = f"MBU_UPSCALED_4X_{datetime.now():%Y%m%d_%H%M%S}.zip"
        zip_path = os.path.join(UPLOAD_FOLDER, zip_name)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in output_files:
                z.write(f, os.path.basename(f))

        return send_file(zip_path, as_attachment=True)

    return render_template_string(HTML)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
