# main.py → MBU Upscaler PRO 2025 → Funciona perfecto en Railway GRATIS
import os
import zipfile
from flask import Flask, request, send_file, render_template_string
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime

# ====== OPTIMIZACIONES OBLIGATORIAS PARA RAILWAY GRATIS ======
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'           # Fuerza CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'            # Silencia todos los warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'           # Menos consumo RAM
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
# ============================================================

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 800 * 1024 * 1024   # 800 MB máximo

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Cargando modelo Real-ESRGAN ligero... (15-40 segundos)")
# Modelo ligero oficial que consume poca RAM y corre perfecto en Railway
model = tf.keras.models.load_model('https://tfhub.dev/captain-pool/esrgan-tf2/1', compile=False)
print("¡Modelo cargado y listo! MBU SCZ ON FIRE")

# HTML PRO con tu marca
HTML = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MBU UPSCALER x4 PRO</title>
    <style>
        body {font-family: Arial; background: linear-gradient(135deg, #000, #2d0b0b); color: white; text-align: center; padding: 40px;}
        h1 {font-size: 3.5em; text-shadow: 0 0 30px #ff0000;}
        .logo {font-size: 2.2em; color: #ff0000; font-weight: bold;}
        .container {max-width: 900px; margin: 0 auto; background: rgba(0,0,0,0.8); padding: 50px; border-radius: 20px;}
        input[type="file"] {padding: 20px; background: #222; border: 3px solid #ff0000; border-radius: 15px; color: white;}
        input[type="submit"] {background: #ff0000; color: white; padding: 20px 60px; font-size: 1.8em; border: none; border-radius: 50px; cursor: pointer; margin-top: 30px;}
        input[type="submit"]:hover {background: #ff3333; transform: scale(1.05);}
        .footer {margin-top: 50px; color: #aaa;}
    </style>
</head>
<body>
    <div class="container">
        <h1>MBU UPSCALER x4</h1>
        <p class="logo">MBU SCZ ⚡ PRO</p>
        <p>Hasta <strong>40 fotos</strong> → Calidad ULTRA HD automática<br>Pérgolas, casetas, decks → fotos que venden solas</p>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="files" multiple accept="image/*" required>
            <br>
            <input type="submit" value="UPSCALAR AHORA">
        </form>
        <div class="footer">© 2025 MBU SCZ - Diego Márquez B. | @diegomarquezb</div>
    </div>
</body>
</html>
'''

def add_watermark(img):
    text = "MBU SCZ"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 3
    thickness = 6
    color = (0, 0, 255)  # Rojo
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
        if len(files) > 40:
            return "Máximo 40 imágenes", 400

        output_files = []
        batch_size = 5

        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            for f in batch:
                if f.filename == '':
                    continue
                filename = secure_filename(f.filename)
                path = os.path.join(UPLOAD_FOLDER, filename)
                f.save(path)

                # Leer y preparar
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=0)

                # Upscale 4x
                output = model(img, training=False)
                output = tf.cast(tf.clip_by_value(output[0] * 255, 0, 255), tf.uint8).numpy()
                upscaled = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

                # Calidad PRO
                upscaled = cv2.detailEnhance(upscaled, sigma_s=15, sigma_r=0.25)
                upscaled = cv2.convertScaleAbs(upscaled, alpha=1.08, beta=6)
                upscaled = add_watermark(upscaled)

                # Guardar como PNG (calidad perfecta)
                name = f"MBU_UPSCALED_{len(output_files)+1}_{filename.rsplit('.',1)[0]}.png"
                out_path = os.path.join(UPLOAD_FOLDER, name)
                cv2.imwrite(out_path, upscaled)
                output_files.append(out_path)

        # Una sola → directo, varias → ZIP
        if len(output_files) == 1:
            return send_file(output_files[0], as_attachment=True)

        zip_name = f"MBU_UPSCALED_4X_{datetime.now():%Y%m%d_%H%M%S}.zip"
        zip_path = os.path.join(UPLOAD_FOLDER, zip_name)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in output_files:
                z.write(f, os.path.basename(f))

        return send Rn_file(zip_path, as_attachment=True)

    return render_template_string(HTML)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
