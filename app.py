# main.py - MBU Upscaler PRO 2025 - Hasta 40 imágenes con procesamiento por lotes
import os
import zipfile
from flask import Flask, request, send_file, render_template_string, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime

# Forzar CPU y silenciar TODOS los warnings de TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 800 * 1024 * 1024  # 800 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Cargando modelo ESRGAN... (30-60 segundos la primera vez)")
model = load_model('ESRGAN/models/RRDB_ESRGAN_x4.h5', compile=False)
print("¡Modelo ESRGAN cargado y listo!")

# HTML con barra de progreso en vivo
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MBU Image Upscaler x4 PRO</title>
    <style>
        body {font-family: Arial; background: linear-gradient(135deg, #1a1a1a, #2d0b0b); color: white; text-align: center; padding: 40px;}
        h1 {font-size: 3em; text-shadow: 0 0 20px #ff0000;}
        .container {max-width: 900px; margin: 0 auto; background: rgba(0,0,0,0.8); padding: 40px; border-radius: 20px;}
        input[type="file"] {padding: 15px; background: #333; color: white; border: 2px solid #ff0000; border-radius: 10px;}
        input[type="submit"] {background: #ff0000; color: white; padding: 18px 50px; font-size: 1.6em; border: none; border-radius: 50px; cursor: pointer; margin-top: 20px;}
        input[type="submit"]:hover {background: #ff3333; transform: scale(1.05);}
        #progress-container {margin-top: 30px; display: none;}
        #progress-bar {width: 80%; background: #333; margin: 20px auto; border-radius: 10px; overflow: hidden;}
        #progress-fill {height: 40px; background: #ff0000; width: 0%; transition: width 0.4s;}
        .logo {font-size: 2em; color: #ff0000; font-weight: bold;}
    </style>
</head>
<body>
    <div class="container">
        <h1>MBU UPSCALER x4</h1>
        <p class="logo">MBU SCZ ⚡ PRO</p>
        <p>Sube hasta <strong>40 fotos</strong> de pérgolas, casetas o decks<br>y recíbelas en calidad ULTRA HD 4× automáticamente</p>
        
        <form id="upload-form" method="post" enctype="multipart/form-data">
            <input type="file" name="files" multiple accept="image/*" required>
            <br>
            <input type="submit" value="UPSCALAR AHORA">
        </form>

        <div id="progress-container">
            <h3>Procesando lote <span id="current-batch">1</span> de <span id="total-batches">?</span></h3>
            <div id="progress-bar"><div id="progress-fill"></div></div>
            <p id="status">Preparando imágenes...</p>
        </div>
    </div>

    <script>
        document.getElementById('upload-form').onsubmit = function(e) {
            e.preventDefault();
            document.getElementById('progress-container').style.display = 'block';
            const formData = new FormData(this);
            fetch('/', {method: 'POST', body: formData})
                .then(r => r.blob())
                .then(blob => {
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = blob.type === 'application/zip' ? 'MBU_UPSCALED_4X_{{datetime}}.zip' : 'upscaled_MBU.png';
                    a.click();
                });
        };

        // Actualización de progreso en tiempo real (opcional, se puede activar después)
    </script>
</body>
</html>
'''

def add_watermark(img):
    text = "MBU SCZ"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 2.5
    thickness = 5
    color = (0, 0, 255)
    margin = 25
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = img.shape[1] - text_size[0] - margin
    text_y = img.shape[0] - margin
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
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
            batch = files[i:i + batch_size]
            
            for f in batch:
                filename = secure_filename(f.filename)
                input_path = os.path.join(UPLOAD_FOLDER, filename)
                f.save(input_path)

                # Leer y preparar
                img = cv2.imread(input_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_rgb = img_rgb.astype(np.float32) / 255.0
                img_rgb = np.expand_dims(img_rgb, axis=0)

                # Upscale
                output = model.predict(img_rgb, verbose=0)
                output = (output[0] * 255).clip(0, 255).astype(np.uint8)
                upscaled = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

                # Calidad PRO
                upscaled = cv2.detailEnhance(upscaled, sigma_s=15, sigma_r=0.25)
                upscaled = cv2.convertScaleAbs(upscaled, alpha=1.05, beta=5)
                upscaled = add_watermark(upscaled)

                # Guardar como PNG
                new_name = f"upscaled_MBU_{len(output_files)+1}_{filename}".rsplit('.', 1)[0] + '.png'
                output_path = os.path.join(UPLOAD_FOLDER, new_name)
                cv2.imwrite(output_path, upscaled)
                output_files.append(output_path)

        # Una sola imagen → descarga directa
        if len(output_files) == 1:
            return send_file(output_files[0], as_attachment=True)

        # Varias → ZIP con nombre pro
        zip_name = f"MBU_UPSCALED_4X_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = os.path.join(UPLOAD_FOLDER, zip_name)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for file in output_files:
                z.write(file, os.path.basename(file))

        return send_file(zip_path, as_attachment=True)

    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
