# main.py → MBU Upscaler x4 – COLORES PERFECTOS 100% – Railway 2025
import os
import zipfile
from flask import Flask, request, send_file, render_template_string
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Cargando ESRGAN x4...")
model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
print("¡MODELO CARGADO! MBU UPSCALER FULL POWER")

HTML = '''
<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MBU UPSCALER x4 PRO</title>
<style>
body {font-family: Arial; background: #000; color: white; text-align: center; padding: 50px;}
h1 {color: #ff0000; font-size: 3.5em; text-shadow: 0 0 30px red;}
.container {max-width: 900px; margin: auto; background: rgba(0,0,0,0.8); padding: 50px; border-radius: 20px;}
input[type="file"] {padding: 20px; background: #222; border: 3px solid #ff0000; border-radius: 15px; color: white;}
input[type="submit"] {background: #ff0000; color: white; padding: 22px 70px; font-size: 2em; border: none; border-radius: 50px; cursor: pointer;}
input[type="submit"]:hover {background: #ff3333;}
</style></head><body>
<div class="container">
<h1>MBU UPSCALER x4</h1>
<p style="font-size:22px">Sube tus fotos → 4× más grandes + sello MBU SCZ</p>
<form method="post" enctype="multipart/form-data">
<input type="file" name="files" multiple accept="image/*" required><br><br>
<input type="submit" value="UPSCALAR AHORA">
</form>
</div></body></html>
'''

def add_watermark(img):
    cv2.putText(img, 'MBU SCZ', (img.shape[1]-420, img.shape[0]-60), 
                cv2.FONT_HERSHEY_DUPLEX, 4, (0,0,255), 10, cv2.LINE_AA)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('files')
        output_files = []

        for f in files:
            filename = secure_filename(f.filename)
            in_path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(in_path)

            # Leer correctamente
            img = cv2.imread(in_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 127.5 - 1.0   # <-- ¡¡ESTO ES LA CLAVE!!
            img = np.expand_dims(img, axis=0)

            # Upscale
            upscaled = model(img)
            upscaled = tf.squeeze(upscaled)

            # Convertir de [-1,1] a [0,255] y corregir canales
            upscaled = (upscaled + 1.0) * 127.5
            upscaled = tf.clip_by_value(upscaled, 0, 255)
            upscaled = upscaled.numpy().astype(np.uint8)

            # Ya está en RGB → pasamos a BGR solo para OpenCV
            final = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
            final = cv2.detailEnhance(final, sigma_s=15, sigma_r=0.25)
            final = cv2.convertScaleAbs(final, alpha=1.12, beta=10)
            final = add_watermark(final)

            out_name = f"MBU_4X_{filename.rsplit('.',1)[0]}.png"
            out_path = os.path.join(UPLOAD_FOLDER, out_name)
            cv2.imwrite(out_path, final)
            output_files.append(out_path)

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
