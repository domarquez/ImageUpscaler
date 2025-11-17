# main.py → MBU UPSCALER PRO 2025 → Real-ESRGAN PyTorch → CALIDAD BRUTAL
import os
import zipfile
from flask import Flask, request, send_file, render_template_string
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import torch
from PIL import Image
from datetime import datetime

# Pip install realesrgan en Railway ya lo hace con requirements.txt
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# Forzar CPU y limpiar logs
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.set_num_threads(2)

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Cargando Real-ESRGAN x4 PRO... (20-35 segundos)")
# Modelo oficial ligero y perfecto
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
upsampler = RealESRGANer(
    scale=4,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/RealESRGAN_x4plus_anime_6B.pth',
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False
)
print("¡REAL-ESRGAN CARGADO! MBU UPSCALER LISTO PARA ROMPER")

HTML = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MBU UPSCALER x4 PRO</title>
    <style>
        body {font-family: Arial; background: linear-gradient(135deg, #000, #3d0b0b); color: white; text-align: center; padding: 40px;}
        h1 {font-size: 3.8em; color: #ff0000; text-shadow: 0 0 30px #ff0000;}
        .container {max-width: 900px; margin: 0 auto; background: rgba(0,0,0,0.9); padding: 60px; border-radius: 20px;}
        input[type="file"] {padding: 20px; background: #222; border: 3px solid #ff0000; border-radius: 15px; color: white;}
        input[type="submit"] {background: #ff0000; color: white; padding: 25px 80px; font-size: 2em; border: none; border-radius: 50px; cursor: pointer; margin-top: 30px;}
        input[type="submit"]:hover {background: #ff3333; transform: scale(1.05);}
    </style>
</head>
<body>
    <div class="container">
        <h1>MBU UPSCALER x4 PRO</h1>
        <p style="font-size: 1.8em;">Sube tus fotos de pérgolas, casetas y decks<br>y recíbelas en calidad ULTRA HD 4× con sello MBU SCZ</p>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="files" multiple accept="image/*" required>
            <br><br>
            <input type="submit" value="UPSCALAR AHORA">
        </form>
    </div>
</body>
</html>
'''

def add_watermark(img):
    cv2.putText(img, 'MBU SCZ', (img.shape[1]-450, img.shape[0]-70), 
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

            # Procesar con Real-ESRGAN
            img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
            output, _ = upsampler.enhance(img, outscale=4)
            output = add_watermark(output)
            output = cv2.convertScaleAbs(output, alpha=1.05, beta=5)

            out_name = f"MBU_4X_{filename.rsplit('.',1)[0]}.png"
            out_path = os.path.join(UPLOAD_FOLDER, out_name)
            cv2.imwrite(out_path, output)
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
