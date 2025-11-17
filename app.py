# main.py - MBU Image Upscaler x4 (Calidad PRO 2025) - @diegomarquezb
import os
import zipfile
from flask import Flask, request, send_file, render_template_string
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime

# Forzar CPU (elimina el warning de CUDA en Railway)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB máx

# Crear carpeta temporal
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Cargando modelo ESRGAN... (30-60 segundos)")
model = load_model('ESRGAN/models/RRDB_ESRGAN_x4.h5', compile=False)
print("Modelo ESRGAN cargado correctamente!")

# Plantilla HTML con tu marca y estilo rojo MBU
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MBU Image Upscaler x4</title>
    <style>
        body {font-family: 'Arial', sans-serif; background: linear-gradient(135deg, #1a1a1a, #2d0b0b); color: white; text-align: center; padding: 40px;}
        h1 {font-size: 3em; margin-bottom: 10px; text-shadow: 0 0 20px #ff0000;}
        .logo {font-size: 1.5em; color: #ff0000; font-weight: bold;}
        .container {max-width: 800px; margin: 0 auto; background: rgba(0,0,0,0.7); padding: 40px; border-radius: 20px; box-shadow: 0 0 30px rgba(255,0,0,0.5);}
        input[type="file"] {margin: 20px 0; padding: 15px; background: #333; color: white; border: 2px solid #ff0000; border-radius: 10px;}
        input[type="submit"] {background: #ff0000; color: white; padding: 18px 40px; font-size: 1.5em; border: none; border-radius: 50px; cursor: pointer; box-shadow: 0 0 20px #ff0000; transition: all 0.3s;}
        input[type="submit"]:hover {background: #ff3333; transform: scale(1.1);}
        .footer {margin-top: 50px; font-size: 0.9em; color: #aaa;}
        .watermark {position: absolute; bottom: 20px; right: 20px; background: rgba(0,0,0,0.7); padding: 5px 15px; border-radius: 10px; font-weight: bold; color: #ff0000;}
    </style>
</head>
<body>
    <div class="container">
        <h1>MBU IMAGE UPSCALER</h1>
        <p class="logo">MBU SCZ ⚡ x4</p>
        <p>Sube tus fotos de pérgolas, casetas, decks y cualquier proyecto<br>y recíbelas en calidad ULTRA HD 4×</p>
        
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="files" multiple accept="image/*" required>
            <br>
            <input type="submit" value="UPSCALAR AHORA">
        </form>
        
        <div class="footer">
            © 2025 MBU SCZ - Diego Márquez B. | @diegomarquezb
        </div>
    </div>
</body>
</html>
'''

def add_watermark(img):
    """Agrega sello MBU SCZ en esquina inferior derecha"""
    text = "MBU SCZ"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 2
    thickness = 4
    color = (0, 0, 255)  # Rojo BGR
    margin = 20
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = img.shape[1] - text_size[0] - margin
    text_y = img.shape[0] - margin
    
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('files')
        output_files = []
        
        for i, f in enumerate(files):
            filename = secure_filename(f.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(input_path)
            
            # Leer y preprocesar
            img = cv2.imread(input_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Upscale 4×
            output = model.predict(img)
            output = output[0] * 255.0
            output = np.clip(output, 0, 255).astype(np.uint8)
            upscaled_img = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            # Mejora de detalles (sharpening PRO)
            upscaled_img = cv2.detailEnhance(upscaled_img, sigma_s=15, sigma_r=0.25)
            
            # Watermark MBU SCZ
            upscaled_img = add_watermark(upscaled_img)
            
            # Guardar en PNG (calidad perfecta)
            ext = os.path.splitext(filename)[1].lower()
            new_name = f"upscaled_MBU_{i+1}_{secure_filename(f.filename)}"
            if ext not in ['.png', '.PNG']:
                new_name = new_name.replace(ext, '.png')
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], new_name)
            cv2.imwrite(output_path, upscaled_img)  # PNG = máxima calidad
            
            output_files.append(output_path)
        
        # Si es una sola imagen → descarga directa
        if len(output_files) == 1:
            return send_file(output_files[0], as_attachment=True)
        
        # Si son varias → ZIP
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], f"MBU_UPSCALED_4X_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in output_files:
                zipf.write(file, os.path.basename(file))
        
        return send_file(zip_path, as_attachment=True)
    
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
