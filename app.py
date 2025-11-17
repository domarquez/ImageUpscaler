# main.py – MBU Upscaler Simple x4 – FUNCIONA 100 % SIN CRASHEAR (Railway Gratis)
from flask import Flask, request, send_file
import cv2
import numpy as np
import os
import zipfile
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def upscale_image(img, scale=4):
    # Upscale bicúbico (calidad profesional, sin magenta ni negro)
    height, width = img.shape[:2]
    new_height, new_width = height * scale, width * scale
    upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Mejora de nitidez (sharpening ligero)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    upscaled = cv2.filter2D(upscaled, -1, kernel)
    
    # Contraste sutil (hace que se vea más profesional)
    upscaled = cv2.convertScaleAbs(upscaled, alpha=1.1, beta=10)
    
    return upscaled

def add_watermark(img):
    # Sello MBU SCZ en esquina inferior derecha
    cv2.putText(img, 'MBU SCZ', (img.shape[1]-300, img.shape[0]-30), 
                cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('files')
        output_files = []

        for f in files:
            if f.filename == '':
                continue
            filename = secure_filename(f.filename)
            in_path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(in_path)

            img = cv2.imread(in_path)
            if img is None:
                continue

            upscaled = upscale_image(img, 4)
            upscaled = add_watermark(upscaled)

            out_name = f"MBU_4X_{filename.rsplit('.',1)[0]}.jpg"
            out_path = os.path.join(UPLOAD_FOLDER, out_name)
            cv2.imwrite(out_path, upscaled, [cv2.IMWRITE_JPEG_QUALITY, 95])
            output_files.append(out_path)

        if len(output_files) == 1:
            return send_file(output_files[0], as_attachment=True)

        zip_name = f"MBU_UPSCALED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = os.path.join(UPLOAD_FOLDER, zip_name)
        with zipfile.ZipFile(zip_path, 'w') as z:
            for f in output_files:
                z.write(f, os.path.basename(f))

        return send_file(zip_path, as_attachment=True)

    return '''
    <!DOCTYPE html>
    <html>
    <head><title>MBU Upscaler</title></head>
    <body style="text-align:center; background:#000; color:white; padding:50px;">
    <h1 style="color:#ff0000; font-size:3em;">MBU UPSCALER x4</h1>
    <p style="font-size:22px;">Sube fotos de pérgolas, casetas y decks → 4× más grandes + sello MBU SCZ</p>
    <form method="post" enctype="multipart/form-data">
    <input type="file" name="files" multiple accept="image/*" required style="padding:15px; font-size:18px;">
    <br><br>
    <input type="submit" value="UPSCALAR AHORA" style="background:#ff0000; color:white; padding:20px 50px; font-size:24px; border:none; border-radius:50px;">
    </form>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
