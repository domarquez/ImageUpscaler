# main.py → MBU UPSCALER IA REAL x4 – CALIDAD BRUTAL + SELLO BLANCO ELEGANTE
import os
import zipfile
from flask import Flask, request, send_file, render_template_string
from werkzeug.utils import secure_filename
import cv2
from datetime import datetime
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Cargando Real-ESRGAN IA REAL x4 (calidad TOP)...")
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
upsampler = RealESRGANer(
    scale=4,
    model_path='weights/RealESRGAN_x4plus.pth',   # ← modelo local (obligatorio)
    model=model,
    tile=400,          # evita crash por memoria
    tile_pad=10,
    pre_pad=0,
    half=False
)
print("¡IA REAL CARGADA! MBU UPSCALER FULL POWER")

# SELLO MBU SCZ BLANCO SEMI-TRANSPARENTE (PRO)
def add_watermark(img):
    text = "MBU SCZ"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 5.5
    thickness = 14
    color = (255, 255, 255)   # Blanco puro
    alpha = 0.78              # 78% opaco → 22% transparente (queda perfecto)

    overlay = img.copy()
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = img.shape[1] - text_size[0] - 70
    text_y = img.shape[0] - 70

    cv2.putText(overlay, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img

HTML = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MBU UPSCALER IA x4</title>
    <style>
        body {font-family: Arial; background: linear-gradient(135deg, #000, #3d0b0b); color: white; text-align: center; padding: 50px;}
        h1 {font-size: 4em; color: #ff0000; text-shadow: 0 0 40px #ff0000;}
        .container {max-width: 900px; margin: 0 auto; background: rgba(0,0,0,0.9); padding: 60px; border-radius: 25px;}
        input[type="file"] {padding: 20px; background: #222; border: 3px solid #ff0000; border-radius: 15px; color: white; font-size: 18px;}
        input[type="submit"] {background: #ff0000; color: white; padding: 25px 90px; font-size: 2.2em; border: none; border-radius: 50px; cursor: pointer; margin-top: 30px; box-shadow: 0 0 40px #ff0000;}
        input[type="submit"]:hover {background: #ff3333; transform: scale(1.05);}
    </style>
</head>
<body>
    <div class="container">
        <h1>MBU UPSCALER IA x4</h1>
        <p style="font-size: 1.9em;">Sube tus fotos de pérgolas, casetas y decks<br>y recíbelas en calidad ULTRA HD con sello MBU SCZ</p>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="files" multiple accept="image/*" required>
            <br><br>
            <input type="submit" value="UPSCALAR AHORA">
        </form>
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('files')
        outputs = []

        for f in files:
            filename = secure_filename(f.filename)
            in_path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(in_path)

            img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
            result, _ = upsampler.enhance(img, outscale=4)
            result = add_watermark(result)

            out_name = f"MBU_IA_4X_{filename.rsplit('.',1)[0]}.png"
            out_path = os.path.join(UPLOAD_FOLDER, out_name)
            cv2.imwrite(out_path, result)
            outputs.append(out_path)

        if len(outputs) == 1:
            return send_file(outputs[0], as_attachment=True)

        zip_path = os.path.join(UPLOAD_FOLDER, f"MBU_IA_4X_{datetime.now():%Y%m%d_%H%M%S}.zip")
        with zipfile.ZipFile(zip_path, 'w') as z:
            for p in outputs:
                z.write(p, os.path.basename(p))
        return send_file(zip_path, as_attachment=True)

    return render_template_string(HTML)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
