"""

"""

import os
import torch
# from torchvision.io import read_image
from PIL import Image
# from torchvision.utils import save_image
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from potholes_utils import UNet, SegNet, data_transforms, decode_mask


unet_model = UNet()
unet_model_save_name = "model_unet_state_dict.pth"
unet_model.load_state_dict(torch.load(unet_model_save_name, map_location=torch.device("cpu")))

segnet_model = SegNet()
segnet_model_save_name = "model_segnet_state_dict.pth"
segnet_model.load_state_dict(torch.load(segnet_model_save_name, map_location=torch.device("cpu")))



app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# For simplicity, we're not implementing async file saving/loading
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# Check if an extension is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route that will process the file upload
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('upload.html', message='No file part')
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return render_template('upload.html', message='No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(saved_path)
            # Here you can call your model's prediction function and save the output
            output_filename = 'processed_' + filename
            process_image(saved_path, os.path.join(app.config['UPLOAD_FOLDER'], output_filename))
            # For demonstration, we just copy the uploaded file
            # os.rename(saved_path, os.path.join(app.config['UPLOAD_FOLDER'], output_filename))
            return render_template('display.html', original_image=filename, mask=output_filename)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Placeholder for your image processing function
def process_image(input_path, output_path):
    # Your model code to process the image
    # For example, it might load the image, predict and save the segmented output
    image = Image.open(input_path) / 255.0
    image = data_transforms(image)

    unet_preds_raw = unet_model(image.unsqueeze(0))
    unet_preds_1d = unet_preds_raw.argmax(axis=1).squeeze(0)
    unet_mask = decode_mask(unet_preds_1d)
    # save_image(unet_mask.float(), output_path)
    Image.fromarray(unet_mask.numpy().astype(np.uint8).transpose(1, 2, 0)).save(output_path)

if __name__ == '__main__':
    app.run(debug=True)
