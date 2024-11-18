from flask import Flask, request, render_template
import os
from beamsearch import generate
from GreedyCaption import generate_caption
import cv2
app = Flask(__name__)

UPLOAD_FOLDER = 'static/img'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def homepage():
    return render_template('homepage.html')


@app.route("/mid", methods=["GET", "POST"])
def mid():
    return render_template('mid.html')

@app.route("/capture", methods=["GET", "POST"])
def captureImage():
    # Capture image from webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'capture.jpg'), frame)
    cap.release()
    
    # Generate caption for the captured image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'capture.jpg')
    description = generate_caption(image_path)  # Assuming generate_caption is defined elsewhere
    
    return render_template('capture.html', cp=description, src=image_path)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    description = None
    p = None
    if request.method == "POST" and 'photo' in request.files:
        file = request.files['photo']
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            p = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            description = generate(p)  # Assuming generate_caption is defined elsewhere
    return render_template('upload.html', cp=description, src=p)








