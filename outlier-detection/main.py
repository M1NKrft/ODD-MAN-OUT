from flask import Flask, request, render_template, redirect, url_for
import os
import shutil
import sys
from PIL import Image

sys.path.append('/home/ansh/ODD-MAN-OUT/outlier-detection/src')
sys.path.append('/home/ansh/ODD-MAN-OUT/outlier-detection/utils')
sys.path.append('.')
from src.detectclassify import detect_and_classify
from src.single_outlier import detect_outlier
from src.clustering import cluster_images 
from classification import classify_image 
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'static/images'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Upload images to folder
@app.route("/", methods=["GET", "POST"])
def upload_images():
    if request.method == "POST":
        # Clear existing images
        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        images = request.files.getlist("images")
        file_paths = []
        for img in images:
            ext = img.filename.split('.')[-1].lower()
            new_filename = os.path.splitext(img.filename)[0] + ".jpg"
            save_path = os.path.join(UPLOAD_FOLDER, new_filename)
            
            if ext in ["jpg", "jpeg"]:
                img.save(save_path)
            else:
                image = Image.open(img)
                rgb_im = image.convert("RGB")
                rgb_im.save(save_path, "JPEG")

            file_paths.append(save_path)
    return render_template("index.html")


@app.route("/detect_outlier")
def detect_outlier_route():
    file_paths = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER)]

    if not file_paths:
        return redirect(url_for("upload_images"))

    outlier_index = detect_outlier(file_paths) 
    outlier_image = file_paths[outlier_index] if outlier_index is not None else None

    return render_template("single_out.html", file_paths=file_paths, outlier_image=outlier_image)

@app.route('/clustering', methods=['POST', 'GET'])
def cluster():
    n_clusters = 3
    
    image_dir = "static/images"
    output_dir = "cluster_folders"

    labels, image_paths, cluster_plot = cluster_images(image_dir, output_dir, n_clusters)

    clustered_images = {}
    for img_path, label in zip(image_paths, labels):
        if label not in clustered_images:
            clustered_images[label] = []
        clustered_images[label].append(img_path)

    return render_template(
        "cluster_result.html",
        clustered_images=clustered_images,
        cluster_plot=cluster_plot
    )

@app.route('/flowerclassify', methods=['POST', 'GET'])
def flowerclassify():
    file_paths = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER)]
    if not file_paths:
        return redirect(url_for("upload_images"))
    outlier_path, flowername, other_path = detect_and_classify()
    print(flowername)
    return render_template("flower.html", outlier_path=outlier_path, flowername=flowername, other_path=other_path, file_paths=file_paths)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)