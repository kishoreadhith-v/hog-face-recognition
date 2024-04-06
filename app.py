import cv2
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO

app = Flask(__name__)


def base64_to_image(base64_string):
    try:
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(base64_string)

        # Convert bytes to PIL image
        img_pil = Image.open(io.BytesIO(image_bytes))

        # Convert PIL image to numpy array
        img_np = np.array(img_pil)

        return img_np
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def extract_hog(image):
    try:
        # Convert image to grayscale
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Resize image to fixed size (if needed)
        img_resized = cv2.resize(img_gray, (64, 128))

        # Calculate HOG descriptor
        hog = cv2.HOGDescriptor()
        hog_descriptor = hog.compute(img_resized)

        return hog_descriptor.flatten()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def compare_faces_with_hog(image1, image2):
    try:
        # Extract HOG features for both images
        hog_features1 = extract_hog(image1)
        hog_features2 = extract_hog(image2)

        # Perform comparison (e.g., using cosine similarity)
        similarity_score = np.dot(hog_features1, hog_features2) / (
            np.linalg.norm(hog_features1) * np.linalg.norm(hog_features2)
        )

        return float(similarity_score)  # Convert to standard Python float
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


@app.route("/compare-faces", methods=["POST"])
def compare_faces_handler():
    try:
        logging.info("Request received at /compare-faces endpoint")

        # Log the request data

        # Parse base64 strings from request
        base64_image1 = request.json["image1"]
        base64_image1_data = base64_image1[0]["data"]
        base64_image2 = request.json["image2"]

        # Convert base64 strings to images
        image1 = base64_to_image(base64_image1_data)
        image2 = base64_to_image(base64_image2)

        # Perform face comparison using HOG descriptors
        similarity_score = compare_faces_with_hog(image1, image2)

        # Return similarity score as JSON response
        return jsonify({"similarity_score": similarity_score}), 200
    except Exception as e:
        # Log the exception
        logging.error(f"Error in /compare-faces endpoint: {str(e)}")

        # Handle errors
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
