import cv2
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

def extract_hog(image):
    try:
        # Convert image bytes to PIL image
        img_pil = Image.open(image)
        
        # Convert PIL image to numpy array
        img_np = np.array(img_pil)
        
        # Convert image to grayscale
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
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
        similarity_score = np.dot(hog_features1, hog_features2) / (np.linalg.norm(hog_features1) * np.linalg.norm(hog_features2))
        
        return float(similarity_score)  # Convert to standard Python float
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

@app.route('/compare-faces', methods=['POST'])
def compare_faces_handler():
    try:
        # Parse image data from request
        image1 = request.files['image1']
        image2 = request.files['image2']
        
        # Perform face comparison using HOG descriptors
        similarity_score = compare_faces_with_hog(image1, image2)
        
        # Return similarity score as JSON response
        return jsonify({'similarity_score': similarity_score}), 200
    except Exception as e:
        # Handle errors
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
