import requests
import base64

# Define the base URL of your Flask application
base_url = "http://127.0.0.1:5000"  # Update with your actual URL

# Load images as binary data
with open("ludwig3.jpg", "rb") as img_file:
    image1_data = img_file.read()

with open("ludwig4.jpg", "rb") as img_file:
    image2_data = img_file.read()

# Encode images to base64 strings
image1_base64 = base64.b64encode(image1_data).decode('utf-8')
image2_base64 = base64.b64encode(image2_data).decode('utf-8')

# Define the API endpoint
endpoint = "/compare-faces"

# Create JSON payload
payload = {
    "image1": image1_base64,
    "image2": image2_base64
}

# Make POST request to the API
response = requests.post(base_url + endpoint, json=payload)

# Check if request was successful
if response.status_code == 200:
    # Print similarity score
    print("Similarity Score:", response.json()['similarity_score'])
else:
    print("Error:", response.json()['error'])
