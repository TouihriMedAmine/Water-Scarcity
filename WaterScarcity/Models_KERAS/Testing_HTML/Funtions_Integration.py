import os
import cgi
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image as PILImage
from http.server import BaseHTTPRequestHandler, HTTPServer
from tensorflow import keras



cnn_model = keras.models.load_model(r"D:\Models_KERAS\Classification.keras", compile=False)
unet_lake_model = keras.models.load_model(r"D:\Models_KERAS\Lake_prediction.keras", compile=False)
unet_river_model = keras.models.load_model(r"D:\Models_KERAS\River_prediction.keras", compile=False)
unet_harbor_model = keras.models.load_model(r"D:\Models_KERAS\harbor_prediction.keras", compile=False)



# Target image size for models
IMG_SIZE = (128, 128)

# Function to predict the class of the image using CNN
def predict_class(image):
    img_resized = cv2.resize(image, IMG_SIZE)
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    prediction = cnn_model.predict(img_resized)
    predicted_class = np.argmax(prediction, axis=1)[0]

    if predicted_class == 0:
        return "harbor"
    elif predicted_class == 1:
        return "lake"
    else:
        return "river"

# Function to predict depth based on the class using U-Net
def predict_depth(image, predicted_class):
    if predicted_class == "lake":
        model = unet_lake_model
    elif predicted_class == "river":
        model = unet_river_model
    else:
        model = unet_harbor_model

    img_resized = cv2.resize(image, IMG_SIZE)
    img_resized = img_resized / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    predicted_mask = model.predict(img_input)[0]
    
    # Convert to correct format for mask and ensure it's the right size
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255
    predicted_mask = cv2.resize(predicted_mask, (img_resized.shape[1], img_resized.shape[0]))
    
    # Make sure blue_channel is also the right format
    blue_channel = (img_resized[:, :, 0] * 255).astype(np.uint8)
    
    # Now apply the mask
    water_region = cv2.bitwise_and(blue_channel, blue_channel, mask=predicted_mask)

    # Calculate blue intensity
    non_zero_pixels = cv2.countNonZero(predicted_mask)
    if non_zero_pixels > 0:
        blue_intensity = np.sum(water_region) / non_zero_pixels / 255.0
    else:
        blue_intensity = 0

    if blue_intensity < 0.33:
        depth_label = "High Depth"
    elif blue_intensity < 0.66:
        depth_label = "Medium Depth"
    else:
        depth_label = "Low Depth"

    return depth_label, round(blue_intensity, 2)
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/":
            # Serve the HTML form
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            with open("index.html", "r") as file:
                self.wfile.write(file.read().encode())

    def do_POST(self):
        if self.path == "/upload":
            # Parse the uploaded image
            ctype, pdict = cgi.parse_header(self.headers['Content-Type'])
            if ctype == 'multipart/form-data':
                pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
                fs = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': self.headers['Content-Type']})
                file_item = fs['image']

                if file_item.filename:
                    # Read the image
                    img_data = file_item.file.read()
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    # Predict the class
                    predicted_class = predict_class(img)

                    # Predict depth
                    predicted_depth, blue_intensity = predict_depth(img, predicted_class)

                    # Send the result back to the client
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()

                    result = f"""
                    <html>
                    <body>
                    <h2>Upload Successful!</h2>
                    <p><b>Predicted Class:</b> {predicted_class}</p>
                    <p><b>Predicted Depth:</b> {predicted_depth}</p>
                    <p><b>Blue Intensity:</b> {blue_intensity}</p>
                    <a href="/">Go Back</a>
                    </body>
                    </html>
                    """
                    self.wfile.write(result.encode())

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8888):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
