import requests
import cv2 as cv
import json
import numpy as np

# from tensorflow.keras.applications.ima

image = cv.imread("test-1.png")
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image = cv.resize(image, (28, 28))
image = np.array(image).reshape(1, 28, 28, 1)
# normalized_image = cv.normalize(image, image, 0, 255, cv.NORM_MINMAX)
data = json.dumps({"signature_name": "serving_default", "instances": image.tolist()})

url = "http://localhost:8501/v1/models/cnn:predict"

response = requests.post(url, data=data, headers={"content-type": "application/json"})

predictions = json.loads(response.text)
print(predictions)
