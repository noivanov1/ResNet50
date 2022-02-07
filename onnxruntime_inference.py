import onnxruntime
import onnx
import numpy as np
#import cv2
import os
from PIL import Image

model_path = 'model_onnx/converted_model.onnx'
onnx_model = onnx.load(model_path)

content = onnx_model.SerializeToString()
sess = onnxruntime.InferenceSession(content)
image = 'photo.bmp'
image = Image.open(image)
img = np.array(image)

img = np.swapaxes(img, 0, 2)
img = np.swapaxes(img, 1, 2)  # change to (c, h,w) order
img = np.expand_dims(img, axis=0).astype(np.float32)

print(img.shape)
print(type(img))
feed1 = {sess.get_inputs()[0].name: img}
out = sess.run(None, feed1)[0][0]
print(out)

with open('onnx.txt', "w") as file:
    for i in range(len(out)):
        file.write(str(out[i]) + '\n')