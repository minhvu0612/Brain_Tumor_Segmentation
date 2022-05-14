import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import streamlit as st
import sys

model = load_model('BrainTumor10Epochs.h5')
type = ['jpeg', 'png', 'jpg', 'webp']

def show(file, width = 200):
	image = Image.open(file)
	image = np.array(image.convert("RGB"))
	st.image(image, use_column_width = width)
	return image

def resultPredict(image):
	image = np.asarray(image)
	img = Image.fromarray(image)
	img = img.resize((64,64))
	img = np.array(img)
	input_img = np.expand_dims(img, axis=0)
	result = (model.predict(input_img) > 0.5).astype("int32")
	if result[0][0] == 1:
		st.error("tumor")
	else:
		st.success("no tumor")

def main():
	st.title("Face Mask Detection")
	st.write("**Using CNN(Deep learning)**")
	active = [
	          "Home", 
	          "Image", 
	          "Training"
	          ]
	choose = st.sidebar.selectbox("Menu", active)
	if choose == "Home":
		# Home page
		st.success("Welcome to Project Tumor!")
	elif choose == "Image":
		# Image page - Upload image and predict
		file_image = st.file_uploader("Choose File", type = type)
		if file_image is not None:
			image = show(file_image)
			if st.button("Run"):
				resultPredict(image)
	elif choose == "Training":
		# Training page - Plot training and history
		pass
main()

