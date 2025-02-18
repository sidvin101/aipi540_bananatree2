import streamlit as st
import numpy as np
import joblib
import cv2
from skimage.feature import hog, local_binary_pattern

def extract_features(uploaded_file):
    """Extract classical ML features from an uploaded file object."""
    # read the file's bytes and convert to a numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # decode the image from the numpy array
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the image. Please check the file format.")
    
    # resize the image to 128x128
    img = cv2.resize(img, (128, 128))
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # extract HOG features
    hog_features = hog(
        gray,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        feature_vector=True
    )

    # extract LBP features
    lbp = local_binary_pattern(gray, P=8, R=2)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 256), density=True)

    # extract color histogram features from each channel
    color_hist = []
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [16], [0, 256])
        color_hist.extend(hist.flatten())

    # Concatenate all features into one vector
    features = np.hstack([hog_features, lbp_hist, color_hist])
    return features

# load the SVM model (ensure that the file 'svm_model_fixed.pkl' is in the correct directory)
model = joblib.load('models/svm_model_fixed.pkl')

# define your class names (update these as needed)
class_names = ['Fusarium', 'Healthy', 'Natural Death', 'Rhizome']

st.title("Classify Your Banana Leaf!")
st.write("Upload an image of a banana leaf and the SVM model will classify it into one of four classes.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # display the uploaded image using Streamlit's built-in function.
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # reset file pointer to beginning, as it may have been advanced by st.image
    uploaded_file.seek(0)
    
    st.write("Processing image and predicting...")
    # extract features from the uploaded image
    features = extract_features(uploaded_file)
    features = features.flatten().reshape(1, -1)
    
    # get prediction from the model
    prediction = model.predict(features)
    predicted_class_index = int(prediction[0])
    predicted_class_name = class_names[predicted_class_index]
    
    # display the prediction
    st.write(f"Prediction: **{predicted_class_name}**")
