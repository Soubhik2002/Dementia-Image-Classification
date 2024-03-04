import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model
import cv2
import os
import uuid
from streamlit_gsheets import GSheetsConnection

conn = st.connection("gsheets", type=GSheetsConnection)
existing_data = conn.read(worksheet="Image", usecols=list(range(7)), ttl=5)
existing_data = existing_data.dropna(how="all")


# Load the Keras model
model = load_model("CNN2Dmodel1.h5")

def preprocess_image(image):
    # Resize the image to match the input shape of the model (224x224)
    image = image.resize((224, 224))
    # Convert image to RGB if it has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Convert image to array and normalize pixel values
    image_array = np.array(image) / 255.0
    
    # Expand dimensions to create a batch of 1 sample
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Function to make predictions
def predict(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Make prediction
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app
def main():
    st.image("./Logo.png", width=100)
    st.markdown("""<h1 style='text-align: center;'>Dementia Classification through Facial Analysis using MOD-2D-CNN</h1>""", unsafe_allow_html=True)
    # st.title("Dementia Classification through Facial Analysis using MOD-2D-CNN")
    st.write("Choose an option to input image:")
    
    # Choose between webcam capture and file upload
    option = st.radio("Select Input Option", ("Webcam", "Upload Image"))
    if option == "Webcam":
        st.write("Please allow access to your webcam.")
        uploaded_file = st.camera_input("Take a picture")
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            # st.image(image, caption='Uploaded Image.', use_column_width=True)
            # Generate a unique filename
            filename = str(uuid.uuid4()) + ".jpg"
            # Define the path where you want to save the image
            save_path = "images/Webcam"
            os.makedirs(save_path, exist_ok=True)  # Create the folder if it doesn't exist

            # Save the image to the specified folder with the unique filename
            image_path = os.path.join(save_path, filename)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)


            # Check if prediction button is clicked
            if st.button("Predict"):
                # Make prediction
                prediction = predict(image)
                # Display prediction result
                if prediction[0][0] > 0.5:
                    Prediction = "Non-Demented"
                    st.write(Prediction)
                else:
                    Prediction = "Demented"
                    st.write(Prediction)
                    
                # Create DataFrame with entered details
                Image_Data = pd.DataFrame(
                    [
                        {
                            "Image_Name": filename,
                            "Result" : Prediction 
                            
                        }
                    ]
                )

                # Concatenate new data with existing data
                updated_df = pd.concat([existing_data, Image_Data], ignore_index=True)

                # Update Google Sheets with new data
                conn.update(worksheet="Image", data=updated_df)

                # Show success message
                st.success("Health Metrics Details Successfully Submitted!")

        
    elif option == "Upload Image":
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Generate a unique filename
            filename = str(uuid.uuid4()) + ".jpg"
            # Define the path where you want to save the image
            save_path = "images/Uploaded"
            os.makedirs(save_path, exist_ok=True)  # Create the folder if it doesn't exist

            # Save the image to the specified folder with the unique filename
            image_path = os.path.join(save_path, filename)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            # Check if prediction button is clicked
            if st.button("Predict"):
                # Make prediction
                prediction = predict(image)
                # Display prediction result
                if prediction[0][0] > 0.5:
                    Prediction = "Non-Demented"
                    st.write(Prediction)
                else:
                    Prediction = "Demented"
                    st.write(Prediction)
                
                # Create DataFrame with entered details
                Image_Data = pd.DataFrame(
                    [
                        {
                            "Image_Name": filename,
                            "Result" : Prediction 
                        }
                    ]
                )

                # Concatenate new data with existing data
                updated_df = pd.concat([existing_data, Image_Data], ignore_index=True)

                # Update Google Sheets with new data
                conn.update(worksheet="Image", data=updated_df)

                # Show success message
                st.success("Health Metrics Details Successfully Submitted!")
                    

if __name__ == '__main__':
    main()



