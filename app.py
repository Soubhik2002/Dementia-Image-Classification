import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model
import os
import uuid
from streamlit_gsheets import GSheetsConnection
from pydrive.auth import GoogleAuth
from oauth2client.service_account import ServiceAccountCredentials
from pydrive.drive import GoogleDrive
from io import BytesIO


# Connect to Google Sheets
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

# Authenticate and create PyDrive client
def authenticate_drive():
    # gauth = GoogleAuth(creds_path='./client_secrets.json')
    # gauth.LocalWebserverAuth() # Creates local webserver and auto handles authentication.
    # drive = GoogleDrive(gauth)
    # return drive
    gauth = GoogleAuth()
    gauth.auth_method = 'service'
    # gauth.credentials = ServiceAccountAuth(filename=creds_path, gauth=gauth)
    gauth.credentials = ServiceAccountCredentials(filename=creds_path, gauth=gauth)
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    return drive


    
# Upload image to Google Drive and return the shareable link
def upload_to_drive(uploaded_file, filename, folder_id):
    drive = authenticate_drive()
    
    # Get file content as bytes
    image_content = uploaded_file.getvalue()
    
    # Create a file-like object
    image_file = BytesIO(image_content)
    
    # Save the image temporarily
    temp_path = "./temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(image_content)
    
    file = drive.CreateFile({'title': filename, 'parents': [{'id': folder_id}]})
    file.SetContentFile(temp_path)
    file.Upload()
    
    # Close the file
    f.close()
    
    # Remove the temporary file
    # os.remove(temp_path)
    
    return file['alternateLink']

# Streamlit app
def main():
    st.image("./Logo.png", width=100)
    st.markdown("""<h1 style='text-align: center;'>Dementia Classification through Facial Analysis using MOD-2D-CNN</h1>""", unsafe_allow_html=True)
    st.write("Choose an option to input image:")
    
    # Choose between webcam capture and file upload
    option = st.radio("Select Input Option", ("Webcam", "Upload Image"))
    if option == "Webcam":
        st.write("Please allow access to your webcam.")
        uploaded_file = st.camera_input("Take a picture")
        
        if uploaded_file is not None:
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
                else:
                    Prediction = "Demented"
                st.write(Prediction)
                
                # Upload image to Google Drive
                folder_id = '1oUYXeaqAcmoZ3Y1I_VMsw5_I3gWlwbu-'  # Replace 'your_folder_id' with actual folder id in Google Drive
                link = upload_to_drive(uploaded_file, str(uuid.uuid4()) + ".jpg", folder_id)

                # Create DataFrame with entered details
                Image_Data = pd.DataFrame(
                    [
                        {
                            "Image_Name": link,
                            "Result": Prediction 
                        }
                    ]
                )

                # Concatenate new data with existing data
                updated_df = pd.concat([existing_data, Image_Data], ignore_index=True)

                # Update Google Sheets with new data
                try:
                    conn.update(worksheet="Image", data=updated_df)
                    # Show success message
                    st.success("Image and Prediction Successfully Uploaded to Google Sheets!")
                except Exception as e:
                    # st.error(f"Error occurred while updating Google Sheets: {e}")
                    st.write("Priti")
        
    elif option == "Upload Image":
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
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
                else:
                    Prediction = "Demented"
                st.write(Prediction)
                
                # Upload image to Google Drive
                folder_id = '1oUYXeaqAcmoZ3Y1I_VMsw5_I3gWlwbu-'  # Replace 'your_folder_id' with actual folder id in Google Drive
                link = upload_to_drive(uploaded_file, str(uuid.uuid4()) + ".jpg", folder_id)

                # Create DataFrame with entered details
                Image_Data = pd.DataFrame(
                    [
                        {
                            "Image_Name": link,
                            "Result": Prediction 
                        }
                    ]
                )

                # Concatenate new data with existing data
                updated_df = pd.concat([existing_data, Image_Data], ignore_index=True)

                # Update Google Sheets with new data
                try:
                    conn.update(worksheet="Image", data=updated_df)
                    # Show success message
                    st.success("Image and Prediction Successfully Uploaded to Google Sheets!")
                except Exception as e:
                    st.error(f"Error occurred while updating Google Sheets: {e}")

if __name__ == '__main__':
    main()
