import streamlit as st
from PIL import Image
from api import process_image

# Set the title of your Streamlit app
st.title("AI Generated Detector App")

# Choose between image upload
file_type = "Image"

# Upload file through Streamlit
uploaded_file = st.file_uploader(f"Choose an {file_type.lower()}...", type=["jpg", "jpeg", "png"])

# Preselect the EfficientNetB4 model
model = "EfficientNetB4"
dataset = "DFDC"
threshold = 0.5

# Display the uploaded image
if uploaded_file is not None:
    if file_type == "Image":
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=200)

            # Check if the user wants to perform the deepfake detection
            if st.button("Check for Deepfake"):
                result, pred = process_image(image=uploaded_file, model=model, dataset=dataset, threshold=threshold)
                st.markdown(
                    f'''
                    <style>
                        .result{{
                            color: {'#ff4b4b' if result == 'fake' else '#6eb52f'};
                        }}
                    </style>
                    <h3>The given {file_type} is: <span class="result"> {result} </span></h3>''', unsafe_allow_html=True)
        except Exception as e:
            print(e)
            st.error(f"Error: Invalid Filetype")
else:
    st.info("Please upload a file.")