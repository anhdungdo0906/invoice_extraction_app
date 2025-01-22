import os
import cv2
import numpy as np
import streamlit as st
import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
from utility_functions import traditional_scan, deep_learning_scan, manual_scan, get_image_download_link
import requests
import json
import base64

# Path to the pre-downloaded model files
model_mbv3_path = os.path.join(os.getcwd(), "model_mbv3_iou_mix_2C049.pth")
model_r50_path = os.path.join(os.getcwd(), "model_r50_iou_mix_2C020.pth")

# Streamlit Configuration
st.set_page_config(
    page_title="Medical Invoice Extraction Tool",
    page_icon="🖌",
    layout="centered",  # centered, wide
    menu_items={
        "About": "A simple tool for extracting the invoice from medical invoice image."
    },
)

@st.cache_resource
def load_model_DL_MBV3(num_classes=2, device=torch.device("cpu"), img_size=384):
    checkpoint_path = model_mbv3_path
    checkpoints = torch.load(checkpoint_path, map_location=device)

    model = deeplabv3_mobilenet_v3_large(num_classes=num_classes, aux_loss=True).to(device)
    model.load_state_dict(checkpoints, strict=False)
    model.eval()
    with torch.no_grad():
        _ = model(torch.randn((1, 3, img_size, img_size)))
    return model

@st.cache_resource
def load_model_DL_R50(num_classes=2, device=torch.device("cpu"), img_size=384):
    checkpoint_path = model_r50_path
    checkpoints = torch.load(checkpoint_path, map_location=device)

    model = deeplabv3_resnet50(num_classes=num_classes, aux_loss=True).to(device)
    model.load_state_dict(checkpoints, strict=False)
    model.eval()
    with torch.no_grad():
        _ = model(torch.randn((1, 3, img_size, img_size)))
    return model

from dotenv import load_dotenv
import os

def extract_invoice_data_with_gpt4_vision(image_path, prompt):
    api_key = os.getenv("OPENAI_API_KEY")  # Lấy API key từ biến môi trường
    if not api_key:
        raise ValueError("API key is missing! Please set it in the .env file.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')

        data = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ],
            "max_tokens": 1000
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)

        if response.status_code == 200:
            response_json = response.json()
            print("Full Response:", json.dumps(response_json, indent=2))  # Log full response for debugging
            try:
                return json.loads(response_json["choices"][0]["message"]["content"])
            except (KeyError, json.JSONDecodeError):
                print("Error parsing response content:", response_json)
                return None
        else:
            print(f"Error: {response.status_code}, Response: {response.text}")
            return None

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return None

def display_extracted_data_as_json(extracted_data):
    st.markdown("### Extracted Invoice Data (JSON)")
    st.json(extracted_data)

def display_extracted_data_as_table(extracted_data):
    st.markdown("### Extracted Invoice Data (Table)")
    if not extracted_data:
        st.write("No data available.")
        return

    st.write("**Customer Name:**", extracted_data.get("customer_name", "N/A"))
    st.write("**Invoice Date:**", extracted_data.get("invoice_date", "N/A"))

    medical_facility = extracted_data.get("medical_facility", {})
    st.write("**Department Name:**", medical_facility.get("department_name", "N/A"))
    st.write("**Hospital Name:**", medical_facility.get("hospital_name", "N/A"))

    doctor = extracted_data.get("doctor", {})
    st.write("**Doctor Title:**", doctor.get("title", "N/A"))
    st.write("**Doctor Name:**", doctor.get("name", "N/A"))

    medications = extracted_data.get("medications", [])
    if medications:
        st.write("**Medications:**")
        for med in medications:
            st.write(f"- **Name:** {med.get('name', 'N/A')}, "
                     f"**Quantity:** {med.get('quantity', 'N/A')}, "
                     f"**Dosage Form:** {med.get('dosage_form', 'N/A')}, "
                     f"**Dosage Unit:** {med.get('dosage_unit', 'N/A')}, "
                     f"**Unit Price:** {med.get('unit_price', 'N/A')}, "
                     f"**Total Price:** {med.get('total_price', 'N/A')}.")
    else:
        st.write("No medications data available.")

    st.write("**Total Amount:**", extracted_data.get("total_amount", "N/A"))

def main(input_file, procedure, image_size=384):
    file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)  # Read bytes
    image = cv2.imdecode(file_bytes, 1)[:, :, ::-1]  # Decode and convert to RGB
    output = None

    st.write("Input image size:", image.shape)

    if procedure == "Manual":
        output = manual_scan(og_image=image)

    else:
        col1, col2 = st.columns((1, 1))

        with col1:
            st.title("Input")
            st.image(image, channels="RGB", use_container_width=True)

        with col2:
            st.title("Scanned")

            if procedure == "Traditional":
                output = traditional_scan(og_image=image)
            else:
                model = model_mbv3 if model_selected == "MobilenetV3-Large" else model_r50
                output = deep_learning_scan(og_image=image, trained_model=model, image_size=image_size)

            st.image(output, channels="RGB", use_container_width=True)

    if output is not None:
        # Save the scanned image temporarily
        scanned_image_path = os.path.join(os.getcwd(), "scanned_image.jpg")
        cv2.imwrite(scanned_image_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

        # Define a prompt for GPT-4 Vision
        prompt = """        
You are an AI assistant tasked with extracting structured information from an invoice image. 
If any field cannot be identified clearly from the image, make a reasonable guess based on the context, but mark it as 'estimated'. 

JSON structure:
{
   "customer_name": "string or 'estimated'",
   "invoice_date": "string (DD/MM/YYYY) or 'estimated'",
   "medical_facility": {
      "department_name": "string or 'estimated'",
      "hospital_name": "string or 'estimated'"
   },
   "doctor": {
      "title": "string or 'estimated'",
      "name": "string or 'estimated'"
   },
   "medications": [
      {
         "name": "string or 'estimated'",
         "quantity": "string or 'estimated'",
         "dosage_form": "string or 'estimated'",
         "dosage_unit": "string or 'estimated'",
         "unit_price": "string or 'estimated'",
         "total_price": "string or 'estimated'"
      }
   ],
   "total_amount": "string or 'estimated'"
}
Only return the JSON object. Do not include additional text.
        """

        # Call GPT-4 Vision API with the scanned image and prompt
        extracted_data = extract_invoice_data_with_gpt4_vision(scanned_image_path, prompt)

        if extracted_data:
            # Allow users to toggle between JSON and Table view
            display_mode = st.radio("Select Display Mode:", ("JSON", "Table"), index=0)

            if display_mode == "JSON":
                display_extracted_data_as_json(extracted_data)
            else:
                display_extracted_data_as_table(extracted_data)

    return output

IMAGE_SIZE = 384
model_mbv3 = load_model_DL_MBV3(img_size=IMAGE_SIZE)
model_r50 = load_model_DL_R50(img_size=IMAGE_SIZE)

st.markdown("<h1 style='text-align: center;'>Medical Invoice Extraction Tool</h1>", unsafe_allow_html=True)

procedure_selected = st.radio("Select Scanning Procedure:", ("Traditional", "Deep Learning", "Manual"), index=1, horizontal=True)

if procedure_selected == "Deep Learning":
    model_selected = st.radio("Select Document Segmentation Backbone Model:", ("MobilenetV3-Large", "ResNet-50"), horizontal=True)

tab1, tab2 = st.tabs(["Upload a Document", "Capture Document"])

with tab1:
    file_upload = st.file_uploader("Upload Document Image:", type=["jpg", "jpeg", "png"])

    if file_upload is not None:
        _ = main(input_file=file_upload, procedure=procedure_selected, image_size=IMAGE_SIZE)

with tab2:
    run = st.checkbox("Start Camera")

    if run:
        file_upload = st.camera_input("Capture Document", disabled=not run)
        if file_upload is not None:
            _ = main(input_file=file_upload, procedure=procedure_selected, image_size=IMAGE_SIZE)
