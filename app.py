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
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import unicodedata

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

# Create the main data folder
data_folder = os.path.join(os.getcwd(), "data")
os.makedirs(data_folder, exist_ok=True)


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


# Nạp biến môi trường từ .env
load_dotenv()


def extract_invoice_data_with_gpt4_vision(image_path, prompt):
    api_key = os.getenv("OPENAI_API_KEY")
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

        print(f"API Response Status Code: {response.status_code}")
        print(f"API Response Content: {response.text}")  # Log full response

        if response.status_code == 200:
            response_text = response.text.strip()
            if not response_text:
                print("Error: Empty response from API")
                return None

            try:
                response_json = response.json()
                print("Full Response:", json.dumps(response_json, indent=2))

                if "choices" in response_json and len(response_json["choices"]) > 0:
                    extracted_data_str = response_json["choices"][0]["message"]["content"]

                    # Clean JSON response
                    if extracted_data_str.startswith("```json"):
                        extracted_data_str = extracted_data_str.replace("```json", "").replace("```", "").strip()

                    extracted_data = json.loads(extracted_data_str)  # Convert to Python dict
                    return extracted_data
                else:
                    print("Error: 'choices' not found in response.")
                    return None
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Error parsing response content: {str(e)}")
                return None
        else:
            print(f"Error: {response.status_code}, Response: {response.text}")
            return None

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return None



def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])


def display_extracted_data_as_json(extracted_data):
    st.markdown("### Extracted Invoice Data (JSON)")
    if extracted_data:
        st.json(extracted_data)
    else:
        st.write("No data available.")


def display_extracted_data_as_table(extracted_data):
    st.markdown("### Extracted Invoice Data (Table)")
    if not extracted_data:
        st.write("No data available.")
        return

    # Customer Information
    customer_name = extracted_data.get("customer_name", "N/A")
    invoice_date = extracted_data.get("invoice_date", "N/A")
    st.write("**Customer Name:**", customer_name)
    st.write("**Invoice Date:**", invoice_date)

    # Medical Facility Information
    medical_facility = extracted_data.get("medical_facility", {})
    department_name = medical_facility.get("department_name", "N/A")
    hospital_name = medical_facility.get("hospital_name", "N/A")
    st.write("**Department Name:**", department_name)
    st.write("**Hospital Name:**", hospital_name)

    # Doctor Information
    doctor = extracted_data.get("doctor", {})
    doctor_title = doctor.get("title", "N/A")
    doctor_name = doctor.get("name", "N/A")
    st.write("**Doctor Title:**", doctor_title)
    st.write("**Doctor Name:**", doctor_name)

    # Medications Information
    medications = extracted_data.get("medications", [])
    if medications:
        st.write("**Medications:**")
        # Prepare data for medications table
        medications_data = []
        for idx, med in enumerate(medications, start=1):
            medications_data.append({
                "No.": idx,
                "Name": med.get('name', 'N/A'),
                "Quantity": med.get('quantity', 'N/A'),
                "Dosage Form": med.get('dosage_form', 'N/A'),
                "Dosage Unit": med.get('dosage_unit', 'N/A'),
                "Unit Price": med.get('unit_price', 'N/A'),
                "Total Price": med.get('total_price', 'N/A'),
            })

        # Display medications as a table using st.dataframe()
        meds_df = pd.DataFrame(medications_data)
        st.dataframe(meds_df, use_container_width=True)  # Avoid showing index column
    else:
        st.write("No medications data available.")

    # Total Amount
    total_amount = extracted_data.get("total_amount", "N/A")
    currency = extracted_data.get("currency", "N/A")
    st.write("**Total Amount:**", total_amount)
    st.write("**Currency:**", currency)


def main(input_file, procedure, image_size=384):
    file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)  # Read bytes
    image = cv2.imdecode(file_bytes, 1)[:, :, ::-1]  # Decode and convert to RGB
    output = None
    extracted_data = None  # Ensure extracted_data is initialized

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
        # Create a patient-specific folder based on patient name and timestamp
        patient_name = "Unknown_Patient"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patient_folder = os.path.join(data_folder, f"{patient_name}_{timestamp}")
        os.makedirs(patient_folder, exist_ok=True)

        # Save the scanned image temporarily
        scanned_image_path = os.path.join(patient_folder, "scanned_image.jpg")
        cv2.imwrite(scanned_image_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

        # Define a prompt for GPT-4 Vision
        with open("prompt.txt", "r", encoding="utf-8") as file:
            prompt = file.read()

        # Check if extracted data is already stored in session state
        if "extracted_data" not in st.session_state:
            extracted_data = extract_invoice_data_with_gpt4_vision(scanned_image_path, prompt)
            st.session_state.extracted_data = extracted_data
        else:
            extracted_data = st.session_state.extracted_data

        # Ensure extracted_data is valid and is not None
        if extracted_data:
            # Use customer_name from extracted_data if available, else fallback to "Unknown Patient"
            customer_name = extracted_data.get("customer_name", "Unknown Patient")

            # Check if customer_name is None, and use a fallback value if necessary
            if customer_name is None:
                customer_name = "Unknown Patient"

            customer_name = remove_accents(customer_name).replace(" ", "_")  # Remove accents and replace spaces

            # Create folder with the patient's name and timestamp
            patient_folder = os.path.join(data_folder, f"{customer_name}_{timestamp}")
            os.makedirs(patient_folder, exist_ok=True)

            # Save images and JSON in the created folder
            scanned_image_path = os.path.join(patient_folder, "scanned_image.jpg")
            cv2.imwrite(scanned_image_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

            json_file_path = os.path.join(patient_folder, "output.json")
            with open(json_file_path, "w", encoding="utf-8") as json_file:
                json.dump(extracted_data, json_file, ensure_ascii=False, indent=4)

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

procedure_selected = st.radio("Select Scanning Procedure:", ("Traditional", "Deep Learning", "Manual"), index=1,
                              horizontal=True)

if procedure_selected == "Deep Learning":
    model_selected = st.radio("Select Document Segmentation Backbone Model:", ("MobilenetV3-Large", "ResNet-50"),
                              horizontal=True)

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
