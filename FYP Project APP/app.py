import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from fpdf import FPDF
import io
import tempfile
from datetime import datetime
import csv

st.set_page_config(page_title="Breast Cancer Diagnostic Assistant", layout="wide")

# ----------------------------
# Password protection
# ----------------------------
PASSWORD = "radiology123"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    password_input = st.sidebar.text_input("Enter password", type="password")
    if st.sidebar.button("Login"):
        if password_input == PASSWORD:
            st.session_state.logged_in = True
            st.sidebar.success("Login successful")
        else:
            st.sidebar.error("Incorrect password")
if not st.session_state.logged_in:
    st.title("🔒 Please login to access the Breast Cancer Diagnostic Assistant")
    st.stop()

st.title("Breast Cancer Diagnostic Assistant")
# ----------------------------
# Sidebar: Patient info + metadata
# ----------------------------
with st.sidebar:
    st.header("Patient Information")
    patient_id = st.text_input("Patient ID")
    patient_age = st.number_input("Age", min_value=0, max_value=120)
    patient_sex = st.selectbox("Sex", ["Female", "Male"])
    family_history = st.checkbox("Family history of breast cancer")
    prior_mammo = st.checkbox("Prior mammogram history")
    notes = st.text_area("Additional Notes (optional)")
    st.markdown("---")
    st.header("Upload Breast Scan Image(s)")
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True
    )
    run_classification = st.button("Run Classification")

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model_saved():
    model = tf.keras.models.load_model("model_saved", compile=False)
    return model

model = load_model_saved()
class_names = ["Benign Calcification", "Benign Mass", "Malignant Calcification", "Malignant Mass"]

# ----------------------------
# Preprocess image
# ----------------------------
def preprocess_image(image_file):
    img = Image.open(image_file).convert("L")
    img = img.resize((150,150))
    img_array = np.array(img).reshape((1,150,150,1))
    img_array = img_array.astype('float32') / 65535
    return img_array

# ----------------------------
# PDF generation with colors
# ----------------------------
def generate_pdf(patient_info, predictions, images):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # --------------------------
    # Header
    # --------------------------
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Breast Cancer Diagnostic Report", ln=True)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Patient ID: {patient_info['id']}", ln=True)
    pdf.cell(0, 10, f"Age: {patient_info['age']}", ln=True)
    pdf.cell(0, 10, f"Sex: {patient_info['sex']}", ln=True)
    pdf.cell(0, 10, f"Family history: {'Yes' if patient_info['family_history'] else 'No'}", ln=True)
    pdf.cell(0, 10, f"Prior mammogram: {'Yes' if patient_info['prior_mammo'] else 'No'}", ln=True)
    if patient_info['notes']:
        pdf.multi_cell(0, 10, f"Notes: {patient_info['notes']}")

    pdf.ln(5)
    
    # --------------------------
    # Table header
    # --------------------------
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(60, 10, "Image", 1)
    pdf.cell(60, 10, "Prediction", 1)
    pdf.cell(40, 10, "Confidence", 1)
    pdf.cell(30, 10, "Priority", 1)
    pdf.ln()

    # --------------------------
    # Table rows
    # --------------------------
    pdf.set_font("Arial", '', 12)
    for pred in predictions:
        pdf.cell(60, 10, pred['filename'], 1)
        pdf.cell(60, 10, pred['class'], 1)
        pdf.cell(40, 10, f"{pred['confidence']:.2%}", 1)
        pdf.cell(30, 10, pred['priority'], 1)
        pdf.ln()

    pdf.ln(10)
    
    # --------------------------
    # Images section
    # --------------------------
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Scan Images", ln=True)
    pdf.ln(5)

    for idx, img in enumerate(images):
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Image {idx+1}: {predictions[idx]['filename']}", ln=True)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            img.save(tmp_file.name)
            pdf.image(tmp_file.name, x=10, w=150)
        pdf.ln(10)

    # --------------------------
    # Output
    # --------------------------
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    buffer = io.BytesIO(pdf_bytes)
    buffer.seek(0)
    return buffer

# ----------------------------
# Audit log
# ----------------------------
def log_audit(user, patient_info, predictions):
    with open("audit_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        for pred in predictions:
            writer.writerow([
                datetime.now().isoformat(),
                user,
                patient_info['id'],
                pred['filename'],
                pred['class'],
                f"{pred['confidence']:.2%}"
            ])

# ----------------------------
# Main panel
# ----------------------------
if run_classification:
    if not patient_id:
        st.error("Please enter Patient ID")
    elif not uploaded_files:
        st.error("Please upload at least one image")
    else:
        st.subheader(f"Patient ID: {patient_id} | Age: {patient_age} | Sex: {patient_sex}")

        patient_info = {
            'id': patient_id,
            'age': patient_age,
            'sex': patient_sex,
            'family_history': 'Yes' if family_history else 'No',
            'prior_mammo': 'Yes' if prior_mammo else 'No',
            'notes': notes
        }

        predictions_list = []
        images_list = []

        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"---\n**Image {idx+1}:**")
            image = Image.open(uploaded_file)
            images_list.append(image)

            col1, col2, col3, col4 = st.columns([2,2,1,1])
            col1.image(image, use_container_width=True)
            input_tensor = preprocess_image(uploaded_file)
            preds = model.serve(input_tensor).numpy()[0]
            pred_idx = np.argmax(preds)
            confidence = preds[pred_idx]
            predicted_class = class_names[pred_idx]

            # Determine priority and color
            if "Malignant" in predicted_class:
                priority = "High"
                color = "#FF4B4B"
            elif confidence < 0.6:
                priority = "Medium"
                color = "#FFA500"
            else:
                priority = "Medium"
                color = "#4CAF50"

            col2.markdown(f"<h4 style='color:{color}'>{predicted_class}</h4>", unsafe_allow_html=True)
            col3.metric("Confidence", f"{confidence:.2%}")
            col4.metric("Priority", priority)

            if confidence < 0.6:
                st.warning("⚠️ Low confidence - please review results carefully.")

            predictions_list.append({
                'filename': uploaded_file.name,
                'class': predicted_class,
                'confidence': confidence,
                'priority': priority
            })

        # Audit log
        log_audit("radiologist_user", patient_info, predictions_list)

        # PDF download
        pdf_buffer = generate_pdf(patient_info, predictions_list, images_list)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer,
            file_name=f"{patient_id}_report.pdf",
            mime="application/pdf"
        )

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("**Disclaimer:** This tool assists radiologists and does NOT replace professional judgement. Use clinical context.")
