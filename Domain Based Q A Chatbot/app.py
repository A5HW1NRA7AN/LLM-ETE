# import streamlit as st
# import os
# from utils import (
#     add_patient,
#     get_patient_data,
#     save_patient_data,
#     get_patient_file_content,
#     process_and_store_documents,
#     get_rag_chain,
#     display_pdf
# )

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Medical Copilot",
#     page_icon="🩺",
#     layout="wide"
# )

# # --- Constants ---
# PATIENT_FILE_PATH = os.path.join("data", "patient_files")

# # --- Sidebar Navigation ---
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["Patient Diagnostic Chat", "Onboard New Patient", "Upload to Knowledge Base"])

# # --- Main Page Content ---

# if page == "Upload to Knowledge Base":
#     st.header("📚 Upload to Knowledge Base")
#     st.write("Upload medical textbooks or guidelines (PDF or TXT) to build the RAG system's knowledge.")
#     uploaded_files = st.file_uploader(
#         "Choose documents...", 
#         accept_multiple_files=True,
#         type=['pdf', 'txt']
#     )
#     if st.button("Process and Add to Knowledge Base"):
#         if uploaded_files:
#             with st.spinner("Processing documents..."):
#                 message = process_and_store_documents(uploaded_files)
#                 st.success(message)
#         else:
#             st.warning("Please upload at least one document.")

# elif page == "Onboard New Patient":
#     st.header("👤 Onboard New Patient")
#     st.write("Enter patient details and upload their medical records.")
#     patient_name = st.text_input("Patient Full Name")
#     uploaded_files = st.file_uploader(
#         "Upload Patient Medical Records (PDF, TXT)...",
#         accept_multiple_files=True,
#         type=['pdf', 'txt']
#     )
#     if st.button("Add Patient"):
#         if patient_name and uploaded_files:
#             success, message = add_patient(patient_name, uploaded_files)
#             if success:
#                 st.success(message)
#             else:
#                 st.error(message)
#         else:
#             st.warning("Please provide a patient name and at least one file.")

# elif page == "Patient Diagnostic Chat":
#     st.header("💬 Patient Diagnostic Chat")
#     patient_data = get_patient_data()
#     patient_names = list(patient_data.keys())

#     if not patient_names:
#         st.warning("No patients found. Please onboard a patient first.")
#     else:
#         selected_patient = st.selectbox("Select a Patient", options=patient_names)

#         if selected_patient:
#             st.subheader(f"Records for {selected_patient}")
            
#             # Initialize session state for the selected patient
#             if "messages" not in st.session_state or st.session_state.get("current_patient") != selected_patient:
#                 st.session_state.messages = patient_data[selected_patient].get("chat_history", [])
#                 st.session_state.current_patient = selected_patient

#             col1, col2 = st.columns([1, 1])

#             with col1:
#                 st.subheader("📄 Document Viewer")
#                 patient_files = patient_data[selected_patient].get("files", [])
#                 pdf_files = [f for f in patient_files if f.lower().endswith('.pdf')]

#                 if pdf_files:
#                     selected_pdf = st.selectbox("Select a PDF to view", options=pdf_files)
#                     pdf_path = os.path.join(PATIENT_FILE_PATH, selected_pdf)
#                     st.markdown(display_pdf(pdf_path), unsafe_allow_html=True)
#                 else:
#                     st.info("No PDF files found for this patient.")

#             with col2:
#                 st.subheader("🤖 Chat with Copilot")
                
#                 # Display chat history
#                 for message in st.session_state.messages:
#                     with st.chat_message(message["role"]):
#                         st.markdown(message["content"])
                
#                 # Chat input
#                 if prompt := st.chat_input("Ask about the patient's condition..."):
#                     # Add user message to chat history
#                     st.session_state.messages.append({"role": "user", "content": prompt})
#                     with st.chat_message("user"):
#                         st.markdown(prompt)

#                     with st.chat_message("assistant"):
#                         with st.spinner("Thinking..."):
#                             # Get patient context
#                             patient_context = get_patient_file_content(selected_patient)
                            
#                             # Get RAG chain
#                             rag_chain = get_rag_chain()
                            
#                             # Prepare inputs for the chain
#                             chain_input = {
#                                 "question": prompt,
#                                 "patient_context": patient_context,
#                                 "chat_history": st.session_state.messages
#                             }
                            
#                             # Invoke chain and display response
#                             response = rag_chain.invoke(chain_input)
#                             st.markdown(response)
                    
#                     # Add assistant response to chat history
#                     st.session_state.messages.append({"role": "assistant", "content": response})
                    
#                     # Save updated chat history
#                     patient_data[selected_patient]["chat_history"] = st.session_state.messages
#                     save_patient_data(patient_data)

import streamlit as st
import os
from PIL import Image
from langchain_community.chat_models import ChatOllama
from utils import (
    add_patient,
    get_patient_data,
    save_patient_data,
    get_patient_file_content,
    process_and_store_documents,
    get_rag_chain,
    get_multimodal_chain,
    image_to_base64,
    display_pdf
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Medical Copilot",
    page_icon="🩺",
    layout="wide"
)

# --- Constants ---
PATIENT_FILE_PATH = os.path.join("data", "patient_files")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Patient Diagnostic Chat", "Onboard New Patient", "Upload to Knowledge Base"])

# --- Main Page Content ---

if page == "Upload to Knowledge Base":
    st.header("📚 Upload to Knowledge Base")
    st.write("Upload medical textbooks or guidelines (PDF or TXT) to build the RAG system's knowledge.")
    uploaded_files = st.file_uploader(
        "Choose documents...", 
        accept_multiple_files=True,
        type=['pdf', 'txt']
    )
    if st.button("Process and Add to Knowledge Base"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                message = process_and_store_documents(uploaded_files)
                st.success(message)
        else:
            st.warning("Please upload at least one document.")

elif page == "Onboard New Patient":
    st.header("👤 Onboard New Patient")
    st.write("Enter patient details and upload their medical records.")
    patient_name = st.text_input("Patient Full Name")
    uploaded_files = st.file_uploader(
        "Upload Patient Medical Records (PDF, TXT, PNG, JPG)...",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'png', 'jpg', 'jpeg']
    )
    if st.button("Add Patient"):
        if patient_name and uploaded_files:
            success, message = add_patient(patient_name, uploaded_files)
            if success:
                st.success(message)
            else:
                st.error(message)
        else:
            st.warning("Please provide a patient name and at least one file.")

elif page == "Patient Diagnostic Chat":
    st.header("💬 Patient Diagnostic Chat")
    patient_data = get_patient_data()
    patient_names = list(patient_data.keys())

    if not patient_names:
        st.warning("No patients found. Please onboard a patient first.")
    else:
        selected_patient = st.selectbox("Select a Patient", options=patient_names)

        if selected_patient:
            st.subheader(f"Records for {selected_patient}")
            
            if "messages" not in st.session_state or st.session_state.get("current_patient") != selected_patient:
                st.session_state.messages = patient_data[selected_patient].get("chat_history", [])
                st.session_state.current_patient = selected_patient

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📄 Document Viewer")
                patient_files = patient_data[selected_patient].get("files", [])
                pdf_files = [f for f in patient_files if f.lower().endswith('.pdf')]

                if pdf_files:
                    selected_pdf = st.selectbox("Select a PDF to view", options=pdf_files)
                    pdf_path = os.path.join(PATIENT_FILE_PATH, selected_pdf)
                    st.markdown(display_pdf(pdf_path), unsafe_allow_html=True)
                else:
                    st.info("No PDF files found for this patient.")

            with col2:
                st.subheader("🤖 Chat with Copilot")
                
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # NEW: Image uploader in the chat column
                uploaded_image = st.file_uploader("Upload an image for analysis", type=['png', 'jpg', 'jpeg'], key=f"uploader_{selected_patient}")

                if prompt := st.chat_input("Ask about the patient's condition..."):
                    # Add user message to UI
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                        if uploaded_image:
                            st.image(uploaded_image, width=200)

                    # Assistant response logic
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = ""
                            # If an image is uploaded, use the multimodal chain
                            if uploaded_image:
                                pil_image = Image.open(uploaded_image)
                                image_b64 = image_to_base64(pil_image)
                                
                                # The multimodal model expects a different input structure
                                from langchain_core.messages import HumanMessage
                                llm = ChatOllama(model="llava:7b")
                                message = HumanMessage(
                                    content=[
                                        {"type": "text", "text": prompt},
                                        {"type": "image_url", "image_url": f"data:image/png;base64,{image_b64}"},
                                    ]
                                )
                                response = llm.invoke([message]).content

                            # Otherwise, use the text-based RAG chain
                            else:
                                patient_context = get_patient_file_content(selected_patient)
                                rag_chain = get_rag_chain()
                                chain_input = {
                                    "question": prompt,
                                    "patient_context": patient_context,
                                    "chat_history": st.session_state.messages
                                }
                                response = rag_chain.invoke(chain_input)
                            
                            st.markdown(response)
                    
                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Save updated chat history
                    patient_data[selected_patient]["chat_history"] = st.session_state.messages
                    save_patient_data(patient_data)

                    # Clear the uploader after processing
                    st.rerun()
