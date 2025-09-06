# import os
# import json
# import base64
# from pathlib import Path

# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.chat_models import ChatOllama
# from langchain.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# # --- Configuration ---
# VECTOR_DB_PATH = "vector_db"
# DATA_PATH = "data"
# PATIENT_FILE_PATH = os.path.join(DATA_PATH, "patient_files")
# PATIENTS_JSON_PATH = os.path.join(DATA_PATH, "patients.json")
# TEXT_MODEL_NAME = "llama3.1"  # Or "mistral", etc.
# EMBEDDING_MODEL_NAME = "nomic-embed-text" # Or "mxbai-embed-large"

# # Ensure directories exist
# os.makedirs(PATIENT_FILE_PATH, exist_ok=True)
# if not os.path.exists(PATIENTS_JSON_PATH):
#     with open(PATIENTS_JSON_PATH, 'w') as f:
#         json.dump({}, f)

# # --- Patient Management ---
# def get_patient_data():
#     """Loads patient data from the JSON file."""
#     with open(PATIENTS_JSON_PATH, 'r') as f:
#         return json.load(f)

# def save_patient_data(data):
#     """Saves patient data to the JSON file."""
#     with open(PATIENTS_JSON_PATH, 'w') as f:
#         json.dump(data, f, indent=4)

# def add_patient(patient_name, uploaded_files):
#     """Adds a new patient and their files."""
#     patient_data = get_patient_data()
#     if patient_name in patient_data:
#         return False, "Patient with this name already exists."

#     patient_record = {"files": [], "chat_history": []}
#     for uploaded_file in uploaded_files:
#         file_path = os.path.join(PATIENT_FILE_PATH, uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
#         patient_record["files"].append(uploaded_file.name)
    
#     patient_data[patient_name] = patient_record
#     save_patient_data(patient_data)
#     return True, f"Patient '{patient_name}' added successfully."

# def get_patient_file_content(patient_name):
#     """Reads and returns the text content of a patient's files."""
#     patient_data = get_patient_data()
#     patient_record = patient_data.get(patient_name, {})
    
#     full_text = ""
#     for filename in patient_record.get("files", []):
#         file_path = os.path.join(PATIENT_FILE_PATH, filename)
#         try:
#             if filename.lower().endswith(".pdf"):
#                 loader = PyPDFLoader(file_path)
#                 pages = loader.load_and_split()
#                 for page in pages:
#                     full_text += page.page_content + "\n"
#             elif filename.lower().endswith(".txt"):
#                 with open(file_path, 'r') as f:
#                     full_text += f.read() + "\n"
#         except Exception as e:
#             print(f"Error reading file {filename}: {e}")
#             full_text += f"\n[Could not read file: {filename}]\n"
            
#     return full_text.strip() if full_text else "No documents found for this patient."

# # --- RAG and LLM Logic ---
# def get_vectorstore():
#     """Initializes and returns the Chroma vector store."""
#     return Chroma(
#         persist_directory=VECTOR_DB_PATH,
#         embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
#     )

# def process_and_store_documents(uploaded_docs):
#     """Processes uploaded documents and adds them to the vector store."""
#     vectorstore = get_vectorstore()
#     for doc in uploaded_docs:
#         # Save temp file
#         temp_path = os.path.join(DATA_PATH, doc.name)
#         with open(temp_path, "wb") as f:
#             f.write(doc.getvalue())
        
#         # Load document
#         if doc.name.endswith('.pdf'):
#             loader = PyPDFLoader(temp_path)
#         else:
#             loader = TextLoader(temp_path)
        
#         documents = loader.load()
        
#         # Split documents
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         splits = text_splitter.split_documents(documents)
        
#         # Add to vector store
#         vectorstore.add_documents(documents=splits)
        
#         # Clean up temp file
#         os.remove(temp_path)
    
#     return f"Successfully added {len(uploaded_docs)} document(s) to the knowledge base."


# def get_rag_chain():
#     """Creates and returns the RAG chain."""
#     vectorstore = get_vectorstore()
#     retriever = vectorstore.as_retriever()
#     llm = ChatOllama(model=TEXT_MODEL_NAME)

#     template = """
#     You are a medical assistant. Use the following context to answer the question. 
#     The context consists of general medical knowledge from textbooks and specific patient records.
#     Combine both sources to provide a comprehensive and helpful answer to the doctor.
#     If you don't know the answer, just say that you don't know.

#     Medical Knowledge Context:
#     {rag_context}

#     Patient Medical Record:
#     {patient_context}

#     Chat History:
#     {chat_history}

#     Question: {question}
    
#     Answer:
#     """
#     prompt = PromptTemplate.from_template(template)

#     def format_chat_history(chat_history):
#         """Helper function to format chat history for the prompt."""
#         if not isinstance(chat_history, list):
#              return ""
#         return "\n".join(f"{msg['role']}: {msg['content']}" for msg in chat_history)

#     # This is the corrected chain structure
#     rag_chain = (
#         {
#             "rag_context": retriever,
#             "patient_context": RunnablePassthrough(),
#             "chat_history": RunnablePassthrough(),
#             "question": RunnablePassthrough(),
#         }
#         | RunnablePassthrough.assign(
#             chat_history=lambda x: format_chat_history(x["chat_history"])
#         )
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
    
#     return rag_chain

# # --- UI Utilities ---
# def display_pdf(file_path):
#     """Displays a PDF file in the Streamlit app."""
#     try:
#         with open(file_path, "rb") as f:
#             base64_pdf = base64.b64encode(f.read()).decode('utf-8')
#         pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
#         return pdf_display
#     except FileNotFoundError:
#         return "PDF file not found."
#     except Exception as e:
#         return f"An error occurred: {e}"

import os
import json
import base64
from pathlib import Path
import io
from PIL import Image

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Configuration ---
VECTOR_DB_PATH = "vector_db"
DATA_PATH = "data"
PATIENT_FILE_PATH = os.path.join(DATA_PATH, "patient_files")
PATIENTS_JSON_PATH = os.path.join(DATA_PATH, "patients.json")
TEXT_MODEL_NAME = "llama3.2:3b"
VISION_MODEL_NAME = "llava:7b"
EMBEDDING_MODEL_NAME = "nomic-embed-text"

# Ensure directories exist
os.makedirs(PATIENT_FILE_PATH, exist_ok=True)
if not os.path.exists(PATIENTS_JSON_PATH):
    with open(PATIENTS_JSON_PATH, 'w') as f:
        json.dump({}, f)

# --- Patient Management ---
def get_patient_data():
    """Loads patient data from the JSON file."""
    with open(PATIENTS_JSON_PATH, 'r') as f:
        return json.load(f)

def save_patient_data(data):
    """Saves patient data to the JSON file."""
    with open(PATIENTS_JSON_PATH, 'w') as f:
        json.dump(data, f, indent=4)

def add_patient(patient_name, uploaded_files):
    """Adds a new patient and their files."""
    patient_data = get_patient_data()
    if patient_name in patient_data:
        return False, "Patient with this name already exists."

    patient_record = {"files": [], "chat_history": []}
    for uploaded_file in uploaded_files:
        # We now accept images during onboarding as well
        file_path = os.path.join(PATIENT_FILE_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        patient_record["files"].append(uploaded_file.name)
    
    patient_data[patient_name] = patient_record
    save_patient_data(patient_data)
    return True, f"Patient '{patient_name}' added successfully."

def get_patient_file_content(patient_name):
    """Reads and returns the text content of a patient's files."""
    patient_data = get_patient_data()
    patient_record = patient_data.get(patient_name, {})
    
    full_text = ""
    for filename in patient_record.get("files", []):
        file_path = os.path.join(PATIENT_FILE_PATH, filename)
        try:
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                pages = loader.load_and_split()
                for page in pages:
                    full_text += page.page_content + "\n"
            elif filename.lower().endswith(".txt"):
                with open(file_path, 'r') as f:
                    full_text += f.read() + "\n"
        except Exception as e:
            print(f"Error reading text file {filename}: {e}")
            full_text += f"\n[Could not read text from file: {filename}]\n"
            
    return full_text.strip() if full_text else "No text documents found for this patient."

# --- RAG and LLM Logic ---
def get_vectorstore():
    """Initializes and returns the Chroma vector store."""
    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    )

def process_and_store_documents(uploaded_docs):
    """Processes uploaded documents and adds them to the vector store."""
    vectorstore = get_vectorstore()
    for doc in uploaded_docs:
        temp_path = os.path.join(DATA_PATH, doc.name)
        with open(temp_path, "wb") as f:
            f.write(doc.getvalue())
        
        loader = PyPDFLoader(temp_path) if doc.name.endswith('.pdf') else TextLoader(temp_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        vectorstore.add_documents(documents=splits)
        os.remove(temp_path)
    
    return f"Successfully added {len(uploaded_docs)} document(s) to the knowledge base."

def get_rag_chain():
    """Creates and returns the RAG chain for text-based queries."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever()
    llm = ChatOllama(model=TEXT_MODEL_NAME)

    template = """
    You are a medical assistant. Use the following context to answer the question. 
    The context consists of general medical knowledge from textbooks and specific patient records.
    Combine both sources to provide a comprehensive and helpful answer to the doctor.
    If you don't know the answer, just say that you don't know.

    Medical Knowledge Context: {rag_context}
    Patient Medical Record: {patient_context}
    Chat History: {chat_history}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    def format_chat_history(chat_history):
        if not isinstance(chat_history, list): return ""
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in chat_history)

    rag_chain = (
        {
            "rag_context": retriever,
            "patient_context": RunnablePassthrough(),
            "chat_history": RunnablePassthrough(),
            "question": RunnablePassthrough(),
        }
        | RunnablePassthrough.assign(
            chat_history=lambda x: format_chat_history(x["chat_history"])
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- NEW: Multimodal Logic ---
def get_multimodal_chain():
    """Creates a chain for multimodal (text + image) queries."""
    llm = ChatOllama(model=VISION_MODEL_NAME)
    
    # This is how you pass images to Llava with LangChain
    prompt_template = """
    You are an expert medical image analyst. A doctor has provided an image and a question.
    Analyze the image carefully based on the provided question and offer your insights.
    Focus on visible patterns, anomalies, or notable features.
    Preface your analysis with a clear disclaimer that you are an AI assistant and not a substitute for a radiologist.

    Question: {question}
    
    Analysis:
    """
    
    class ImageInput:
        def __init__(self, image_b64):
            self.image_b64 = image_b64
        def to_dict(self):
            return {"image": self.image_b64}

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # The chain structure for llava needs to be different to accommodate the image
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    return chain


def image_to_base64(image_file):
    """Converts a PIL image to a base64 string."""
    buffered = io.BytesIO()
    image_file.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# --- UI Utilities ---
def display_pdf(file_path):
    """Displays a PDF file in the Streamlit app."""
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        return f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
    except FileNotFoundError:
        return "PDF file not found."
    except Exception as e:
        return f"An error occurred: {e}"
