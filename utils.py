import streamlit as st
import spacy
from spacy import displacy
from transformers import pipeline
import re

# --- MODEL LOADING (with caching) ---
# These functions are cached, so models are loaded from disk after the first download.

@st.cache_resource
def load_ner_pipeline(language):
    """Loads a named entity recognition pipeline based on the selected language."""
    if language == "English":
        model_name = "dbmdz/distilbert-base-uncased-finetuned-conll03-english"
    else:
        # CORRECTED MODEL: This is a valid and robust multilingual NER model.
        model_name = "Davlan/xlm-roberta-base-ner-hrl"
    return pipeline("ner", model=model_name, aggregation_strategy="simple")

@st.cache_resource
def load_summarization_pipeline(language):
    """Loads a text summarization pipeline based on the selected language."""
    if language == "English":
        model_name = "t5-small"
    else:
        model_name = "csebuetnlp/mT5_multilingual_XLSum"
    return pipeline("summarization", model=model_name)

@st.cache_resource
def load_qa_pipeline(language):
    """Loads a question answering pipeline based on the selected language."""
    if language == "English":
        model_name = "deepset/tinyroberta-squad2"
    else:
        model_name = "distilbert-base-multilingual-cased-sent-squad"
    return pipeline("question-answering", model=model_name)

@st.cache_resource
def load_translation_pipeline(source_lang_short, target_lang_short):
    """Loads a bilingual translation pipeline using efficient Helsinki-NLP models."""
    model_name = f"Helsinki-NLP/opus-mt-{source_lang_short}-{target_lang_short}"
    return pipeline("translation", model=model_name)


# --- HELPER FUNCTIONS ---

def get_language_code(language, for_helsinki=False):
    """Returns the language code required for multilingual models."""
    if for_helsinki:
        lang_map = {"English": "en", "Hindi": "hi", "Spanish": "es"}
    else:
        # Fallback for other model types, though not currently used in the main app
        lang_map = {"English": "en_XX", "Hindi": "hi_IN", "Spanish": "es_XX"}
    return lang_map.get(language)

def render_ner(text, ner_results):
    """Formats NER output for visualization with displaCy."""
    ents = []
    for entity in ner_results:
        ents.append({
            "start": entity['start'],
            "end": entity['end'],
            "label": entity['entity_group']
        })
    doc = {"text": text, "ents": ents, "title": "Named Entities"}
    return displacy.render(doc, style="ent", manual=True, jupyter=False, options={'colors': {'ORG': '#a6e22e', 'PER': '#f92672', 'LOC': '#66d9ef', 'MISC': '#fd971f'}})

# --- TEXT PROCESSING FOR LONG INPUTS ---

def process_long_text(text, task_pipeline, chunk_size=500):
    """
    Splits long text into chunks and applies a pipeline task to each chunk.
    This is essential for models with a token limit.
    """
    # Split text by sentences to maintain context
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding the next sentence exceeds the chunk size, process the current chunk
        if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Process each chunk with the pipeline
    results = []
    # Add a progress bar for long text processing
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        try:
            # The pipeline returns a list of dictionaries
            result = task_pipeline(chunk)
            results.append(result[0])
        except Exception as e:
            st.warning(f"Could not process chunk {i+1}. It might be too long or contain unsupported characters. Error: {e}")
        progress_bar.progress((i + 1) / len(chunks))
    
    # Hide the progress bar after completion
    progress_bar.empty()

    # Combine the results from each chunk
    if task_pipeline.task == "summarization":
        return " ".join([r['summary_text'] for r in results])
    elif task_pipeline.task == "translation":
        return " ".join([r['translation_text'] for r in results])
    
    return "Task not supported for long text processing or an error occurred."