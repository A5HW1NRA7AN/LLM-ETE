import streamlit as st
import spacy
from spacy import displacy
from transformers import pipeline
import torch # Although not called directly, it's a required backend for transformers.

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Multilingual NLP Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODEL LOADING (with caching) ---
# Using st.cache_resource to cache the models and prevent reloading on every interaction.
@st.cache_resource
def load_ner_pipeline(language):
    """Loads a named entity recognition pipeline based on the selected language."""
    if language == "English":
        model_name = "dslim/bert-base-NER"
    else: # For Hindi and Spanish, we use a multilingual model
        model_name = "xlm-roberta-large-finetuned-conll03-english"
    return pipeline("ner", model=model_name, aggregation_strategy="simple")

@st.cache_resource
def load_summarization_pipeline(language):
    """Loads a text summarization pipeline based on the selected language."""
    if language == "English":
        model_name = "sshleifer/distilbart-cnn-12-6"
    else: # For Hindi and Spanish
        model_name = "facebook/mbart-large-50"
    return pipeline("summarization", model=model_name)

@st.cache_resource
def load_qa_pipeline(language):
    """Loads a question answering pipeline based on the selected language."""
    if language == "English":
        model_name = "distilbert-base-cased-distilled-squad"
    else: # For Hindi and Spanish
        model_name = "deepset/xlm-roberta-base-squad2"
    return pipeline("question-answering", model=model_name)


# --- HELPER FUNCTIONS ---
def get_language_code(language):
    """Returns the language code required for the multilingual summarization model."""
    lang_map = {
        "Hindi": "hi_IN",
        "Spanish": "es_XX"
    }
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


# --- UI LAYOUT ---
st.title("Multilingual NLP Suite 🤖")
st.markdown("Perform Named Entity Recognition, Summarization, and Question Answering on your text. Supports English, Hindi, and Spanish.")

# --- SIDEBAR FOR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Controls")

    input_method = st.radio(
        "Choose your input method:",
        ("Paste Text", "Upload a .txt file"),
        key="input_method"
    )

    language = st.selectbox(
        "Select Language:",
        ("English", "Hindi", "Spanish"),
        key="language"
    )

    task = st.radio(
        "Select NLP Task:",
        ("Named Entity Recognition", "Summarization", "Question Answering"),
        key="task"
    )

    # Context-specific input for Q&A
    question = ""
    if task == "Question Answering":
        question = st.text_input("Ask a question based on the text:", key="question")


# --- MAIN PANEL FOR INPUT AND OUTPUT ---
st.header("Input Text")
context_text = ""
if input_method == "Paste Text":
    context_text = st.text_area("Paste your paragraph here:", height=250, key="context_text_area")
else:
    uploaded_file = st.file_uploader("Upload your .txt file", type=['txt'])
    if uploaded_file is not None:
        context_text = uploaded_file.read().decode('utf-8')
        st.text_area("File Content:", value=context_text, height=250, disabled=True)


# --- PROCESSING AND DISPLAYING RESULTS ---
if st.button("Process", key="process_button"):
    # Input validation
    if not context_text.strip():
        st.warning("Please provide some text to process.")
    elif task == "Question Answering" and not question.strip():
        st.warning("Please ask a question for the Q&A task.")
    else:
        with st.spinner(f"Running {task}... This may take a moment."):
            try:
                st.header("Results")

                # --- Named Entity Recognition ---
                if task == "Named Entity Recognition":
                    ner_pipeline = load_ner_pipeline(language)
                    ner_results = ner_pipeline(context_text)

                    if not ner_results:
                        st.info("No entities were found in the text.")
                    else:
                        st.subheader("Entity Visualization")
                        html = render_ner(context_text, ner_results)
                        st.write(html, unsafe_allow_html=True)

                        st.subheader("Evaluation: Entities and Confidence Scores")
                        st.dataframe(
                            [
                                {"Entity": r["word"], "Type": r["entity_group"], "Confidence": f"{r['score']:.4f}"}
                                for r in ner_results
                            ]
                        )

                # --- Summarization ---
                elif task == "Summarization":
                    summarizer = load_summarization_pipeline(language)

                    # Multilingual model requires language codes
                    if language in ["Hindi", "Spanish"]:
                        lang_code = get_language_code(language)
                        summary_result = summarizer(context_text, src_lang=lang_code, tgt_lang=lang_code, max_length=150, min_length=30)
                    else:
                        summary_result = summarizer(context_text, max_length=150, min_length=30, do_sample=False)

                    summary_text = summary_result[0]['summary_text']

                    st.subheader("Generated Summary")
                    st.success(summary_text)

                    st.subheader("Evaluation: Summary Metrics")
                    original_words = len(context_text.split())
                    summary_words = len(summary_text.split())
                    compression = 100 * (1 - (summary_words / original_words))

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Original Words", original_words)
                    col2.metric("Summary Words", summary_words)
                    col3.metric("Compression Ratio", f"{compression:.2f}%")


                # --- Question Answering ---
                elif task == "Question Answering":
                    qa_pipeline = load_qa_pipeline(language)
                    answer_result = qa_pipeline(question=question, context=context_text)

                    st.subheader("Answer")
                    st.info(f"**Question:** {question}")
                    st.success(f"**Answer:** {answer_result['answer']}")

                    st.subheader("Evaluation: Confidence Score")
                    st.metric("Confidence", f"{answer_result['score']*100:.2f}%")

                    # Highlight the answer in the context
                    st.subheader("Answer in Context")
                    start_idx = answer_result['start']
                    end_idx = answer_result['end']
                    highlighted_text = (
                        context_text[:start_idx] +
                        f"<mark style='background-color:#a6e22e; padding: 2px 5px; border-radius: 3px;'>{context_text[start_idx:end_idx]}</mark>" +
                        context_text[end_idx:]
                    )
                    st.markdown(highlighted_text, unsafe_allow_html=True)


            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("This could be due to model compatibility or resource limitations. Please try with a shorter text or a different model if the problem persists.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Hugging Face Transformers.")
