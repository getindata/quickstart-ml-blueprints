import base64
import os

import streamlit as st
import yaml
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from streamlit.runtime.uploaded_file_manager import UploadedFile


def _display_pdf(file: UploadedFile) -> None:
    """Display PDF file in Streamlit. Not always works on Chrome, works in other browsers.

    Args:
        file (UploadedFile): uploaded file
    """
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="1230" \
        height="650" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


# Create Kedro session to be able to call Kedro pipelines and extract Kedro context to get access to parameters
bootstrap_project(os.getcwd())
session = KedroSession.create()
context = session.load_context()

# Page settings
st.set_page_config(layout="wide")

# Sidebar widgets
with st.sidebar:
    st.title("Reading Assistant")

    uploaded_file = st.file_uploader("Choose your .pdf file", type="pdf")
    api = st.selectbox("Select an API:", ["OpenAI", "VertexAI PaLM", "Azure OpenAI"])

    if api == "OpenAI":
        model_choices = [
            "text-davinci-003",
            "text-davinci-002",
            "text-curie-001",
            "text-babbage-001",
            "text-ada-001",
        ]
    elif api == "VertexAI PaLM":
        model_choices = ["text-bison@001"]
    else:
        model_choices = context.params.get("azure_openai_deployments")

    model = st.selectbox("Choose LLM:", model_choices)
    mode = st.selectbox("Choose mode:", ["explain", "summarize"])
    input_text = st.text_area("Paste term or text:", value="", height=200)

    # Save Streamlit input to local parameters file
    with open("./conf/local/parameters.yml", "w") as f:
        yaml.dump(
            {"api": api, "model": model, "mode": mode, "input_text": input_text}, f
        )

    if st.button("Get Answer!"):
        # Run Kedro pipeline to use LLM
        answer = session.run("run_assistant_pipeline")["answer"]
    else:
        answer = "Paste input text and click [Get Answer!] button"

# Main pane widgets
if uploaded_file is not None:
    _display_pdf(uploaded_file)
    answer_prefix = "Explanation:\n\n" if mode == "explain" else "Summary:\n\n"
    st.write(answer_prefix + answer)
