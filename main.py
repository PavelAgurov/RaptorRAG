"""
    Main APP and UI
"""
# pylint: disable=C0301,C0103,C0303,C0411

import pandas as pd
import streamlit as st
import logging

from utils_streamlit import streamlit_hack_remove_top_space
from utils.app_logger import init_streamlit_logger

from backend.core import Core

init_streamlit_logger()

# ------------------------------- Session
if 'document_contents' not in st.session_state:
    st.session_state.document_contents = []
if 'document_names' not in st.session_state:
    st.session_state.document_names = []
if 'tokens' not in st.session_state:
    st.session_state.tokens = 0
if 'display_error' not in st.session_state:
    st.session_state.display_error = None
if 'backend_core' not in st.session_state:
    all_secrets = {s[0]:s[1] for s in st.secrets.items()}
    st.session_state.backend_core = Core(all_secrets)

# ------------------------------- UI
st.set_page_config(page_title= "Demo POC", layout="wide")
streamlit_hack_remove_top_space()

st.markdown("# Demo POC")

logger : logging.Logger = logging.getLogger()

tabExtraction, tabTemplate, tabSettings = st.tabs(["Extraction", "Template", "Settings"])

with tabExtraction:
    st.warning("It's PoC, so do not upload documents with personal or sensitive information!")
    with st.form("my-form", clear_on_submit=True, border=True):
        uploaded_files = st.file_uploader(
            "Drag your documents here (docx)",
            type=["docx"],
            accept_multiple_files= True
        )
        submitted_uploaded_files = st.form_submit_button("Upload selected documents")

    if st.session_state.document_contents:
        st.info(f"{len(st.session_state.document_contents)} documents uploaded")
    if st.session_state.display_error:
        st.error(st.session_state.display_error)
    st.session_state.display_error = None

    btnProcessDocuments = st.button("Process Documents")

with st.sidebar:
    token_count_container = st.empty()

progress_bar = st.progress(0, "")

#-------------------------------------- Functions

def report_progress(percent_complete : int, status_str : str):
    """Report status"""
    progress_bar.progress(percent_complete, text = status_str)

def show_used_tokens(currently_used = 0):
    """Show token counter"""
    token_count_container.markdown(f'Used {currently_used} tokens. Total used {st.session_state.tokens} tokens.')

#-------------------------------------- APP

show_used_tokens(0)

# upload file
if submitted_uploaded_files:
    document_names   = []
    document_contents = []
    for uploaded_file in uploaded_files:
        document_names.append(uploaded_file.name)
        document_contents.append(uploaded_file)
    st.session_state.document_names    = document_names
    st.session_state.document_contents = [d.read() for d in document_contents]
    st.rerun()

if btnProcessDocuments and not st.session_state.document_contents:
    st.session_state.display_error = "Please upload documents first"
    st.rerun()

if btnProcessDocuments and st.session_state.document_contents:
    data_output = st.session_state.backend_core.run(st.session_state.document_names, st.session_state.document_contents, report_progress)
    show_used_tokens(data_output.tokens)
    report_progress(0, "Done")

    if data_output.output:
        df = pd.DataFrame(data_output.output, columns=['Question', 'Answer'])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("No data found")




