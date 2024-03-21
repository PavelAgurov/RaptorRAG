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
from data.question_list import DEFAULT_QUESTION_LIST

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
if 'llm_core' not in st.session_state:
    st.session_state.llm_core = None
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'question_list' not in st.session_state:
    st.session_state.question_list = DEFAULT_QUESTION_LIST

# ------------------------------- UI
st.set_page_config(page_title= "Demo POC", layout="wide")
streamlit_hack_remove_top_space()

st.markdown("# Demo POC")

logger : logging.Logger = logging.getLogger()

tabExtraction, tabTemplate, tabSettings = st.tabs(["Extraction", "Template", "Settings"])

with tabExtraction:
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

    btnBuildIndex = st.button("Build Index")
    progress_bar = st.progress(0, "")

    btnBuildAnswers = st.button("Build answers")
    
    if st.session_state.result_df is not None:
        st.dataframe(st.session_state.result_df, use_container_width=True, hide_index=True)

with tabSettings:
    question_list = st.text_area("Enter questions (one per line)", value = '\n'.join(st.session_state.question_list), height=400)

with st.sidebar:
    st.warning("""
               It's PoC:
               - Do not upload documents with personal or sensitive information
               - Data is not stored on the disk and removed after session (close browser or refresh page)
               """
    )
    token_count_container = st.container(border=True).empty()


#-------------------------------------- Functions

def report_progress(percent_complete : int, status_str : str):
    """Report status"""
    progress_bar.progress(percent_complete, text = status_str)

def show_used_tokens(currently_used = 0):
    """Show token counter"""
    st.session_state.tokens += currently_used
    token_count_container.markdown(f'Used {currently_used} tokens. Total used {st.session_state.tokens} tokens.')

#-------------------------------------- APP

show_used_tokens(0)

if question_list:
    st.session_state.question_list = question_list.split("\n")

# upload file
if submitted_uploaded_files:
    document_names   = []
    document_contents = []
    for uploaded_file in uploaded_files:
        document_names.append(uploaded_file.name)
        document_contents.append(uploaded_file)
    st.session_state.document_names    = document_names
    st.session_state.document_contents = [d.read() for d in document_contents]
    st.session_state.result_df = None
    st.session_state.llm_core  = None
    st.rerun()

if btnBuildIndex and not st.session_state.document_contents:
    st.session_state.display_error = "Please upload documents first"
    st.rerun()

if btnBuildIndex and st.session_state.document_contents:
    llm_core, tokens_used = st.session_state.backend_core.build_index(st.session_state.document_names, st.session_state.document_contents, report_progress)
    show_used_tokens(tokens_used)
    st.session_state.llm_core  = llm_core
    st.session_state.result_df = None
    report_progress(0, "Done")

if btnBuildAnswers:
    if not st.session_state.llm_core:
        st.session_state.llm_core = st.session_state.backend_core.get_default_llm_core()
    data_output, tokens_used = st.session_state.backend_core.query_document(st.session_state.llm_core, st.session_state.question_list, report_progress)
    show_used_tokens(tokens_used)
    if data_output:
        st.session_state.result_df = pd.DataFrame(data_output, columns=['Column', 'Question', 'Answer', "Score"])
    else:
        st.session_state.display_error = "No data found"
    st.rerun()