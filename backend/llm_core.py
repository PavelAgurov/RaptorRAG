"""
    LLM Core
"""
# pylint: disable=C0301,C0103,C0303,C0411,W1203,C0412

import logging
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.llm_base_core import LLMBaseCore

logger : logging.Logger = logging.getLogger()

class LLMCore(LLMBaseCore):
    """
        LLM core class
    """
    
    chain : any

    def __init__(self, secrets : dict[str, Any]):

        LLMBaseCore.__init__(self, secrets)

        # Init LLM
        index_model_name = secrets.get('INDEX_BASE_MODEL_NAME')
        index_max_tokens = secrets.get('INDEX_MAX_TOKENS')
        logger.info(f"LLM index model name: {index_model_name}")
        logger.info(f"LLM index max tokens: {index_max_tokens}")
        llm_index = self.create_llm(index_max_tokens, index_model_name)

        query_model_name = secrets.get('QUERY_BASE_MODEL_NAME')
        query_max_tokens = secrets.get('QUERY_MAX_TOKENS')
        logger.info(f"LLM query model name: {query_model_name}")
        logger.info(f"LLM query max tokens: {query_max_tokens}")
        llm_query = self.create_llm(query_max_tokens, query_model_name)

        # Init chains
        base_prompt = ChatPromptTemplate.from_template("You are helpful assistant to answer question '{question}'.")
        self.chain_index  = base_prompt | llm_index | StrOutputParser()

    def parse_documents(self, document_names : list[str]) -> list[str]:
        """
            Parse documents
        """
        
        loader = UnstructuredFileLoader(document_names)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )

        chunks = [d.page_content for d in text_splitter.split_documents(docs)]
        return chunks

    def test_chat(self, query : str) -> str :
        """
            Run LLM
        """
        logger.debug(f"LLM input: {query}")
        with get_openai_callback() as cb:
            answer = self.chain.invoke({
                "question" : query
            })
        tokens_used = cb.total_tokens
        
        return answer, tokens_used
    


