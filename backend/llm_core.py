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
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import TokenTextSplitter

from backend.llm_base_core import LLMBaseCore
from backend import prompts

logger : logging.Logger = logging.getLogger()

class LLMCore(LLMBaseCore):
    """
        LLM core class
    """
    
    __index_max_tokens : int
    __query_max_tokens : int
    __embedding_model  : Embeddings
    __chain_summary    : any
    __vector_store     : Chroma
    __query_prompt     : ChatPromptTemplate
    __llm_query        : any
    __retriever_size   : int
    __chunk_size       : int
    __chunk_offset     : int

    def __init__(self, secrets : dict[str, Any]):

        LLMBaseCore.__init__(self, secrets)
        
        # Init LLM
        index_model_name = secrets.get('INDEX_BASE_MODEL_NAME')
        self.__index_max_tokens = secrets.get('INDEX_MAX_TOKENS')
        logger.info(f"LLM index model name: {index_model_name}")
        logger.info(f"LLM index max tokens: {self.__index_max_tokens}")
        llm_index = self.create_llm(self.__index_max_tokens, index_model_name)

        query_model_name = secrets.get('QUERY_BASE_MODEL_NAME')
        self.__query_max_tokens = secrets.get('QUERY_MAX_TOKENS')
        logger.info(f"LLM query model name: {query_model_name}")
        logger.info(f"LLM query max tokens: {self.__query_max_tokens}")
        self.__llm_query = self.create_llm(self.__query_max_tokens, query_model_name)

        underlying_embeddings = OpenAIEmbeddings()
        cache_embeddings_storage = LocalFileStore(".cache-embeddings")
        self.__embedding_model = CacheBackedEmbeddings.from_bytes_store(
           underlying_embeddings, cache_embeddings_storage, namespace= underlying_embeddings.model
        )
        
        self.__retriever_size = secrets.get('RETRIEVER_SIZE', 10)
        self.__chunk_size     = secrets.get('CHUNK_SIZE', 200)
        self.__chunk_offset   = secrets.get('CHUNK_OFFSET', 0)

        # Init chains
        index_prompt = ChatPromptTemplate.from_template(prompts.SUMMARY_PROMPT_TEMPLATE)
        self.__chain_summary = index_prompt | llm_index | StrOutputParser()
        
        self.__query_prompt = ChatPromptTemplate.from_template(prompts.QUERY_PROMPT_TEMPLATE)
        
        self.__vector_store = None

    def parse_documents(self, document_names : list[str]) -> list[str]:
        """
            Parse documents
        """
        
        loader = UnstructuredFileLoader(document_names)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size      = self.__chunk_size,
            chunk_overlap   = self.__chunk_offset,
            length_function = len,
            is_separator_regex=False,
        )

        chunks = [d.page_content for d in text_splitter.split_documents(docs)]
        return chunks

    def embedding_text(self, text : str) -> list[float]:
        """
            Embed document
        """
        return self.__embedding_model.embed_documents([text])[0]

    def build_summary(self, text : str) -> str :
        """
            Run LLM for summarization
        """
        logger.debug("LLM build_summary")
        
        text_splitter = TokenTextSplitter(chunk_size= self.__index_max_tokens, chunk_overlap=0)
        texts = text_splitter.split_text(text)
        
        total_tokens  = 0
        total_summary = []
        for text in texts:
            with get_openai_callback() as cb:
                summary = self.__chain_summary.invoke({
                    "text" : text
                })
            total_tokens += cb.total_tokens
            total_summary.append(summary)
        
        return '\n'.join(total_summary), total_tokens
    
    def fill_vector_store(self, texts : list[str], collection_name : str) -> None:
        """
            Fill vector store
        """
        self.__vector_store = Chroma.from_texts(texts=texts, persist_directory=".chroma", embedding= self.__embedding_model, collection_name= collection_name)
        self.__vector_store.persist()

    def query(self, query : str, collection_name : str) -> tuple[str, int]:
        """
            Query LLM
        """

        if not self.__vector_store:
            self.__vector_store = Chroma(persist_directory=".chroma", embedding_function= self.__embedding_model, collection_name= collection_name)
            
        retriever = self.__vector_store.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs={
                "k": self.__retriever_size,
                "score_threshold" : 0.5
            })
        
        def format_docs_call(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        chain_query = (
            {"context": retriever | format_docs_call, "question": RunnablePassthrough()}
            | self.__query_prompt
            | self.__llm_query
            | StrOutputParser()
        )
        
        with get_openai_callback() as cb:
            json_result = chain_query.invoke(query)
        tokens_used = cb.total_tokens
        
        return json_result, tokens_used
