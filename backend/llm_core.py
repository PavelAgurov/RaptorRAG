"""
    LLM Core
"""
# pylint: disable=C0301,C0103,C0303,C0411,W1203,C0412

import logging
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter, MarkdownTextSplitter
from langchain_core.embeddings import Embeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from chromadb.config import Settings as ChromadbSettings

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
    __llm_query        : ChatOpenAI
    __llm_index        : ChatOpenAI
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
        self.__llm_index = self.create_llm(self.__index_max_tokens, index_model_name)
        logger.info(f"LLM index created: {self.__llm_index}")

        query_model_name = secrets.get('QUERY_BASE_MODEL_NAME')
        self.__query_max_tokens = secrets.get('QUERY_MAX_TOKENS')
        logger.info(f"LLM query model name: {query_model_name}")
        logger.info(f"LLM query max tokens: {self.__query_max_tokens}")
        self.__llm_query = self.create_llm(self.__query_max_tokens, query_model_name)
        logger.info(f"LLM query created: {self.__llm_query}")

        underlying_embeddings = self.create_openai_embeddings()
        cache_embeddings_storage = LocalFileStore(".cache-embeddings")
        self.__embedding_model = CacheBackedEmbeddings.from_bytes_store(
           underlying_embeddings, cache_embeddings_storage, namespace= underlying_embeddings.model
        )
        
        self.__retriever_size = secrets.get('RETRIEVER_SIZE', 10)
        self.__chunk_size     = secrets.get('CHUNK_SIZE', 200)
        self.__chunk_offset   = secrets.get('CHUNK_OFFSET', 0)

        # Init chains
        index_prompt = ChatPromptTemplate.from_template(prompts.SUMMARY_PROMPT_TEMPLATE)
        self.__chain_summary = index_prompt | self.__llm_index | StrOutputParser()
        
        self.__query_prompt = ChatPromptTemplate.from_template(prompts.QUERY_PROMPT_TEMPLATE)
        
        self.__vector_store = None

    def parse_documents(self, document_names : list[str]) -> list[str]:
        """
            Parse documents
        """
        
        md_doc_names = [f for f in document_names if f.lower().endswith('.md')]
        md_doc_chunks = self.__get_md_chunks(md_doc_names)
        
        another_doc_names = [f for f in document_names if f not in md_doc_names]
        another_chunks = self.__get_unstructured_chunks(another_doc_names)
        
        return md_doc_chunks + another_chunks

    def __get_md_chunks(self, doc_names : list[str]) -> list[str]:
        """
            Parse Markdown documents and return chunks
        """
        if len(doc_names) == 0:
            return []
        
        logger.error(doc_names)
        
        document_list = []
        for doc_name in doc_names:
            doc_loader = UnstructuredMarkdownLoader(doc_name)
            docs = doc_loader.load()
            document_list.extend(docs)
            
        markdown_splitter = MarkdownTextSplitter()
        return [d.page_content for d in markdown_splitter.split_documents(document_list)]
        
    def __get_unstructured_chunks(self, doc_names : list[str]) -> list[str]:
        """
            Parse unstructured documents and return chunks
        """
        if len(doc_names) == 0:
            return []
        
        doc_loader = UnstructuredFileLoader(doc_names)
        docs = doc_loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size      = self.__chunk_size,
            chunk_overlap   = self.__chunk_offset,
#            separators      = ["!", "?", "\n"],
            separators      = ["\n\n"],
            keep_separator  = True,
            length_function = len,
            is_separator_regex=False,
        )
        return [d.page_content for d in text_splitter.split_documents(docs)]

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
        chunks = text_splitter.split_text(text)
        
        total_tokens  = 0
        total_summary = []
        for chunk in chunks:
            with get_openai_callback() as cb:
                summary = self.__chain_summary.invoke({
                    "text" : chunk
                })
            total_tokens += cb.total_tokens
            total_summary.append(summary)
        
        return '\n'.join(total_summary), total_tokens
    
    def fill_vector_store(self, texts : list[str], collection_name : str) -> None:
        """
            Fill vector store
        """
        # clean collection before fillng
        try:
            tmp = Chroma(persist_directory=".chroma", embedding_function= self.__embedding_model, collection_name= collection_name)
            tmp.delete_collection()
        except:  # pylint: disable=W0702
            pass
        self.__vector_store = Chroma.from_texts(
            texts=texts, 
            persist_directory=".chroma", 
            embedding= self.__embedding_model, 
            collection_name= collection_name,
            client_settings = ChromadbSettings(anonymized_telemetry=False)
        )
        self.__vector_store.persist()


    def vectordb_get_collection_size(self, collection_name : str) -> str:
        """
            Return size of selected collection
        """
        if not self.__vector_store:
            self.__vector_store = Chroma(
                persist_directory=".chroma", 
                embedding_function= self.__embedding_model, 
                collection_name= collection_name,
                client_settings = ChromadbSettings(anonymized_telemetry=False)
            )
        return self.__vector_store._collection.count() # pylint: disable=W01212

    def query(self, query : str, collection_name : str) -> tuple[str, int]:
        """
            Query LLM
        """

        if not self.__vector_store:
            self.__vector_store = Chroma(
                persist_directory=".chroma", 
                embedding_function= self.__embedding_model, 
                collection_name= collection_name,
                client_settings = ChromadbSettings(anonymized_telemetry=False)
            )
            
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
