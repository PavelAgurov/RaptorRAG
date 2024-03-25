"""
    Backeend core class
"""

# pylint: disable=C0301,C0103,C0303,C0411,W1203

import os
import logging
import uuid
import pandas as pd
import json
from enum import Enum

from backend.llm_core import LLMCore
from backend import clustering
from utils import file_utils

logger : logging.Logger = logging.getLogger()


class OperationMode(Enum):
    """Mode of query processing"""
    NONE    = 0
    COMBINE = 1
    def __str__(self):
        return str(self.name)
    
class Core:
    """
        Backe-end core class
    """
    secrets : dict[str, any] = None
    
    __DOWNLOAD_FOLDER = ".downloads"
    __DOCUMENT_INDEX_COLLECTION = "document_index"
    __CHUNK_STORAGE   = ".chunks"
    __CHUNK_CLUSTERS  = ".clusters"
    __SUMMARY_STORAGE = ".summary"
    __FINAL_SUMMARY_STORAGE = ".final_summary"
    __DISABLE_RAPTOR = False
    
    def __init__(self, secrets : dict[str, any]):
        """
            Constuctur
        """
        self.secrets = secrets
        self.__DISABLE_RAPTOR = secrets.get('DISABLE_RAPTOR', False)

    def get_default_llm_core(self):
        """
            Return default llm core
        """
        return LLMCore(self.secrets)
    
    def is_raptor_disabled(self) -> bool:
        """
            return disable raptor flag
        """
        return self.__DISABLE_RAPTOR

    def build_index(self, document_names : list[str], document_contents : list[any], report_progress : callable) -> tuple[LLMCore, int]:
        """
            Run processing
        """

        # clean up previous session - delete all files
        file_utils.recreate_dirs([
            self.__DOWNLOAD_FOLDER,
            self.__CHUNK_STORAGE, 
            self.__CHUNK_CLUSTERS,
            self.__SUMMARY_STORAGE,
            self.__FINAL_SUMMARY_STORAGE
        ])

        # store documents in tmp files
        report_progress(10, "Store documents...")
        full_document_names = []
        for document_name, document_content in zip(document_names, document_contents):
            full_document_name = os.path.join(self.__DOWNLOAD_FOLDER, document_name)
            logger.info(f"Store document {document_name} to {full_document_name}")
            with open(full_document_name, 'wb') as f:
                f.write(document_content)
            full_document_names.append(full_document_name)
        report_progress(11, f"Stored {len(full_document_names)} documents")
        
        report_progress(20, "Create LLM...")
        llm_core = LLMCore(self.secrets)
        
        # TODO: save links between chunks and doc names
        report_progress(30, "Parse documents...")
        chunks = llm_core.parse_documents(full_document_names)

        # save chunks
        for index, chunk in enumerate(chunks):
            chunk_path = os.path.join(self.__CHUNK_STORAGE, f"chunk_{index}.txt")
            file_utils.save_text_file_utf8(chunk_path, chunk)
            
        if self.__DISABLE_RAPTOR:
            llm_core.fill_vector_store(chunks, self.__DOCUMENT_INDEX_COLLECTION)
            return llm_core, 0

        # embeddings
        global_embeddings = []
        chunk_length = len(chunks)
        for index, chunk in enumerate(chunks):
            report_progress(40, f"Build embeddings {index+1}/{chunk_length}...")
            global_embeddings.append(llm_core.embedding_text(chunk))
            
        # clustering
        report_progress(50, "Cluster embeddings...")
        global_embeddings_reduced = clustering.reduce_cluster_embeddings(global_embeddings, dim = 2)
        labels, _ = clustering.gmm_clustering(global_embeddings_reduced, threshold=0.5)
        simple_labels = [label[0] if len(label) > 0 else -1 for label in labels]
        df = pd.DataFrame({
            'Text': chunks,
            'Embedding': list(global_embeddings_reduced),
            'Cluster': simple_labels
        })
        clustered_texts = {}
        for cluster in df['Cluster'].unique():
            cluster_texts = df[df['Cluster'] == cluster]['Text'].tolist()
            clustered_texts[cluster] = "\n\n---\n\n".join(cluster_texts)
            # Save clusters
            cluster_path = os.path.join(self.__CHUNK_CLUSTERS, f"cluster_{cluster}.txt")
            file_utils.save_text_file_utf8(cluster_path, clustered_texts[cluster])

        total_tokens_used = 0

        # build summary
        summaries = {}
        cluster_texts_length = len(clustered_texts.items())
        cluster_index = 0
        for cluster, text in clustered_texts.items():
            report_progress(60, f"Build summary for cluster {cluster_index+1}/{cluster_texts_length}...")
            summary, tokens_used = llm_core.build_summary(text)
            summaries[cluster] = summary
            total_tokens_used += tokens_used
            # Save summaries
            summary_path = os.path.join(self.__SUMMARY_STORAGE, f"summary_{cluster_index}.txt")
            file_utils.save_text_file_utf8(summary_path, summary)
            cluster_index += 1
        
        embedded_summaries = []
        summary_index = 0
        summaries_length = len(summaries.values())
        for summary in summaries.values():
            report_progress(70, f"Embedding summary {summary_index+1}/{summaries_length}...")
            embedded_summaries.append(llm_core.embedding_text(summary))
            summary_index+= 1

        report_progress(80, "Clustering summary...")
        labels, _ = clustering.gmm_clustering_list(embedded_summaries, threshold=0.5)
        simple_labels = [label[0] if len(label) > 0 else -1 for label in labels]           
        clustered_summaries = {}
        for i, label in enumerate(simple_labels):
            if label not in clustered_summaries:
                clustered_summaries[label] = []
                clustered_summaries[label].append(list(summaries.values())[i])
        
        final_summaries = {}
        final_summary_index = 0
        summaries_length = len(clustered_summaries.items())
        for cluster, texts in clustered_summaries.items():
            report_progress(90, f"Final summary {final_summary_index+1}/{summaries_length}...")
            combined_text = ' '.join(texts)
            final_summary, tokens_used = llm_core.build_summary(combined_text)
            total_tokens_used += tokens_used
            final_summaries[cluster] = final_summary
            # Save final summaries
            final_summary_path = os.path.join(self.__FINAL_SUMMARY_STORAGE, f"final_summary_{final_summary_index}.txt")
            file_utils.save_text_file_utf8(final_summary_path, final_summary)
            final_summary_index += 1
        
        report_progress(95, "Build vector store...")
        texts_from_df = df['Text'].tolist()
        texts_from_clustered_texts = list(clustered_texts.values())
        texts_from_final_summaries = list(final_summaries.values())
        combined_texts = texts_from_df + texts_from_clustered_texts + texts_from_final_summaries
        
        #fill vector store with all texts
        llm_core.fill_vector_store(combined_texts, self.__DOCUMENT_INDEX_COLLECTION)

        # RAPTOR result        
        return llm_core, total_tokens_used


    def query_document(self, llm_core : LLMCore, query_list : list[str], report_progress : callable) -> tuple[list, int]:
        """
            Query document
        """

        report_progress(0.0, "Start query...")
        
        total_tokens_used = 0
        
        result_output   = []
        result_answered = {}
        index = 0
        for query_str in query_list:
            report_progress(float(index+1) / len(query_list), f"Build output {index+1}/{len(query_list)}...")
            
            query_str = query_str.strip()
            if len(query_str) == 0: # skip empty strings
                index+= 1
                continue

            if query_str.startswith('#'): # skip comments
                index+= 1
                continue
            
            query_parsed = query_str.split(":")
            if len(query_parsed)!= 2:
                result_output.append(["Error", query_str, "", -1, None])
                index+= 1
                continue
            
            query_code = query_parsed[0].strip()
            query_text = query_parsed[1].strip()
            
            operation_mode = OperationMode.NONE
            if query_code.endswith("+"):
                operation_mode = OperationMode.COMBINE
                query_code = query_code[:-1]
            
            # we alredy have answer for this column
            if result_answered.get(query_code, False) and operation_mode != OperationMode.COMBINE:
                index += 1
                continue

            answer_text = ""
            answer_score = 0
            answer_llm = ""
            try:
                answer_llm, tokens_used = llm_core.query(query_text, self.__DOCUMENT_INDEX_COLLECTION)
                total_tokens_used += tokens_used
            except: # pylint: disable=W0718,W0702
                answer_text = "Error: LLM query failed"
                answer_score = -1
            
            if answer_score != -1:
                try:
                    answer_json = json.loads(answer_llm)
                    answer_text = answer_json['answer']
                    answer_score = answer_json['score']
                    result_answered[query_code] = True
                except: # pylint: disable=W0718,W0702
                    answer_text = "Error parsing answer"
                    answer_score = -1
            
            result_output.append([query_code, query_text, answer_text, answer_score, operation_mode])
            index += 1
        
        report_progress(0.0, "Done")
        
        return result_output, total_tokens_used

    def single_query(self, llm_core: LLMCore, query: str) -> tuple[list, int]:
        """
            Query for a single query
        """

        answer_text = ""
        answer_score = -1

        try:
            answer_llm, tokens_used = llm_core.query(query, self.__DOCUMENT_INDEX_COLLECTION)
        except: # pylint: disable=W0718,W0702
            answer_text = "Error: LLM query failed"
            answer_score = -1

        
        
        return [answer_llm, answer_score], tokens_used
