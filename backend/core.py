"""
    Backeend core class
"""

# pylint: disable=C0301,C0103,C0303,C0411,W1203

import os
import logging
import uuid
import pandas as pd
import json

from backend.llm_core import LLMCore
from backend import clustering

logger : logging.Logger = logging.getLogger()

class Core:
    """
        Backe-end core class
    """
    secrets : dict[str, any] = None
    
    DOWNLOAD_FOLDER = ".downloads"
    
    def __init__(self, secrets : dict[str, any]):
        """
            Constuctur
        """
        self.secrets = secrets
        os.makedirs(self.DOWNLOAD_FOLDER, exist_ok=True)

    def get_default_llm_core(self):
        """
            Return default llm core
        """
        return LLMCore(self.secrets)

    def build_index(self, document_names : list[str], document_contents : list[any], report_progress : callable) -> tuple[LLMCore, int]:
        """
            Run processing
        """

        report_progress(10, "Store documents...")
        document_temp_names = []
        for document_name, document_content in zip(document_names, document_contents):
            unique_suffix = str(uuid.uuid4().hex)
            unique_document_name = f'{self.DOWNLOAD_FOLDER}\\{document_name}_{unique_suffix}.docx'
            logger.info(f"Store document {document_name} to {unique_document_name}")
            with open(unique_document_name, 'wb') as f:
                f.write(document_content)
            document_temp_names.append(unique_document_name)
        report_progress(20, f"Stored {len(document_temp_names)} documents")
        
        total_tokens_used = 0
        try:
            report_progress(30, "Create LLM...")
            llm_core = LLMCore(self.secrets)
            
            report_progress(40, "Parse documents...")
            chunks = llm_core.parse_documents(document_temp_names)

            global_embeddings = []
            chunk_length = len(chunks)
            for index, chunk in enumerate(chunks):
                report_progress(50, f"Build embeddings {index+1}/{chunk_length}...")
                global_embeddings.append(llm_core.embedding_text(chunk))
            
            report_progress(60, "Cluster embeddings...")
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
                clustered_texts[cluster] = " --- ".join(cluster_texts)

            summaries = {}
            cluster_texts_length = len(clustered_texts.items())
            index = 0
            for cluster, text in clustered_texts.items():
                report_progress(70, f"Build summary for cluster {index+1}/{cluster_texts_length}...")
                summary, tokens_used = llm_core.build_summary(text)
                summaries[cluster] = summary
                total_tokens_used += tokens_used
                index += 1
           
            embedded_summaries = []
            index = 0
            summaries_length = len(summaries.values())
            for summary in summaries.values():
                report_progress(80, f"Embedding summary {index+1}/{summaries_length}...")
                embedded_summaries.append(llm_core.embedding_text(summary))
                index+= 1

            report_progress(85, "Clustering summary...")
            labels, _ = clustering.gmm_clustering_list(embedded_summaries, threshold=0.5)
            simple_labels = [label[0] if len(label) > 0 else -1 for label in labels]           
            clustered_summaries = {}
            for i, label in enumerate(simple_labels):
                if label not in clustered_summaries:
                    clustered_summaries[label] = []
                    clustered_summaries[label].append(list(summaries.values())[i])
           
            final_summaries = {}
            index = 0
            summaries_length = len(clustered_summaries.items())
            for cluster, texts in clustered_summaries.items():
                report_progress(90, f"Final summary {index+1}/{summaries_length}...")
                combined_text = ' '.join(texts)
                summary, tokens_used = llm_core.build_summary(combined_text)
                total_tokens_used += tokens_used
                final_summaries[cluster] = summary
                index += 1
           
            report_progress(95, "Build vector store...")
            texts_from_df = df['Text'].tolist()
            texts_from_clustered_texts = list(clustered_texts.values())
            texts_from_final_summaries = list(final_summaries.values())
            combined_texts = texts_from_df + texts_from_clustered_texts + texts_from_final_summaries
            llm_core.fill_vector_store(combined_texts)
            
            return llm_core, total_tokens_used
        finally:
            # clean up temp files
            for document_temp_name in document_temp_names:
                os.remove(document_temp_name)


    def query_document(self, llm_core : any, question_list : list[str], report_progress : callable) -> tuple[dict[str, any], int]:
        """
            Query document
        """

        report_progress(0.0, "Start query...")
        
        total_tokens_used = 0
        
        result_output   = []
        result_answered = {}
        index = 0
        for question_str in question_list:
            report_progress(float(index+1) / len(question_list), f"Build output {index+1}/{len(question_list)}...")
            
            question_parsed = question_str.split(":")
            if len(question_parsed)!= 2:
                result_output.append(["Error", question_str, "", -1])
                index+= 1
                continue
            
            question_code  = question_parsed[0].strip()
            question_query = question_parsed[1].strip()
            
            # we alredy have answer for this column
            if result_answered.get(question_code, False):
                index += 1
                continue

            answer_text = ""
            answer_score = 0
            answer_llm = ""
            try:
                answer_llm, tokens_used = llm_core.query(question_query)
                total_tokens_used += tokens_used
            except: # pylint: disable=W0718,W0702
                answer_text = "Error: LLM query failed"
                answer_score = -1
            
            if answer_score != -1:
                try:
                    answer_json = json.loads(answer_llm)
                    answer_text = answer_json['answer']
                    answer_score = answer_json['score']
                    result_answered[question_code] = True
                except: # pylint: disable=W0718,W0702
                    answer_text = "Error parsing answer"
                    answer_score = -1
            
            result_output.append([question_code, question_query, answer_text, answer_score])
            index += 1
        
        report_progress(0.0, "Done")
        
        return result_output, total_tokens_used


