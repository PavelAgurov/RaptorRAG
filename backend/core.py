"""
    Backeend core class
"""

# pylint: disable=C0301,C0103,C0303,C0411,W1203

import os
import logging
import uuid
import pandas as pd

from backend.classes.dataoutput import DataOutput
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

    def run(self, document_names : list[str], document_contents : list[any], report_progress : callable) -> DataOutput:
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
           
            report_progress(90, "Build output...")
            result = DataOutput([['q1', 'a1'], ['q2', 'a2'] ], total_tokens_used)
            return result
        finally:
            # clean up temp files
            for document_temp_name in document_temp_names:
                os.remove(document_temp_name)


