"""
    Backeend core class
"""

# pylint: disable=C0301,C0103,C0303,C0411,W1203

import os
import logging
import uuid

from backend.classes.dataoutput import DataOutput
from backend.llm_core import LLMCore

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
            logger.error(f"Store document {document_name} to {unique_document_name}")
            with open(unique_document_name, 'wb') as f:
                f.write(document_content)
            document_temp_names.append(unique_document_name)
        report_progress(20, f"Stored {len(document_temp_names)} documents")
        
        try:
            report_progress(30, "Create LLM...")
            llm_core = LLMCore(self.secrets)
            
            report_progress(40, "Parse documents...")
            chunks = llm_core.parse_documents(document_temp_names)
            report_progress(50, f"Extracted {len(chunks)} chunks")
            
            report_progress(90, "Build output...")
            result = DataOutput([['q1', 'a1'], ['q2', 'a2'] ], 100)
            return result
        finally:
            # clean up temp files
            for document_temp_name in document_temp_names:
                os.remove(document_temp_name)
