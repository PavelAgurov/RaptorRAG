"""
    Functions to work with documents
"""

# pylint: disable=C0301,C0103,C0303,C0411,W1203

import os
import logging

logger : logging.Logger = logging.getLogger()


def save_documents(store_folder : str, document_names : list[str], document_contents : list[any]) -> list[str]:
    """
        Store documents in the given folder, returns stored document file names
    """
    full_document_names = []
    for document_name, document_content in zip(document_names, document_contents):
        full_document_name = os.path.join(store_folder, document_name)
        logger.info(f"Store document {document_name} to {full_document_name}")
        with open(full_document_name, 'wb') as f:
            f.write(document_content)
        full_document_names.append(full_document_name)
    return full_document_names
