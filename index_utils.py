"""
Builds a Pinecone index from local documents for LLM applications involving RAG.
Each document in the index also has metadata containing ACL
information such as groups that have read permission on that document.
"""
import dataclasses
import logging
import os
import traceback
from dataclasses import dataclass, field

# Retrieval and indexing of local data.
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, exceptions  # Import exceptions

from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Google Generative AI Embeddings model.
_MODEL_NAME: str = "models/embedding-001"

# Pinecone index constants.
PUBLIC_USERS_GROUP: str = "all_users"


@dataclass
class Document:
    # unique id of this document is its path.
    file_id: str

    # filename, e.g. "abc.pdf"
    name: str

    # List of groups that have read access to this document.
    read_access: List[str] = field(default_factory=list)

    # Last modified time of this file or document.
    modified_time: Optional[float] = None

    # Size in bytes, of the data in this document.
    size: Optional[int] = None

    # If this document has been written to the index,
    # the list of record ids associated with this doc.
    index_record_ids: Optional[Set[str]] = None

    def __eq__(self, other):
        """Note that index_record_ids is not checked for equality."""
        return (
                other.file_id == self.file_id and
                other.name == self.name and
                sorted(other.read_access) == sorted(self.read_access) and
                other.modified_time == self.modified_time and
                other.size == self.size)


# An index manifest represents the state of an index at a point in time.
# It stores metadata of all the documents in the index.
IndexManifest = Dict[str, Document]


def read_documents_from_local(
        folder_path: str,
        file_permissions: Dict[str, List[str]],  # e.g., {"document1.pdf": ["eng_users"]}
        include_files_with_extensions: Optional[List[str]] = None
) -> List[Document]:
    """
    Retrieves the metadata of all documents in the provided local folder
    (recursively traversing sub-folders).

    Args:
        folder_path: The path to the local folder.
        file_permissions: A dictionary mapping filenames to a list of groups with read access.
        include_files_with_extensions: If set, only include documents with the specified extensions.

    Returns: A list of documents from the provided folder.
    """
    if include_files_with_extensions is None:
        include_files_with_extensions = ["pdf"]

    if not os.path.isdir(folder_path):
        logging.error(f"Provided path '{folder_path}' is not a directory.")
        return []

    result = []
    for root, _, files in os.walk(folder_path):
        for name in files:
            if not any(name.endswith(ext) for ext in include_files_with_extensions):
                continue

            file_path = os.path.join(root, name)
            try:
                stat = os.stat(file_path)
                # Default to public if no specific permissions are set
                access_groups = file_permissions.get(name, [PUBLIC_USERS_GROUP])

                doc = Document(
                    file_id=file_path,
                    name=name,
                    read_access=access_groups,
                    modified_time=stat.st_mtime,
                    size=stat.st_size
                )
                result.append(doc)
                logging.info(f"Reading from local disk: found document:\n\t{doc}")
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")

    return result


def clear_index(
        pinecone_api_key: str, pinecone_index_name: str) -> None:
    """
    Deletes all vectors from the specified Pinecone index's default namespace.
    """
    os.environ['PINECONE_API_KEY'] = pinecone_api_key
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)

    try:
        # Try to delete all vectors in the default namespace
        index.delete(delete_all=True)
        logging.warning(f"Successfully cleared all documents in index '{pinecone_index_name}'.")
    except exceptions.NotFoundException:
        # This exception occurs if the namespace is not found, which means it's already empty.
        logging.warning(f"Index '{pinecone_index_name}' or its default namespace was already empty. No action needed.")
    except Exception as e:
        # Re-raise any other unexpected errors
        logging.error(f"Could not clear index '{pinecone_index_name}'. Unexpected error: {e}")
        raise


def add_documents_to_index(
        documents: List[Document],
        google_api_key: str,
        pinecone_api_key: str,
        pinecone_index_name: str,
) -> IndexManifest:
    """
    Builds the Pinecone index with the supplied documents.
    Returns the updated IndexManifest with the newly added documents.
    """
    os.environ['PINECONE_API_KEY'] = pinecone_api_key
    os.environ['GOOGLE_API_KEY'] = google_api_key

    embedding = GoogleGenerativeAIEmbeddings(model=_MODEL_NAME)
    vectorstore = PineconeVectorStore(index_name=pinecone_index_name, embedding=embedding)

    index_manifest: IndexManifest = {}

    for document in documents:
        try:
            logging.info(f"Processing '{document.name}'.")

            loader = PyPDFLoader(document.file_id)  # file_id is the path
            pages = loader.load_and_split()

            for page in pages:
                page.metadata["read_access"] = sorted(list(set(document.read_access)))
                page.metadata["name"] = document.name
                page.metadata["file_id"] = document.file_id
                if document.modified_time is not None:
                    page.metadata["modified_time"] = str(document.modified_time)
                if document.size is not None:
                    page.metadata["size"] = str(document.size)

            if pages:
                logging.info(f"\tUploading {len(pages)} records for '{document.name}' to the index.")
                added_record_ids = vectorstore.add_documents(pages)

                # Update manifest
                doc_copy = dataclasses.replace(document, index_record_ids=set(added_record_ids))
                index_manifest[document.file_id] = doc_copy

                logging.info(f"\tSuccessfully uploaded records for '{document.name}'.")

        except Exception as e:
            logging.error(f"Failed to process and index document {document.name}. Error: {e}")
            logging.error(traceback.format_exc())

    return index_manifest


def get_all_indexed_documents(
        pinecone_api_key: str,
        pinecone_index_name: str,
        namespace: str = None
) -> Dict[str, Dict[str, Any]]:
    """A simplified way to get metadata for all vectors to reconstruct a manifest."""
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)

    # This is a workaround as Pinecone doesn't have a direct "list all vectors" API.
    # We fetch a dummy vector hoping to get stats, then query for all.
    # NOTE: This can be slow and memory-intensive for very large indexes.
    try:
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        if total_vectors == 0:
            return {}

        # Fetch all vector IDs. This is not a standard API feature.
        # A common trick is to query a zero vector with top_k=total_vectors
        # This is not guaranteed to work and depends on Pinecone's implementation.
        # For a robust solution, you should maintain the manifest separately.
        # For this simplified version, we assume this might not be feasible and
        # will rely on rebuilding the index when needed.
        logging.warning("Fetching all documents from index is not efficiently supported. Rebuilding is recommended.")
        return {}  # Return empty to force rebuilds, which is simpler.

    except Exception as e:
        logging.error(f"Could not retrieve index stats: {e}")
        return {}
