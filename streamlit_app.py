import json
import logging
import os
import re
import traceback

import streamlit as st
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pinecone import Pinecone

import index_utils
from typing import List, Dict, Tuple

# Gemini Embeddings model.
_MODEL_NAME: str = "models/embedding-001"


def escape_markdown(text: str) -> str:
    """Escapes markdown characters in a string."""
    return text.replace("$", "\\$").replace("{", "\\{").replace("}", "\\}")


@st.cache_resource(ttl=3600)
def get_llm(google_api_key) -> ChatGoogleGenerativeAI:
    """Returns a cached instance of the Gemini LLM."""
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2, google_api_key=google_api_key)


@st.cache_resource(ttl=3600)
def get_vectorstore_indexwrapper(
        google_api_key: str, pinecone_api_key: str, pinecone_index_name: str) -> VectorStoreIndexWrapper:
    """Returns a cached instance of the VectorStoreIndexWrapper."""
    os.environ['PINECONE_API_KEY'] = pinecone_api_key
    os.environ['GOOGLE_API_KEY'] = google_api_key

    embed = GoogleGenerativeAIEmbeddings(model=_MODEL_NAME)

    vector_store = PineconeVectorStore.from_existing_index(
        index_name=pinecone_index_name,
        embedding=embed
    )

    return VectorStoreIndexWrapper(vectorstore=vector_store)


def query_rag_with_rbac(
        input_text: str,
        llm: ChatGoogleGenerativeAI,
        index: VectorStoreIndexWrapper,
        groups: List[str]) -> Tuple[str, List[str]]:
    """
    Queries the RAG system with RBAC filters.
    Returns a tuple of <answer, list of sources>.
    """
    if not groups:
        return "You do not belong to any user group. Please select at least one group.", []

    response = index.query_with_sources(
        input_text,
        llm=llm,
        retriever_kwargs={
            "search_kwargs": {
                "filter": {
                    "read_access": {"$in": groups}
                }
            }
        },
        reduce_k_below_max_tokens=True
    )

    answer = escape_markdown(response.get("answer", "I don't know.").strip())
    sources = [source.strip() for source in response.get("sources", "").split(",") if source.strip()]
    if not sources:
        sources = ["N/A"]

    logging.info(f"query_rag_with_rbac: user groups {groups}.")
    logging.info(f"query_rag_with_rbac: answer:\n{answer}")
    logging.info(f"query_rag_with_rbac: sources:\n{sources}")

    return (answer, sources)


def main():
    st.title('RAG with RBAC from Local `documents` Folder')
    st.markdown(
        "An app demonstrating Retrieval Augmented Generation with Role-Based Access Control using PDF files from the repository's `documents` folder.")

    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        script_dir = os.getcwd()

    local_folder_path = os.path.join(script_dir, "documents")

    st.sidebar.header("Configuration")
    google_api_key = st.sidebar.text_input('Google AI API Key', type='password', key='google_api_key')
    pinecone_api_key = st.sidebar.text_input('Pinecone API Key', type='password', key='pinecone_api_key')
    pinecone_index_name = st.sidebar.text_input('Pinecone Index Name', key='pinecone_index_name')

    st.sidebar.info(f"**Documents source:** `{local_folder_path}`")

    if not all([google_api_key, pinecone_api_key, pinecone_index_name]):
        st.warning("Please provide all required API and index details in the sidebar.")
        return

    if not os.path.isdir(local_folder_path):
        st.error(
            f"The `documents` folder was not found. Please create it at `{local_folder_path}` and add your PDF files.")
        return

    st.sidebar.header("Access Control")

    # Use os.walk to find files in subdirectories as well
    available_files = []
    for root, _, files in os.walk(local_folder_path):
        for name in files:
            if name.endswith('.pdf'):
                available_files.append(name)
    available_files.sort()

    standard_groups = ["CCSK", "ZeroTrust"]
    all_groups = sorted(standard_groups + [index_utils.PUBLIC_USERS_GROUP])

    st.sidebar.markdown("#### Standard User Groups")
    st.sidebar.markdown(f"`{', '.join(standard_groups)}`")

    # --- FIX APPLIED HERE: Corrected the filename key ---
    predefined_permissions = {
        "CCSK Study Guide20250225.pdf": ["CCSK"],  # Note: Space removed
        "Zero_Trust_Implementation_Study_Guide.pdf": ["ZeroTrust"],
        "Zero_Trust_Planning_Study_Guide.pdf": ["ZeroTrust"],
        "Zero_Trust_Strategy_Study_Guide.pdf": ["ZeroTrust"]
    }

    if 'file_permissions' not in st.session_state:
        st.session_state.file_permissions = {}
        for fname in available_files:
            st.session_state.file_permissions[fname] = predefined_permissions.get(fname,
                                                                                  [index_utils.PUBLIC_USERS_GROUP])

    with st.sidebar.expander("View/Modify File Permissions for this Session", expanded=False):
        for fname in available_files:
            if fname not in st.session_state.file_permissions:
                st.session_state.file_permissions[fname] = predefined_permissions.get(fname,
                                                                                      [index_utils.PUBLIC_USERS_GROUP])

            st.session_state.file_permissions[fname] = st.multiselect(
                f"**{fname}**",
                options=all_groups,
                default=st.session_state.file_permissions.get(fname, [])
            )

    if st.sidebar.button("Index/Update Documents"):
        with st.spinner("Indexing documents... This may take a while."):
            try:
                st.write("Clearing existing index...")
                index_utils.clear_index(pinecone_api_key, pinecone_index_name)

                st.write("Reading local files and permissions...")
                documents_to_index = index_utils.read_documents_from_local(
                    local_folder_path,
                    st.session_state.file_permissions
                )

                if documents_to_index:
                    st.write(f"Found {len(documents_to_index)} documents. Adding to Pinecone...")
                    index_utils.add_documents_to_index(
                        documents=documents_to_index,
                        google_api_key=google_api_key,
                        pinecone_api_key=pinecone_api_key,
                        pinecone_index_name=pinecone_index_name
                    )
                    st.sidebar.success(f"Successfully indexed {len(documents_to_index)} documents!")
                    st.cache_resource.clear()
                else:
                    st.sidebar.warning("No PDF documents found to index in the `documents` folder.")

            except Exception as e:
                st.sidebar.error(f"Indexing failed: {e}")
                logging.error(traceback.format_exc())

    st.header("💬 Ask a Question")

    current_user_groups = st.multiselect(
        "Select your user group(s) for this query:",
        options=all_groups,
        default=[]
    )
    st.info(f"Querying with access rights for: **{', '.join(current_user_groups) or 'None'}**")

    with st.form("ask_llm_form"):
        text = st.text_area(
            label="Your question",
            value="What is the SAAS?",
            label_visibility="hidden"
        )
        submitted = st.form_submit_button(label="Submit")

        if submitted:
            if not current_user_groups:
                st.error("Please select at least one user group before submitting a question.")
            else:
                try:
                    llm = get_llm(google_api_key=google_api_key)
                    index = get_vectorstore_indexwrapper(
                        google_api_key=google_api_key,
                        pinecone_api_key=pinecone_api_key,
                        pinecone_index_name=pinecone_index_name
                    )

                    with st.spinner("Searching for answers with your access rights..."):
                        answer, sources = query_rag_with_rbac(
                            input_text=text,
                            llm=llm,
                            index=index,
                            groups=current_user_groups
                        )

                        st.markdown("#### Answer")
                        st.write(answer)

                        st.markdown("#### Sources")
                        st.write("\n".join(f"- `{s}`" for s in sources))


                except Exception as ex:
                    st.error(f"An error occurred: {ex}")
                    logging.error(traceback.format_exc())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()