# RAG with RBAC (Bellamy).
LLM applications typically use vector databases for RAG (retrieval augmented generation), but most vector databases today don't
natively support role based access controls (RBAC). They do support multi-tenancy and some access protections but these are not intended for the application layer, they are intended for service accounts and infrastructure layers. With role based access control (RBAC), change management is non-trivial - how do we keep the index up to date with files and metadata in storage (Google Drive or elsewhere)? This demo shows an implementation of how to do this. 

This is forked from: https://github.com/abhishek-kumar/rag_with_rbac

## Features
- **Local Document Ingestion**: Ingests PDF files from a local `documents` folder.
- **Role-Based Access Control**: Assigns specific documents to user groups ("CCSK", "ZeroTrust") and filters context based on the selected group.

## Running with the Dev Container

This project is configured to run inside a VS Code Dev Container, which provides a consistent and isolated development environment.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Visual Studio Code](https://code.visualstudio.com/)
- [VS Code Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Setup & Launch
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/TikkaMasala1/rag_with_rbac_bellamy
    cd rag_with_rbac_bellamy
    ```

2.  **Open in Dev Container**:
    - Open the cloned project folder in Visual Studio Code.
    - A pop-up will appear in the bottom-right corner asking to "Reopen in Container". Click it.
    - VS Code will now build the container. This may take a few minutes on the first run.

3.  **Configure Secrets**:
    - In the root of the project, create a new file named `.env`.
    - Copy the contents of `.env.example` (or the block below) into your new `.env` file and add your secret keys. This file is included in `.gitignore` and will not be committed to your repository.
    ```env
    # .env file

    # Get from [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
    GOOGLE_API_KEY="YOUR_GOOGLE_AI_API_KEY"

    # Get from [https://app.pinecone.io/](https://app.pinecone.io/)
    PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
    PINECONE_INDEX_NAME="your-pinecone-index-name"
    ```

4.  **Add Your Documents**:
    - Place all your PDF files into the `documents` folder. The application is pre-configured to look for specific filenames for the "CCSK" and "ZeroTrust" groups.

5.  **Run the Application**:
    - Once the dev container is running, open a new terminal in VS Code (`Terminal > New Terminal`).
    - Run the following command to start the Streamlit application:
    ```bash
    streamlit run streamlit_app.py
    ```
    - A new tab will open in your browser with the running application. If it doesn't, you can click the URL provided in the terminal (e.g., `http://localhost:8501`).

### How to Use the App
1.  **Index Documents**: The first time you run the app (and anytime you change documents or permissions), click the **"🔄 Index/Update Documents"** button in the sidebar. This clears the remote Pinecone index and uploads the documents with their correct access control settings.
2.  **Select a User Group**: In the main panel, use the dropdown menu to select the user group you want to query as (e.g., "CCSK" or "ZeroTrust").
3.  **Ask a Question**: Type your question in the text area and click "Submit". The application will retrieve information only from the documents that your selected group has access to and generate an answer.
