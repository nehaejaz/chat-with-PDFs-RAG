# Chat with PDFs - Retrieval-Augmented Generation (RAG) Project

## Overview

This project enables users to interact with PDF documents using a Retrieval-Augmented Generation (RAG) approach. Users can upload PDFs, which are converted into vectorized data stored in a database. The system answers user queries by leveraging the vector database and a large language model (LLMs).



## Features

- **Upload PDF Documents**: Upload one or more PDF files for analysis.
- **Vector Database Creation**: Automatically converts uploaded PDFs into embeddings for efficient search and retrieval.
- **Chat with Documents**: Ask questions about the content of the PDFs, and the system provides accurate, context-aware answers.
- **Fast and Scalable**: Designed for quick query responses, even with large documents.



## Getting Started

- **Python**: Ensure Python 3.8 or later is installed.

**1. Clone repository**
```bash
git clone https://github.com/nehaejaz/RAG_chatbot.git
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Run Application**
```bash
streamlit run app.py
```

4. Go to http://localhost:8000 to access the user interface.


