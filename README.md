# 📓 NodeBook: Graph-Augmented RAG Engine

NodeBook is an advanced **Retrieval-Augmented Generation (RAG)** platform that combines the document-grounded Q&A capabilities of **Google’s NotebookLM** with the interactive, interconnected knowledge mapping of **Obsidian**.

Unlike standard RAG, which only performs vector searches, **NodeBook** extracts a formal Knowledge Graph from your documents, allowing you to visualize relationships between entities while chatting with your data.

---

## 🌟 Key Features

- **Obsidian-Style Knowledge Graph**: Uses a force-directed graph (PyVis) to map entities and relationships extracted via the **REBEL** (Relation Extraction By End-to-end Language generation) model.
- **Neural Reranking**: Implemented a **Cross-Encoder** (MS-Marco) to re-score retrieved documents, significantly increasing the accuracy of the final answer.
- **Context-Grounded Summarization**: Uses **BART-Large-CNN** to synthesize raw text chunks into fluent, human-like answers.
- **Multi-Document Support**: Process and index multiple PDFs simultaneously into a persistent **ChromaDB** vector store.
- **Citations & Grounding**: Every answer includes expandable source citations to prevent hallucinations and provide transparency.

---

## ⚙️ The NLP Pipeline

1. **Document Ingestion**: PDFs are parsed using `PyMuPDF` and segmented into semantic chunks with a sliding-window overlap to preserve context.
2. **Vector Indexing**: Chunks are embedded using `all-MiniLM-L6-v2` and stored in a local `ChromaDB` instance.
3. **Relation Extraction**: Each chunk passes through the `rebel-large` Seq2Seq model to extract `(Subject, Relation, Object)` triplets.
4. **Hybrid Retrieval**:
    - **Step A**: Retrieve top 10 candidates via Semantic Vector Search.
    - **Step B**: Re-rank candidates using a Neural Cross-Encoder.
5. **Summarization**: The top 3 candidates are passed into a BART model with custom prompt engineering to generate the final response.

---

## 🛠️ Tech Stack

- **Models**: `Babelscape/rebel-large` (Graph), `facebook/bart-large-cnn` (Summary), `cross-encoder/ms-marco-MiniLM-L-6-v2` (Reranker).
- **Vector DB**: `ChromaDB`.
- **Logic/Graph**: `NetworkX`, `Python 3.10`.
- **Frontend**: `Streamlit`, `PyVis (JavaScript-based visualization)`.

---

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/NodeBook.git
   cd NodeBook

2. **Install Dependencies**
   ```bash
    pip install -r requirements.txt

3. **Run the Application**
   ```bash
    streamlit run app/main.py
