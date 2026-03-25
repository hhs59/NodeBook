import logging
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class VectorStore:

    def __init__(self, db_path: str = "data/db/chroma"):
        logger.info(f"Initializing VectorStore at {db_path}")
        
        self.client = chromadb.PersistentClient(path=db_path)
        
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name="document_chunks",
            embedding_function=self.embedding_fn
        )
        logger.info("ChromaDB collection 'document_chunks' is ready.")


    def add_chunks(self, chunks: List[str]):

        if not chunks:
            logger.warning("No chunks provided to add_chunks.")
            return

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        metadatas = [{'chunk_index': i} for i in range(len(chunks))]
        
        logger.info(f"Adding {len(chunks)} chunks with metadata to the vector store...")
        try:
            self.collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            logger.info("Successfully indexed chunks.")
        except Exception as e:
            logger.error(f"Failed to add chunks to ChromaDB: {e}")
            raise

    def search(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        
        logger.info(f"Searching for: '{query}'")
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

if __name__ == "__main__":
    logger.info("Running VectorStore local test.")
    
    store = VectorStore()
    
    test_chunks = [
        "Yann LeCun is a professor at NYU and Chief AI Scientist at Meta.",
        "The Turing Award is often referred to as the Nobel Prize of Computing.",
        "The Eiffel Tower is located in Paris, France."
    ]
    
    store.add_chunks(test_chunks)
    
    query = "Who is the AI scientist at Meta?"
    results = store.search(query, n_results=1)
    
    print("\n--- SEARCH RESULTS ---")
    print(f"Query: {query}")
    print(f"Matched Text: {results['documents'][0][0]}")