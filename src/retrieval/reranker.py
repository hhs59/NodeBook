import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class DocumentReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading Reranker Model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, passages: list[str]) -> list[str]:
        """
        Scores and sorts passages based on how well they answer the query.
        """
        if not passages:
            return []
            
        pairs = [[query, passage] for passage in passages]
        
        scores = self.model.predict(pairs)
        
        ranked_passages = [p for _, p in sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)]
        
        return ranked_passages