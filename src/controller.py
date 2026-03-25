import logging
import os
import torch
from typing import List, Dict, Any

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.ingestion.parser import extract_text_from_pdf
from src.ingestion.chunker import chunk_text
from src.graph_engine.extractor import RelationExtractor
from src.vector_engine.vector_db import VectorStore
from src.retrieval.reranker import DocumentReranker

logger = logging.getLogger(__name__)

class NodeBookController:
    def __init__(self):
        logger.info("Initializing NodeBook Engine...")
        
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.graph_extractor = RelationExtractor()
        self.vector_store = VectorStore()
        self.reranker = DocumentReranker()
        
        model_name = "facebook/bart-large-cnn"
        logger.info(f"Loading Summarizer: {model_name}")
        self.sum_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sum_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        self.knowledge_graph = []
        logger.info("NodeBook Engine is fully loaded and ready.")

    def process_document(self, file_path: str):
        raw_text = extract_text_from_pdf(file_path)
        chunks = chunk_text(raw_text, chunk_size=400, overlap=50)
        self.vector_store.add_chunks(chunks)
        
        for i, chunk in enumerate(chunks):
            triplets = self.graph_extractor.extract_triplets(chunk)
            for t in triplets:
                t['source'] = os.path.basename(file_path)
                t['chunk_id'] = i
                self.knowledge_graph.append(t)
        
        logger.info(f"Indexing complete. Total triplets: {len(self.knowledge_graph)}")


    def ask_question(self, query: str) -> Dict[str, Any]:
        logger.info(f"Answering query: {query}")
        
        initial_results = self.vector_store.search(query, n_results=10)
        initial_chunks = initial_results.get('documents', [[]])[0]
        
        if not initial_chunks:
            return {"answer": "I couldn't find any relevant information in the uploaded documents.", "relevant_nodes": []}

        best_chunks = self.reranker.rerank(query, initial_chunks)[:3]
        context = " ".join(best_chunks)

        prompt_template = f"""
        You are a helpful AI assistant. Your task is to answer the user's question based ONLY on the provided context.
        Do not use any outside knowledge. Do not make anything up.
        If the context does not contain the information needed to answer the question, you must say: "I cannot find the answer in the provided documents."

        [Question]:
        {query}

        [Context]:
        {context}

        [Answer]:
        """

        inputs = self.sum_tokenizer(prompt_template, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        
        summary_ids = self.sum_model.generate(
            inputs["input_ids"], 
            num_beams=4, 
            max_length=200, 
            min_length=40, 
            early_stopping=True
        )
        answer = self.sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        query_keywords = query.lower().split()
        relevant_nodes = [
            t for t in self.knowledge_graph 
            if any(k in t['head'].lower() or k in t['tail'].lower() for k in query_keywords)
        ]
        
        return {
            "answer": answer,
            "answer_context": best_chunks,
            "relevant_nodes": relevant_nodes
        }