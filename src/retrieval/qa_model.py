from transformers import pipeline

class LocalSummarizer:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device="cpu")

    def generate_answer(self, query: str, context: str) -> str:
        prompt = f"Question: {query} Context: {context}"
        summary = self.summarizer(prompt, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']