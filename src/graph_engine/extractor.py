import logging
import torch
from typing import List, Dict, Any
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class RelationExtractor:
    
    def __init__(self, model_name: str = "Babelscape/rebel-large"):
        logger.info(f"Initializing RelationExtractor with model: {model_name}")
        try:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
            logger.info(f"Using device: {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            
            logger.info("Model and Tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}. Error: {e}")
            raise

    def extract_triplets(self, text: str) -> List[Dict[str, str]]:
        if not text or not isinstance(text, str):
            return []

        inputs = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        gen_outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,
            num_beams=3,
        )
        
        decoded_text = self.tokenizer.decode(gen_outputs[0], skip_special_tokens=False)
        return self._parse_rebel_output(decoded_text)

    def _parse_rebel_output(self, raw_text: str) -> List[Dict[str, str]]:
        triplets = []
        current = "x"
        subject, relation, object_ = "", "", ""
        
        text = raw_text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
        tokens = text.split()
        
        for token in tokens:
            if token == "<triplet>":
                current = "t"
                if relation:
                    triplets.append({"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()})
                subject, relation, object_ = "", "", ""
            elif token == "<subj>":
                current = "s"
                if relation:
                    triplets.append({"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()})
                object_ = ""
            elif token == "<obj>":
                current = "o"
                relation = ""
            else:
                if current == "t": subject += " " + token
                elif current == "s": object_ += " " + token
                elif current == "o": relation += " " + token

        if subject and relation and object_:
            triplets.append({"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()})
            
        return triplets