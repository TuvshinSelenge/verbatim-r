# BGE Reranker
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

class BGEReranker:
    """BGE-based reranker for improved chunk ranking."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.device = (
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Loading BGE reranker: {BGE_RERANKER_MODEL} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(BGE_RERANKER_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(BGE_RERANKER_MODEL)
        self.model.to(self.device)
        self.model.eval()
        self._initialized = True
        print("BGE Reranker ready")
    
    @torch.inference_mode()
    def rerank(self, query: str, chunks: list, top_k: int = 5, text_key: str = "text"):
        """
        Return (top_chunks, ranking) where:
        - top_chunks: the best `top_k` chunks in descending relevance
        - ranking: list of {"id": original_index, "score": float} in descending relevance
        """
        if not chunks:
            return [], []

        pairs = []
        for c in chunks:
            if isinstance(c, dict):
                txt = c.get(text_key, "") or c.get("text", "")
            else:
                txt = getattr(c, "text", "")
            pairs.append([query, txt])

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=1024,  # speed/accuracy tradeoff
        ).to(self.device)

        scores = (
            self.model(**inputs, return_dict=True)
            .logits.view(-1,)
            .float()
            .detach()
            .cpu()
            .tolist()
        )

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        ranking = [{"id": int(i), "score": float(s)} for i, s in ranked]

        for i, s in ranked:
            c = chunks[i]
            if isinstance(c, dict):
                c["rerank_score"] = float(s)
            else:
                setattr(c, "rerank_score", float(s))

        top_chunks = [chunks[i] for i, _ in ranked[:top_k]]
        return top_chunks, ranking
    
    @staticmethod
    def _get_text(c) -> str:
        if isinstance(c, dict):
            return c.get("text", "")
        return getattr(c, "text", "")
    
    @staticmethod
    def _normalize_text(t: str) -> str:
        return (t or "").replace("\ufffd", "").replace("\r", "\n")
