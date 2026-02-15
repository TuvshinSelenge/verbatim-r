import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Change only this value to switch reranker model.
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

        self.model_name = BGE_RERANKER_MODEL
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Loading BGE reranker: {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        tokenizer_max = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_max, int) and tokenizer_max > 100000:
            tokenizer_max = None
        config_max = getattr(self.model.config, "max_position_embeddings", None)
        max_candidates = [v for v in [tokenizer_max, config_max, 512] if isinstance(v, int) and v > 0]
        self.max_length = min(max_candidates)

        self._initialized = True
        print(f"BGE Reranker ready (max_length={self.max_length})")

    @torch.inference_mode()
    def rerank(self, query: str, chunks: list, top_k: int = 5, text_key: str = "text"):
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
            max_length=self.max_length,
        ).to(self.device)

        scores = (
            self.model(**inputs, return_dict=True).logits.view(
                -1,
            )
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
