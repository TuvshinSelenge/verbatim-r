# Query Generator
from typing import List
from openai import OpenAI

BANK_NAME = "Raiffeisen Bank International AG"
BANK_SHORT = "RBI"
LLM_MODEL = "google/gemini-3-flash-preview"

class QueryGenerator:
    """Generates multiple search queries from a user question."""
    
    def __init__(self, client: OpenAI = None, model: str = LLM_MODEL):
        self.client = client or OpenAI()
        self.model = model
    
    def generate_queries(self, question: str) -> List[str]:
        """Generate multiple search queries for the given question."""
        import json
        
        prompt = f"""
You generate search queries to retrieve relevant chunks from a bank annual report for: {BANK_NAME} ({BANK_SHORT}).

Return JSON only with this schema:
{{"queries": ["q1","q2","q3"]}}

<Rules>
- Make 3 queries max
- Do NOT start queries with: Confirm / Please / Advise
- Use short keyword-ish phrases that would appear in annual reports
- Include 1 exact-name query with the bank name or short name when helpful
- Include variants with synonyms / acronyms (e.g. O-SII, G-SIB, systemic risk buffer, credit rating Moody's, etc.)
</Rules>

Question: {question}
"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        queries = [q.strip() for q in data.get("queries", []) if q and q.strip()]
        
        # Deduplication keeping order
        out, seen = [], set()
        for q in queries:
            if q not in seen:
                seen.add(q)
                out.append(q)
        return out
