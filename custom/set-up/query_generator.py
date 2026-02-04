# Query Generator with robust JSON parsing for Gemini models
import json
import re
from typing import List
from openai import OpenAI

BANK_NAME = "Raiffeisen Bank International AG"
BANK_SHORT = "RBI"
LLM_MODEL = "google/gemini-3-flash-preview"


def safe_parse_queries(raw: str, fallback_query: str) -> List[str]:
    """
    Parse {"queries": [...]} from raw model output.
    Works for pure JSON, JSON in code blocks, or JSON mixed with text.
    Falls back to [fallback_query] if parsing fails.
    
    This handles Gemini models which often wrap JSON in ```json ... ``` blocks.
    """
    if not raw or not raw.strip():
        return [fallback_query]

    content = raw.strip()

    # 1) Markdown code block (```json ... ``` or ``` ... ```)
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    if m:
        try:
            data = json.loads(m.group(1).strip())
            qs = data.get("queries", [])
            return [q.strip() for q in qs if q and q.strip()] or [fallback_query]
        except Exception:
            pass

    # 2) First JSON object in text (handles "thinking" text before JSON)
    m = re.search(r'(\{[\s\S]*\})', content)
    if m:
        try:
            data = json.loads(m.group(1))
            qs = data.get("queries", [])
            return [q.strip() for q in qs if q and q.strip()] or [fallback_query]
        except Exception:
            pass

    # 3) Try whole string as direct JSON
    try:
        data = json.loads(content)
        qs = data.get("queries", [])
        return [q.strip() for q in qs if q and q.strip()] or [fallback_query]
    except Exception:
        return [fallback_query]


class QueryGenerator:
    """Generates multiple search queries from a user question."""
    
    def __init__(self, client: OpenAI = None, model: str = LLM_MODEL):
        self.client = client or OpenAI()
        self.model = model
    
    def generate_queries(self, question: str) -> List[str]:
        """Generate multiple search queries for the given question."""
        
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
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
        except Exception as e:
            print(f"  Query generation error: {e}")
            raw = ""
        
        # Use robust parsing that handles Gemini's markdown-wrapped JSON
        queries = safe_parse_queries(raw, fallback_query=question)
        
        # Deduplication keeping order
        out, seen = [], set()
        for q in queries:
            if q not in seen:
                seen.add(q)
                out.append(q)
        return out
