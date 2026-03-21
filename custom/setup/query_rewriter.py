import os
from typing import Optional
from openai import OpenAI


class QueryRewriter:
    def __init__(
        self,
        bank_name: str = "Raiffeisen Bank International AG",
        bank_short_name: str = "RBI",
        openai_client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        verbose: bool = False,
    ):
        self.bank_name = bank_name
        self.bank_short_name = bank_short_name
        self.client = openai_client or OpenAI()
        self.model = model or os.environ.get("QUERY_REWRITER_MODEL", "gpt-5.1")
        self.verbose = verbose

    def rewrite(
        self,
        query: str,
        bank_name: Optional[str] = None,
        bank_short_name: Optional[str] = None,
    ) -> str:
        system = self._system_prompt(
            bank_name if bank_name is not None else self.bank_name,
            bank_short_name if bank_short_name is not None else self.bank_short_name,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query.strip()},
            ],
        )
        return response.choices[0].message.content.strip()

    def _system_prompt(self, bank_name: str, bank_short_name: str) -> str:
        return f"""
        You are an expert in Due Diligence / finance questions. You rewrite Due Diligence / finance questions into simple questions suitable for searching
        the {bank_name} Annual Report.

        <Rules>
        - Output ONLY questions.
        - No bullet points, no numbering, no prefixes, no commentary.
        - Translate the abbreviations into the full form.
        - Look at the <Example> for inspiration. It is asking about its systemic importance.
        - If the question is asking about contracting party, organisation, entity or firm then the question is asking about {bank_name}.
        - When the text uses a short ticker or acronym (e.g. {bank_short_name}), treat it as referring to {bank_name} when rewriting.
        - Try to grasp the meaning of the question and rewrite it into a simple question.
        </Rules>

     <Example>
     - User: "Please provide documentation or evidence to confirm whether your service infrastructure undergoes independent third-party audits, such as SOC2 Type II or ISO 27001, on an annual basis."
     - Rewritten: Does your service undergo independent third-party security audits every year?
     </Example>
        """.strip()
