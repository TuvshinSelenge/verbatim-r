import json
import re
from typing import List, Optional, Set
from pydantic import BaseModel
from openai import OpenAI


class QueryRewriter:
    def __init__(self,
        bank_name: str = "Raiffeisen Bank International AG",
        bank_short_name: str = "RBI",
        openai_client: Optional[OpenAI] = None,
        model: str = "gpt-5.1",
        verbose: bool = False
    ):
        self.bank_name = bank_name
        self.bank_short_name = bank_short_name
        self.client = openai_client or OpenAI()
        self.model = model
        self.verbose = verbose

    def rewrite(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": self._system_prompt(query)},
                {"role": "user", "content": query.strip()}
            ],
        )
        return response.choices[0].message.content.strip()

        
    def _system_prompt(self, query: str) -> str:
        return f"""
        You are an expert in Due Diligence / finance questions. You rewrite Due Diligence / finance questions into simple questions suitable for searching
        the {self.bank_name} Annual Report.

        <Rules>
        - Output ONLY questions.
        - No bullet points, no numbering, no prefixes, no commentary.
        - Translate the abbreviations into the full form.
        - Look at the <Example> for inspiration. It is asking about its systemic importance.
        - If the question is asking about contracting party, organisation, entity or firm then the question is asking about {self.bank_name}.
        - Try to grasp the meaning of the question and rewrite it into a simple question.
        </Rules>

        <Example>
        - User: "Please advise the name of the contracting party providing custody and / or client money services in your jurisdiction and responding to this questionnaire. If applicable, please also advise the name of the local delegate if different from the contracting entity."
        - Rewritten: What is the name of the reporting entity?
        - User: "Please confirm if your organisation is considered a systematically important financial institution as defined by the Financial Stability Board / Basel Committee on Banking Supervision or as defined in the jurisdiction where it is registered."
        - Rewritten: Is your organisation considered a systematically important financial institution?
        </Example>
        """.strip()