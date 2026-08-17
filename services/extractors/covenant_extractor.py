import re
import logging
from typing import Dict, Optional, Tuple
from shared.schemas import DocumentInfo, CovenantClause, CovenantExtractionResult
from shared.llm_client import LLMClient

logger = logging.getLogger(__name__)

COVENANT_ANALYSIS_PROMPT = """You are a financial covenant analyst. Analyze the loan agreement clause and extract the metric composition.
Do NOT calculate numbers. Only describe what transactions/items are INCLUDED in the metric.

Return ONLY a JSON object (no markdown, no extra text):
{
  "numerator_definition": "Short plain-text description of what expenses/amounts form the numerator or tested sum. Example: 'All capital expenditure payments reclassified by auditor as Capex' or 'Total debt service payments including interest and principal'. Use the actual clause wording.",
  "denominator_definition": "Description of denominator if ratio-based test, or null if it is an absolute limit test",
  "references_audit_adjustment": true or false — does the clause mention auditor reclassification or adjustment,
  "references_kyc": true or false — does the clause mention related parties, affiliates or beneficial owners
}

Rules:
- numerator_definition MUST always be a non-empty string describing the category of transactions to aggregate
- If the clause tests a ratio, fill denominator_definition with the denominator description
- If the clause tests an absolute amount limit, set denominator_definition to null
"""


from services.retrieval.bm25_retriever import BM25Retriever

class CovenantExtractor:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        self.bm25 = BM25Retriever()

    def extract_covenants(self, doc: DocumentInfo) -> CovenantExtractionResult:
        text = doc.raw_text or ""
        account_id = doc.account_id or "UNKNOWN"
        company_name = doc.company_name or "UNKNOWN"

        covenants_map = self._extract_covenants_from_text(text, account_id, company_name)

        return CovenantExtractionResult(
            account_id=account_id,
            company_name=company_name,
            covenants=covenants_map
        )

    def _extract_covenant_body(self, text: str) -> str:
        match_body = re.search(r"(?:Статья 6|Article 6|6\.1)[\s\S]{1,6000}(?=(?:Статья 7|Article 7|\Z))", text, re.IGNORECASE)
        if match_body:
            return match_body.group(0)
        
        # Fallback to BM25 top snippets for covenant keywords
        bm25_query = "ковенант финансовый ковенант 6.1 6.2 6.3 капитальные затраты capex лимит отношение доля долг"
        return self.bm25.get_top_snippets(text, query=bm25_query, top_k=3, max_tokens_approx=1500)

    def _extract_covenants_from_text(self, snippet: str, account_id: str, company_name: str) -> Dict[str, CovenantClause]:
        covenants_map = {}

        for clause_key in ["6.1", "6.2", "6.3"]:
            escaped_key = re.escape(clause_key)
            pattern = rf"(?:Пункт|Clause|Section|^|\n)\s*{escaped_key}[\s\.\s][\s\S]{{1,1500}}(?=(?:(?:Пункт|Clause|Section|\n)\s*6\.[123]|Статья|Article|\Z))"
            match = re.search(pattern, snippet, re.IGNORECASE)
            if match:
                raw_clause_text = match.group(0).strip()
                # Strip clause number header (e.g. "6.1.") to prevent threshold regex matching the clause number itself
                clean_text = re.sub(rf"^(?:Пункт|Clause|Section|\n)?\s*{escaped_key}[\s\.\:]*", "", raw_clause_text, flags=re.IGNORECASE).strip()

                # Extract threshold float number (Currency Amount or Ratio multiplier)
                threshold = 0.0
                ratio_match = re.search(r"(\d+\.\d+)\s*[xх]?", clean_text, re.IGNORECASE)
                amount_match = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", clean_text, re.IGNORECASE)

                text_lower = clean_text.lower()
                if "коэффициент" in text_lower or "отношение" in text_lower or "ratio" in text_lower or "доля" in text_lower:
                    if ratio_match:
                        threshold = float(ratio_match.group(1))
                    elif amount_match:
                        threshold = float(amount_match.group(1).replace(",", ""))
                else:
                    if amount_match:
                        threshold = float(amount_match.group(1).replace(",", ""))
                    elif ratio_match:
                        threshold = float(ratio_match.group(1))

                # Extract operator
                operator = "<="
                text_lower = raw_clause_text.lower()
                if "не менее" in text_lower or "поддерживать" in text_lower or "обеспечить" in text_lower or "at least" in text_lower or "minimum" in text_lower:
                    operator = ">="
                elif "не превышал" in text_lower or "не допускать" in text_lower or "не превышала" in text_lower or "not exceed" in text_lower or "maximum" in text_lower:
                    operator = "<="

                # Determine metric category
                if "коэффициент" in text_lower or "отношение" in text_lower or "доля" in text_lower or "ratio" in text_lower or "покрытия" in text_lower:
                    metric_name = "RATIO_TEST"
                elif "выручк" in text_lower or "поступлен" in text_lower or "revenue" in text_lower:
                    metric_name = "REVENUE_LIMIT"
                elif "связан" in text_lower or "аффилир" in text_lower or "related-party" in text_lower:
                    metric_name = "RELATED_PARTY_LIMIT"
                elif "персонал" in text_lower or "оплат" in text_lower or "накладных" in text_lower:
                    metric_name = "OVERHEAD_PERSONNEL_LIMIT"
                elif "капитальн" in text_lower or "capex" in text_lower:
                    metric_name = "CAPEX_LIMIT"
                else:
                    metric_name = "GENERIC_LIMIT"

                # Check marginal single transaction condition
                is_marginal = ("отдельная" in text_lower or "отдельный" in text_lower or "одной транзакцией" in text_lower or "одиночн" in text_lower or "определяющей результат" in text_lower)

                # Check audit adjustments & KYC references
                ref_audit = any(w in text_lower for w in ["корректировк", "аудит", "переклассифи", "восстановл", "отсечен"])
                ref_kyc = any(w in text_lower for w in ["связан", "аффилир", "дочерн", "kyc", "бенфициа"])

                # LLM extraction with raw_clause_text fallback
                num_def, den_def = self._extract_definitions(raw_clause_text)

                if not num_def or not str(num_def).strip():
                    num_def = raw_clause_text
                    logger.warning(f"LLM returned empty numerator_definition for {clause_key}. Using raw clause text as fallback.")

                covenants_map[clause_key] = CovenantClause(
                    clause_number=clause_key,
                    title=f"Clause {clause_key}",
                    metric_name=metric_name,
                    operator=operator,
                    threshold=threshold,
                    period="ANNUAL",
                    is_marginal_single_txn=is_marginal,
                    raw_clause_text=raw_clause_text,
                    numerator_definition=num_def,
                    denominator_definition=den_def,
                    references_audit_adjustment=ref_audit,
                    references_kyc=ref_kyc
                )
            else:
                covenants_map[clause_key] = CovenantClause(
                    clause_number=clause_key,
                    title=f"Clause {clause_key}",
                    metric_name="GENERIC",
                    operator="<=",
                    threshold=0.0,
                    period="ANNUAL",
                    raw_clause_text=f"Пункт {clause_key} по умолчанию"
                )

        return covenants_map

    def _parse_ratio_formula(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        text_clean = text.replace('\n', ' ')
        m = re.search(r'отношение\s+(.*?)\s+к\s+(.*?)(?=\s+(?:не превышало|не превышает|не превышал|составляло|составлял|составлять|величина|быть|равно|превышал|\b\d+\.\d+x|\.|$))', text_clean, re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        m2 = re.search(r'(.*?)\s+не превышал[ао]?\s+\d+\.\d+x\s+(.*?)(?=\s+(?:за|для|\.|$))', text_clean, re.IGNORECASE)
        if m2:
            return m2.group(1).strip(), m2.group(2).strip()
        m3 = re.search(r'доля\s+(.*?)\s+в\s+(.*?)(?=\s+(?:не превышала|не превышает|составляла|\b\d+\.\d+x|\.|$))', text_clean, re.IGNORECASE)
        if m3:
            return m3.group(1).strip(), m3.group(2).strip()
        return text_clean, None

    def _extract_definitions(self, clause_text: str) -> tuple:
        """Extract numerator/denominator definitions via LLM with deterministic regex fallback.
        Returns (numerator_def, denominator_def).
        """
        if self.llm_client.is_configured() and clause_text.strip():
            try:
                res = self.llm_client.completion_json(clause_text, system_prompt=COVENANT_ANALYSIS_PROMPT)
                if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
                    res = res[0]
                if isinstance(res, dict):
                    num_def = res.get("numerator_definition") or res.get("numerator") or res.get("metric_definition")
                    den_def = res.get("denominator_definition") or res.get("denominator")
                    if num_def:
                        return num_def, den_def
            except Exception as e:
                logger.error(f"LLM covenant definition extraction failed: {e}")

        # Deterministic regex fallback
        return self._parse_ratio_formula(clause_text)
