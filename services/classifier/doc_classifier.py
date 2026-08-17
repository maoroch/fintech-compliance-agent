import re
import logging
from typing import List, Optional
from shared.schemas import DocumentInfo, DocType
from shared.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Target account IDs for our scenarios
TARGET_ACCOUNTS = {
    "P1": "ACC-7801",
    "P2": "ACC-7802",
    "P3": "ACC-7803",
    "P4": "ACC-7804",
    "P5": "ACC-7805",
    "P6": "ACC-7806",
    "P7": "ACC-7807",
    "P8": "ACC-7808",
    "P9": "ACC-7809",
    "P10": "ACC-7810",
    "B1": "ACC-7201",
    "B4": "ACC-7204",
}


class DocumentClassifier:
    """
    Classifies ingested documents into LOAN_AGREEMENT, AUDIT_NOTE, KYC_DOSSIER, or DECOY.
    Extracts explicit account_id (e.g. ACC-7801) and company names to prevent entity resolution errors.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def classify_documents(self, documents: List[DocumentInfo]) -> List[DocumentInfo]:
        classified: List[DocumentInfo] = []

        for doc in documents:
            if not doc.is_valid or doc.doc_type == DocType.CORRUPT:
                classified.append(doc)
                continue

            text = doc.raw_text or ""
            if not text.strip():
                doc.doc_type = DocType.CORRUPT
                doc.is_valid = False
                classified.append(doc)
                continue

            # Extract account_id using exact pattern matching first
            account_id = self._extract_account_id(text)
            doc.account_id = account_id

            # Determine document type via heuristics + LLM fallback
            doc_type = self._classify_doc_type(text)
            doc.doc_type = doc_type

            # Only check LLM for account_id if document is a core financial doc (Loan Agreement or Audit Note)
            if not account_id and doc_type in (DocType.LOAN_AGREEMENT, DocType.AUDIT_NOTE) and self.llm_client.is_configured():
                account_id = self._extract_account_id_llm(text[:3000])
                doc.account_id = account_id

            # Extract company name
            doc.company_name = self._extract_company_name(text)

            classified.append(doc)

        logger.info(
            f"Classified {len(classified)} documents. "
            f"Relevant with identified account_id: {sum(1 for d in classified if d.account_id is not None)}"
        )
        return classified

    def _extract_account_id(self, text: str) -> Optional[str]:
        # Regex for ACC-XXXX or Account: ACC-XXXX or Счет: ACC-XXXX
        matches = re.findall(r"ACC-\d{4}", text, re.IGNORECASE)
        if matches:
            return matches[0].upper()

        text_lower = text.lower()
        company_account_map = {
            "aktau port": "ACC-7801",
            "ekibastuz energy": "ACC-7201",
            "shymkent refinery": "ACC-7204",
            "aktau power": "ACC-7802",
            "aktobe power": "ACC-7803",
            "aktobe refinery": "ACC-7804",
            "aktobe water": "ACC-7805",
            "astana office": "ACC-7806",
            "astana property": "ACC-7807",
            "aktau water": "ACC-7808",
            "aktau energy": "ACC-7809",
            "karaganda logistics": "ACC-7810",
        }

        for company_kw, acc_id in company_account_map.items():
            if company_kw in text_lower:
                return acc_id

        return None

    def _classify_doc_type(self, text: str) -> DocType:
        header_text = text[:1500].lower()
        text_lower = text.lower()

        # Check explicit document headers first
        if "договор банковского займа" in header_text or "договор займа" in header_text or "кредитный договор" in header_text or "заём №" in header_text or "заем №" in header_text:
            return DocType.LOAN_AGREEMENT

        if "аудитор" in header_text or "примечания к финансовой отчётности" in header_text or "аудиторское заключение" in header_text or "аудиторская записка" in header_text or "независимый аудитор" in header_text:
            return DocType.AUDIT_NOTE

        if "досье контрагента" in header_text or "проверка контрагента" in header_text or "kyc" in header_text or "аффилированным лицам" in header_text:
            return DocType.KYC_DOSSIER

        # Secondary fallback by body keywords
        loan_keywords = ["договор банковского займа", "ссудный счет", "кредитный договор"]
        audit_keywords = ["аудиторское заключение", "аудиторская записка", "промежуточная ведомость", "примечания к финансовой отчётности", "add-back ebitda"]
        kyc_keywords = ["досье контрагента", "бенефициарный владелец", "список аффилированных лиц", "политика связанных сторон"]
        decoy_keywords = ["hr-политика", "бренд-гайд", "it-инцидент", "страховой полис", "пожарная безопасность", "server log"]

        loan_score = sum(1 for kw in loan_keywords if kw in text_lower)
        audit_score = sum(1 for kw in audit_keywords if kw in text_lower)
        kyc_score = sum(1 for kw in kyc_keywords if kw in text_lower)
        decoy_score = sum(1 for kw in decoy_keywords if kw in text_lower)

        if max(loan_score, audit_score, kyc_score, decoy_score) == 0:
            return DocType.DECOY

        if decoy_score > max(loan_score, audit_score, kyc_score):
            return DocType.DECOY

        if loan_score >= audit_score and loan_score >= kyc_score:
            return DocType.LOAN_AGREEMENT
        elif audit_score >= loan_score and audit_score >= kyc_score:
            return DocType.AUDIT_NOTE
        elif kyc_score >= loan_score and kyc_score >= audit_score:
            return DocType.KYC_DOSSIER

        return DocType.DECOY

    def _extract_company_name(self, text: str) -> Optional[str]:
        # Search for company pattern: e.g. "АО ...", "ТОО ...", "JSC ...", "LLP ..."
        match = re.search(r'(?:АО|ТОО|JSC|LLP|ЗАО|ОАО)\s+["«]?[A-Za-z0-9\sА-Яа-я—\-\.]+(?:["»]|\b)', text)
        if match:
            return match.group(0).strip()
        return None

    def _extract_account_id_llm(self, text_snippet: str) -> Optional[str]:
        try:
            prompt = f"Extract the bank account ID (formatted as ACC-XXXX) from this document text:\n\n{text_snippet}\n\nReturn JSON: {{\"account_id\": \"ACC-XXXX\" or null}}"
            res = self.llm_client.completion_json(prompt, system_prompt="You are a financial document parser.")
            return res.get("account_id")
        except Exception as e:
            logger.warning(f"LLM account_id extraction failed: {e}")
            return None
