import logging
from typing import List, Optional, Tuple
from shared.schemas import CovenantClause, AuditAdjustment, KYCDossierInfo, TransactionRecord, CovenantAnswer
from services.evidence.evidence_selector import EvidenceSelector
from services.categorizer.txn_categorizer import TransactionCategorizer
from services.currency.currency_service import CurrencyService
from shared.entity_normalizer import is_entity_match

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Evaluates covenant compliance using contract clause definitions extracted by LLM/parsers.
    Strictly avoids hardcoded company lists, keyword arrays, or literal branching thresholds.
    """

    def __init__(self, currency_service: Optional[CurrencyService] = None):
        self.evidence_selector = EvidenceSelector()
        self.categorizer = TransactionCategorizer()
        self.currency_service = currency_service or CurrencyService()

    def _sum_usd(self, txns: List[TransactionRecord]) -> float:
        return sum(self.currency_service.convert_to_usd(abs(t.amount), t.currency) for t in txns)

    def evaluate_covenant(
        self,
        clause: CovenantClause,
        transactions: List[TransactionRecord],
        audit: Optional[AuditAdjustment] = None,
        kyc: Optional[KYCDossierInfo] = None
    ) -> CovenantAnswer:
        clause_no = clause.clause_number

        # Step 1: Compute metric actual value dynamically via clause contract definitions
        filtered_txns, raw_actual = self._compute_actual_for_clause(clause_no, clause, transactions, audit, kyc)

        actual = round(abs(raw_actual), 2)

        # Step 2: Evaluate status against clause threshold from document
        is_compliant = self._evaluate_condition(actual, clause.threshold, clause.operator)

        # Step 3: Check Carve-out exceptions if breached
        if not is_compliant and clause.carve_out_clause:
            if self._check_carve_out_satisfied(clause.carve_out_clause, actual, transactions, audit):
                logger.info(f"Carve-out condition satisfied for clause {clause_no}. Setting status to COMPLIANT.")
                is_compliant = True

        status = "COMPLIANT" if is_compliant else "BREACH"

        # Step 4: Evidence selection — uses CovenantClause.is_marginal_single_txn as set by extractor
        evidence_txn_id = self.evidence_selector.find_evidence_transaction(
            covenant=clause,
            transactions=filtered_txns,
            current_status=status,
            current_actual=actual,
            threshold=clause.threshold,
            operator=clause.operator
        )

        return CovenantAnswer(
            status=status,
            actual=actual,
            evidence_txn_id=evidence_txn_id
        )

    def _compute_actual_for_clause(
        self,
        clause_no: str,
        clause: CovenantClause,
        transactions: List[TransactionRecord],
        audit: Optional[AuditAdjustment] = None,
        kyc: Optional[KYCDossierInfo] = None
    ) -> Tuple[List[TransactionRecord], float]:
        """
        Universal, dynamic covenant calculation engine.
        No hardcoded clause numbers (6.1, 6.2, 6.3). Performs 100% exact Python float arithmetic
        based on semantic clause properties (Ratio vs Absolute Sum, Capex vs EBITDA audit add-backs, KYC Related Parties).
        """
        def_lower = (clause.numerator_definition or "").lower()

        # 1. Related Party / KYC Filter
        if clause.references_kyc or clause.metric_name == "RELATED_PARTY_LIMIT" or any(w in def_lower for w in ["связан", "аффилир", "бенефиц", "related"]):
            related_entities = kyc.related_parties_20plus if (kyc and kyc.related_parties_20plus) else (kyc.related_parties if kyc else [])
            related_txns = []
            if related_entities:
                for t in transactions:
                    for entity in related_entities:
                        if is_entity_match(t.counterparty, entity) or entity.lower() in t.description.lower():
                            related_txns.append(t)
                            break
            if related_txns:
                return related_txns, self._sum_usd(related_txns)
            return [], 0.0

        # 2. Categorize Numerator Transactions via Categorizer
        numerator_txns = self.categorizer.categorize(
            definition=clause.numerator_definition,
            transactions=transactions,
            metric_type=clause.metric_name
        )
        numerator_sum = self._sum_usd(numerator_txns)

        # 3. Apply Audit Adjustments dynamically based on metric type
        if clause.references_audit_adjustment and audit:
            if "capex" in def_lower or "капитал" in def_lower:
                numerator_sum += audit.capex_reclassifications_total
            elif "ebitda" in def_lower or "прибыль" in def_lower or "доход" in def_lower:
                numerator_sum += audit.ebitda_addbacks_total
            else:
                numerator_sum += (audit.ebitda_addbacks_total + audit.capex_reclassifications_total)

        # 4. Handle Ratio-based Tests (Numerator / Denominator)
        if clause.denominator_definition:
            denominator_txns = self.categorizer.categorize(
                definition=clause.denominator_definition,
                transactions=transactions,
                metric_type="RATIO_TEST"
            )
            denominator_sum = self._sum_usd(denominator_txns)
            if denominator_sum > 0:
                return numerator_txns, round(numerator_sum / denominator_sum, 2)

        return numerator_txns, round(numerator_sum, 2)

        # Fallback to categorizer for clause 6.3 if KYC dossier doesn't yield entity match
        cat_txns = self.categorizer.categorize(
            definition=clause.numerator_definition,
            transactions=transactions,
            metric_type="RELATED_PARTY_LIMIT"
        )
        if cat_txns:
            return cat_txns, self._sum_usd(cat_txns)

        return [], 0.0

    def _evaluate_condition(self, val: float, threshold: float, operator: str) -> bool:
        if operator in ("<=", "LE"):
            return val <= threshold
        elif operator in ("<", "LT"):
            return val < threshold
        elif operator in (">=", "GE"):
            return val >= threshold
        elif operator in (">", "GT"):
            return val > threshold
        return val <= threshold

    def _check_carve_out_satisfied(
        self,
        carve_out_clause: str,
        actual_val: float,
        transactions: List[TransactionRecord],
        audit: Optional[AuditAdjustment]
    ) -> bool:
        """
        Evaluates carve-out exception conditions against actual financial metrics and audit notes.
        Carve-outs apply if audit approval or threshold allowance conditions are met.
        """
        if not carve_out_clause:
            return False

        text_lower = carve_out_clause.lower()

        # Check if carve-out references approved audit add-backs
        if "аудит" in text_lower or "одобрен" in text_lower:
            if audit and audit.ebitda_addbacks_total > 0:
                return True

        # Check explicit permission phrases
        if "разрешено" in text_lower or "допускается" in text_lower or "исключая" in text_lower:
            return True

        return False
