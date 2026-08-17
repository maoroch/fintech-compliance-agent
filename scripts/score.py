import os
import json
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def score_submission(submission_path: str = "submission.json", ground_truth_path: str = "docs/agentic-bank-public/ground_truth.json"):
    if not os.path.exists(submission_path):
        logger.error(f"Submission file not found: {submission_path}")
        return 0.0

    if not os.path.exists(ground_truth_path):
        logger.error(f"Ground truth file not found: {ground_truth_path}")
        return 0.0

    with open(submission_path, "r", encoding="utf-8") as f:
        sub_data = json.load(f)

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    gt_scenarios = gt_data.get("scenarios", {})
    sub_answers = sub_data.get("answers", {})

    total_possible = len(gt_scenarios) * 3  # 36 cells
    total_score = 0.0
    cell_details = []

    for scenario_id, gt_scen in gt_scenarios.items():
        gt_covs = gt_scen.get("covenants", {})
        sub_covs = sub_answers.get(scenario_id, {})

        for clause_no, gt_cell in gt_covs.items():
            sub_cell = sub_covs.get(clause_no, {})

            gt_status = gt_cell.get("status")
            gt_actual = float(gt_cell.get("actual", 0.0))
            gt_evidence = gt_cell.get("evidence_txn_id")

            sub_status = sub_cell.get("status")
            sub_actual = float(sub_cell.get("actual", 0.0)) if sub_cell.get("actual") is not None else None
            sub_evidence = sub_cell.get("evidence_txn_id")

            # 1. Status Evaluation (0.50 points)
            if sub_status != gt_status:
                cell_score = 0.0
                cell_details.append({
                    "cell": f"{scenario_id}:{clause_no}",
                    "status_match": False,
                    "actual_error": None,
                    "evidence_match": False,
                    "score": 0.0
                })
                continue

            status_score = 0.50

            # 2. Actual Evaluation (0.30 points)
            if sub_actual is None:
                rel_error = 1.0
            elif gt_actual == 0.0:
                rel_error = 0.0 if abs(sub_actual) < 0.01 else 1.0
            else:
                rel_error = abs(sub_actual - gt_actual) / abs(gt_actual)

            actual_factor = max(0.0, 1.0 - (rel_error / 0.05))
            actual_score = 0.30 * actual_factor

            # 3. Evidence Evaluation (0.20 points)
            if gt_evidence is not None:
                evidence_score = 0.20 if sub_evidence == gt_evidence else 0.0
            else:
                # If ground truth evidence is null, evidence points decay with actual error
                evidence_score = 0.20 * actual_factor

            cell_score = status_score + actual_score + evidence_score
            total_score += cell_score

            cell_details.append({
                "cell": f"{scenario_id}:{clause_no}",
                "status_match": True,
                "actual_error": round(rel_error, 4),
                "evidence_match": (sub_evidence == gt_evidence),
                "score": round(cell_score, 4)
            })

    normalized_score = (total_score / total_possible) * 100.0

    print("=" * 60)
    print(f"LOCAL SCORE EVALUATION (Ground Truth Benchmark)")
    print("=" * 60)
    for cd in cell_details:
        print(f"  Cell {cd['cell']:<12}: StatusMatch={cd['status_match']} | Score={cd['score']:.2f}")
    print("=" * 60)
    print(f"Scenarios evaluated : {len(gt_scenarios)}")
    print(f"Total cells          : {total_possible}")
    print(f"Raw Points Earned    : {total_score:.2f} / {total_possible:.2f}")
    print(f"Accuracy Score       : {normalized_score:.2f}%")
    print("=" * 60)

    return normalized_score


if __name__ == "__main__":
    sub_p = sys.argv[1] if len(sys.argv) > 1 else "submission.json"
    gt_p = sys.argv[2] if len(sys.argv) > 2 else "docs/agentic-bank-public/ground_truth.json"
    score_submission(sub_p, gt_p)
