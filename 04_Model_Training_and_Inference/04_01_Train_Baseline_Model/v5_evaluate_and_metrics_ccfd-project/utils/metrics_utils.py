"""
Credit Card Fraud Detection - Metrics Utilities (Basics)

Functions to evaluate ML models using BOTH:
1. Technical metrics (F1, Precision, Recall) - for data scientists
2. Business metrics (Net Benefit in $$$) - for executives/stakeholders

Functions:
- calculate_business_metrics: Convert confusion matrix to business ROI
- print_metrics: Comprehensive performance report
- print_confusion_matrix: Readable confusion matrix display
- save_metrics_json: Save metrics to JSON file
"""


import json
import os
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# Business Constants
DEFAULT_AVG_FRAUD_AMOUNT = 1500      # Average amount stolen per fraud ($)
DEFAULT_INVESTIGATION_COST = 25      # Cost to investigate one transaction ($)


def _print_header(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")


def calculate_business_metrics(y_true, y_pred, avg_fraud_amount=None, investigation_cost=None):
    if avg_fraud_amount is None:
        avg_fraud_amount = DEFAULT_AVG_FRAUD_AMOUNT
    if investigation_cost is None:
        investigation_cost = DEFAULT_INVESTIGATION_COST

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    benefit_tp = tp * (avg_fraud_amount - investigation_cost)
    cost_fp = fp * investigation_cost
    cost_fn = fn * avg_fraud_amount
    net_benefit = benefit_tp - cost_fp - cost_fn

    return {
        'net_benefit': net_benefit,
        'fraud_caught': tp,
        'fraud_missed': fn,
        'false_alarms': fp,
        'true_negatives': tn,
        'total_saved': benefit_tp,
        'total_cost': cost_fp + cost_fn,
        'confusion_matrix': cm
    }


def print_confusion_matrix(cm):
    print("\nCONFUSION MATRIX:")
    print(f"                Predicted")
    print(f"                Legit  Fraud")
    print(f"   Actual Legit  {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"   Actual Fraud  {cm[1][0]:4d}  {cm[1][1]:4d}")


def print_metrics(algorithm_name, y_true, y_pred, y_pred_proba=None,
                  avg_fraud_amount=None, investigation_cost=None):
    _print_header(f"{algorithm_name} - PERFORMANCE METRICS")

    # Technical Metrics
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print("\nTECHNICAL METRICS:")
    print(f"   F1 Score:     {f1:.4f}")
    print(f"   Precision:    {precision:.4f}")
    print(f"   Recall:       {recall:.4f}")

    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        print(f"   ROC AUC:      {roc_auc:.4f}")
    else:
        roc_auc = None

    # Business Metrics
    business_metrics = calculate_business_metrics(
        y_true, y_pred, avg_fraud_amount, investigation_cost
    )

    print("\nBUSINESS METRICS:")
    print(f"   Net Benefit:    ${business_metrics['net_benefit']:,.0f}")
    print(f"   Fraud Caught:   {business_metrics['fraud_caught']}")
    print(f"   Fraud Missed:   {business_metrics['fraud_missed']}")
    print(f"   False Alarms:   {business_metrics['false_alarms']}")

    # Confusion Matrix
    print_confusion_matrix(business_metrics['confusion_matrix'])

    # Classification Report
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraud']))

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'business_metrics': business_metrics
    }

