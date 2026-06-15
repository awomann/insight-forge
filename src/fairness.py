# fairness.py has one job—detect whether the data and insights treat all segments equally

# IMPORTS

import pandas as pd
import numpy as np


# THRESHOLD
# Segments deviating more than 20% from the overall average get flagged

DISPARITY_THRESHOLD = 0.20


# REGIONAL FAIRNESS
# Checks whether any geographic region deviates significantly from the average for a given metric

def assess_regional_fairness(df: pd.DataFrame, metric: str = "profit") -> dict:
    regional = df.groupby("region")[metric].mean()
    overall_mean = df[metric].mean()
    
    disparities = []
    for region, value in regional.items():
        deviation = abs(value - overall_mean) / overall_mean
        if deviation > DISPARITY_THRESHOLD:
            disparities.append({
                "segment": region,
                "value": round(value, 2),
                "overall_mean": round(overall_mean, 2),
                "deviation_pct": round(deviation * 100, 1),
                "flag": "⚠️ Disparity detected"
            })
    
    return {
        "metric": metric,
        "overall_mean": round(overall_mean, 2),
        "segment_values": regional.round(2).to_dict(),
        "disparities": disparities,
        "is_fair": len(disparities) == 0
    }


# SEGMENT FAIRNESS
# Checks whether Consumer, Corporate, and Home Office segments are served equally

def assess_segment_fairness(df: pd.DataFrame, metric: str = "profit") -> dict:
    by_segment = df.groupby("customer_segment")[metric].mean()
    overall_mean = df[metric].mean()
    
    disparities = []
    for segment, value in by_segment.items():
        deviation = abs(value - overall_mean) / overall_mean
        if deviation > DISPARITY_THRESHOLD:
            disparities.append({
                "segment": segment,
                "value": round(value, 2),
                "overall_mean": round(overall_mean, 2),
                "deviation_pct": round(deviation * 100, 1),
                "flag": "⚠️ Disparity detected"
            })
    
    return {
        "metric": metric,
        "overall_mean": round(overall_mean, 2),
        "segment_values": by_segment.round(2).to_dict(),
        "disparities": disparities,
        "is_fair": len(disparities) == 0
    }

# FULL FAIRNESS CHECK
# Runs all fairness checks and returns a consolidated report

def run_full_fairness_check(df: pd.DataFrame) -> dict:
    regional = assess_regional_fairness(df)
    segment = assess_segment_fairness(df)
    
    all_disparities = regional["disparities"] + segment["disparities"]
    
    return {
        "overall_fair": len(all_disparities) == 0,
        "total_flags": len(all_disparities),
        "checks": {
            "regional": regional,
            "segment": segment
        },
        "summary": (
            "No significant fairness issues detected."
            if len(all_disparities) == 0
            else f"{len(all_disparities)} disparity flag(s) detected. Review before acting on recommendations."
        )
    }
