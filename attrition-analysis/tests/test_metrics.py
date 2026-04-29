import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "department": ["Sales", "Sales", "HR", "HR", "IT", "IT"],
            "overtime": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "monthly_income": [3000, 6000, 4000, 8000, 5000, 7000],
            "job_satisfaction": [1, 4, 2, 4, 1, 3],
            "attrition": ["Yes", "No", "Yes", "No", "Yes", "No"],
        }
    )


# --- attrition_rate ---

def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    assert attrition_rate(df) == 50.0


def test_attrition_rate_all_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


def test_attrition_rate_no_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


def test_attrition_rate_single_leaver():
    df = pd.DataFrame({"employee_id": [1], "attrition": ["Yes"]})
    assert attrition_rate(df) == 100.0


# --- attrition_by_department ---

def test_attrition_by_department_columns(sample_df):
    result = attrition_by_department(sample_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_rates(sample_df):
    result = attrition_by_department(sample_df)
    sales = result[result["department"] == "Sales"].iloc[0]
    assert sales["employees"] == 2
    assert sales["leavers"] == 1
    assert sales["attrition_rate"] == 50.0


def test_attrition_by_department_sorted_descending(sample_df):
    result = attrition_by_department(sample_df)
    rates = result["attrition_rate"].tolist()
    assert rates == sorted(rates, reverse=True)


def test_attrition_by_department_all_leavers():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2],
            "department": ["Sales", "Sales"],
            "attrition": ["Yes", "Yes"],
        }
    )
    result = attrition_by_department(df)
    assert result.iloc[0]["attrition_rate"] == 100.0


# --- attrition_by_overtime ---

def test_attrition_by_overtime_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_rates(sample_df):
    result = attrition_by_overtime(sample_df)
    overtime_yes = result[result["overtime"] == "Yes"].iloc[0]
    overtime_no = result[result["overtime"] == "No"].iloc[0]
    assert overtime_yes["employees"] == 3
    assert overtime_yes["leavers"] == 3
    assert overtime_yes["attrition_rate"] == 100.0
    assert overtime_no["employees"] == 3
    assert overtime_no["leavers"] == 0
    assert overtime_no["attrition_rate"] == 0.0


def test_attrition_by_overtime_no_overtime_group():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2],
            "overtime": ["No", "No"],
            "attrition": ["Yes", "No"],
        }
    )
    result = attrition_by_overtime(df)
    assert len(result) == 1
    assert result.iloc[0]["attrition_rate"] == 50.0


# --- average_income_by_attrition ---

def test_average_income_by_attrition_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values(sample_df):
    result = average_income_by_attrition(sample_df)
    leavers_income = result[result["attrition"] == "Yes"].iloc[0]["avg_monthly_income"]
    stayers_income = result[result["attrition"] == "No"].iloc[0]["avg_monthly_income"]
    # leavers: 3000, 4000, 5000 → mean 4000.0
    # stayers: 6000, 8000, 7000 → mean 7000.0
    assert leavers_income == 4000.0
    assert stayers_income == 7000.0


def test_average_income_by_attrition_leavers_earn_less(sample_df):
    result = average_income_by_attrition(sample_df)
    leavers_income = result[result["attrition"] == "Yes"].iloc[0]["avg_monthly_income"]
    stayers_income = result[result["attrition"] == "No"].iloc[0]["avg_monthly_income"]
    assert leavers_income < stayers_income


# --- satisfaction_summary ---

def test_satisfaction_summary_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_sorted_by_satisfaction(sample_df):
    result = satisfaction_summary(sample_df)
    scores = result["job_satisfaction"].tolist()
    assert scores == sorted(scores)


def test_satisfaction_summary_rate_uses_group_headcount(sample_df):
    # satisfaction 1: 2 employees (IDs 1 and 5), both left → rate should be 100%
    # satisfaction 4: 2 employees (IDs 2 and 4), none left → rate should be 0%
    result = satisfaction_summary(sample_df)
    rate_1 = result[result["job_satisfaction"] == 1].iloc[0]["attrition_rate"]
    rate_4 = result[result["job_satisfaction"] == 4].iloc[0]["attrition_rate"]
    assert rate_1 == 100.0
    assert rate_4 == 0.0


def test_satisfaction_summary_rate_not_share_of_total_leavers(sample_df):
    # Guard against the old bug: rate must never exceed 100
    result = satisfaction_summary(sample_df)
    assert (result["attrition_rate"] <= 100.0).all()
    assert (result["attrition_rate"] >= 0.0).all()


def test_satisfaction_summary_partial_attrition():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "job_satisfaction": [2, 2, 3, 3],
            "attrition": ["Yes", "No", "Yes", "No"],
        }
    )
    result = satisfaction_summary(df)
    for _, row in result.iterrows():
        assert row["attrition_rate"] == 50.0
