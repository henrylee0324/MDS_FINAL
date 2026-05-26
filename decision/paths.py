"""Central paths for the decision demo pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'

_CSV_NAME = 'head200_MSD_with_all_features_categorical_encoded.csv'
_CSV_CANDIDATES = [
    DATA_DIR / _CSV_NAME,
    ROOT / 'sample_data' / _CSV_NAME,
    WORKSPACE_ROOT / _CSV_NAME,
    PROJECT_ROOT / _CSV_NAME,
]


def resolve_head200_csv(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    for p in _CSV_CANDIDATES:
        if p.exists():
            return p
    return _CSV_CANDIDATES[0]


DEFAULT_CSV = _CSV_CANDIDATES[0]
MOCK_CSV = DATA_DIR / 'head200_mock_for_demo.csv'

SYNTHETIC_CSV = OUTPUT_DIR / 'synthetic_new_songs.csv'
SCORES_CSV = OUTPUT_DIR / 'song_scores.csv'
MODEL_PKL = OUTPUT_DIR / 'hgb_scorer.pkl'

KNAPSACK_SOLUTION_CSV = OUTPUT_DIR / 'knapsack_solution.csv'
KNAPSACK_REPORT_TXT = OUTPUT_DIR / 'knapsack_report.txt'
MULTIPERIOD_SOLUTION_CSV = OUTPUT_DIR / 'knapsack_multperiod_solution.csv'
MULTIPERIOD_REPORT_TXT = OUTPUT_DIR / 'knapsack_multperiod_report.txt'
STRATEGY_REPORT_MD = OUTPUT_DIR / 'strategy_report.md'
STRATEGY_SUMMARY_CSV = OUTPUT_DIR / 'strategy_summary.csv'
SENSITIVITY_CSV = OUTPUT_DIR / 'strategy_budget_sensitivity.csv'
STRATEGY_EXCEL = OUTPUT_DIR / 'HitSong_策略決策報告.xlsx'
COMPARISON_CSV = OUTPUT_DIR / 'decision_comparison.csv'
CONSTRAINT_CHECK_CSV = OUTPUT_DIR / 'constraint_check.csv'
MULTIPERIOD_SOLUTION_CSV = OUTPUT_DIR / 'knapsack_multperiod_solution.csv'
