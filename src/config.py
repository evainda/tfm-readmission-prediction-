from pathlib import Path

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data folders
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Target variable
TARGET_VARIABLE = "readmission"

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Dataset files
FILES = {
    "patients": DATA_RAW / "mimic_iv" / "hosp" / "patients.csv.gz",
    "admissions": DATA_RAW / "mimic_iv" / "hosp" / "admissions.csv.gz",
    "diagnoses": DATA_RAW / "mimic_iv" / "hosp" / "diagnoses_icd.csv.gz",
    "procedures": DATA_RAW / "mimic_iv" / "hosp" / "procedures_icd.csv.gz",
    "prescriptions": DATA_RAW / "mimic_iv" / "hosp" / "prescriptions.csv.gz"
}