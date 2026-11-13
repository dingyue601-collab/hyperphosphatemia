# Processed inputs

Four RDS files are required. They are created within the institution and are
not included in this repository. No name, medical-record number, EMPI, visit
identifier, birth date, free text or contact information is required.

| File | Contents |
|---|---|
| `Development_inputs.rds` | List: `training`, `validation_x`, `candidates`. Training has `record`, `patient`, `fold`, `y` and 26 candidates. Test input has the keys and candidates, without outcomes. |
| `Testing_target_frame.rds` | All clinically eligible admissions: anonymous keys, `tested_early` and measurement-model covariates. No phosphate concentration or diagnostic outcome. |
| `Membership.rds` | `record`, `partition`, `fold`, in target-frame order. Partitions are `Training` and `Validation`; the latter is reported as Test. |
| `Validation_labels.rds` | Anonymous `record`, `patient`, binary `y` and `phosphate` in mmol/L. Read only by terminal evaluation. |

`record` is a unique anonymous admission key; `patient` groups repeat admissions.
Fold values 1–5 identify training patients. Every patient's admissions belong to
one partition and one development fold. Test predictors retain the input order.
The study dimensions are 4,575 training admissions (1,736 patients), 2,026 test
admissions (745 patients), and 14,361 eligible target admissions. The scripts
check these dimensions when reproducing this study.

Candidate names and order are specified by `candidates` and preserved in all
imputation matrices. Binary fields use 0/1; proteinuria uses 0/1/2. Continuous
fields are numeric, with missing values represented by `NA`. Age is in years;
creatinine and uric acid are in μmol/L; hemoglobin is in g/L; blood pressure is
in mmHg. Proteinuria is the same-admission dipstick category. Glucose remains
an imputation auxiliary but is not selected into the final prediction or
measurement model.

Measurement-model prescreening includes age, female sex, admission year,
creatinine, SBP, DBP, HbA1c, hemoglobin, white blood cells, platelets, albumin,
proteinuria, hypertension, coronary disease and uric acid. The nine covariates
with no more than 30% missingness in the training target are retained.
Continuous covariates use training-derived spline bases; stabilized weights
are not truncated. Predictive imputation does not use diagnostic outcomes.

Stored partition/fold assignments are part of the processed inputs and should
not be regenerated during numerical replay.
