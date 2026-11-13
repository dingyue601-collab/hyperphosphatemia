# T2DKD Hyperphosphatemia: Analysis Code and TabM Calculator

This project uses electronic health record data from Guangzhou First People's Hospital to develop models that identify early-admission hyperphosphatemia in hospitalized patients with type 2 diabetic kidney disease. After variable reduction, the final model requires only age and serum creatinine. The outcome is serum phosphate >1.45 mmol/L on admission calendar day 0 or 1.

- Manuscript: A Parsimonious Model for Early Identification of Hyperphosphatemia in Hospitalized Patients With Type 2 Diabetic Kidney Disease
- Repository: [hyperphosphatemia](https://github.com/dingyue601-collab/hyperphosphatemia/)
- Online calculator: [Hyperphosphatemia alert](https://hyperphosphatemiaclem.streamlit.app/)

## Repository structure

```text
analysis/               Imputation, IPW, selection, training, evaluation and SHAP
calculator/             Bilingual Streamlit interface and inference code
calculator/models/      30 fitted two-variable TabM models and preprocessors
calculator/vendor/      Third-party implementations and licenses
run_analysis.py         Numerical analysis entry point
Data_schema.md          Processed input structure and variable definitions
```

The public code starts from research data prepared and deidentified within the hospital. It does not include hospital data extraction, identity linkage, patient records, or figure and table production code.

## Environment

The analyses used R 4.5.2 and Python 3.12.14. Dependencies are listed in `analysis/Dependencies.R` and `requirements-analysis.txt`; calculator dependencies are listed separately in `calculator/requirements.txt`. The bundled implementations are TabM 0.0.3 and rtdl_num_embeddings 0.0.12. TabM training used CUDA 12.6, 32 members and float32.

## Core analysis code

| Stage | Contents |
|---|---|
| `prepare` | Multiple imputation, measurement IPW and patient-grouped LASSO |
| `benchmark` | Seven-algorithm comparison, tuning and calibration selection |
| `reduce` | Subset refitting and variable reduction |
| `final` | Final model fitting, two-input logistic reference and non-IPW sensitivity model |
| `evaluate` | Test performance, patient-cluster bootstrap and sensitivity analyses |
| `explain` | Exact empirical SHAP for the two-variable TabM ensemble probability |

Patients were split approximately 7:3 (seed 1234), with all admissions from the same patient kept in one partition. Development used patient-grouped five-fold cross-validation (seed 123) and 30 imputations with 20 iterations each. LASSO retained variables selected in at least 18 imputations. Predictive imputation did not use outcomes. Measurement models were fitted only in the training partition and then applied to held-out-fold or test records.

TabM parameter tuning evaluated 30 trials and confirmed the two leading configurations. Variable reduction screened all 127 nonempty subsets on imputations 1, 7, 13, 19 and 25, with at most 11 candidates confirmed on 30 imputations. The selected TabM configuration was refitted for 14 epochs; the non-IPW sensitivity model used 18 epochs. Training performance is the mean of held-out-fold metrics. Calibration and the maximum weighted Youden threshold were determined from training out-of-fold predictions and held fixed during testing. The 5,000 patient-cluster bootstrap samples used fixed predictions.

The supplementary logistic reference uses the same age and creatinine inputs, patient split, imputations and IPW, with linear log-odds terms. Its training-side calibration decision and Youden threshold are determined separately.

With the processed inputs described in `Data_schema.md`, run the analysis in a private working directory:

```sh
python run_analysis.py --data /path/to/processed_inputs --workspace /path/to/work --rscript /path/to/Rscript --stage all
```

## Calculator

The calculator includes 30 TabM models and their preprocessors. Inputs are age in years and observed serum creatinine in μmol/L or mg/dL, using a conversion factor of 88.4.

From the repository root, run the following in a Python 3.12 environment with the calculator dependencies installed:

```sh
python -m streamlit run calculator/app.py --server.address 127.0.0.1 --server.port 8501
```

The interface defaults to English and loads a hypothetical example. It supports Chinese, unit switching and clearing inputs, and does not store inputs. Prediction first averages member probabilities within each model, then averages probabilities across the 30 models. The screening threshold is 12.92%; classification uses the unrounded value `0.129243104966978`.

Run `python -X utf8 -B calculator/test_app.py` to check interactions, including unit deselection and conversion.

## License

The project code is licensed under [Apache-2.0](LICENSE). The bundled TabM and numerical-embedding implementations retain their upstream licenses and notices; other dependencies retain their respective licenses. Load model and preprocessing files only from trusted sources.

