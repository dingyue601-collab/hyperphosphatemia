options(encoding = "UTF-8", stringsAsFactors = FALSE, warn = 1)

Sys.setenv(OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1")

suppressPackageStartupMessages({
    library(data.table)
    library(jsonlite)
    library(ranger)
    library(glmnet)
    library(splines)
    library(MASS)
    library(nnet)
})

setDTthreads(1L)

root <- Sys.getenv("HYPERPHOS_RUN")
source(file.path(root,"analysis/common.R"))
out <- file.path(root, "results/analysis")

base <- file.path(root, "results/predictors")

cohort <- file.path(root, "data/cleaned")

dest <- file.path(out, "test_IPW")

dir.create(dest, showWarnings = FALSE)

iterations <- 20L

binary_variables <- c("female", "hypertension", "coronary_disease", "tested_early")

ordered_variables <- "proteinuria_grade"

input <- readRDS(file.path(cohort, "Development_inputs.rds"))

test <- as.data.table(input$validation_x)

train <- as.data.table(input$training)

excluded <- c("potassium", "sodium", "chloride", "calcium", "magnesium", "total_CO2", "bicarbonate", 
    "anion_gap", "calcium_phosphate_product", "phosphate", "first_phosphate", "NLR", "nlr")

predictors <- setdiff(input$candidates, excluded)

stopifnot(nrow(test) == 2026L, uniqueN(test$patient) == 745L, length(predictors) == 26L, !any(c("y", 
    "label", "phosphate", "first_phosphate") %in% names(test)), !length(intersect(train$patient, test$patient)))

variables <- c("age", "female", "year", "creatinine", "SBP", "DBP", "proteinuria_grade", "hypertension", 
    "coronary_disease")

source <- as.data.table(readRDS(file.path(cohort, "Testing_target_frame.rds")))[, c("record", "patient", 
    "tested_early", variables), with = FALSE]

membership <- readRDS(file.path(cohort, "Membership.rds"))

stopifnot(nrow(source) == 14361L, identical(source$record, membership$record))

target <- source[membership$partition == "Validation"]

training_target <- source[membership$partition == "Training"]

stopifnot(nrow(target) == 4436L, sum(target$tested_early) == nrow(test), setequal(target[tested_early == 
    1L, record], test$record), !length(intersect(target$patient, training_target$patient)))

selected <- match(test$record, target$record)

stopifnot(!anyNA(selected), identical(target$patient[selected], test$patient), all(target$tested_early[selected] == 
    1L))

raw <- as.data.frame(target[, c(variables, "tested_early"), with = FALSE])

weights <- matrix(NA_real_, nrow(test), 30L)

diagnostics <- vector("list", 30L)

for (m in seq_len(30L)) {
    started <- Sys.time()
    fitted <- readRDS(file.path(out, "weights/fold00", sprintf("m%02d.rds", m)))
    stopifnot(identical(fitted$variables, variables), identical(fitted$record, train$record), !any(c(excluded, 
        "diagnostic_y", "y", "label") %in% fitted$imputer$columns), abs(fitted$numerator - mean(training_target$tested_early)) < 
        1e-12)
    cache <- file.path(dest, sprintf("m%02d.rds", m))
    if (file.exists(cache)) {
        result <- readRDS(cache)
        stopifnot(identical(result$record, test$record), identical(result$target_record, target$record))
        check_observed(result$completed_target, raw)
    }
    else {
        completed <- impute_query(fitted$imputer, raw, 550000L + m)
        design <- selection_design(completed, variables, fitted$basis)
        probability <- drop(predict(fitted$model, design$x, type = "response", s = 0.01))
        stopifnot(all(is.finite(probability)), all(probability > 0 & probability < 1))
        result <- list(imputation = m, record = test$record, target_record = target$record, probability = probability, 
            completed_target = completed, numerator = fitted$numerator, weights = fitted$numerator/probability[selected], 
            test_labels_read = FALSE, fitted_source = "Frozen full-training MICE conditional models, spline bases, ridge model and numerator")
        saveRDS(result, paste0(cache, ".tmp"))
        stopifnot(file.rename(paste0(cache, ".tmp"), cache))
    }
    stopifnot(length(result$weights) == nrow(test), all(is.finite(result$weights)), all(result$weights > 
        0), max(abs(result$weights - fitted$numerator/result$probability[selected])) < 1e-12)
    weights[, m] <- result$weights
    diagnostics[[m]] <- data.table(Imputation = m, Target_admissions = nrow(target), Tested_admissions = nrow(test), 
        ESS_admissions = sum(result$weights)^2/sum(result$weights^2), Weight_min = min(result$weights), 
        Weight_median = median(result$weights), Weight_max = max(result$weights), Testing_probability_min = min(result$probability), 
        Testing_probability_max = max(result$probability))
    completed_x <- readRDS(file.path(base, "test_inputs", sprintf("test_m%02d.rds", m)))
    stopifnot(setequal(names(completed_x), predictors))
    check_observed(completed_x, as.data.frame(test[, names(completed_x), with = FALSE]))
    csv <- fread(file.path(base, "test_inputs", sprintf("test_m%02d.csv", m)))
    stopifnot(max(abs(as.matrix(csv) - as.matrix(completed_x[, names(csv), drop = FALSE]))) < 1e-09)
    write_json(list(stage = "Applying frozen training selection model to test covariates", imputation = m, 
        pid = Sys.getpid(), updated = format(Sys.time(), "%F %T"), test_labels_read = FALSE), file.path(dest, 
        "Run_status.json"), pretty = TRUE, auto_unbox = TRUE)
    cat(format(Sys.time(), "%F %T"), "TEST_IPW_QUERY_COMPLETE", m, "SECONDS", round(as.numeric(difftime(Sys.time(), 
        started, units = "secs")), 2), "\n")
    flush.console()
}

w <- as.data.table(weights)

setnames(w, sprintf("m%02d", 1:30))

fwrite(w, file.path(out, "Test_IPW_weights.csv"))

fwrite(test[, .(record, patient)], file.path(out, "Test_reference_without_labels.csv"))

fwrite(rbindlist(diagnostics), file.path(dest, "Test_IPW_diagnostics.csv"), bom = TRUE)

check <- as.matrix(fread(file.path(out, "Test_IPW_weights.csv")))

stopifnot(identical(dim(check), c(2026L, 30L)), max(abs(check - weights)) < 1e-09)

write_json(list(completed = TRUE, records = nrow(test), patients = uniqueN(test$patient), target_records = nrow(target), 
    imputations = 30L, refitting_on_test = FALSE, truncation = "None", test_labels_read = FALSE, predictive_test_matrices_checked = 30L), 
    file.path(dest, "Complete.json"), pretty = TRUE, auto_unbox = TRUE)

cat("FROZEN_TRAINING_SELECTION_MODEL_TEST_WEIGHTS_COMPLETE; NO DIAGNOSTIC TEST LABELS\n")

