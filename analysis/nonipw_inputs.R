options(encoding = "UTF-8", stringsAsFactors = FALSE, warn = 1)

suppressPackageStartupMessages({
    library(data.table)
    library(jsonlite)
})

setDTthreads(1L)

root <- Sys.getenv("HYPERPHOS_RUN")
parent <- file.path(root, "results/analysis")

out <- file.path(parent, "nonIPW_sensitivity")

dir.create(file.path(out, "inputs"), recursive = TRUE, showWarnings = FALSE)

input <- readRDS(file.path(parent, "Development_inputs.rds"))

d <- as.data.table(input$reference)

selection <- as.data.table(input$selection)

stopifnot(nrow(d) == 4575L, uniqueN(d$patient) == 1736L, sum(d$y) == 519L, all(d[, uniqueN(fold), by = patient]$V1 == 
    1L), identical(input$test_labels_read, FALSE))

excluded <- c("potassium", "sodium", "chloride", "calcium", "magnesium", "total_CO2", "bicarbonate", 
    "anion_gap", "calcium_phosphate_product", "phosphate", "diagnostic_y", "NLR", "nlr")

features_by_fold <- setNames(lapply(0:5, function(k) selection[Fold == k & Analysis == "unweighted" & 
    Included == TRUE, Variable]), as.character(0:5))

stopifnot(all(lengths(features_by_fold) > 0L), !length(intersect(unlist(features_by_fold), excluded)))

manifest <- lapply(input$blocks, function(b) {
    features <- features_by_fold[[as.character(b$fold)]]
    z <- readRDS(b$predictive_imputations)
    stopifnot(identical(z$train, b$train), identical(z$query, b$query), !length(intersect(d$patient[b$train], 
        d$patient[b$query])))
    entries <- lapply(seq_len(30L), function(m) {
        a <- as.data.table(z$completed[[m]])[, ..features]
        stopifnot(nrow(a) == length(b$train), all(is.finite(as.matrix(a))), !any(c("y", "label", "diagnostic_y") %in% 
            names(a)))
        a[, `:=`(label = d$y[b$train], sample_weight = 1)]
        ap <- file.path(out, "inputs", sprintf("fold%02d_m%02d_train.csv", b$fold, m))
        fwrite(a, ap)
        aa <- fread(ap)
        stopifnot(identical(names(aa), names(a)), nrow(aa) == nrow(a), max(abs(as.matrix(aa) - as.matrix(a))) < 
            1e-09)
        qp <- NULL
        if (b$fold) {
            q <- as.data.table(z$queries[[m]])[, ..features]
            stopifnot(nrow(q) == length(b$query), all(is.finite(as.matrix(q))), !any(c("y", "label", 
                "sample_weight") %in% names(q)))
            qp <- file.path(out, "inputs", sprintf("fold%02d_m%02d_query.csv", b$fold, m))
            fwrite(q, qp)
            qq <- fread(qp)
            stopifnot(identical(names(qq), names(q)), nrow(qq) == nrow(q), max(abs(as.matrix(qq) - as.matrix(q))) < 
                1e-09)
        }
        list(imputation = m, train = ap, query = qp)
    })
    cats <- intersect(features, c("female", "hypertension", "coronary_disease", "proteinuria_grade"))
    yp <- wp <- NULL
    if (b$fold) {
        yp <- file.path(out, "inputs", sprintf("fold%02d_labels.csv", b$fold))
        fwrite(d[b$query, .(label = y)], yp)
        wp <- file.path(out, "inputs", sprintf("fold%02d_weights.csv", b$fold))
        fwrite(as.data.table(matrix(1, length(b$query), 30L)), wp)
    }
    list(fold = b$fold, features = features, train_indices = b$train, query_indices = b$query, categorical_levels = setNames(lapply(cats, 
        function(v) if (v == "proteinuria_grade") 0:2 else 0:1), cats), query_labels = yp, query_weights = wp, 
        entries = entries)
})

fwrite(d[, .(record, patient, fold, label = y)], file.path(out, "Training_reference.csv"))

w <- as.data.table(matrix(1, nrow(d), 30L))

setnames(w, sprintf("m%02d", 1:30))

fwrite(w, file.path(out, "Training_OOF_weights.csv"))

job <- list(seed = 123L, allocation_seed = 1234L, imputations = 30L, folds = 5L, blocks = manifest, features = features_by_fold[["0"]], 
    excluded = excluded, algorithms = "TabM", primary_analysis = "Unweighted sensitivity; own fitting-partition unweighted LASSO", 
    calibration = "Training cross-fitted Platt decision only", test_labels_read = FALSE)

write_json(job, file.path(out, "Training_job.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)

readback <- fromJSON(file.path(out, "Training_job.json"), simplifyVector = FALSE)

stopifnot(length(readback$blocks) == 6L, identical(unlist(readback$features), features_by_fold[["0"]]), 
    !readback$test_labels_read, all(as.matrix(fread(file.path(out, "Training_OOF_weights.csv"))) == 1))

write_json(list(completed = TRUE, analysis = "Non-IPW sensitivity", train_records = nrow(d), train_patients = uniqueN(d$patient), 
    imputations = 30L, selection_source = "Fitting-partition unweighted LASSO", predictors = features_by_fold, 
    predictive_imputation = "Y-independent fold-specific matrices", test_labels_read = FALSE), 
    file.path(out, "Input_export_complete.json"), pretty = TRUE, auto_unbox = TRUE)

cat("NONIPW_TRAINING_EXPORT_AND_ALL_MATRIX_READBACK_PASS; NO TEST LABELS\n")

