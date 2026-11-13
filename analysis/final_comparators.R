options(encoding = "UTF-8", stringsAsFactors = FALSE, warn = 1)

Sys.setenv(OMP_NUM_THREADS = "4", MKL_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1")

suppressPackageStartupMessages({
    library(data.table)
    library(jsonlite)
    library(catboost)
})

setDTthreads(1L)

root <- Sys.getenv("HYPERPHOS_RUN")
source(file.path(root,"analysis/common.R"))
out <- file.path(root, "results/analysis")

base <- file.path(root, "results/predictors")

seed <- 123L

device <- fromJSON(file.path(out, "CatBoost", "Device_benchmark.json"))$device
stopifnot(length(device) == 1L, device %in% c("CPU", "GPU"))

load_functions(file.path(root, "analysis/models.R"), 
    "fit_one")

fit_regular <- fit_one

load_functions(file.path(root, "analysis/catboost.R"), 
    c("categories", "fit_one"))

fit_catboost <- fit_one

d <- fread(file.path(out, "Training_reference.csv"))

test <- fread(file.path(out, "Test_reference_without_labels.csv"))

job <- fromJSON(file.path(out, "Training_job.json"), simplifyVector = FALSE)

full <- Filter(function(b) b$fold == 0L, job$blocks)[[1L]]

features <- unlist(full$features)

stopifnot(nrow(d) == 4575L, uniqueN(d$patient) == 1736L, sum(d$label) == 519L, nrow(test) == 2026L, uniqueN(test$patient) == 
    745L, !length(intersect(d$patient, test$patient)), !any(c("y", "label") %in% names(test)), length(features) == 
    7L)

thresholds <- fread(file.path(out, "Training_Youden_performance.csv"))

for (family in c("Logistic", "RF", "SVM", "XGBoost", "LightGBM", "CatBoost")) {
    dest <- file.path(out, "final", paste0(family, "7"))
    dir.create(dest, recursive = TRUE, showWarnings = FALSE)
    complete <- file.path(dest, "Complete.json")
    if (file.exists(complete)) {
        stopifnot(fromJSON(complete)$completed)
        next
    }
    configuration <- fromJSON(file.path(out, family, "Selected_configuration.json"))
    pa <- configuration$parameters
    frames <- list(lapply(1:30, function(m) {
        a <- fread(full$entries[[m]]$train)
        q <- fread(file.path(base, "test_inputs", sprintf("test_m%02d.csv", m)))
        stopifnot(identical(a$label, d$label), !any(c("label", "y", "phosphate") %in% names(q)), nrow(q) == 
            2026L, all(a$sample_weight > 0))
        query <- rbind(as.data.frame(a[, ..features]), as.data.frame(q[, ..features]))
        aa <- as.data.frame(a[, ..features])
        if (family == "CatBoost") {
            x <- categories(aa)
            z <- categories(query)
        } else {
            x <- design(aa)
            z <- design(query)
        }
        stopifnot(identical(colnames(x), colnames(z)))
        list(train = x, query = z, y = d$label, labels = rep(NA_real_, nrow(query)), weight = a$sample_weight/mean(a$sample_weight), 
            center = if (is.matrix(x)) colMeans(x) else NULL, scale = if (is.matrix(x)) apply(x, 2L, 
                sd) else NULL)
    }))
    probability <- matrix(NA_real_, 6601L, 30L)
    for (m in 1:30) {
        cache <- file.path(dest, sprintf("m%02d.rds", m))
        if (file.exists(cache)) 
            p <- readRDS(cache)
        else {
            p <- if (family == "CatBoost") 
                fit_catboost(1L, m, pa)
            else fit_regular(1L, m, pa)
            saveRDS(p, paste0(cache, ".tmp"))
            stopifnot(file.rename(paste0(cache, ".tmp"), cache))
        }
        stopifnot(length(p) == 6601L, all(is.finite(p)), all(p >= 0 & p <= 1))
        probability[, m] <- p
        write_json(list(stage = "Full-training comparator fitting and label-free prediction", algorithm = family, 
            imputation = m, pid = Sys.getpid(), updated = format(Sys.time(), "%F %T"), test_labels_read = FALSE), 
            file.path(out, "Conventional_final_status.json"), pretty = TRUE, auto_unbox = TRUE)
        cat(format(Sys.time(), "%F %T"), "FINAL_COMPARATOR", family, "MI", m, "\n")
        flush.console()
    }
    for (partition in c("Training", "Test")) {
        ii <- if (partition == "Training") 
            1:4575
        else 4575L + 1:2026
        p <- as.data.table(probability[ii, , drop = FALSE])
        setnames(p, sprintf("m%02d", 1:30))
        path <- file.path(dest, paste0(partition, "_probabilities.csv"))
        fwrite(p, path)
        stopifnot(max(abs(as.matrix(fread(path)) - as.matrix(p))) < 1e-09)
    }
    write_json(list(completed = TRUE, family = family, features = features, imputations = 30L, records = list(Training = 4575L, 
        Test = 2026L), learning = "Stabilized training IPW", calibration = thresholds[Model == family, 
        Calibration], threshold = thresholds[Model == family, Threshold], probability_files = "Raw MI probabilities; apply the frozen training OOF calibration after averaging", 
        parameters = pa, runtime_device = if (family == "CatBoost") device else "CPU",
        fitting_seeds = seed + 100L + 1:30, test_labels_read = FALSE), complete, pretty = TRUE,
        auto_unbox = TRUE, digits = NA)
    cat("FINAL_COMPARATOR_COMPLETE", family, "NO TEST LABELS\n")
    flush.console()
}

write_json(list(completed = TRUE, algorithms = c("Logistic", "RF", "SVM", "XGBoost", "LightGBM", "CatBoost"), 
    train_records = 4575L, test_records = 2026L, imputations = 30L, test_labels_read = FALSE), file.path(out, 
    "Conventional_final_complete.json"), pretty = TRUE, auto_unbox = TRUE)

cat("SIX_FIXED_COMPARATOR_FINAL_PREDICTIONS_COMPLETE; NO TEST PERFORMANCE EVALUATION\n")

