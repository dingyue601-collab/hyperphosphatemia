options(encoding = "UTF-8", stringsAsFactors = FALSE, warn = 1)

suppressPackageStartupMessages({
    library(data.table)
    library(jsonlite)
    library(matrixStats)
})

setDTthreads(1L)

root <- Sys.getenv("HYPERPHOS_RUN")
source(file.path(root,"analysis/common.R"))
out <- file.path(root, "results/analysis")

B <- 5000L

M <- 30L

K <- 5L

metric_names <- c("Accuracy", "Precision", "Sensitivity", "Specificity", "NPV", "F1", "Flagged_fraction", 
    "Youden", "Balanced_accuracy")

models <- c("Logistic", "RF", "SVM", "XGBoost", "LightGBM", "CatBoost", "TabM")

note <- function(...) {
    cat(format(Sys.time(), "%F %T"), ..., "\n")
    flush.console()
}

if (identical(commandArgs(TRUE), "selftest")) {
    y <- c(0, 1, 0, 1, 1, 0)
    p <- c(0.1, 0.8, 0.3, 0.4, 0.8, 0.2)
    w <- cbind(1:6, 6:1)
    cut <- weighted_youden(y, p, w)
    stopifnot(is.finite(cut), is.finite(weighted_youden(y, rep(0.3, 6L), w)))
    cl <- rep(1:3, each = 2L)
    mult <- matrix(c(1, 1, 1, 2, 0, 1, 0, 2, 1), 3L, 3L)
    stacked <- w[, rep(1:2, times = 3L)] * mult[cl, rep(1:3, each = 2L)]
    z <- curve(y, p, stacked)
    cc <- clinical(y, p, stacked, c(Youden = cut))
    for (i in 1:3) {
        manual <- curve(y, p, w * mult[cl, i])
        stopifnot(abs(colMeans(matrix(z$AUC, 2L, 3L))[i] - mean(manual$AUC)) < 1e-12, abs(colMeans(matrix(z$AP, 
            2L, 3L))[i] - mean(manual$AP)) < 1e-12)
        cmanual <- clinical(y, p, w * mult[cl, i], c(Youden = cut))
        stopifnot(max(abs(cc$metrics[, 1L, (2L * i - 1L):(2L * i)] - cmanual$metrics[, 1L, ]), na.rm = TRUE) < 
            1e-12)
    }
    note("WEIGHTED_YOUDEN_AND_PATIENT_BY_MI_BOOTSTRAP_LAYOUT_SELFTEST_PASS")
    quit(save = "no")
}

d <- fread(file.path(out, "Training_reference.csv"))

weights <- as.matrix(fread(file.path(out, "Training_OOF_weights.csv")))

stopifnot(nrow(d) == 4575L, uniqueN(d$patient) == 1736L, sum(d$label) == 519L, identical(dim(weights), 
    c(nrow(d), M)), all(is.finite(weights)), all(weights > 0), all(d[, uniqueN(fold), by = patient]$V1 == 
    1L), !anyDuplicated(d$record))

y <- d$label

calibration <- probabilities <- list()

summaries <- thresholds <- folds <- list()

for (model in models) {
    complete <- fromJSON(file.path(out, model, "Complete.json"))
    pp <- as.matrix(fread(file.path(out, model, "OOF_probabilities.csv")))
    stopifnot(complete$completed, !complete$test_labels_read, complete$imputations == M, identical(dim(pp), 
        c(nrow(d), M)), all(is.finite(pp)), all(pp >= 0 & pp <= 1))
    raw <- rowMeans(pp)
    cal <- choose_platt(y, raw, weights, d$fold)
    p <- cal$probability
    crossfit <- if (is.null(cal$crossfit_probability)) 
        raw
    else cal$crossfit_probability
    cutoff <- weighted_youden(y, p, weights)
    by_fold <- rbindlist(lapply(1:K, function(k) {
        ii <- which(d$fold == k)
        w <- weights[ii, , drop = FALSE]
        z <- curve(y[ii], raw[ii], w)
        a <- probability_scores(y[ii], raw[ii], w)
        b <- probability_scores(y[ii], crossfit[ii], w)
        data.table(Model = model, Fold = k, AUC = mean(z$AUC), AP = mean(z$AP), Raw_Brier = a[["Brier"]], 
            Crossfit_Brier = b[["Brier"]], Raw_log_loss = a[["Log_loss"]], Crossfit_log_loss = b[["Log_loss"]])
    }))
    expected <- if (!is.null(complete$mean_auc)) 
        complete$mean_auc
    else complete$configuration$mean_auc
    if (!is.null(expected)) 
        stopifnot(abs(mean(by_fold$AUC) - expected) < 1e-10)
    a <- clinical(y, p, weights, c(Youden = cutoff))
    count <- function(name) mean(a$counts[[name]])
    threshold <- data.table(Model = model, Threshold = cutoff, Calibration = if (cal$use_platt) 
        "Platt"
    else "None", Weighted_TP = count("TP"), Weighted_FP = count("FP"), Weighted_TN = count("TN"), Weighted_FN = count("FN"), 
        Observed_TP = sum(p >= cutoff & y == 1L), Observed_FP = sum(p >= cutoff & y == 0L), Observed_TN = sum(p < 
            cutoff & y == 0L), Observed_FN = sum(p < cutoff & y == 1L))
    for (v in metric_names) threshold[, `:=`((v), mean(a$metrics[v, 1L, ]))]
    summaries[[model]] <- data.table(Model = model, Mean_fold_AUC = mean(by_fold$AUC), Mean_fold_AP = mean(by_fold$AP), 
        Mean_fold_Brier = mean(by_fold$Crossfit_Brier), Raw_mean_fold_Brier = mean(by_fold$Raw_Brier), 
        Mean_fold_log_loss = mean(by_fold$Crossfit_log_loss), Calibration = if (cal$use_platt) 
            "Platt"
        else "None", Calibration_intercept = cal$coefficients[1L], Calibration_slope = cal$coefficients[2L], 
        Threshold = cutoff, Learning = "Stabilized IPW", Evaluation = "Training fold-specific IPW")
    thresholds[[model]] <- threshold
    folds[[model]] <- by_fold
    calibration[[model]] <- cal
    probabilities[[model]] <- list(raw = raw, probability = p, crossfit = crossfit, threshold = cutoff)
}

summary <- rbindlist(summaries)

setorder(summary, -Mean_fold_AUC)

models <- summary$Model

fwrite(summary, file.path(out, "Training_CV_performance.csv"), bom = TRUE)

fwrite(rbindlist(thresholds)[match(models, Model)], file.path(out, "Training_Youden_performance.csv"), 
    bom = TRUE)

fwrite(rbindlist(folds), file.path(out, "Training_fold_metrics.csv"), bom = TRUE)

saveRDS(list(reference = d, weights = weights, models = probabilities, calibration = calibration), file.path(out, 
    "Training_predictions_and_calibration.rds"))

cluster <- match(d$patient, unique(d$patient))

G <- max(cluster)

set.seed(123L)

multiplicity <- vapply(seq_len(B), function(i) tabulate(sample.int(G, G, replace = TRUE), nbins = G), 
    integer(G))

boot_names <- c("Mean_fold_AUC", "Mean_fold_AP", "Mean_fold_Brier", metric_names)

bootstraps <- list()

intervals <- list()

for (model in models) {
    path <- file.path(out, model, "Patient_bootstrap_5000.rds")
    p <- probabilities[[model]]
    if (file.exists(path)) {
        boot <- readRDS(path)
        stopifnot(identical(dim(boot), c(B, length(boot_names))), all(is.finite(boot[, 1:3])))
    }
    else {
        boot <- matrix(NA_real_, B, length(boot_names), dimnames = list(NULL, boot_names))
        for (start in seq.int(1L, B, 25L)) {
            bb <- start:min(B, start + 24L)
            nb <- length(bb)
            w <- weights[, rep(seq_len(M), times = nb), drop = FALSE] * multiplicity[cluster, rep(bb, 
                each = M), drop = FALSE]
            a <- b <- c <- numeric(nb)
            for (k in 1:K) {
                ii <- which(d$fold == k)
                ww <- w[ii, , drop = FALSE]
                z <- curve(y[ii], p$raw[ii], ww)
                a <- a + colMeans(matrix(z$AUC, M, nb))/K
                b <- b + colMeans(matrix(z$AP, M, nb))/K
                loss <- as.numeric(crossprod(ww, (p$crossfit[ii] - y[ii])^2))/colSums(ww)
                c <- c + colMeans(matrix(loss, M, nb))/K
            }
            boot[bb, 1:3] <- cbind(a, b, c)
            z <- clinical(y, p$probability, w, c(Youden = p$threshold))$metrics
            for (v in metric_names) boot[bb, v] <- colMeans(matrix(z[v, 1L, ], M, nb))
            if (start == 1L || max(bb)%%1000L == 0L) 
                note("PATIENT_BOOTSTRAP", model, max(bb), "OF", B)
        }
        stopifnot(all(is.finite(boot[, 1:3])))
        saveRDS(boot, paste0(path, ".tmp"))
        stopifnot(file.rename(paste0(path, ".tmp"), path))
    }
    bootstraps[[model]] <- boot
    for (v in boot_names) {
        point <- if (v %in% names(summary)) 
            summary[Model == model][[v]]
        else thresholds[[model]][[v]]
        ci <- quantile(boot[, v], c(0.025, 0.975), na.rm = TRUE)
        intervals[[length(intervals) + 1L]] <- data.table(Model = model, Metric = v, Estimate = point, 
            Lower = unname(ci[1L]), Upper = unname(ci[2L]), Bootstrap_replicates = B)
    }
}

fwrite(rbindlist(intervals), file.path(out, "Training_performance_patient_CI.csv"), bom = TRUE)

pairs <- combn(models, 2L, simplify = FALSE)

comparisons <- rbindlist(lapply(pairs, function(pair) {
    a <- pair[1L]
    b <- pair[2L]
    delta <- summary[Model == a, Mean_fold_AUC] - summary[Model == b, Mean_fold_AUC]
    values <- bootstraps[[a]][, "Mean_fold_AUC"] - bootstraps[[b]][, "Mean_fold_AUC"]
    ci <- quantile(values, c(0.025, 0.975))
    se <- sd(values)
    p <- if (se > 0) 
        2 * pnorm(-abs(delta/se))
    else if (delta == 0) 
        1
    else 0
    data.table(Comparison = paste(a, b, sep = " - "), Difference = delta, Lower = unname(ci[1L]), Upper = unname(ci[2L]), 
        Bootstrap_SE = se, P = p)
}))

comparisons[, `:=`(Holm_P, p.adjust(P, method = "holm"))]

stopifnot(nrow(comparisons) == 21L, all(comparisons$Holm_P >= comparisons$P - 1e-15))

fwrite(comparisons, file.path(out, "Training_AUC_pairwise_Holm.csv"), bom = TRUE)

write_json(list(completed = TRUE, algorithms = models, patient_bootstrap_replicates = B, auc_comparisons = 21L, 
    multiplicity = "Holm; paired patient-bootstrap SE normal approximation", feature_selection = "Fitting-fold Y-auxiliary weighted LASSO", 
    calibration = "Training-only cross-fitted Platt decision", evaluation = "Development-side internal CV used for selection; not final-test performance", 
    uncertainty = "Conditional on fixed predictions, calibrators and selection weights", test_labels_read = FALSE, 
    optimism_correction = FALSE), file.path(out, "Training_report_complete.json"), auto_unbox = TRUE, 
    pretty = TRUE)

note("TRAINING_SEVEN_MODELS_CALIBRATION_YOUDEN_AND_BOOTSTRAP_COMPLETE; NO TEST EVALUATION")

