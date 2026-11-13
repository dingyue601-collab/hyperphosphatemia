options(encoding = "UTF-8", stringsAsFactors = FALSE, warn = 1)

Sys.setenv(OMP_NUM_THREADS = "4", MKL_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1")

suppressPackageStartupMessages({
    library(data.table)
    library(jsonlite)
    library(matrixStats)
})

setDTthreads(1L)

root <- Sys.getenv("HYPERPHOS_RUN")
source(file.path(root,"analysis/common.R"))
out <- file.path(root, "results/analysis")

python <- Sys.getenv("HYPERPHOS_PYTHON", Sys.which("python"))
args <- commandArgs(TRUE)

family <- if (length(args)) args[1L] else "Logistic"

selftest <- family == "selftest"

stopifnot(family %in% c("selftest", "export", "Logistic", "RF", "SVM", "XGBoost", "LightGBM"))

seed <- 123L

M <- 30L

K <- 5L

tune_mi <- c(1L, 7L, 13L, 19L, 25L)

note <- function(...) {
    cat(format(Sys.time(), "%F %T"), ..., "\n")
    flush.console()
}

atomic_rds <- function(x, path) {
    saveRDS(x, paste0(path, ".tmp"))
    stopifnot(file.rename(paste0(path, ".tmp"), path))
}

if (selftest) {
    set.seed(seed)
    n <- 300L
    d <- data.table(patient = rep(seq_len(n/2L), each = 2L), fold = rep(rep(1:5, length.out = n/2L), 
        each = 2L))
    x <- data.frame(age = runif(n, 35, 85), creatinine = exp(rnorm(n, 4.8, 0.4)), proteinuria_grade = sample(0:2, 
        n, TRUE))
    d[, `:=`(y, rbinom(n, 1L, plogis(-1 + 0.04 * (x$age - 60) + 0.005 * (x$creatinine - 120))))]
    M <- 2L
    blocks <- lapply(1:5, function(k) list(fold = k, train = which(d$fold != k), query = which(d$fold == 
        k)))
    frames <- lapply(blocks, function(b) lapply(1:M, function(m) {
        aa <- design(x[b$train, ])
        qq <- design(x[b$query, ])
        list(train = aa, query = qq, y = d$y[b$train], labels = d$y[b$query], weight = rep(c(0.5, 1, 
            2), length.out = length(b$train)), query_weight = rep(c(0.7, 1.3), length.out = length(b$query)), 
            center = colMeans(aa), scale = apply(aa, 2L, sd))
    }))
} else {
    input <- readRDS(file.path(out, "Development_inputs.rds"))
    d <- as.data.table(input$reference)
    selected <- as.data.table(input$selection)
    stopifnot(nrow(d) == 4575L, uniqueN(d$patient) == 1736L, sum(d$y) == 519L, all(d[, uniqueN(fold), 
        by = patient]$V1 == 1L), identical(input$test_labels_read, FALSE))
    features_by_fold <- setNames(lapply(0:5, function(k) selected[Fold == k & Analysis == "primary" & 
        Included == TRUE, Variable]), as.character(0:5))
    stopifnot(all(lengths(features_by_fold) > 0L))
    blocks <- input$blocks[vapply(input$blocks, function(b) b$fold > 0L, TRUE)]
    if (family == "export") {
        folder <- file.path(out, "inputs")
        dir.create(folder, showWarnings = FALSE)
        oof_weights <- matrix(NA_real_, nrow(d), M)
        manifest <- lapply(input$blocks, function(b) {
            features <- features_by_fold[[as.character(b$fold)]]
            z <- readRDS(b$predictive_imputations)
            stopifnot(identical(z$train, b$train), identical(z$query, b$query), !length(intersect(d$patient[b$train], 
                d$patient[b$query])))
            query_weights <- if (b$fold) 
                matrix(NA_real_, length(b$query), M)
            else NULL
            entries <- lapply(seq_len(M), function(m) {
                w <- readRDS(file.path(b$weights, sprintf("m%02d.rds", m)))
                stopifnot(identical(w$record, d$record[b$train]), identical(w$query_record, d$record[b$query]))
                a <- as.data.table(z$completed[[m]])[, ..features]
                a[, `:=`(label = d$y[b$train], sample_weight = w$train_weights)]
                ap <- file.path(folder, sprintf("fold%02d_m%02d_train.csv", b$fold, m))
                fwrite(a, ap)
                qp <- NULL
                if (b$fold) {
                  q <- as.data.table(z$queries[[m]])[, ..features]
                  stopifnot(!any(c("y", "label", "sample_weight") %in% names(q)))
                  qp <- file.path(folder, sprintf("fold%02d_m%02d_query.csv", b$fold, m))
                  fwrite(q, qp)
                  query_weights[, m] <<- w$query_weights
                }
                list(imputation = m, train = ap, query = qp)
            })
            cats <- intersect(features, c("female", "hypertension", "coronary_disease", "proteinuria_grade"))
            yp <- wp <- NULL
            if (b$fold) {
                yp <- file.path(folder, sprintf("fold%02d_labels.csv", b$fold))
                fwrite(d[b$query, .(label = y)], yp)
                wp <- file.path(folder, sprintf("fold%02d_weights.csv", b$fold))
                fwrite(as.data.table(query_weights), wp)
                oof_weights[b$query, ] <<- query_weights
            }
            list(fold = b$fold, features = features, train_indices = b$train, query_indices = b$query, 
                categorical_levels = setNames(lapply(cats, function(v) if (v == "proteinuria_grade") 0:2 else 0:1), 
                  cats), query_labels = yp, query_weights = wp, entries = entries)
        })
        stopifnot(all(is.finite(oof_weights)), all(oof_weights > 0))
        fwrite(d[, .(record, patient, fold, label = y)], file.path(out, "Training_reference.csv"))
        ww <- as.data.table(oof_weights)
        setnames(ww, sprintf("m%02d", 1:M))
        fwrite(ww, file.path(out, "Training_OOF_weights.csv"))
        write_json(list(seed = seed, allocation_seed = 1234L, imputations = M, folds = K, blocks = manifest, 
            features = features_by_fold[["0"]], excluded = c("potassium", "sodium", "chloride", "calcium", 
                "magnesium", "total_CO2", "bicarbonate", "anion_gap", "calcium_phosphate_product", "phosphate", 
                "diagnostic_y"), algorithms = c("Logistic", "RF", "SVM", "XGBoost", "LightGBM", "CatBoost", "TabM"), primary_analysis = "Stabilized testing IPW; fold-specific feature selection", 
            calibration = "Training cross-fitted Platt decision only", test_labels_read = FALSE), file.path(out, 
            "Training_job.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
        note("TRAINING_INPUTS_EXPORTED; NO TEST LABELS")
        quit(save = "no")
    }
    frames <- lapply(blocks, function(b) {
        features <- features_by_fold[[as.character(b$fold)]]
        z <- readRDS(b$predictive_imputations)
        stopifnot(identical(z$train, b$train), identical(z$query, b$query), !length(intersect(d$patient[b$train], 
            d$patient[b$query])))
        lapply(seq_len(M), function(m) {
            w <- readRDS(file.path(b$weights, sprintf("m%02d.rds", m)))
            stopifnot(identical(w$record, d$record[b$train]), identical(w$query_record, d$record[b$query]))
            aa <- design(z$completed[[m]][, features, drop = FALSE])
            qq <- design(z$queries[[m]][, features, drop = FALSE])
            stopifnot(identical(colnames(aa), colnames(qq)), all(is.finite(aa)), all(is.finite(qq)))
            list(train = aa, query = qq, y = d$y[b$train], labels = d$y[b$query], weight = w$train_weights/mean(w$train_weights), 
                query_weight = w$query_weights, center = colMeans(aa), scale = apply(aa, 2L, sd))
        })
    })
}

fit_one <- function(k, m, pa) {
    a <- frames[[k]][[m]]
    fitseed <- seed + 100L * k + m
    set.seed(fitseed)
    if (family == "Logistic") {
        fit <- glm.fit(cbind(1, a$train), a$y, weights = a$weight, family = quasibinomial())
        stopifnot(fit$converged, all(is.finite(fit$coefficients)))
        p <- plogis(drop(cbind(1, a$query) %*% fit$coefficients))
    }
    else if (family == "RF") {
        fit <- ranger::ranger(x = as.data.frame(a$train), y = factor(a$y, levels = 0:1), probability = TRUE, 
            case.weights = a$weight, num.trees = 500L, mtry = max(1L, round(pa$fraction * ncol(a$train))), 
            min.node.size = as.integer(pa$node), sample.fraction = pa$sample, num.threads = 4L, seed = fitseed)
        p <- predict(fit, as.data.frame(a$query), num.threads = 4L)$predictions[, "1"]
    }
    else if (family == "SVM") {
        ss <- a$scale
        ss[ss <= 0 | !is.finite(ss)] <- 1
        x <- sweep(sweep(a$train, 2L, a$center), 2L, ss, "/")
        q <- sweep(sweep(a$query, 2L, a$center), 2L, ss, "/")
        solver_log <- tempfile("weighted_svm_")
        fit <- local({
            con <- file(solver_log, open = "wt", encoding = "UTF-8")
            sink(con, type = "message")
            on.exit({
                sink(type = "message")
                close(con)
            })
            WeightSVM::wsvm(x, factor(a$y, levels = 0:1), weight = a$weight, type = "C-classification", 
                kernel = pa$kernel, cost = pa$cost, gamma = pa$gamma, scale = FALSE, probability = TRUE, 
                cachesize = 300)
        })
        if (any(grepl("reaching max number of iterations", readLines(solver_log, warn = FALSE), fixed = TRUE))) 
            stop("SVM solver reached its iteration limit")
        p <- attr(predict(fit, q, probability = TRUE), "probabilities")[, "1"]
    }
    else if (family == "XGBoost") {
        fit <- xgboost::xgb.train(list(objective = "binary:logistic", tree_method = "hist", device = "cpu", 
            nthread = 4L, max_depth = as.integer(pa$depth), eta = pa$eta, min_child_weight = pa$node, 
            lambda = pa$l2, alpha = pa$l1, subsample = pa$sample, colsample_bytree = pa$columns, seed = fitseed), 
            xgboost::xgb.DMatrix(a$train, label = a$y, weight = a$weight, nthread = 1L), nrounds = as.integer(pa$rounds), 
            verbose = 0)
        p <- predict(fit, xgboost::xgb.DMatrix(a$query, nthread = 1L))
    }
    else {
        fit <- lightgbm::lgb.train(list(objective = "binary", metric = "auc", verbosity = -1L, num_threads = 4L, 
            device_type = "cpu", learning_rate = pa$eta, num_leaves = as.integer(pa$leaves), min_data_in_leaf = as.integer(pa$node), 
            lambda_l2 = pa$l2, lambda_l1 = pa$l1, feature_fraction = pa$columns, bagging_fraction = pa$sample, 
            bagging_freq = 1L, deterministic = TRUE, force_col_wise = TRUE, seed = fitseed), lightgbm::lgb.Dataset(a$train, 
            label = a$y, weight = a$weight), nrounds = as.integer(pa$rounds))
        p <- predict(fit, a$query)
    }
    p <- as.numeric(p)
    stopifnot(length(p) == length(a$labels), all(is.finite(p)), all(p >= 0 & p <= 1))
    p
}

if (selftest) {
    configurations <- list(Logistic = list(), RF = list(fraction = 0.5, node = 5L, sample = 0.8), SVM = list(kernel = "radial", 
        cost = 1, gamma = 0.1), XGBoost = list(depth = 3L, eta = 0.05, rounds = 10L, node = 2, l2 = 1, 
        l1 = 0.001, sample = 0.8, columns = 0.8), LightGBM = list(leaves = 7L, eta = 0.05, rounds = 10L, 
        node = 10L, l2 = 1, l1 = 0.001, sample = 0.8, columns = 0.8))
    for (family in names(configurations)) {
        p <- fit_one(1L, 1L, configurations[[family]])
        a <- frames[[1L]][[1L]]
        w <- rep(1:3, length.out = length(p))
        q <- curve(a$labels, p, matrix(w, ncol = 1L))
        ii <- rep(seq_along(p), w)
        r <- curve(a$labels[ii], p[ii], matrix(1, length(ii), 1L))
        stopifnot(max(abs(c(q$AUC - r$AUC, q$AP - r$AP, q$Brier - r$Brier))) < 1e-12)
        note("WEIGHTED_FIT_AND_INTEGER_REPLICATION_PASS", family)
    }
    quit(save = "no")
}

dest <- file.path(out, family)

dir.create(dest, showWarnings = FALSE)

score <- function(number, pa, mm, stage) {
    prefix <- file.path(dest, sprintf("trial%03d", number))
    dir.create(prefix, showWarnings = FALSE)
    values <- briers <- numeric(K)
    for (k in 1:K) {
        path <- file.path(prefix, sprintf("fold%02d.rds", k))
        p <- if (file.exists(path)) 
            readRDS(path)
        else matrix(NA_real_, length(blocks[[k]]$query), M)
        for (m in mm) if (!all(is.finite(p[, m]))) {
            p[, m] <- fit_one(k, m, pa)
            atomic_rds(p, path)
            note(stage, family, "TRIAL", number, "FOLD", k, "MI", m)
        }
        q <- rowMeans(p[, mm, drop = FALSE])
        y <- d$y[blocks[[k]]$query]
        weights <- do.call(cbind, lapply(mm, function(m) frames[[k]][[m]]$query_weight))
        metric <- curve(y, q, weights)
        values[k] <- mean(metric$AUC)
        briers[k] <- mean(metric$Brier)
    }
    list(trial = number, parameters = pa, mean_auc = mean(values), mean_brier = mean(briers), fold_auc = values, 
        imputations = mm)
}

if (family == "Logistic") {
    best <- score(0L, list(), 1:M, "FIT")
    best$completed_trials <- 0L
    records <- list()
} else {
    reticulate::use_python(python, required = TRUE)
    optuna <- reticulate::import("optuna", convert = TRUE)
    optuna$logging$set_verbosity(optuna$logging$WARNING)
    fail_trial <- reticulate::py_eval("lambda study, number: study.tell(number, state=__import__('optuna').trial.TrialState.FAIL)")
    budget <- c(RF = 30L, SVM = 40L, XGBoost = 60L, LightGBM = 60L)[[family]]
    study <- optuna$create_study(study_name = family, direction = "maximize", storage = paste0("sqlite:///", 
        gsub("\\", "/", file.path(dest, "Optuna.sqlite3"), fixed = TRUE)), load_if_exists = TRUE, sampler = optuna$samplers$TPESampler(seed = seed, 
        n_startup_trials = 10L))
    suggest <- function(t) switch(family, RF = list(trees = 500L, fraction = t$suggest_float("fraction", 
        0.25, 1), node = t$suggest_int("node", 3L, 50L), sample = t$suggest_float("sample", 0.6, 1)), 
        SVM = list(kernel = t$suggest_categorical("kernel", list("linear", "radial")), cost = t$suggest_float("cost", 
            0.01, 100, log = TRUE), gamma = t$suggest_float("gamma", 0.001, 1, log = TRUE)), XGBoost = list(depth = t$suggest_int("depth", 
            2L, 6L), eta = t$suggest_float("eta", 0.01, 0.15, log = TRUE), rounds = t$suggest_int("rounds", 
            100L, 800L, step = 50L), node = t$suggest_float("node", 1, 40, log = TRUE), l2 = t$suggest_float("l2", 
            0.01, 100, log = TRUE), l1 = t$suggest_float("l1", 1e-05, 10, log = TRUE), sample = t$suggest_float("sample", 
            0.6, 1), columns = t$suggest_float("columns", 0.6, 1)), LightGBM = list(leaves = t$suggest_int("leaves", 
            5L, 31L), eta = t$suggest_float("eta", 0.01, 0.15, log = TRUE), rounds = t$suggest_int("rounds", 
            100L, 800L, step = 50L), node = t$suggest_int("node", 15L, 150L), l2 = t$suggest_float("l2", 
            0.01, 100, log = TRUE), l1 = t$suggest_float("l1", 1e-05, 10, log = TRUE), sample = t$suggest_float("sample", 
            0.6, 1), columns = t$suggest_float("columns", 0.6, 1)))
    write_json(list(family = family, max_trials = budget, tuning_imputations = tune_mi, confirmation_imputations = 1:M, 
        objective = "Mean fold IPW AUC after MI probability averaging", feature_selection = "Separate Y-auxiliary weighted LASSO within each fitting fold", 
        stopping = "Budget or two 10-trial blocks with AUC improvement below .001 after 30 trials", learning_weights = if (family == 
            "RF") "ranger case.weights weighted sampling" else "Per-observation fitting weights", precision = "Installed R backend", 
        device = "CPU", optuna = optuna$`__version__`, R = as.character(getRversion()), packages = setNames(lapply(c("ranger", 
            "WeightSVM", "xgboost", "lightgbm"), function(p) as.character(packageVersion(p))), c("ranger", 
            "WeightSVM", "xgboost", "lightgbm")), test_labels_read = FALSE), file.path(dest, "Search_specification.json"), 
        pretty = TRUE, auto_unbox = TRUE)
    repeat {
        records <- lapply(list.files(dest, "^trial[0-9]+_complete.rds$", full.names = TRUE), readRDS)
        done <- length(study$trials)
        running <- Filter(function(t) as.integer(t$state) == 0L, study$trials)
        if (done >= budget && !length(running)) 
            break
        if (!length(running) && done >= 30L && done%%10L == 0L && length(records) == done) {
            values <- vapply(records[order(vapply(records, `[[`, 0L, "trial"))], `[[`, 0, "mean_auc")
            bests <- cummax(values)
            if (bests[done] - bests[done - 10L] < 0.001 && bests[done - 10L] - bests[done - 20L] < 0.001) 
                break
        }
        t <- if (length(running)) 
            optuna$trial$Trial(study, running[[1L]]$`_trial_id`)
        else study$ask()
        number <- as.integer(t$number)
        pa <- suggest(t)
        result <- tryCatch(score(number, pa, tune_mi, "TUNE"), error = function(e) e)
        if (inherits(result, "error")) {
            write_json(list(trial = number, parameters = pa, error = conditionMessage(result)), file.path(dest, 
                sprintf("trial%03d_failed.json", number)), auto_unbox = TRUE, pretty = TRUE)
            fail_trial(study, number)
            note("TRIAL_FAILED", family, number, conditionMessage(result))
            next
        }
        study$tell(t, result$mean_auc)
        atomic_rds(result, file.path(dest, sprintf("trial%03d_complete.rds", number)))
        note("TRIAL_COMPLETE", family, number, "IPW_AUC", result$mean_auc)
    }
    records <- lapply(list.files(dest, "^trial[0-9]+_complete.rds$", full.names = TRUE), readRDS)
    stopifnot(length(records) > 0L)
    ranked <- order(-vapply(records, `[[`, 0, "mean_auc"), vapply(records, `[[`, 0, "mean_brier"))
    finalists <- lapply(head(ranked, 2L), function(j) {
        z <- records[[j]]
        ans <- score(z$trial, z$parameters, 1:M, "CONFIRM")
        atomic_rds(ans, file.path(dest, sprintf("trial%03d_confirmed.rds", z$trial)))
        ans
    })
    best <- finalists[[order(-vapply(finalists, `[[`, 0, "mean_auc"), vapply(finalists, `[[`, 0, "mean_brier"))[1L]]]
    best$completed_trials <- length(records)
}

write_json(best, file.path(dest, "Selected_configuration.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)

probability <- matrix(NA_real_, nrow(d), M)

for (k in 1:K) probability[blocks[[k]]$query, ] <- readRDS(file.path(dest, sprintf("trial%03d/fold%02d.rds", 
    best$trial, k)))

stopifnot(all(is.finite(probability)), all(probability >= 0 & probability <= 1))

pp <- as.data.table(probability)

setnames(pp, sprintf("m%02d", 1:M))

fwrite(pp, file.path(dest, "OOF_probabilities.csv"))

atomic_rds(list(probability = probability, features_by_fold = features_by_fold, parameters = best$parameters), 
    file.path(dest, "OOF_predictions.rds"))

write_json(list(completed = TRUE, family = family, rows = nrow(d), folds = K, imputations = M, completed_trials = length(records), 
    mean_auc = best$mean_auc, mean_brier = best$mean_brier, features_by_fold = features_by_fold, training = "Stabilized IPW", 
    test_labels_read = FALSE), file.path(dest, "Complete.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)

note("MODEL_COMPLETE", family, "IPW_AUC", best$mean_auc)

