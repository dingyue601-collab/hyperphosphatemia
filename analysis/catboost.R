options(encoding = "UTF-8", stringsAsFactors = FALSE, warn = 1)

Sys.setenv(OMP_NUM_THREADS = "4", MKL_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1")

suppressPackageStartupMessages({
    library(data.table)
    library(jsonlite)
    library(matrixStats)
    library(catboost)
})

setDTthreads(1L)

root <- Sys.getenv("HYPERPHOS_RUN")
source(file.path(root,"analysis/common.R"))
out <- file.path(root, "results/analysis")

python <- Sys.getenv("HYPERPHOS_PYTHON", Sys.which("python"))
mode <- commandArgs(TRUE)

mode <- if (length(mode)) mode[1L] else "run"

stopifnot(mode %in% c("run", "selftest", "benchmark"))

family <- "CatBoost"

seed <- 123L

M <- 30L

K <- 5L

tune_mi <- c(1L, 7L, 13L, 19L, 25L)

budget <- 60L

note <- function(...) {
    cat(format(Sys.time(), "%F %T"), ..., "\n")
    flush.console()
}

load_functions(file.path(root, "analysis/models.R"), 
    c("atomic_rds", "score"))

dest <- file.path(out, family)

categories <- function(x) {
    x <- as.data.frame(x)
    for (v in names(x)) {
        if (v %in% c("female", "hypertension", "coronary_disease", "proteinuria_grade")) {
            lev <- if (v == "proteinuria_grade") 
                0:2
            else 0:1
            stopifnot(all(x[[v]] %in% lev))
            x[[v]] <- factor(x[[v]], levels = lev)
        }
        else x[[v]] <- as.double(x[[v]])
    }
    stopifnot(!anyNA(x))
    x
}

fit_one <- function(k, m, pa) {
    a <- frames[[k]][[m]]
    learn <- catboost.load_pool(a$train, label = a$y, weight = a$weight)
    query <- catboost.load_pool(a$query)
    params <- c(list(loss_function = "Logloss", task_type = device, thread_count = 4L, random_seed = seed + 
        100L * k + m, logging_level = "Silent", allow_writing_files = FALSE, use_best_model = FALSE, 
        one_hot_max_size = 3L, border_count = 128L, bootstrap_type = "Bayesian"), pa)
    if (device == "GPU") 
        params$devices <- "0"
    fit <- catboost.train(learn, params = params)
    p <- as.numeric(catboost.predict(fit, query, prediction_type = "Probability", thread_count = 4L))
    stopifnot(length(p) == length(a$labels), all(is.finite(p)), all(p >= 0 & p <= 1))
    p
}

default <- list(depth = 4L, learning_rate = 0.05, iterations = 300L, l2_leaf_reg = 3, random_strength = 1, 
    bagging_temperature = 1, boosting_type = "Plain")

if (mode == "selftest") {
    device <- "CPU"
    set.seed(seed)
    n <- 300L
    x <- categories(data.frame(age = runif(n, 35, 85), creatinine = exp(rnorm(n, 4.8, 0.5)), proteinuria_grade = sample(0:2, 
        n, TRUE), hypertension = sample(0:1, n, TRUE)))
    y <- rbinom(n, 1L, plogis(-1 + 0.04 * (x$age - 60) + 0.005 * (x$creatinine - 120)))
    ii <- 1:220
    qq <- 221:n
    a <- list(train = x[ii, ], query = x[qq, ], y = y[ii], labels = y[qq], weight = rep(c(0.5, 1, 2), 
        length.out = length(ii)), query_weight = rep(1, length(qq)))
    frames <- list(list(a))
    pa <- default
    pa$iterations <- 30L
    p <- fit_one(1L, 1L, pa)
    frames[[1L]][[1L]]$weight <- rep(1, length(ii))
    unweighted <- fit_one(1L, 1L, pa)
    stopifnot(max(abs(p - unweighted)) > 1e-08)
    w <- rep(1:3, length.out = length(qq))
    z <- curve(y[qq], p, matrix(w, ncol = 1L))
    rr <- rep(seq_along(qq), w)
    q <- curve(y[qq][rr], p[rr], matrix(1, length(rr), 1L))
    stopifnot(max(abs(c(z$AUC - q$AUC, z$AP - q$AP, z$Brier - q$Brier))) < 1e-12)
    note("CATBOOST_NATIVE_CASE_WEIGHT_AND_INTEGER_REPLICATION_SELFTEST_PASS", "version", as.character(packageVersion("catboost")), 
        "weight_effect", max(abs(p - unweighted)))
    quit(save = "no")
}

input <- readRDS(file.path(out, "Development_inputs.rds"))

d <- as.data.table(input$reference)

selected <- as.data.table(input$selection)

stopifnot(nrow(d) == 4575L, uniqueN(d$patient) == 1736L, sum(d$y) == 519L, all(d[, uniqueN(fold), by = patient]$V1 == 
    1L), identical(input$test_labels_read, FALSE))

features_by_fold <- setNames(lapply(0:5, function(k) selected[Fold == k & Analysis == "primary" & Included == 
    TRUE, Variable]), as.character(0:5))

blocks <- input$blocks[vapply(input$blocks, function(b) b$fold > 0L, TRUE)]

frames <- lapply(blocks, function(b) {
    features <- features_by_fold[[as.character(b$fold)]]
    stopifnot(length(features) > 0L, !length(intersect(features, c("NLR", "nlr", "potassium", "sodium", 
        "calcium", "magnesium", "total_CO2", "chloride", "bicarbonate", "anion_gap", "calcium_phosphate_product", 
        "phosphate", "diagnostic_y"))))
    z <- readRDS(b$predictive_imputations)
    stopifnot(identical(z$train, b$train), identical(z$query, b$query), !length(intersect(d$patient[b$train], 
        d$patient[b$query])))
    lapply(seq_len(M), function(m) {
        w <- readRDS(file.path(b$weights, sprintf("m%02d.rds", m)))
        stopifnot(identical(w$record, d$record[b$train]), identical(w$query_record, d$record[b$query]))
        list(train = categories(z$completed[[m]][, features, drop = FALSE]), query = categories(z$queries[[m]][, 
            features, drop = FALSE]), y = d$y[b$train], labels = d$y[b$query], weight = w$train_weights/mean(w$train_weights), 
            query_weight = w$query_weights)
    })
})

dir.create(dest, showWarnings = FALSE)

if (mode == "benchmark") {
    results <- lapply(c("CPU", "GPU"), function(which) {
        device <<- which
        ans <- tryCatch({
            warm <- default
            warm$iterations <- 5L
            invisible(fit_one(1L, 1L, warm))
            elapsed <- numeric(3L)
            for (j in 1:3) elapsed[j] <- system.time(p <- fit_one(1L, 1L, default))[["elapsed"]]
            list(device = which, supported = TRUE, elapsed = elapsed, median_seconds = median(elapsed), 
                finite_probability = all(is.finite(p)))
        }, error = function(e) list(device = which, supported = FALSE, error = conditionMessage(e)))
        note("CATBOOST_DEVICE_BENCHMARK", which, toJSON(ans, auto_unbox = TRUE))
        ans
    })
    usable <- Filter(function(x) isTRUE(x$supported), results)
    stopifnot(length(usable) > 0L)
    chosen <- usable[[which.min(vapply(usable, `[[`, 0, "median_seconds"))]]$device
    write_json(list(device = chosen, results = results, parameters = default, training_fold = 1L, imputation = 1L, 
        catboost = as.character(packageVersion("catboost")), test_labels_read = FALSE), file.path(dest, 
        "Device_benchmark.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
    note("CATBOOST_DEVICE_SELECTED", chosen)
    quit(save = "no")
}

benchmark <- fromJSON(file.path(dest, "Device_benchmark.json"))

device <- benchmark$device

stopifnot(device %in% c("CPU", "GPU"))

reticulate::use_python(python, required = TRUE)

optuna <- reticulate::import("optuna", convert = TRUE)

optuna$logging$set_verbosity(optuna$logging$WARNING)

fail_trial <- reticulate::py_eval("lambda study, number: study.tell(number, state=__import__('optuna').trial.TrialState.FAIL)")

study <- optuna$create_study(study_name = family, direction = "maximize", storage = paste0("sqlite:///", 
    gsub("\\", "/", file.path(dest, "Optuna.sqlite3"), fixed = TRUE)), load_if_exists = TRUE, sampler = optuna$samplers$TPESampler(seed = seed, 
    n_startup_trials = 10L))

suggest <- function(t) list(depth = t$suggest_int("depth", 2L, 6L), learning_rate = t$suggest_float("learning_rate", 
    0.01, 0.15, log = TRUE), iterations = t$suggest_int("iterations", 100L, 800L, step = 50L), l2_leaf_reg = t$suggest_float("l2_leaf_reg", 
    0.1, 100, log = TRUE), random_strength = t$suggest_float("random_strength", 0, 3), bagging_temperature = t$suggest_float("bagging_temperature", 
    0, 2), boosting_type = t$suggest_categorical("boosting_type", list("Plain", "Ordered")))

write_json(list(family = family, max_trials = budget, tuning_imputations = tune_mi, confirmation_imputations = 1:M, 
    objective = "Mean fold IPW AUC after MI probability averaging", feature_selection = "Separate fitting-fold Y-auxiliary weighted LASSO", 
    stopping = "Budget or two 10-trial blocks with AUC improvement below .001 after 30 trials", learning_weights = "Native object weights, normalized by fitting-partition mean; no class weights", 
    categories = "Native factor input; one-hot for all observed binary and three-level predictors", device = device, 
    R = as.character(getRversion()), catboost = as.character(packageVersion("catboost")), optuna = optuna$`__version__`, 
    test_labels_read = FALSE), file.path(dest, "Search_specification.json"), pretty = TRUE, auto_unbox = TRUE)

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

write_json(best, file.path(dest, "Selected_configuration.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)

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
    device = device, catboost = as.character(packageVersion("catboost")), test_labels_read = FALSE), 
    file.path(dest, "Complete.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)

note("MODEL_COMPLETE", family, "IPW_AUC", best$mean_auc)

