options(encoding = "UTF-8", stringsAsFactors = FALSE, warn = 1)

Sys.setenv(OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1")

suppressPackageStartupMessages({
    library(data.table)
    library(mice)
    library(ranger)
    library(glmnet)
    library(jsonlite)
    library(splines)
})

setDTthreads(1L)

root <- Sys.getenv("HYPERPHOS_RUN")
source(file.path(root,"analysis/common.R"))
base <- file.path(root, "results/predictors")

cohort <- file.path(root, "data/cleaned")

out <- file.path(root, "results/analysis")

args <- commandArgs(TRUE)

mode <- if (length(args)) args[1L] else "run"

stopifnot(mode %in% c("selftest", "inspect", "weights", "select", "run"))

M <- 30L

K <- 5L

iterations <- 20L

seed <- 123L

workers <- 4L

lambda <- exp(seq(log(100), log(1e-05), length.out = 180L))

binary_variables <- c("female", "hypertension", "coronary_disease", "tested_early", "diagnostic_y")

ordered_variables <- "proteinuria_grade"

excluded <- c("potassium", "sodium", "chloride", "calcium", "magnesium", "total_CO2", "bicarbonate", 
    "anion_gap", "calcium_phosphate_product", "phosphate", "first_phosphate")

note <- function(...) {
    cat(format(Sys.time(), "%F %T"), ..., "\n")
    flush.console()
}

if (mode == "selftest") {
    set.seed(seed)
    n <- 150L
    raw <- data.frame(age = runif(n, 35, 85), creatinine = exp(rnorm(n, 4.8, 0.5)), SBP = rnorm(n, 130, 
        18), female = rbinom(n, 1, 0.4), proteinuria_grade = sample(0:2, n, TRUE), tested_early = rbinom(n, 
        1, 0.55))
    raw$creatinine[1:20] <- NA_real_
    raw$proteinuria_grade[21:35] <- NA_real_
    a <- complete_one(raw, 124L, 2L)
    stopifnot(a$method[["tested_early"]] == "", a$predictorMatrix["creatinine", "tested_early"] == 1L, 
        all(a$predictorMatrix["tested_early", ] == 0L), !"diagnostic_y" %in% names(a$completed))
    vars <- setdiff(names(raw), "tested_early")
    x <- selection_design(a$completed, vars)
    q <- selection_design(a$completed[1:12, ], vars, x$basis)
    stopifnot(max(abs(x$x[1:12, ] - q$x)) < 1e-10)
    raw$diagnostic_y <- rbinom(n, 1, 0.25)
    b <- complete_one(raw[, setdiff(names(raw), "tested_early")], 125L, 2L)
    stopifnot(isTRUE(all.equal(b$completed$diagnostic_y, raw$diagnostic_y, check.attributes = FALSE)), 
        b$predictorMatrix["creatinine", "diagnostic_y"] == 1L)
    imputer <- freeze_imputer(a$completed, raw[, setdiff(names(raw), "diagnostic_y")], 127L)
    query <- raw[1:12, setdiff(names(raw), "diagnostic_y")]
    query_with_unused_y <- query
    query_with_unused_y$diagnostic_y <- rep(1, nrow(query))
    qa <- impute_query(imputer, query, 128L)
    qb <- impute_query(imputer, query_with_unused_y, 128L)
    stopifnot(identical(qa, qb), !"diagnostic_y" %in% names(qa))
    folds <- group_folds(rep(seq_len(n/2L), each = 2L), 126L)
    stopifnot(all(tapply(folds, rep(seq_len(n/2L), each = 2L), function(z) length(unique(z))) == 1L))
    fit <- lasso_one(b$completed[, vars], raw$diagnostic_y, rep(1, n), folds)
    stopifnot(is.finite(fit$lambda_1se), !"diagnostic_y" %in% names(fit$coefficients))
    set.seed(129L)
    lp <- rnorm(2500)
    yy <- rbinom(length(lp), 1L, plogis(lp - 1))
    pp <- plogis(2.5 * (lp - 1))
    ff <- group_folds(seq_along(yy), 130L)
    cal <- choose_platt(yy, pp, cbind(rep(1, length(yy)), rep(2, length(yy))), ff)
    stopifnot(cal$use_platt, cal$coefficients[2L] > 0, mean(cal$fold_scores$Platt_Brier) < mean(cal$fold_scores$Raw_Brier), 
        all(is.finite(cal$probability)), length(cal$crossfit_probability) == length(yy))
    flat <- choose_platt(yy, rep(0.25, length(yy)), matrix(1, length(yy), 1L), ff)
    stopifnot(!flat$use_platt)
    note("IPW_TRAINING_SELECTION_SELFTEST_PASS")
    quit(save = "no")
}

dir.create(out, recursive = TRUE, showWarnings = FALSE)

dir.create(file.path(out, "weights"), showWarnings = FALSE)

dir.create(file.path(out, "selection"), showWarnings = FALSE)

input <- readRDS(file.path(cohort, "Development_inputs.rds"))

d <- as.data.table(input$training)

common <- setdiff(input$candidates, excluded)

test_x <- as.data.table(input$validation_x)

stopifnot(nrow(d) == 4575L, uniqueN(d$patient) == 1736L, sum(d$y) == 519L, nrow(test_x) == 2026L, uniqueN(test_x$patient) == 
    745L, length(common) == 26L, !any(c("y", "phosphate", "first_phosphate") %in% names(test_x)), !length(intersect(d$patient, 
    test_x$patient)), all(d[, uniqueN(fold), by = patient]$V1 == 1L), !anyNA(d$fold))

rm(test_x, input)

prior <- c("age", "female", "year", "creatinine", "SBP", "DBP", "HbA1c", "hemoglobin", "wbc", "platelets", 
    "albumin", "proteinuria_grade", "hypertension", "coronary_disease", "uric_acid")

source <- as.data.table(readRDS(file.path(cohort, "Testing_target_frame.rds")))[, c("record", "patient", 
    "tested_early", prior), with = FALSE]

membership <- readRDS(file.path(cohort, "Membership.rds"))

stopifnot(identical(source$record, membership$record), nrow(source) == 14361L, uniqueN(source$patient) == 
    3235L, sum(source$tested_early) == 6601L)

source[, `:=`(partition = membership$partition, fold = membership$fold)]

target <- source[partition == "Training"]

rm(source, membership)

stopifnot(nrow(target) == 9925L, all(target[, uniqueN(fold), by = patient]$V1 == 1L), setequal(target[tested_early == 
    1L, record], d$record), identical(target$patient[match(d$record, target$record)], d$patient))

screen <- data.table(Variable = prior, Training_target_missing_pct = 100 * vapply(target[, ..prior], 
    function(x) mean(is.na(x)), 0))

screen[, `:=`(Included, Training_target_missing_pct <= 30)]

ps_variables <- prior[prior %in% screen[Included == TRUE, Variable]]

stopifnot(length(ps_variables) == 9L, !any(ps_variables %in% excluded))

setorder(screen, -Training_target_missing_pct)

fwrite(screen, file.path(out, "IPW_covariate_missingness.csv"), bom = TRUE)

fwrite(data.table(Variable = common, Training_missing_pct = 100 * vapply(d[, ..common], function(x) mean(is.na(x)), 
    0)), file.path(out, "Predictor_missingness.csv"), bom = TRUE)

write_json(list(status = "Preparing training-side IPW and feature selection", imputations = M, iterations = iterations, 
    allocation_seed = 1234L, fold_seed = seed, training_admissions = nrow(d), training_patients = uniqueN(d$patient), 
    test_labels_read = FALSE, updated = format(Sys.time(), "%F %T")), file.path(out, "Run_status.json"), 
    auto_unbox = TRUE, pretty = TRUE)

if (mode == "inspect") {
    print(screen)
    quit(save = "no")
}

block_indices <- function(k) {
    list(train = if (k == 0L) seq_len(nrow(d)) else which(d$fold != k), query = if (k == 0L) integer() else which(d$fold == 
        k), target_train = if (k == 0L) seq_len(nrow(target)) else which(target$fold != k), target_query = if (k == 
        0L) integer() else which(target$fold == k))
}

weight_one <- function(task) {
    k <- task$fold
    m <- task$imputation
    folder <- file.path(out, "weights", sprintf("fold%02d", k))
    dir.create(folder, showWarnings = FALSE)
    path <- file.path(folder, sprintf("m%02d.rds", m))
    if (file.exists(path)) 
        return(path)
    started <- Sys.time()
    index <- block_indices(k)
    a <- target[index$target_train]
    q <- target[index$target_query]
    stopifnot(!length(intersect(a$patient, q$patient)), all(vapply(a[, ..ps_variables], function(x) mean(is.na(x)), 
        0) <= 0.3))
    raw <- as.data.frame(a[, c(ps_variables, "tested_early"), with = FALSE])
    z <- complete_one(raw, 250000L + 1000L * k + m)
    matrix_a <- selection_design(z$completed, ps_variables)
    fit <- glmnet(matrix_a$x, a$tested_early, family = "binomial", alpha = 0, lambda = 0.01, standardize = TRUE)
    p <- drop(predict(fit, matrix_a$x, type = "response", s = 0.01))
    numerator <- mean(a$tested_early)
    if (nrow(q) || k == 0L) {
        imputer <- freeze_imputer(z$completed, raw, 350000L + 1000L * k + m)
    }
    else imputer <- NULL
    pq <- numeric()
    if (nrow(q)) {
        completed_q <- impute_query(imputer, as.data.frame(q[, c(ps_variables, "tested_early"), with = FALSE]), 
            450000L + 1000L * k + m)
        matrix_q <- selection_design(completed_q, ps_variables, matrix_a$basis)
        pq <- drop(predict(fit, matrix_q$x, type = "response", s = 0.01))
    }
    stopifnot(all(is.finite(c(p, pq))), all(c(p, pq) > 0 & c(p, pq) < 1))
    ia <- match(d$record[index$train], a$record)
    iq <- match(d$record[index$query], q$record)
    stopifnot(!anyNA(ia), !anyNA(iq), all(a$tested_early[ia] == 1L), all(q$tested_early[iq] == 1L))
    w <- numerator/p
    wq <- numerator/pq
    balance <- rbindlist(lapply(ps_variables, function(v) {
        rbindlist(lapply(c("Observed", "Imputed", "Missing"), function(component) {
            value <- if (component == "Observed") 
                a[[v]]
            else if (component == "Imputed") 
                z$completed[[v]]
            else as.numeric(is.na(a[[v]]))
            components <- if (v == "proteinuria_grade" && component != "Missing") 
                lapply(0:2, function(j) as.numeric(value == j))
            else list(value)
            rbindlist(lapply(seq_along(components), function(j) {
                x <- components[[j]]
                ii <- which(a$tested_early == 1L & is.finite(x))
                scale <- sd(x, na.rm = TRUE)
                if (!is.finite(scale) || scale == 0) 
                  return(NULL)
                rbindlist(lapply(c("Unweighted", "Stabilized_IPW"), function(kind) {
                  ww <- if (kind == "Unweighted") 
                    rep(1, length(ii))
                  else w[ii]
                  mean_target <- mean(x, na.rm = TRUE)
                  mean_measured <- weighted.mean(x[ii], ww)
                  data.table(Variable = v, Component = component, Level = if (length(components) > 1L) 
                    as.character(j - 1L)
                  else "", Method = kind, Target_mean = mean_target, Measured_mean = mean_measured, SMD = (mean_measured - 
                    mean_target)/scale)
                }))
            }))
        }))
    }))
    saveRDS(list(fold = k, imputation = m, variables = ps_variables, train = index$train, query = index$query, 
        target_train = index$target_train, target_query = index$target_query, record = d$record[index$train], 
        query_record = d$record[index$query], train_weights = w[ia], query_weights = wq[iq], target_probability = p, 
        query_target_probability = pq, numerator = numerator, model = fit, basis = matrix_a$basis, imputer = if (k == 
            0L) imputer else NULL, completed = z$completed, balance = balance, method = z$method, loggedEvents = z$loggedEvents, 
        chainMean = z$chainMean, seconds = as.numeric(difftime(Sys.time(), started, units = "secs"))), 
        path)
    note("IPW_CHAIN_COMPLETE", k, m)
    path
}

select_one <- function(task) {
    k <- task$fold
    m <- task$imputation
    folder <- file.path(out, "selection", sprintf("fold%02d", k))
    dir.create(folder, showWarnings = FALSE)
    path <- file.path(folder, sprintf("m%02d.rds", m))
    if (file.exists(path)) 
        return(path)
    started <- Sys.time()
    index <- block_indices(k)
    variables <- common[vapply(d[index$train, ..common], function(x) mean(is.na(x)) <= 0.3 && uniqueN(na.omit(x)) > 
        1L, TRUE)]
    raw <- as.data.frame(d[index$train, ..variables])
    raw$diagnostic_y <- d$y[index$train]
    z <- complete_one(raw, 550000L + 1000L * k + m)
    x <- z$completed[, variables, drop = FALSE]
    weights <- readRDS(file.path(out, "weights", sprintf("fold%02d/m%02d.rds", k, m)))
    stopifnot(identical(weights$record, d$record[index$train]), !"diagnostic_y" %in% names(x))
    folds <- if (k == 0L) 
        d$fold
    else group_folds(d$patient[index$train], seed + 100L * k)
    primary <- lasso_one(x, raw$diagnostic_y, weights$train_weights, folds)
    sensitivity <- lasso_one(x, raw$diagnostic_y, rep(1, nrow(x)), folds)
    saveRDS(list(fold = k, imputation = m, train = index$train, query = index$query, variables = variables, 
        primary = primary, unweighted = sensitivity, method = z$method, predictorMatrix = z$predictorMatrix, 
        loggedEvents = z$loggedEvents, chainMean = z$chainMean, chainVar = z$chainVar, completed = x, 
        seconds = as.numeric(difftime(Sys.time(), started, units = "secs"))), path)
    note("SELECTION_CHAIN_COMPLETE", k, m)
    path
}

run_parallel <- function(fun, tasks) {
  cl <- parallel::makePSOCKcluster(min(workers,length(tasks)),outfile=file.path(out,"Workers.log"))
  on.exit(parallel::stopCluster(cl))
  parallel::clusterCall(cl,function(paths) .libPaths(paths),.libPaths())
  parallel::clusterEvalQ(cl,{
    Sys.setenv(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    suppressPackageStartupMessages({library(data.table);library(mice);library(ranger);library(glmnet);library(splines);library(MASS);library(nnet)})
    setDTthreads(1L);NULL
  })
  parallel::clusterExport(cl,c("out","d","target","common","ps_variables","M","K","iterations","seed","lambda",
    "binary_variables","ordered_variables","note","check_observed","complete_one","freeze_imputer","impute_query",
    "as_categories","design","selection_design","group_folds","lasso_one","block_indices","weight_one","select_one"))
  parallel::parLapplyLB(cl,tasks,fun)
}

tasks <- lapply(0:K, function(k) lapply(seq_len(M), function(m) list(fold = k, imputation = m)))

tasks <- unlist(tasks, recursive = FALSE)

if (mode %in% c("weights", "run")) {
    note("IPW_TRAINING_ONLY_START", "CHAINS", length(tasks), "WORKERS", workers)
    invisible(run_parallel(weight_one, tasks))
    note("ALL_IPW_BLOCKS_COMPLETE")
    write_json(list(status = "Training-side weights complete; selecting predictors", chains = length(tasks), 
        test_labels_read = FALSE, updated = format(Sys.time(), "%F %T")), file.path(out, "Run_status.json"), 
        auto_unbox = TRUE, pretty = TRUE)
    if (mode == "weights") 
        quit(save = "no")
}

if (mode %in% c("select", "run")) {
    stopifnot(all(vapply(tasks, function(z) file.exists(file.path(out, "weights", sprintf("fold%02d/m%02d.rds", 
        z$fold, z$imputation))), TRUE)))
    note("OUTCOME_AUXILIARY_LASSO_START", "CHAINS", length(tasks))
    invisible(run_parallel(select_one, tasks))
}

frequencies <- diagnostics <- blocks <- balance <- list()

for (k in 0:K) {
    b <- lapply(seq_len(M), function(m) readRDS(file.path(out, "selection", sprintf("fold%02d/m%02d.rds", 
        k, m))))
    w <- lapply(seq_len(M), function(m) readRDS(file.path(out, "weights", sprintf("fold%02d/m%02d.rds", 
        k, m))))
    index <- block_indices(k)
    raw_predictive <- readRDS(file.path(base, "selection", sprintf("MI_fold%02d.rds", k)))
    stopifnot(identical(raw_predictive$train, index$train), identical(raw_predictive$query, index$query), 
        length(raw_predictive$completed) == M, setequal(raw_predictive$variables, common))
    raw <- as.data.frame(d[index$train, ..common])
    for (m in seq_len(M)) {
        check_observed(raw_predictive$completed[[m]], raw)
        if (length(index$query)) 
            check_observed(raw_predictive$queries[[m]], as.data.frame(d[index$query, ..common]))
        for (kind in c("primary", "unweighted")) {
            values <- b[[m]][[kind]]$coefficients[-1L]
            mapped <- sub("^proteinuria_grade[12]$", "proteinuria_grade", names(values))
            frequencies[[length(frequencies) + 1L]] <- data.table(Fold = k, Imputation = m, Analysis = kind, 
                Variable = common, Selected = common %in% unique(mapped[abs(values) > 1e-10]))
        }
        ww <- w[[m]]$train_weights
        group_weight <- rowsum(ww, d$patient[index$train], reorder = FALSE)[, 1L]
        diagnostics[[length(diagnostics) + 1L]] <- data.table(Fold = k, Imputation = m, Target_admissions = length(index$target_train), 
            Measured_admissions = length(ww), ESS = sum(ww)^2/sum(ww^2), Patient_weight_total_ESS = sum(group_weight)^2/sum(group_weight^2), 
            Minimum = min(ww), Median = median(ww), Maximum = max(ww), P99 = unname(quantile(ww, 0.99)), 
            Weight_sum = sum(ww))
        balance[[length(balance) + 1L]] <- cbind(Fold = k, Imputation = m, w[[m]]$balance)
    }
    blocks[[length(blocks) + 1L]] <- list(fold = k, train = index$train, query = index$query, predictive_imputations = file.path(base, 
        "selection", sprintf("MI_fold%02d.rds", k)), weights = file.path(out, "weights", sprintf("fold%02d", 
        k)))
}

frequencies <- rbindlist(frequencies)

counts <- frequencies[, .(Selected_imputations = sum(Selected), Total_imputations = .N), by = .(Fold, 
    Analysis, Variable)]

counts[, `:=`(Included, Selected_imputations >= 18L)]

setorder(counts, Fold, Analysis, -Selected_imputations, Variable)

fwrite(counts, file.path(out, "LASSO_selection_frequencies.csv"), bom = TRUE)

fwrite(counts[Fold == 0L & Included == TRUE], file.path(out, "Selected_predictors.csv"), bom = TRUE)

fwrite(rbindlist(diagnostics), file.path(out, "IPW_weight_diagnostics.csv"), bom = TRUE)

fwrite(rbindlist(balance), file.path(out, "IPW_covariate_balance.csv"), bom = TRUE)

saveRDS(list(reference = d[, .(record, patient, fold, y)], candidates = common, blocks = blocks, selection = counts, 
    imputations = M, iterations = iterations, selection_auxiliary = "Training-fitting-block diagnostic Y only; never a model predictor", 
    predictive_imputation = "Outcome-free, fitting-block models applied to matching query predictors", 
    calibration = "Training OOF cross-fitted Platt; one-SE Brier rule and non-worsening log loss", test_labels_read = FALSE), 
    file.path(out, "Development_inputs.rds"))

write_json(list(status = "Training IPW and fold-specific LASSO selection complete", chains = length(tasks), 
    final_predictors = counts[Fold == 0L & Analysis == "primary" & Included == TRUE, Variable], inner_fold_counts = counts[Analysis == 
        "primary" & Included == TRUE, .N, by = Fold], test_labels_read = FALSE, updated = format(Sys.time(), 
        "%F %T")), file.path(out, "Run_status.json"), auto_unbox = TRUE, pretty = TRUE)

note("IPW_PRIMARY_SELECTION_COMPLETE; NO TEST PERFORMANCE EVALUATION")

