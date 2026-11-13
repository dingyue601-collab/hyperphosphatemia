suppressPackageStartupMessages({library(data.table);library(mice);library(ranger);library(glmnet);library(jsonlite);library(splines);library(MASS);library(nnet);library(matrixStats)})
M <- 30L; K <- 5L; iterations <- 20L; seed <- 123L
lambda <- exp(seq(log(100),log(.00001),length.out=180L))
binary_variables <- c("female","hypertension","coronary_disease","tested_early","diagnostic_y")
ordered_variables <- "proteinuria_grade"
metric_names <- c("Accuracy","Precision","Sensitivity","Specificity","NPV","F1","Flagged_fraction","Youden","Balanced_accuracy")
load_functions <- function(file,functions) {for(e in parse(file,encoding="UTF-8")) if(is.call(e)&&identical(e[[1L]],as.name("<-"))&&is.symbol(e[[2L]])&&as.character(e[[2L]])%in%functions)eval(e,envir=.GlobalEnv)}
check_observed <- function(x, raw) {
    stopifnot(identical(names(x), names(raw)), nrow(x) == nrow(raw), !anyNA(x))
    for (v in names(raw)) {
        ii <- !is.na(raw[[v]])
        stopifnot(all(is.finite(x[[v]])), isTRUE(all.equal(x[[v]][ii], raw[[v]][ii], check.attributes = FALSE)))
    }
    for (v in intersect(binary_variables, names(x))) stopifnot(all(x[[v]] %in% 0:1))
    if ("proteinuria_grade" %in% names(x)) 
        stopifnot(all(x$proteinuria_grade %in% 0:2))
    invisible(TRUE)
}

impute_fit <- function(raw, s) {
    typed <- raw
    for (v in intersect(binary_variables, names(typed))) typed[[v]] <- factor(typed[[v]], levels = 0:1)
    for (v in intersect(ordered_variables, names(typed))) typed[[v]] <- ordered(typed[[v]], levels = 0:2)
    init <- typed
    method <- setNames(ifelse(colSums(is.na(typed)) > 0, "rf", ""), names(typed))
    for (v in names(typed)) {
        observed <- typed[[v]][!is.na(typed[[v]])]
        fill <- if (is.factor(observed)) 
            names(which.max(table(observed)))
        else median(observed)
        init[[v]][is.na(init[[v]])] <- fill
        if (anyNA(typed[[v]]) && is.factor(typed[[v]])) 
            method[v] <- if (is.ordered(typed[[v]])) 
                "polr"
            else if (nlevels(typed[[v]]) == 2L) 
                "logreg"
            else "polyreg"
    }
    predictor <- matrix(1L, ncol(raw), ncol(raw), dimnames = list(names(raw), names(raw)))
    diag(predictor) <- 0L
    imp <- mice(typed, m = M, maxit = iterations, method = method, predictorMatrix = predictor, data.init = init, 
        seed = s, printFlag = FALSE, rfPackage = "ranger", ntree = 10L, num.threads = 1L, polr.to.loggedEvents = TRUE, 
        remove.constant = FALSE, remove.collinear = FALSE, eps = 0)
    completed <- lapply(seq_len(M), function(m) {
        z <- mice::complete(imp, m)
        done <- raw
        for (v in names(raw)) {
            missing <- is.na(raw[[v]])
            values <- if (is.factor(z[[v]])) 
                as.numeric(as.character(z[[v]]))
            else z[[v]]
            done[[v]][missing] <- values[missing]
        }
        check_observed(done, raw)
        done
    })
    list(completed = completed, loggedEvents = imp$loggedEvents, chainMean = imp$chainMean, chainVar = imp$chainVar, 
        method = method)
}

freeze_imputer <- function(completed, raw, s) {
    forests <- setNames(vector("list", ncol(raw)), names(raw))
    for (j in seq_along(raw)) {
        v <- names(raw)[j]
        observed <- which(!is.na(raw[[v]]))
        predictors <- setdiff(names(raw), v)
        frame <- completed[observed, predictors, drop = FALSE]
        if (v %in% ordered_variables) {
            frame$response <- ordered(raw[[v]][observed], levels = 0:2)
            fit <- tryCatch(MASS::polr(response ~ ., data = frame, method = "logistic", Hess = FALSE, 
                control = list(maxit = 100L)), error = function(e) NULL)
            kind <- "polr"
            if (is.null(fit)) {
                fit <- nnet::multinom(response ~ ., data = frame, trace = FALSE, maxit = 100L, MaxNWts = 50000L)
                kind <- "polyreg"
            }
            forests[[v]] <- list(fit = fit, kind = kind, predictors = predictors)
        }
        else if (v %in% binary_variables) {
            frame$response <- raw[[v]][observed]
            forests[[v]] <- list(fit = glm(response ~ ., data = frame, family = binomial()), kind = "logreg", 
                predictors = predictors)
        }
        else {
            fit <- ranger(x = frame, y = raw[[v]][observed], num.trees = 10L, num.threads = 1L, seed = s + 
                j, write.forest = TRUE)
            nodes <- predict(fit, frame, type = "terminalNodes", num.threads = 1L)$predictions
            donors <- lapply(seq_len(ncol(nodes)), function(k) split(raw[[v]][observed], nodes[, k]))
            forests[[v]] <- list(fit = fit, kind = "rf", predictors = predictors, donors = donors)
        }
    }
    fill <- vapply(completed, median, 0)
    for (v in intersect(c(binary_variables, ordered_variables), names(completed))) fill[v] <- as.numeric(names(which.max(table(completed[[v]]))))
    list(forests = forests, medians = fill, columns = names(raw))
}

impute_query <- function(object, raw, s) {
    raw <- raw[, object$columns, drop = FALSE]
    x <- raw
    set.seed(s)
    for (v in names(x)) x[[v]][is.na(x[[v]])] <- object$medians[[v]]
    missing <- which(colSums(is.na(raw)) > 0)
    for (it in seq_len(iterations)) for (j in missing) {
        v <- names(x)[j]
        ii <- which(is.na(raw[[v]]))
        f <- object$forests[[v]]
        query <- x[ii, f$predictors, drop = FALSE]
        if (f$kind == "rf") {
            nodes <- predict(f$fit, query, type = "terminalNodes", num.threads = 1L)$predictions
            x[[v]][ii] <- vapply(seq_along(ii), function(r) {
                pool <- unlist(lapply(seq_len(ncol(nodes)), function(k) f$donors[[k]][[as.character(nodes[r, 
                  k])]]), use.names = FALSE)
                stopifnot(length(pool) > 0)
                pool[sample.int(length(pool), 1L)]
            }, 0)
        }
        else {
            probability <- if (f$kind == "logreg") {
                p <- predict(f$fit, query, type = "response")
                cbind(`0` = 1 - p, `1` = p)
            }
            else predict(f$fit, query, type = "probs")
            if (is.null(dim(probability))) 
                probability <- matrix(probability, nrow = 1L, dimnames = list(NULL, names(probability)))
            stopifnot(nrow(probability) == length(ii), all(is.finite(probability)), all(probability >= 
                0))
            x[[v]][ii] <- vapply(seq_along(ii), function(r) as.numeric(sample(colnames(probability), 
                1L, prob = probability[r, ])), 0)
        }
    }
    check_observed(x, raw)
    x
}

eligible <- function(x) names(x)[vapply(x, function(z) mean(is.na(z)) <= 0.3 && length(unique(na.omit(z))) > 
    1L, TRUE)]

as_categories <- function(x) {
    x <- as.data.frame(x)
    if ("proteinuria_grade" %in% names(x)) 
        x$proteinuria_grade <- factor(x$proteinuria_grade, levels = 0:2)
    x
}

design <- function(x) model.matrix(~., as_categories(x))[, -1L, drop = FALSE]

complete_one <- function(raw, s, maxit = iterations) {
    typed <- raw
    for (v in intersect(binary_variables, names(raw))) typed[[v]] <- factor(raw[[v]], levels = 0:1)
    for (v in intersect(ordered_variables, names(raw))) typed[[v]] <- ordered(raw[[v]], levels = 0:2)
    method <- setNames(ifelse(colSums(is.na(typed)) > 0, "rf", ""), names(typed))
    init <- typed
    for (v in names(typed)) {
        observed <- na.omit(typed[[v]])
        fill <- if (is.factor(observed)) 
            names(which.max(table(observed)))
        else median(observed)
        init[[v]][is.na(init[[v]])] <- fill
        if (method[[v]] != "" && is.factor(typed[[v]])) 
            method[[v]] <- if (is.ordered(typed[[v]])) 
                "polr"
            else "logreg"
    }
    predictor <- matrix(1L, ncol(raw), ncol(raw), dimnames = list(names(raw), names(raw)))
    diag(predictor) <- 0L
    for (v in intersect(c("diagnostic_y", "tested_early"), names(raw))) {
        stopifnot(!anyNA(raw[[v]]))
        method[[v]] <- ""
        predictor[v, ] <- 0L
    }
    z <- mice(typed, m = 1L, maxit = maxit, method = method, predictorMatrix = predictor, data.init = init, 
        seed = s, printFlag = FALSE, rfPackage = "ranger", ntree = 10L, num.threads = 1L, polr.to.loggedEvents = TRUE, 
        remove.constant = FALSE, remove.collinear = FALSE, eps = 0)
    completed <- mice::complete(z, 1L)
    for (v in names(completed)) if (is.factor(completed[[v]])) 
        completed[[v]] <- as.numeric(as.character(completed[[v]]))
    check_observed(completed, raw)
    list(completed = completed, method = method, predictorMatrix = predictor, chainMean = z$chainMean, 
        chainVar = z$chainVar, loggedEvents = z$loggedEvents)
}

selection_design <- function(a, variables, basis = NULL) {
    fitting <- is.null(basis)
    if (fitting) 
        basis <- setNames(vector("list", length(variables)), variables)
    pieces <- lapply(variables, function(v) {
        x <- a[[v]]
        if (v == "proteinuria_grade") {
            z <- cbind(as.numeric(x == 1L), as.numeric(x == 2L))
        }
        else if (v %in% binary_variables) {
            z <- matrix(x, ncol = 1L)
        }
        else {
            if (v == "creatinine") 
                x <- log1p(x)
            if (fitting) {
                basis[[v]] <<- ns(x, df = 3L)
                z <- basis[[v]]
            }
            else z <- predict(basis[[v]], x)
        }
        colnames(z) <- paste0(v, "_", seq_len(ncol(z)))
        z
    })
    x <- do.call(cbind, pieces)
    stopifnot(all(is.finite(x)))
    list(x = x, basis = basis)
}

group_folds <- function(patient, s) {
    set.seed(s)
    p <- unique(patient)
    assignment <- sample(rep(seq_len(K), length.out = length(p)))
    assignment[match(patient, p)]
}

lasso_one <- function(x, y, weights, foldid) {
    x <- design(x)
    z <- cv.glmnet(x, y, family = "binomial", alpha = 1, weights = weights/mean(weights), foldid = foldid, 
        nfolds = K, lambda = lambda, standardize = TRUE, type.measure = "deviance", maxit = 100000L, 
        parallel = FALSE)
    coefficient <- as.matrix(coef(z, s = "lambda.1se"))[, 1L]
    list(coefficients = coefficient, lambda_1se = z$lambda.1se, lambda_min = z$lambda.min, curve = data.table(lambda = z$lambda, 
        deviance = z$cvm, SE = z$cvsd))
}

effective_weights <- function(w) {
    w <- as.matrix(w)
    stopifnot(all(is.finite(w)), all(w > 0))
    rowMeans(sweep(w, 2L, colSums(w), "/"))
}

probability_scores <- function(y, p, w) {
    p <- pmin(pmax(p, 1e-08), 1 - 1e-08)
    w <- effective_weights(w)
    c(Brier = sum(w * (y - p)^2), Log_loss = -sum(w * (y * log(p) + (1 - y) * log1p(-p))))
}

choose_platt <- function(y, p, weights, fold) {
    weights <- as.matrix(weights)
    stopifnot(length(y) == length(p), nrow(weights) == length(y), length(fold) == length(y), all(y %in% 
        0:1), all(is.finite(p)), all(p >= 0 & p <= 1), setequal(unique(fold), 1:5))
    lp <- qlogis(pmin(pmax(p, 1e-08), 1 - 1e-08))
    fit_one <- function(ii) {
        ww <- effective_weights(weights[ii, , drop = FALSE]) * length(ii)
        f <- tryCatch(glm(y[ii] ~ lp[ii], weights = ww, family = quasibinomial()), error = function(e) NULL)
        if (is.null(f) || !isTRUE(f$converged)) 
            return(NULL)
        cc <- unname(coef(f))
        if (length(cc) != 2L || any(!is.finite(cc)) || cc[2L] <= 0) 
            return(NULL)
        cc
    }
    calibrated <- rep(NA_real_, length(y))
    comparison <- vector("list", 5L)
    for (k in 1:5) {
        ii <- which(fold != k)
        qq <- which(fold == k)
        cc <- fit_one(ii)
        if (is.null(cc)) 
            return(list(use_platt = FALSE, coefficients = c(0, 1), probability = p, reason = "A training cross-fit did not yield a finite positive-slope calibrator"))
        calibrated[qq] <- plogis(cc[1L] + cc[2L] * lp[qq])
        a <- probability_scores(y[qq], p[qq], weights[qq, , drop = FALSE])
        b <- probability_scores(y[qq], calibrated[qq], weights[qq, , drop = FALSE])
        comparison[[k]] <- data.table(Fold = k, Raw_Brier = a[["Brier"]], Platt_Brier = b[["Brier"]], 
            Raw_Log_loss = a[["Log_loss"]], Platt_Log_loss = b[["Log_loss"]])
    }
    comparison <- rbindlist(comparison)
    delta <- comparison$Platt_Brier - comparison$Raw_Brier
    improvement <- -mean(delta)
    one_se <- sd(delta)/sqrt(5)
    use <- improvement > one_se && mean(comparison$Platt_Log_loss) <= mean(comparison$Raw_Log_loss)
    cc <- if (use) 
        fit_one(seq_along(y))
    else c(0, 1)
    if (is.null(cc)) {
        use <- FALSE
        cc <- c(0, 1)
    }
    list(use_platt = use, coefficients = cc, probability = if (use) plogis(cc[1L] + cc[2L] * lp) else p, 
        crossfit_probability = if (use) calibrated else p, fold_scores = comparison, Brier_improvement = improvement, 
        paired_fold_SE = one_se, rule = "Choose Platt only when Brier improvement exceeds one paired-fold SE and log loss does not worsen")
}

ratio <- function(x, n) ifelse(n > 0, x/n, NA_real_)

clinical <- function(y, p, w, cutoffs) {
    yes <- outer(p, cutoffs, ">=")
    tp <- t(crossprod(w, yes * y))
    fp <- t(crossprod(w, yes * (1 - y)))
    n1 <- as.numeric(crossprod(w, y))
    n0 <- as.numeric(crossprod(w, 1 - y))
    fn <- matrix(n1, length(cutoffs), ncol(w), byrow = TRUE) - tp
    tn <- matrix(n0, length(cutoffs), ncol(w), byrow = TRUE) - fp
    se <- ratio(tp, tp + fn)
    sp <- ratio(tn, tn + fp)
    ans <- array(NA_real_, c(length(metric_names), length(cutoffs), ncol(w)), dimnames = list(metric_names, 
        names(cutoffs), NULL))
    ans["Accuracy", , ] <- ratio(tp + tn, tp + fp + tn + fn)
    ans["Precision", , ] <- ratio(tp, tp + fp)
    ans["Sensitivity", , ] <- se
    ans["Specificity", , ] <- sp
    ans["NPV", , ] <- ratio(tn, tn + fn)
    ans["F1", , ] <- ratio(2 * tp, 2 * tp + fp + fn)
    ans["Flagged_fraction", , ] <- ratio(tp + fp, tp + fp + tn + fn)
    ans["Youden", , ] <- se + sp - 1
    ans["Balanced_accuracy", , ] <- (se + sp)/2
    list(metrics = ans, counts = list(TP = tp, FP = fp, TN = tn, FN = fn))
}

curve <- function(y, p, w, with_ap = TRUE) {
    ix <- order(p)
    ties <- cumsum(c(TRUE, diff(p[ix]) != 0))
    pos <- rowsum(w[ix, , drop = FALSE] * y[ix], ties, reorder = FALSE)
    neg <- rowsum(w[ix, , drop = FALSE] * (1 - y[ix]), ties, reorder = FALSE)
    n1 <- colSums(pos)
    n0 <- colSums(neg)
    stopifnot(all(n1 > 0), all(n0 > 0))
    a <- colSums(pos * (colCumsums(neg) - neg/2))/(n1 * n0)
    if (!with_ap) 
        return(a)
    revix <- rev(seq_len(nrow(pos)))
    cp <- colCumsums(pos[revix, , drop = FALSE])
    cn <- colCumsums(neg[revix, , drop = FALSE])
    precision <- ratio(cp, cp + cn)
    precision[!is.finite(precision)] <- 0
    list(AUC = a, AP = colSums(pos[revix, , drop = FALSE] * precision)/n1, Brier = as.numeric(crossprod(w, 
        (p - y)^2))/(n1 + n0))
}

weighted_youden <- function(y, p, w) {
    ix <- order(-p)
    pos <- rowMeans(sweep(w * y, 2L, colSums(w * y), "/"))
    neg <- rowMeans(sweep(w * (1 - y), 2L, colSums(w * (1 - y)), "/"))
    a <- data.table(p = p[ix], pos = pos[ix], neg = neg[ix])[, .(pos = sum(pos), neg = sum(neg)), by = p]
    a[, `:=`(Sensitivity = cumsum(pos), Specificity = 1 - cumsum(neg))]
    a[, `:=`(J, Sensitivity + Specificity - 1)]
    best <- which(abs(a$J - max(a$J)) < 1e-12)
    j <- best[which.max(a$Sensitivity[best])]
    cut <- if (j < nrow(a)) 
        mean(a$p[c(j, j + 1L)])
    else min(p)
    check <- clinical(y, p, w, c(Youden = cut))$metrics
    stopifnot(abs(mean(check["Youden", 1L, ]) - a$J[j]) < 1e-10)
    cut
}

transform_probability <- function(raw, coefficients) {
    stopifnot(length(coefficients) == 2L, all(is.finite(coefficients)), coefficients[2L] > 0)
    if (all(unname(coefficients) == c(0, 1))) 
        return(raw)
    plogis(coefficients[1L] + coefficients[2L] * qlogis(pmin(pmax(raw, 1e-08), 1 - 1e-08)))
}

calibration_assessment <- function(y, p, w) {
    ww <- effective_weights(w) * length(y)
    lp <- qlogis(pmin(pmax(p, 1e-08), 1 - 1e-08))
    messages <- character()
    fits <- withCallingHandlers(list(glm(y ~ offset(lp), weights = ww, family = quasibinomial()), glm(y ~ 
        lp, weights = ww, family = quasibinomial())), warning = function(z) {
        messages <<- c(messages, conditionMessage(z))
        invokeRestart("muffleWarning")
    })
    values <- c(unname(coef(fits[[1L]])[1L]), unname(coef(fits[[2L]])[2L]))
    values[!is.finite(values)] <- NA_real_
    if (!all(vapply(fits, function(x) isTRUE(x$converged), TRUE))) 
        values[] <- NA_real_
    list(intercept = values[1L], slope = values[2L], note = paste(unique(messages), collapse = "; "), 
        converged = all(vapply(fits, function(x) isTRUE(x$converged), TRUE)))
}

score <- function(d, p, w, cut, model, source, evaluation, subset = "All_available") {
    y <- d$y
    w <- as.matrix(w)
    n <- length(y)
    M <- ncol(w)
    stopifnot(nrow(d) == length(p), nrow(w) == n, length(unique(y)) == 2L, all(is.finite(p)), all(p >= 
        0 & p <= 1), all(is.finite(w)), all(w > 0), is.finite(cut))
    cluster <- match(d$patient, unique(d$patient))
    G <- max(cluster)
    a <- curve(y, p, w)
    cl <- clinical(y, p, w, c(Youden = cut))
    cal <- calibration_assessment(y, p, w)
    point <- c(AUC = mean(a$AUC), AP = mean(a$AP), Brier = mean(a$Brier), setNames(vapply(metric_names, 
        function(v) mean(cl$metrics[v, 1L, ]), 0), metric_names))
    z <- data.table(Model = model, Source = source, Evaluation = evaluation, Subset = subset, Admissions = n, 
        Patients = G, Events = sum(y), Observed_prevalence = mean(y), Weighted_prevalence = mean(colSums(w * 
            y)/colSums(w)), ESS_admissions = mean(colSums(w)^2/colSums(w^2)), Threshold = cut, Calibration_assessment_intercept = cal$intercept, 
        Calibration_assessment_slope = cal$slope, Calibration_assessment_converged = cal$converged, Calibration_assessment_note = cal$note)
    for (v in names(point)) z[, `:=`((v), point[[v]])]
    for (v in c("TP", "FP", "TN", "FN")) z[, `:=`(paste0("Weighted_", v), mean(cl$counts[[v]]))]
    z[, `:=`(Observed_TP = sum(p >= cut & y == 1L), Observed_FP = sum(p >= cut & y == 0L), Observed_TN = sum(p < 
        cut & y == 0L), Observed_FN = sum(p < cut & y == 1L))]
    key <- paste(source, evaluation, subset, model, sep = "_")
    cache <- file.path(bootstrap_dir, paste0(key, ".rds"))
    if (file.exists(cache)) {
        saved <- readRDS(cache)
        stopifnot(identical(saved$record, d$record), identical(saved$patient, d$patient), identical(saved$y, 
            y), isTRUE(all.equal(saved$probability, p, tolerance = 1e-14)), isTRUE(all.equal(saved$weights, 
            w, tolerance = 1e-12)), saved$threshold == cut, identical(dim(saved$bootstrap), c(B, length(point))))
        boot <- saved$bootstrap
        rejected <- saved$degenerate_draws_replaced
    }
    else {
        boot <- matrix(NA_real_, B, length(point), dimnames = list(NULL, names(point)))
        done <- rejected <- 0L
        set.seed(123L)
        while (done < B) {
            count <- min(25L, B - done)
            mult <- vapply(seq_len(count), function(i) tabulate(sample.int(G, G, replace = TRUE), nbins = G), 
                integer(G))
            frequency <- mult[cluster, , drop = FALSE]
            keep <- as.numeric(crossprod(frequency, y)) > 0 & as.numeric(crossprod(frequency, 1 - y)) > 
                0
            rejected <- rejected + sum(!keep)
            mult <- mult[, keep, drop = FALSE]
            nb <- ncol(mult)
            if (!nb) 
                next
            ww <- w[, rep(seq_len(M), times = nb), drop = FALSE] * mult[cluster, rep(seq_len(nb), each = M), 
                drop = FALSE]
            q <- curve(y, p, ww)
            clinical_boot <- clinical(y, p, ww, c(Youden = cut))$metrics
            rows <- done + seq_len(nb)
            boot[rows, 1:3] <- cbind(colMeans(matrix(q$AUC, M, nb)), colMeans(matrix(q$AP, M, nb)), colMeans(matrix(q$Brier, 
                M, nb)))
            for (v in metric_names) boot[rows, v] <- colMeans(matrix(clinical_boot[v, 1L, ], M, nb))
            done <- done + nb
        }
        stopifnot(all(is.finite(boot[, 1:3])))
        saved <- list(record = d$record, patient = d$patient, y = y, probability = p, weights = w, threshold = cut, 
            bootstrap = boot, degenerate_draws_replaced = rejected, method = "5000 fixed-prediction patient-cluster draws; model, MI and selection weights not refitted")
        saveRDS(saved, paste0(cache, ".tmp"))
        stopifnot(file.rename(paste0(cache, ".tmp"), cache))
    }
    for (v in names(point)) {
        values <- boot[, v]
        values <- values[is.finite(values)]
        ci <- if (length(values)) 
            quantile(values, c(0.025, 0.975))
        else c(NA_real_, NA_real_)
        z[, `:=`(paste0(v, "_lower"), unname(ci[1L]))]
        z[, `:=`(paste0(v, "_upper"), unname(ci[2L]))]
    }
    z[, `:=`(Bootstrap_replicates = B, Degenerate_draws_replaced = rejected)]
    note("TERMINAL_FIXED_PREDICTION_CI_COMPLETE", key)
    list(summary = z, bootstrap = boot, prediction = p, reference = d, weights = w)
}

