options(encoding = "UTF-8", stringsAsFactors = FALSE, warn = 1)

suppressPackageStartupMessages({
    library(data.table)
    library(jsonlite)
})

setDTthreads(1L)

root <- Sys.getenv("HYPERPHOS_RUN")
base <- file.path(root, "results/analysis")

dest <- file.path(base, "two_predictor")

dir.create(file.path(dest, "inputs"), recursive = TRUE, showWarnings = FALSE)

features <- c("age", "creatinine")

M <- 30L

input <- readRDS(file.path(base, "Development_inputs.rds"))

d <- as.data.table(input$reference)

stopifnot(nrow(d) == 4575L, uniqueN(d$patient) == 1736L, sum(d$y) == 519L, identical(input$test_labels_read, 
    FALSE), all(d[, uniqueN(fold), by = patient]$V1 == 1L))

weights <- as.matrix(fread(file.path(base, "Training_OOF_weights.csv")))

design2 <- function(a) cbind(`(Intercept)` = 1, age = a$age / 10, creatinine = a$creatinine / 100)
fit2 <- function(a) {
    fit <- glm.fit(design2(a), a$label,
        weights = a$sample_weight / mean(a$sample_weight), family = quasibinomial("logit"),
        control = glm.control(epsilon = 1e-8, maxit = 100L), singular.ok = FALSE)
    stopifnot(fit$converged, fit$rank == 3L, all(is.finite(coef(fit))))
    unname(coef(fit))
}

probability <- matrix(NA_real_, nrow(d), M)
coefficients <- matrix(NA_real_, M, 3L)
coefficient_rows <- list()
manifest <- list()

for (b in input$blocks) {
    z <- readRDS(b$predictive_imputations)
    stopifnot(identical(z$train, b$train), identical(z$query, b$query), !length(intersect(d$patient[b$train], 
        d$patient[b$query])))
    if (b$fold) 
        stopifnot(identical(b$query, which(d$fold == b$fold)))
    entries <- lapply(seq_len(M), function(m) {
        w <- readRDS(file.path(b$weights, sprintf("m%02d.rds", m)))
        stopifnot(identical(w$record, d$record[b$train]), identical(w$query_record, d$record[b$query]))
        a <- as.data.table(z$completed[[m]])[, ..features]
        a[, `:=`(label = d$y[b$train], sample_weight = w$train_weights)]
        stopifnot(all(is.finite(as.matrix(a))), all(a$sample_weight > 0))
        ap <- file.path(dest, "inputs", sprintf("fold%02d_m%02d_train.csv", b$fold, m))
        fwrite(a, ap)
        cc <- fit2(a)
        if (b$fold == 0L) coefficients[m, ] <<- cc
        coefficient_rows[[length(coefficient_rows) + 1L]] <<- data.table(Fold = b$fold, MI = m,
            Intercept = cc[1L], Age_per_10_years = cc[2L], Creatinine_per_100_umol_L = cc[3L])
        qp <- NULL
        if (b$fold) {
            q <- as.data.table(z$queries[[m]])[, ..features]
            stopifnot(all(is.finite(as.matrix(q))), max(abs(w$query_weights - weights[b$query, m])) < 
                1e-12)
            qp <- file.path(dest, "inputs", sprintf("fold%02d_m%02d_query.csv", b$fold, m))
            fwrite(q, qp)
            probability[b$query, m] <<- plogis(drop(design2(q) %*% cc))
        }
        list(imputation = m, train = ap, query = qp)
    })
    manifest[[length(manifest) + 1L]] <- list(fold = b$fold, features = features, train_indices = b$train, 
        query_indices = b$query, categorical_levels = list(), entries = entries)
    cat(format(Sys.time(), "%F %T"), "FIXED2_INPUT_BLOCK_COMPLETE", b$fold, "\n")
    flush.console()
}

stopifnot(all(is.finite(probability)), all(probability >= 0 & probability <= 1),
    all(is.finite(coefficients)))
colnames(coefficients) <- c("Intercept", "Age_per_10_years", "Creatinine_per_100_umol_L")
colnames(probability) <- sprintf("m%02d", 1:M)
fwrite(as.data.table(probability), file.path(dest, "Logistic2_OOF_probabilities.csv"))
fwrite(rbindlist(coefficient_rows), file.path(dest, "Logistic_coefficients_all_folds.csv"))
fwrite(data.table(MI = 1:M, coefficients), file.path(dest, "Logistic_final_coefficients.csv"))
saveRDS(coefficients, file.path(dest, "Logistic_final_coefficients.rds"))
test_probability <- vapply(1:M, function(m) {
    q <- fread(file.path(root, "results/predictors/test_inputs", sprintf("test_m%02d.csv", m)),
        select = features)
    stopifnot(nrow(q) == 2026L, all(is.finite(as.matrix(q))))
    plogis(drop(design2(q) %*% coefficients[m, ]))
}, numeric(2026L))
colnames(test_probability) <- sprintf("m%02d", 1:M)
fwrite(as.data.table(test_probability), file.path(dest, "Logistic2_Test_probabilities.csv"))

fwrite(d[, .(record, patient, fold, label = y)], file.path(dest, "Training_reference.csv"))

write_json(list(features = features, imputations = M, blocks = manifest, test_labels_read = FALSE, predictor_choice = "Age and creatinine; fixed in all five folds", 
    evaluation_limit = "Fixed-input patient-separated OOF evaluation"), 
    file.path(dest, "Training_job.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)

write_json(list(completed = TRUE, blocks = 6L, imputations = M, fixed_features = features, fixed_predictor_count = 2L, 
    test_labels_read = FALSE), file.path(dest, "Inputs_complete.json"), auto_unbox = TRUE, pretty = TRUE, 
    digits = NA)

