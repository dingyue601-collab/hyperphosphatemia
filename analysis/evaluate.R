options(stringsAsFactors=FALSE)
root <- Sys.getenv("HYPERPHOS_RUN")
source(file.path(root,"analysis/common.R"))
out <- file.path(root,"results/analysis")
dest <- file.path(out,"two_predictor")
B <- 5000L
note <- function(...)cat(...,"\n")
bootstrap_dir <- file.path(out,"Patient_bootstrap")
dir.create(bootstrap_dir,showWarnings=FALSE)
ref <- as.data.table(readRDS(file.path(root,"data/cleaned/Validation_labels.rds")))
test <- fread(file.path(out,"Test_reference_without_labels.csv"))
ref <- ref[match(test$record,ref$record)]
stopifnot(identical(ref$record,test$record),nrow(ref)==2026L,uniqueN(ref$patient)==745L,sum(ref$y)==258L)
weights <- as.matrix(fread(file.path(out,"Test_IPW_weights.csv")))
training <- readRDS(file.path(out,"Training_predictions_and_calibration.rds"))
two <- readRDS(file.path(dest,"Developed_models.rds"))
thresholds <- fread(file.path(out,"Training_Youden_performance.csv"))
results <- list()
for(name in c("Logistic","RF","SVM","XGBoost","LightGBM","CatBoost","TabM")) {
  folder <- file.path(out,"final",paste0(name,"7"))
  p <- rowMeans(as.matrix(fread(file.path(folder,"Test_probabilities.csv"))))
  p <- transform_probability(p,training$calibration[[name]]$coefficients)
  results[[paste0(name,"7")]] <- score(ref,p,weights,thresholds[Model==name,Threshold],
                                         paste0(name,"7"),"Test","IPW")
}
for(name in c("TabM2","Logistic2","NonIPW_TabM2")) {
  path <- switch(name,TabM2=file.path(dest,"Test_probabilities.csv"),
    Logistic2=file.path(dest,"Logistic2_Test_probabilities.csv"),
    NonIPW_TabM2=file.path(dest,name,"Test_probabilities.csv"))
  p <- transform_probability(rowMeans(as.matrix(fread(path))),two[[name]]$calibration$coefficients)
  w <- if(name=="NonIPW_TabM2")matrix(1,nrow(ref),30L)else weights
  results[[name]] <- score(ref,p,w,two[[name]]$threshold,name,"Test",if(name=="NonIPW_TabM2")"Unweighted" else "IPW")
}
summary <- rbindlist(lapply(results,`[[`,"summary"))
fwrite(summary,file.path(out,"Test_performance.csv"))
contrasts <- rbindlist(lapply(c("AUC","AP","Brier"),function(metric) {
  a <- results$TabM2;b <- results$Logistic2
  stopifnot(identical(a$reference$record,b$reference$record),identical(a$weights,b$weights))
  interval <- quantile(a$bootstrap[,metric]-b$bootstrap[,metric],c(.025,.975))
  data.table(Metric=c(AUC="AUROC",AP="AUPRC",Brier="Brier")[[metric]],
    TabM_minus_Logistic=a$summary[[metric]]-b$summary[[metric]],
    Lower=unname(interval[1L]),Upper=unname(interval[2L]))
}))
fwrite(contrasts,file.path(out,"Paired_two_input_differences.csv"))
p <- results$TabM2$prediction
cut <- two$TabM2$threshold
pool <- split(seq_len(nrow(ref)),ref$patient)
set.seed(123L)
random <- sort(vapply(pool,function(ii)ii[sample.int(length(ii),1L)],0L))
equal <- weights
for(m in 1:30) equal[,m] <- weights[,m]/ave(weights[,m],ref$patient,FUN=sum)
inside <- which(ref$phosphate>=1.40 & ref$phosphate<=1.50)
outside <- setdiff(seq_len(nrow(ref)),inside)
sensitivity <- list()
cases <- list(Random_one_admission=list(index=random,w=weights),
  Patient_equal_IPW=list(index=seq_len(nrow(ref)),w=equal),
  Excluding_boundary=list(index=outside,w=weights),
  Patient_equal_IPW_excluding_boundary=list(index=outside,w=equal),
  Within_boundary=list(index=inside,w=weights))
for(name in names(cases)) {
  z <- cases[[name]];ii <- z$index
  if(name=="Patient_equal_IPW_excluding_boundary") {
    w <- weights[ii,,drop=FALSE]
    for(m in 1:30)w[,m] <- w[,m]/ave(w[,m],ref$patient[ii],FUN=sum)
  } else w <- z$w[ii,,drop=FALSE]
  sensitivity[[name]] <- score(ref[ii],p[ii],w,cut,"TabM2","Test","IPW",name)
}
sensitivity$NonIPW <- results$NonIPW_TabM2
fwrite(rbindlist(lapply(sensitivity,`[[`,"summary")),file.path(out,"Test_sensitivity.csv"))
saveRDS(list(models=results,sensitivity=sensitivity),file.path(out,"Evaluation_results.rds"))
