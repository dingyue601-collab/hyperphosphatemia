options(stringsAsFactors=FALSE)
root <- Sys.getenv("HYPERPHOS_RUN")
source(file.path(root,"analysis/common.R"))
out <- file.path(root,"results/analysis")
dest <- file.path(out,"two_predictor")
d <- fread(file.path(dest,"Training_reference.csv"))
w <- as.matrix(fread(file.path(out,"Training_OOF_weights.csv")))
models <- list();fold_results <- list()
for(name in c("TabM2","Logistic2","NonIPW_TabM2")) {
  file <- if(name=="NonIPW_TabM2")file.path(dest,name,"TabM2_OOF_probabilities.csv")else
    file.path(dest,paste0(name,"_OOF_probabilities.csv"))
  pp <- as.matrix(fread(file));p <- rowMeans(pp)
  weights <- if(name=="NonIPW_TabM2")matrix(1,nrow(d),30L)else w
  cal <- choose_platt(d$label,p,weights,d$fold)
  if(is.null(cal$crossfit_probability)) cal$crossfit_probability <- p
  cutoff <- weighted_youden(d$label,cal$probability,weights)
  models[[name]] <- list(raw=p,probability=cal$probability,crossfit=cal$crossfit_probability,
                         calibration=cal,threshold=cutoff,features=c("age","creatinine"))
  fold_results[[name]] <- rbindlist(lapply(1:5,function(k) {
    ii <- which(d$fold==k);z <- curve(d$label[ii],p[ii],weights[ii,,drop=FALSE])
    loss <- probability_scores(d$label[ii],cal$crossfit_probability[ii],weights[ii,,drop=FALSE])
    data.table(Model=name,Fold=k,AUROC=mean(z$AUC),AUPRC=mean(z$AP),Brier=loss[["Brier"]])
  }))
}
saveRDS(models,file.path(dest,"Developed_models.rds"))
fwrite(rbindlist(fold_results),file.path(dest,"Training_fold_performance.csv"))
write_json(lapply(models,function(x)list(threshold=x$threshold,features=x$features,
  calibration_coefficients=unname(x$calibration$coefficients))),file.path(dest,"Frozen_decisions.json"),auto_unbox=TRUE,pretty=TRUE,digits=NA)
write_json(list(completed=TRUE,fixed_features_all_folds=c("age","creatinine"),
  models=lapply(models,function(x)list(threshold=x$threshold,
    calibration=if(x$calibration$use_platt)"Platt" else "None",
    calibration_coefficients=unname(x$calibration$coefficients)))),
  file.path(dest,"Development_complete.json"),auto_unbox=TRUE,pretty=TRUE,digits=NA)

