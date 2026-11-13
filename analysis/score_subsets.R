options(stringsAsFactors=FALSE)
root <- Sys.getenv("HYPERPHOS_RUN")
source(file.path(root,"analysis/common.R"))
out <- file.path(root,"results/analysis")
dest <- file.path(out,"subsets")
mode <- commandArgs(TRUE)[1L]
d <- fread(file.path(out,"Training_reference.csv"))
weights <- as.matrix(fread(file.path(out,"Training_OOF_weights.csv")))
candidates <- fread(file.path(dest,"Candidates.csv"))
fold_scores <- function(p,w) rbindlist(lapply(1:5,function(k) {
  ii <- which(d$fold==k);z <- curve(d$label[ii],p[ii],w[ii,,drop=FALSE])
  loss <- probability_scores(d$label[ii],p[ii],w[ii,,drop=FALSE])
  data.table(Fold=k,AUROC=mean(z$AUC),AUPRC=mean(z$AP),Brier=loss[["Brier"]],Log_loss=loss[["Log_loss"]])
}))
if(mode=="search") {
  result <- rbindlist(lapply(candidates$Mask,function(mask) {
    pp <- as.matrix(fread(file.path(dest,"search",sprintf("subset%03d.csv",mask))))
    stopifnot(identical(dim(pp),c(nrow(d),5L)),all(is.finite(pp)))
    f <- fold_scores(rowMeans(pp),weights[,c(1,7,13,19,25),drop=FALSE])
    cbind(candidates[Mask==mask],data.table(Mean_fold_AUROC=mean(f$AUROC),
      Mean_fold_AUPRC=mean(f$AUPRC),Mean_fold_Brier=mean(f$Brier)))
  }))
  setorder(result,Variables,-Mean_fold_AUROC,-Mean_fold_AUPRC,Mean_fold_Brier,Mask)
  fwrite(result,file.path(dest,"Search_5MI_all_subsets.csv"))
  top <- result[,head(.SD,if(.BY$Variables<=4L)2L else 1L),by=Variables]
  write_json(list(masks=unique(c(top$Mask,127L))),file.path(dest,"Confirmation_queue.json"),auto_unbox=TRUE,pretty=TRUE)
} else {
  masks <- fromJSON(file.path(dest,"Confirmation_queue.json"))$masks
  result <- rbindlist(lapply(masks,function(mask) {
    pp <- as.matrix(fread(file.path(dest,"confirmation",sprintf("subset%03d.csv",mask))))
    stopifnot(identical(dim(pp),c(nrow(d),30L)),all(is.finite(pp)))
    p <- rowMeans(pp);cal <- choose_platt(d$label,p,weights,d$fold)
    f <- fold_scores(p,weights)
    crossfit <- if(is.null(cal$crossfit_probability))p else cal$crossfit_probability
    calibrated <- fold_scores(crossfit,weights)
    cl <- clinical(d$label,cal$probability,weights,c(Youden=weighted_youden(d$label,cal$probability,weights)))
    lp <- qlogis(pmin(pmax(cal$probability,1e-8),1-1e-8))
    assessment <- unname(coef(glm(d$label~lp,weights=effective_weights(weights)*nrow(d),family=quasibinomial())))
    row <- cbind(candidates[Mask==mask],data.table(Mean_fold_AUROC=mean(f$AUROC),Mean_fold_AUPRC=mean(f$AUPRC),
      Mean_fold_Brier=mean(calibrated$Brier),Mean_fold_Log_loss=mean(calibrated$Log_loss),
      Calibration=if(cal$use_platt)"Platt" else "None",OOF_calibration_intercept=assessment[1L],OOF_calibration_slope=assessment[2L],
      Youden_threshold=weighted_youden(d$label,cal$probability,weights)))
    for(v in metric_names)row[,(v):=mean(cl$metrics[v,1L,])]
    row
  }))
  base <- result[Mask==127L,Mean_fold_AUROC]
  result[,AUROC_loss:=base-Mean_fold_AUROC]
  setorder(result,Variables,-Mean_fold_AUROC,-Mean_fold_AUPRC,Mean_fold_Brier)
  fwrite(result,file.path(dest,"Confirmed_30MI_subsets.csv"))
  best <- result[,.SD[1L],by=Variables]
  selected <- rbindlist(lapply(c(.01,.02),function(limit) {
    z <- best[AUROC_loss<=limit+1e-12][1L];z[,Loss_limit:=limit];z
  }))
  fwrite(selected,file.path(dest,"Selected_retrained_subsets.csv"))
}
