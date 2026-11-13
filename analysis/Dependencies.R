packages <- c("data.table","mice","ranger","glmnet","jsonlite","splines","MASS",
              "nnet","matrixStats","reticulate","WeightSVM","xgboost","lightgbm","catboost")
available <- vapply(packages,requireNamespace,TRUE,quietly=TRUE)
if(any(!available))stop("Unavailable R packages: ",paste(packages[!available],collapse=", "))
data.frame(Package=packages,Version=vapply(packages,function(p)as.character(packageVersion(p)),""))
