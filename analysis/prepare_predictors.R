options(encoding="UTF-8", stringsAsFactors=FALSE)
root <- Sys.getenv("HYPERPHOS_RUN")
source(file.path(root,"analysis/common.R"))
setDTthreads(1L)
out <- file.path(root,"results/predictors")
dir.create(file.path(out,"selection"),recursive=TRUE,showWarnings=FALSE)
dir.create(file.path(out,"test_inputs"),showWarnings=FALSE)
input <- readRDS(file.path(root,"data/cleaned/Development_inputs.rds"))
d <- as.data.table(input$training)
v <- as.data.table(input$validation_x)
common <- input$candidates
stopifnot(length(common)==26L,nrow(d)==4575L,nrow(v)==2026L,
          !length(intersect(d$patient,v$patient)),!any(c("y","label","phosphate")%in%names(v)),
          all(d[,uniqueN(fold),by=patient]$V1==1L))
saveRDS(input,file.path(out,"Development_inputs.rds"))
fwrite(d[,.(record,patient,fold,label=y)],file.path(out,"Training_reference.csv"))
for(k in 0:5) {
  path <- file.path(out,"selection",sprintf("MI_fold%02d.rds",k))
  if(file.exists(path)) next
  ii <- if(k==0L)seq_len(nrow(d))else which(d$fold!=k)
  qq <- if(k==0L)integer()else which(d$fold==k)
  raw <- as.data.frame(d[ii,..common])
  s <- seed+10000L+k*100L
  completed <- impute_fit(raw,s)
  queries <- vector("list",M)
  if(k>0L) for(m in 1:M) {
    fitted <- freeze_imputer(completed$completed[[m]],raw,s+m*1000L)
    queries[[m]] <- impute_query(fitted,as.data.frame(d[qq,..common]),s+m*1000L+50L)
  }
  saveRDS(list(fold=k,train=ii,query=qq,variables=common,completed=completed$completed,
               queries=queries,imputation_method=completed$method,loggedEvents=completed$loggedEvents,
               chainMean=completed$chainMean,chainVar=completed$chainVar),path)
}
full <- readRDS(file.path(out,"selection/MI_fold00.rds"))
for(m in 1:M) {
  path <- file.path(out,"test_inputs",sprintf("test_m%02d.rds",m))
  raw <- as.data.frame(d[,..common]); query <- as.data.frame(v[,..common])
  if(file.exists(path)) {
    z <- readRDS(path);check_observed(z,query)
  } else {
    fitted <- freeze_imputer(full$completed[[m]],raw,seed+500000L+m*1000L)
    z <- impute_query(fitted,query,seed+800000L+m*1000L)
    check_observed(z,query);saveRDS(z,path)
  }
  fwrite(as.data.table(z),sub("rds$","csv",path))
}
write_json(list(completed=TRUE,imputations=M,rows=nrow(v),test_labels_read=FALSE),
           file.path(out,"Test_inputs_complete.json"),auto_unbox=TRUE,pretty=TRUE)
