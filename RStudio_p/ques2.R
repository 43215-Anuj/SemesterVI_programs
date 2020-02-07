library(editrules)
library(dplyr)
library(naniar)

setwd("C:\\Users\\anujk\\Documents\\RStudio_p")
data_iris <- read.csv("dirty_iris.csv")

print(data_iris)
#Count total number of record
total = nrow(data_iris)
print(paste0("Total number of records are: ",total))
#Exatract missing dataframe
miss <- is.na(data_iris)
#Count number of missing record
missing = sum(miss)
#count complete record
complete = total - missing
print(paste0("Records with Complete data are : ",complete))
#Percentage of complete record
percent_complete = (complete * 100)/total
print(paste0("percentage of records completed :",percent_complete,"%"))

#ALTERNATIVE WAY
complete_data <- complete.cases(data_iris)
print(complete_data)
hard_complete = sum(complete_data)
print(hard_complete)
na.omit(data_iris)
#ALTERNATIVE ENDS

#PERCENTAGE OF MISSING DATA
print(1-mean(miss))
data_iris

#let special value be infinit/Inf
is.na(data_iris) <- sapply(data_iris, is.infinite)
print(data_iris)

#alternative better way to replace Inf with NA
replaceSpecial <- function(data_i){
  data_i[data_i==Inf]<-NA
  data_i[data_i==is.null]<-NA
  data_i[data_i==is.nan]<-NA
  return(data_i)
}

#Consult the Rules file "iris_constrants.txt"
E = editfile("C:/Users/anujk/Documents/RStudio_p/iris_constrants.txt")
Result <- violatedEdits(E,data_iris)
print(Result)
summary(Result)
plot(Result)
plot(E)

boxplot(data_iris$Sepal.Length)
boxplot.stats(data_iris$Sepal.Length, conf=1.5, do.conf = TRUE)
  