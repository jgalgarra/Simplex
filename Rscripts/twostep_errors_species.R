# Compute errors for species using the TWO STEP predictor by species
# Author: Javier Garcia-Algarra
# February 2021
#
# Results: tables/datos_err_iter.csv            Cumulative errors by iteration
#          tables/datos_err_onlyspecies.csv     Cumulative errors by species
#          tables/datos_err_species.csv         Errors by species and interation

library(readxl)
library(Metrics)

lHojas <- c("Linear_Regression","Random_Forest","XGBoost")

datoserr <- data.frame("Species"=c(),"Iteration"=c(),
                       "Method"=c(),"RMSE"=c(),"RSE"=c(),"R2"=c())

file <- paste0("../results/TWOSTEP_byspecies_100.xlsx")
tdir <- "../tables"
if (!dir.exists(tdir))
  dir.create(tdir)
for (Hoja in lHojas)
{
  twostep <- read_excel(file,sheet = Hoja)
  names(twostep) <- c("Species","Value","Prediction","Iteration")
  liters <- unique(twostep$Iteration)
  lspecies <- unique(twostep$Species)
  for (s in lspecies){
    for (iter in liters){
     print(paste(Hoja,s,"iteration",iter))
     datoss <- twostep[(twostep$Species == s) & (twostep$Iteration==iter),]
     rmses <- rmse(datoss$Value,datoss$Prediction)
     rses <- rse(datoss$Value,datoss$Prediction)
     r2 <- 1-rses # First revision change. Provide R2 instead of RSE
     datoserr <- rbind(datoserr,data.frame("Species"=s,"Iteration"=iter,
                                           "Method"=Hoja,
                                           "RMSE"=rmses,"RSE"=rses,"R2"=r2))
    }
  }
  write.csv2(datoserr,paste0(tdir,"/datos_err_species.csv"),row.names = FALSE)
}

datosall <- data.frame("Iteration"=c(),"Method"=c(),"RMSE"=c(),"RSE"=c(), "R2"=c())

for (Hoja in lHojas)
{
  twostep <- read_excel(file,sheet = Hoja)
  names(twostep) <- c("Species","Value","Prediction","Iteration")
  liters <- unique(twostep$Iteration)
    for (iter in liters){
      print(paste(Hoja,"iteration",iter))
      datoss <- twostep[(twostep$Iteration==iter),]
      rmses <- rmse(datoss$Value,datoss$Prediction)
      rses <- rse(datoss$Value,datoss$Prediction)
      r2 <- 1-rses # First revision change. Provide R2 instead of RSE
      datosall <- rbind(datosall,data.frame("Iteration"=iter,
                                            "Method"=Hoja,
                                            "RMSE"=rmses,"RSE"=rses,"R2"=r2))
    }  
  write.csv2(datosall,paste0(tdir,"/datos_err_iter.csv"),row.names = FALSE)
}


datosall <- data.frame("Species"=c(),"Method"=c(),"RMSE"=c(),"RSE"=c(),"R2"=c())
for (Hoja in lHojas)
{
  twostep <- read_excel(file,sheet = Hoja)
  names(twostep) <- c("Species","Value","Prediction","Iteration")
  lspecies <- unique(twostep$Species)
  for (s in lspecies){
    print(paste(Hoja,"Species",s))
    datoss <- twostep[(twostep$Species==s),]
    rmses <- rmse(datoss$Value,datoss$Prediction)
    rses <- rse(datoss$Value,datoss$Prediction)
    r2 <- 1-rses # First revision change. Provide R2 instead of RSE
    datosall <- rbind(datosall,data.frame("Species"=s,
                                          "Method"=Hoja,
                                          "RMSE"=rmses,"RSE"=rses,"R2"=r2))
  }
  write.csv2(datosall,paste0(tdir,"/datos_err_onlyspecies.csv"),row.names = FALSE)
}
