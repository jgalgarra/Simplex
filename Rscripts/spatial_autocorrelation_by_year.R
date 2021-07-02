library(dplyr)
library(spdep)
library(writexl)

environment_train <- read.csv("../datasets/abund_merged_dataset_onlyenvironment.csv")
years <- unique(environment_train[c("year")])

moran_list <- c()
year_list <- c()

for (y in 1:5){
    
  year_i <- years[y,1]
  
  print(paste("YEAR: ",year_i))
  
  year_esp <- filter(environment_train, year == year_i)
  
  distancias.matriz <- as.matrix(dist(cbind(year_esp$x, year_esp$y)))
  
  #now, I make the inverse matrix and the diagonal equal to 0
  plots.dists.inv <- 1/distancias.matriz
  diag(plots.dists.inv) <- 0
  
  plots.dists.inv[plots.dists.inv == Inf] <- 10000
  
  matrix_w <- mat2listw(plots.dists.inv)
  
  if (any(year_esp$individuals != 0)){
    
    year_esp_list <- c(year_esp_list, year_i)
    
    moran_stat <- moran.test(year_esp$individuals,matrix_w)
    moran_list <- c(moran_list, moran_stat$estimate[1])
    print(moran_stat)
  
}
}

i_moran_species_year <- data.frame(year_esp_list, moran_list)
write_xlsx(i_moran_species_year,"../results/spatial_autocorrelation_year.xlsx")


