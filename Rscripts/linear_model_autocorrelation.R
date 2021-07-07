library(nlme)
library(dplyr)
library(spdep)
library(writexl)
library(sjstats)
library(CatEncoders)

environment_train <- read.csv("../datasets/abund_merged_dataset_onlyenvironment.csv")
distances <- read.csv("../datasets/distances.csv", sep = ';')

environment_train <- inner_join(environment_train, distances, by=c("plot","subplot"))

years <- unique(environment_train[c("year")])
species <- unique(environment_train[c("species")])
rownames(species) <- 1:23


error_list <- c()
year_esp_list <- c()


for (y in 1:5){
  for (s in 1:23){
    
    year_i <- years[y,1]
    specie_i <- species[s,1]
    
    print(paste("YEAR: ",year_i," and SPECIE: ",specie_i))
    
    year_esp <- filter(environment_train, year == year_i & species == specie_i & individuals != 0)
    
    if (dim(year_esp)[1] != 0 & dim(year_esp)[1] > 6 & specie_i != "PUPA" & year_i != 2017){
      year_esp <- select(year_esp, c('individuals','ph','salinity','precip',
                                     'cl','co3','c','mo','n','cn','p','ca','mg','k','na','x_coor2','y_coor2','plot'))
      
      
      # STANDARIZATION
      
      year_esp_std <- select(year_esp, -c('individuals','plot'))
      year_esp_std <- scale(year_esp_std)
      
      year_esp_df <- data.frame(year_esp_std)
      year_esp_df$individuals <- year_esp$individuals
      year_esp_df$precip <- year_esp$precip
      year_esp_df$plot <- year_esp$plot
      
      year_esp_list <- c(year_esp_list, paste(year_i,'_',specie_i))
      
      ctrl <- lmeControl(opt='optim')
      model_g <- lme(individuals ~ ., data= year_esp_df , random = ~1 |plot, control=ctrl,
                     corr = corSpatial(form = ~ x_coor2 + y_coor2, type ="gaussian", nugget = T), method = "ML")
      
      rmse <- rmse(model_g, normalized = TRUE)
      mse <- mse(model_g)
      
      print(paste("RMSE: ",rmse))
      print(paste("MSE: ",mse))
      error_list <- c(error_list, rmse, mse)
    }
  }
}


error_species_year <- data.frame(year_esp_list, error_list)
write_xlsx(error_species_year,"../results/error_species_year.xlsx")




