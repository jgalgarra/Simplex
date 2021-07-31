library("Metrics")
library(grid)
library(gridExtra)
library(cowplot)
library(ggplot2)
twosteppred <- read.csv("../Pyscripts/rfpred.csv")
abiopred <- read.csv("../Pyscripts/abiopred.csv")

rmse_twostep <- rmse(twosteppred$real,twosteppred$prediction)
rse_twostep <- rse(twosteppred$real,twosteppred$prediction)
twosteppred$error <- (twosteppred$real - twosteppred$prediction)

rmse_abio <- rmse(abiopred$real,abiopred$prediction)
rse_abio <- rse(abiopred$real,abiopred$prediction)
abiopred$error <- (abiopred$real - abiopred$prediction)

print(sprintf("twostep      RMSE: %.02f RSE: %.03f\n",rmse_twostep,rse_twostep))
print(sprintf("ABIOTIC RMSE: %.02f RSE: %.03f\n",rmse_abio,rse_abio))

plot_scatter <- function(datos){
  p <- ggplot(data=datos)+geom_point(aes(x=real,y=prediction))+scale_x_log10()
  return(p)
}

mybreaks=c(0,1,3,5,10,50,100,200,500)

plot_error <- function(datos,relleno="blue",titulo=""){
  p <- ggplot(data=datos)+geom_point(aes(x=real,y=error),size=1, 
                                     shape=19, color=relleno, 
                                     alpha=0.05)+ggtitle(titulo)+
       scale_x_log10(breaks=mybreaks)+ylim(c(-250,500))+theme_bw()
  return(p)
}

ptwostep <- plot_scatter(twosteppred)
pabio <- plot_scatter(abiopred)

perror_twostep <- plot_error(twosteppred,titulo=sprintf("TWO STEP RMSE: %.02f RSE: %.03f\n",rmse_twostep,rse_twostep)) 

perror_abio <- plot_error(abiopred,relleno="red",titulo=sprintf("ABIOTIC RMSE: %.02f RSE: %.03f\n",rmse_abio,rse_abio)) 

g <- plot_grid(
  perror_twostep, perror_abio,
  ncol = 2,labels=c("A","B"),
  label_size = 16
)