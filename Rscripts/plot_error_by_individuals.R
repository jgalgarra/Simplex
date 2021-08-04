# Plot prediction errors of the ABIOTIC and TWO STEP methods
# Author: Javier Garcia-Algarra
# August 2021
#
# The script reads one simulation run of ABIOTIC and TWO STEP methods and
# plot the prediction errors (e = y - y') by number of individuals for all
# species
#
# Results: plots/Errors_indivs (both .png and. tiff) 


library("Metrics")
library(grid)
library(gridExtra)
library(cowplot)
library(ggplot2)
twosteppred <- read.csv("../results/TWO_STEP_rfpred.csv")
abiopred <- read.csv("../results/abiopred.csv")

rmse_twostep <- rmse(twosteppred$real,twosteppred$prediction)
rse_twostep <- rse(twosteppred$real,twosteppred$prediction)
twosteppred$error <- (twosteppred$real - twosteppred$prediction)

rmse_abio <- rmse(abiopred$real,abiopred$prediction)
rse_abio <- rse(abiopred$real,abiopred$prediction)
abiopred$error <- (abiopred$real - abiopred$prediction)

print(sprintf("twostep      RMSE: %.02f RSE: %.03f",rmse_twostep,rse_twostep))
print(sprintf("ABIOTIC RMSE: %.02f RSE: %.03f",rmse_abio,rse_abio))

plot_scatter <- function(datos){
  p <- ggplot(data=datos)+geom_point(aes(x=real,y=prediction))+scale_x_log10()
  return(p)
}

mybreaks=c(0,0.1,1,3,5,10,50,100,200,500)

plot_error <- function(datos,relleno="blue",titulo=""){
  p <- ggplot(data=datos)+
       geom_point(aes(x=real,y=error),size=2, shape=19, color=relleno, alpha=0.05)+
       geom_text(x=1, y=275, label=titulo, hjust=0) + 
       scale_x_sqrt(breaks=mybreaks)+
       ylim(c(-250,300))+theme_bw()+ylab("Error")+xlab("Individuals")+
       theme(        axis.text = element_text(face="bold", size=11),
                     axis.title.x = element_text(face="bold", size=13),
                     axis.title.y  = element_text(face="bold", size=13))
    
  return(p)
}

ptwostep <- plot_scatter(twosteppred)
pabio <- plot_scatter(abiopred)

perror_twostep <- plot_error(twosteppred,titulo=sprintf("TWO STEP RMSE: %.02f R2: %.03f",rmse_twostep,1-rse_twostep)) 

perror_abio <- plot_error(abiopred,relleno="red",titulo=sprintf("ABIOTIC RMSE: %.02f R2: %.03f",rmse_abio,1-rse_abio)) 


odir <- "../plots"
if (!dir.exists(odir))
  dir.create(odir)
ppi <- 300
nfile <- paste0(odir,"/Errors_indivs")
png(paste0(nfile,".png"), width=15*ppi, height=5*ppi, res=ppi)
g <- plot_grid(
  perror_twostep, perror_abio,
  ncol = 2,labels=c("A","B"),
  label_size = 16
)
print(g)
dev.off()

tiff(paste0(nfile,".tiff"), units="in", width=15, height=5, res=ppi)
print(g)
dev.off()