# Plot errors by species
# Author: Javier Garcia-Algarra
# February 2021
#
# Results: plots/Errores_species_[MODEL] (both .png and. tiff) 


library(ggplot2)
library(tidyverse)
library(scales)
library(cowplot)

tdir <- "../tables"
ResumenErrores <- read.csv2(paste0(tdir,"/Model_Errors.csv"))
ResumenErrores[ResumenErrores$Method=="Linear Regressor",]$Method = "Linear_Regression"
ResumenErrores[ResumenErrores$Method=="Random Forest",]$Method = "Random_Forest"

datos_err_iter <- read.csv2(paste0(tdir,"/datos_err_iter.csv"))
metodos <- unique(datos_err_iter$Method)
for (m in metodos){
  d <- datos_err_iter[datos_err_iter$Method==m,]
  d$medianRMSE <- 0
  print(paste("Method", m, "medianRMSE",median(d$RMSE)))
}

for (m in metodos){
  d <- datos_err_iter[datos_err_iter$Method==m,]
  d$medianRSE <- 0
  print(paste("Method", m, "medianRSE",median(d$RSE)))
}

for (m in metodos){
  d <- datos_err_iter[datos_err_iter$Method==m,]
  d$medianR2 <- 0
  print(paste("Method", m, "medianR2",median(d$R2)))
}

datos_err_species <- read.csv2(paste0(tdir,"/datos_err_species.csv"))
metodos <- unique(datos_err_species$Method)
for (m in metodos){
d <- datos_err_species[datos_err_species$Method==m,]
d$medianRMSE <- 0
for (s in unique(d$Species))
  d[d$Species==s,]$medianRMSE=median(d[d$Species == s ,]$RMSE)
d$Species <- reorder(d$Species,d$medianRMSE)
etqs <- c(0,1,0.5,1, 5, 10, 50)
q <- ggplot(d,aes(x=RMSE,y=Species, color = Species)) + 
  scale_x_log10(breaks = etqs, labels = etqs,limits = c(0.1, 100))+
  geom_jitter(height = 0.2,alpha=0.15)+theme_bw()+ggtitle("")+ylab("")+
  theme_bw()+
  theme(panel.border = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(linetype = 2, color="ivory3", size = 0.3),
        legend.position = "none",
        axis.line = element_line(colour = "black"),
        plot.title = element_text(lineheight=1.5, face="bold"),
        axis.text = element_text(face="bold", size=10),
        axis.title.x = element_text(face="bold", size=11),
        axis.title.y  = element_text(face="bold", size=11) )

  d <- datos_err_species[datos_err_species$Method==m,]
  d$medianRSE <- 0
  for (s in unique(d$Species))
    d[d$Species==s,]$medianRSE=median(d[d$Species == s ,]$RSE)
  d$medianR2 <- 0
  for (s in unique(d$Species))
    d[d$Species==s,]$medianR2=median(d[d$Species == s ,]$R2)
  
  d$Species <- reorder(d$Species,d$medianRSE)
 
  etqs <- c(0.01,0.3, 1, 10, 100, 1000)
  r <- ggplot(d,aes(x=RSE,y=Species, color = Species))+
            geom_jitter(height = 0.2,alpha=0.15)+theme_bw()+ggtitle("")+ylab("")+
       scale_x_log10(breaks = etqs, labels = etqs,limits = c(0.01, 120000))+
       theme_bw()+
       theme(panel.border = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.grid.minor.y = element_blank(),
          panel.grid.major.y = element_blank(),
          panel.grid.major.x = element_line(linetype = 2, color="ivory3", size = 0.3),
          legend.position = "none",
          axis.line = element_line(colour = "black"),
          plot.title = element_text(lineheight=1.5, face="bold"),
          axis.text = element_text(face="bold", size=10),
          axis.title.x = element_text(face="bold", size=11),
          axis.title.y  = element_text(face="bold", size=11))
  
  s <- ggplot(d,aes(x=R2,y=Species, color = Species))+
    geom_jitter(height = 0.2,alpha=0.15)+theme_bw()+ggtitle("")+ylab("")+
    theme_bw()+
    theme(panel.border = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.grid.minor.y = element_blank(),
          panel.grid.major.y = element_blank(),
          panel.grid.major.x = element_line(linetype = 2, color="ivory3", size = 0.3),
          legend.position = "none",
          axis.line = element_line(colour = "black"),
          plot.title = element_text(lineheight=1.5, face="bold"),
          axis.text = element_text(face="bold", size=10),
          axis.title.x = element_text(face="bold", size=11),
          axis.title.y  = element_text(face="bold", size=11))

  ppi <- 300
  tdir <- "../plots"
  if (!dir.exists(tdir))
    dir.create(tdir)
  nfile <- paste0(tdir,"/Errores_species_",m)
  png(paste0(nfile,".png"), width=10*ppi, height=5*ppi, res=ppi)
  g <- plot_grid(
    q, r,
    ncol = 2,labels=c("A","B"),
    label_size = 16
  )
  print(g)
  dev.off()
  tiff(paste0(nfile,".tiff"), units="in", width=10, height=5, res=ppi)
  print(g)
  dev.off()
}


