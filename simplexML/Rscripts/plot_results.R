library(readxl)
library(ggplot2)
library(grid)
library(gridExtra)
library(tidyverse)
library(cowplot)

ploterror <- function(datos,texto){
  p <- ggplot() + geom_density(aes(x= ERROR, color = Set, fill = Set),  alpha = .1,
                               data=datos, position = "identity", adjust= 1)+
    xlab(texto)+
    ylab("Density\n") + scale_fill_manual(values=c("blue","red","green4")) +
    scale_color_manual(values=c("blue","red","green4"))+ theme_bw()+
    theme(panel.border = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.grid.minor.y = element_blank(),
          panel.grid.major.y = element_line(linetype = 2, color="ivory3"),
          panel.grid.major.x = element_blank(), 
          legend.title = element_blank(),
          axis.line = element_line(colour = "black"),
          plot.title = element_text(lineheight=1.5, face="bold"),
          axis.text = element_text(face="bold", size=12),
          axis.title.x = element_text(face="bold", size=12),
          axis.title.y  = element_text(face="bold", size=12) )

}

plothistoerror <- function(datos,texto,metodo,logaritmico="no"){
  med_df <- datos %>%
    group_by(Set) %>%
    summarize(median=median(ERROR))
  p <- ggplot(datos,aes(x=ERROR)) + geom_histogram(aes(y=..density.. , fill = Set),  alpha = .2,
                                 data=datos, position="identity", bins =40)+ 
    xlab(texto)+ylab("Density\n")+    scale_y_continuous(expand=c(0,0))+
    geom_vline(data = med_df, aes(xintercept = median, 
                                  color = Set), size=0.3,alpha=0.8)+ 
    
    scale_fill_manual(values=c("blue","red","green4"))+#,name=gsub("_"," ",metodo))+ 
    scale_color_manual(values=c("blue","red","green4"))+
    guides(fill=guide_legend(gsub("_"," ",metodo)), fill = FALSE)+
    guides(color="none", color = FALSE)+
    theme_bw()+
    theme(panel.border = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.grid.minor.y = element_blank(),
          panel.grid.major.y = element_line(linetype = 2, color="ivory3", size = 0.2),
          panel.grid.major.x = element_blank(), 
          legend.title = element_text(face="bold", size=12),
          axis.line = element_line(colour = "black"),
          plot.title = element_text(lineheight=1.5, face="bold"),
          axis.text = element_text(face="bold", size=11),
          axis.title.x = element_text(face="bold", size=12),
          axis.title.y  = element_text(face="bold", size=12) )
  if (logaritmico == "yes")
    p <- p+scale_x_sqrt(expand = c(0, 0))#  x_log10(expand = c(0, 0))
  else
    p <- p+ scale_x_continuous(limits = c(c(min(datos$ERROR)),max(datos$ERROR)), expand = c(0, 0))
  return(p)
  return(p)
}

AddResumen <- function(sumario,nexper,modelo,metodo,index){
  resumen_errores_int <- data.frame("nexper"=nexper,"model"=modelo,
                                    "Method"=metodo,"Index"=index,
                                    "Min"=sumario[[1]],
                                    "Q1"=sumario[[2]],
                                    "Median"=sumario[[3]],
                                    "Mean"=sumario[[4]],
                                    "Q3"=sumario[[5]],"Max"=sumario[[6]])
  return(resumen_errores_int)
}

lHojas <- c("Linear Regressor","Random Forest","XGBoost")
#lexper <- c(100,300)
lexper <- c(100)
resumen_errores <- data.frame("nexper"=c(),"model"=c(),"Method"=c(),"Index"=c(),
                              "Min"=c(),"Q1"=c(),"Median"=c(),"Mean"=c(),"Q3"=c(),"Max"=c())

pathresults = "../results"
for (nexper in lexper)
{
  allerrors <- data.frame("MSE"=c(),  "RMSE"=c(), "RSE"=c(),  "Set"=c(), 
                          "Metodo"=c(), "Exper" = c())
  for (Hoja in lHojas)
  {
    abundancia_edaf <- read_excel(paste0(pathresults,"/ABIOTIC_",nexper,".xlsx"),
                                  sheet = Hoja)
    abundancia_edaf$Set <- "ABIOTIC"
    resumen_errores <- rbind(resumen_errores,AddResumen(summary(abundancia_edaf$RSE),
                                                        nexper,"ABIOTIC",Hoja,"RSE"))
    resumen_errores <- rbind(resumen_errores,AddResumen(summary(abundancia_edaf$RMSE),
                                                        nexper,"ABIOTIC",Hoja,"RMSE"))
    
    twostep <- read_excel(paste0(pathresults,"/TWOSTEP_",nexper,".xlsx"),
                              sheet = Hoja)
    twostep$Set <- "TWOSTEP"
    resumen_errores <- rbind(resumen_errores,AddResumen(summary(twostep$RSE),
                                                        nexper,"TWOSTEP",Hoja,"RSE"))
    resumen_errores <- rbind(resumen_errores,AddResumen(summary(twostep$RMSE),
                                                        nexper,"TWOSTEP",Hoja,"RMSE"))
    
    abundancia_edaf_comp <- read_excel(paste0(pathresults,"/ALLFEATURES_",nexper,".xlsx"),
                                       sheet = Hoja)
    abundancia_edaf_comp$Set <- "ALLFEATURES"
    resumen_errores <- rbind(resumen_errores,AddResumen(summary(abundancia_edaf_comp$RSE),
                                                        nexper,"ALLFEATURES",Hoja,"RSE"))
    resumen_errores <- rbind(resumen_errores,AddResumen(summary(abundancia_edaf_comp$RMSE),
                                                        nexper,"ALLFEATURES",Hoja,"RMSE"))
    
    
    
    datos <- rbind(abundancia_edaf,abundancia_edaf_comp,twostep)
    
    datoserr <- datos
    datoserr$ERROR <- datos$RSE
    print(paste("Method",Hoja))
    print("RSE")
    print(summary(datos$RSE))

    prse <- ploterror(datoserr,"RSE")
    phrse <- plothistoerror(datoserr,"RSE",Hoja,logaritmico="yes")   
    datoserr$ERROR <- datos$RMSE
    prmse <- ploterror(datoserr,"RMSE")
    phrmse <- plothistoerror(datoserr,"RMSE",Hoja)
    print("RMSE")
    print(summary(datos$RMSE))
    
    odir <- "../plots"
    if (!dir.exists(odir))
      dir.create(odir)
    ppi <- 300
    png(paste0(odir,"/Errors_hist",Hoja,"_",nexper,".png"), width=8*ppi, height=8*ppi, res=ppi)
    g <- plot_grid(
      phrse, phrmse,
      ncol = 1,labels=c("A","B"),
      label_size = 16
    )
    print(g)
    dev.off()

  }
  tdir <- "../tables"
  if (!dir.exists(tdir))
    dir.create(tdir)
  write.csv2(resumen_errores,paste0(tdir,"/Model_Errors.csv"),row.names = FALSE)
}