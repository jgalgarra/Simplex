library(ggplot2)
library(ggpmisc)
library(cowplot)

dspecies <- read.csv("../datasets/abund_merged_dataset_onlycompetitors.csv")
dabiotic <- read.csv("../datasets/abund_merged_dataset_onlyenvironment.csv")
alldata <- dabiotic
# Merge biotic and abiotic data
for (j in 5:ncol(dspecies))
    alldata <- cbind(alldata,dspecies[,j])
names(alldata) <- c(names(dabiotic),names(dspecies)[5:ncol(dspecies)])

nz <- alldata[alldata$individuals>0,]

#nz <- alldata


# Data frame with nonzero individuals
nz$meanindividuals <- 0
nz$varindividuals <- 0
nz$medianindividuals <- 0
for (i in unique(nz$species)){
  nz[nz$species==i,]$meanindividuals = mean(nz[nz$species==i,]$individuals)
  nz[nz$species==i,]$varindividuals = var(nz[nz$species==i,]$individuals)
  nz[nz$species==i,]$medianindividuals = median(nz[nz$species==i,]$individuals)
}

nz$species <-  reorder(nz$species, as.numeric(nz$medianindividuals))

#Species abundance
bab <- ggplot(data=nz)+
       geom_boxplot(aes(x=species,y=individuals,color = species, fill=species),alpha=0.5)+
  scale_y_log10()+labs(x="Species",y="Individuals")+
  theme_bw()+
  theme(panel.border = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(linetype = 2, color="ivory3", size = 0.3),
        legend.position = "none",
        axis.line = element_line(colour = "black"),
        plot.title = element_text(lineheight=1.5, face="bold"),
        axis.text = element_text(face="bold", size=9),
        axis.title.x = element_text(face="bold", size=11),
        axis.title.y  = element_text(face="bold", size=11))+coord_flip()


taylor_data <- as.data.frame(cbind(as.character(nz$species),as.numeric(nz$medianindividuals),
                                   as.numeric(nz$meanindividuals),as.numeric(nz$varindividuals)))
names(taylor_data) <- c("species","medianindividuals","meanindividuals","varindividuals")
taylor_data <- taylor_data[!duplicated(taylor_data), ]
taylor_data$medianindividuals <- as.numeric(taylor_data$medianindividuals)
taylor_data$meanindividuals <- as.numeric(taylor_data$meanindividuals)
taylor_data$varindividuals <- as.numeric(taylor_data$varindividuals)
taylor_data$species <- factor(taylor_data$species, levels=levels(nz$species))

write.csv2(taylor_data,"../tables/abundance_distribution.csv",row.names = FALSE)

rlin <- lm(log10(taylor_data$varindividuals)~log10(taylor_data$meanindividuals))
print("Linear Regression")
print(summary(rlin))
brks <- c(1,10,100,1000, 10000, 100000)
labl <- rep(" ",length(brks))
for (i in 1:length(brks))
  labl[i] = sprintf("%7d",brks[i])
my.formula <- y ~ x
tplot <- ggplot(data=taylor_data,aes(x=meanindividuals,y=varindividuals))+
  geom_point(aes(fill=species,color=species),size=2,alpha=0.8,shape=21)+
  stat_smooth(method = "lm", col = "lightblue", alpha=0.5, formula = my.formula, se = FALSE)+
  stat_poly_eq(formula = my.formula, 
               aes(label = paste(..eq.label.., ..rr.label.., sep = "~~~")), 
               parse = TRUE)+
  scale_x_log10(limits = c(min(taylor_data$meanindividuals), 200))+
  scale_y_log10(breaks = brks, labels = labl,limits = c(0.7, 100000))+
  labs(x="Mean(Individuals)",y="Var(Individuals)")+
  theme_bw()+
  theme(panel.border = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(linetype = 2, color="ivory3", size = 0.3),
        legend.position = "none",
        axis.line = element_line(colour = "black"),
        plot.title = element_text(lineheight=1.5, face="bold"),
        axis.text = element_text(face="bold", size=10),
        axis.title.x = element_text(face="bold", size=11),
        axis.title.y  = element_text(face="bold", size=11))


ppi <- 300
tdir <- "../plots"
nfile = paste0(tdir,"/EXPLO_individuals")
if (!dir.exists(tdir))
  dir.create(tdir)
png(paste0(nfile,".png"), width=10*ppi, height=4*ppi, res=ppi)

g <- plot_grid(
  bab, tplot,
  ncol = 2,labels=c("A","B"), label_x = 0, hjust = 0,
  label_size = 16
)
print(g)
dev.off()

tiff(paste0(nfile,".tiff"), units="in", width=10, height=4, res=ppi)
print(g)
dev.off()