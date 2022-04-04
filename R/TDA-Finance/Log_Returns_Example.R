# Working directory is where this script is at.
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)
source("TDA_Finance.R")

# Load required packages
library(quantmod)   # for getSymbols
library(TDA)        # for ripsDiag
library(PKNCA)      # for pk.calc.auc.all
library(factoextra) # Para agrupamientos
library(ggplot2)    # Gráficas

#Función extra para normalizar ejes de para graficar clustering.
Normalize_Data <- function(val)
{
  return ((val - min(val)) / (max(val) - min(val)))
}

analisis_cluster <- function(cluster, data){
  tmp <- cluster
  tmp <- as.data.frame(tmp)
  tmp <- cbind(tmp,data)
  tmp <- subset(tmp, data > 0.5)
  y <- as.data.frame(table(tmp$tmp))
  colnames(y) <- c("Cluster", "Num_elementos")
  print(y)
  res <- y$Cluster[which.max(y$Num_elementos)]
  #print(as.numeric(as.character(res)))
  return(as.numeric(as.character(res)))
}

#import an one dimension time series as an xts object. 
#Here we use Bitcoin data from the specified dates

#getSymbols("BTC-USD", from="2017-09-16", to="2018-05-07", warnings = FALSE)
#getSymbols("BTC-USD", from="2020-11-01", to="2021-07-01", warnings = FALSE)
getSymbols("BTC-USD", from="2014-01-01", to="2021-07-01", warnings = FALSE)
data <- `BTC-USD`[,1]
data <- na.fill(data,"extend")

#Load data Shannon entropy of log(close_price) bitcoin (generado en python)
data <- read.zoo("Shannon_Transf_BTC-USD_MA_1.csv", header = TRUE, sep = ",")
data <- as.xts(data)
colnames(data) <- "Close"
plot(data)
head(data)

#This calculates the norm of the data in one line.
#
#Implicitly, this is using the default values for parameters:
# - embedding dim = 4
# - scaling is "log" which means that we will take the log of the data
# - max_scale is defaulted to zero, which leads to estimating max_scale 
#   using the find_diam function.
# - K_max = 10
# - window = 50
# - returns = TRUE means that we will take log-returns of the price 
#   time series before processing it through the TDA pipeline.
#   Note that when returns = TRUE, the scaling parameter has no effect.
#
#Thus, this is equivalent to: 
# norm_data <- 
# analyze_1d(data, dim=4, scaling_method="log", max_scale=0, 
# K_max=10, window=50, returns = TRUE)
norm_data <- analyze_1d(data, returns = TRUE, K_max = 10, window = 50)

#Here we plot the norm
plot(norm_data, type = "l")
plot(log_r(data), type = "l" )
#autoplot(norm_data)

output(norm_data, "BTC.zoo")

# Cluster generados por el log(price) eje x y la norma C1 en el eje y.
plot(diff(norm_data))
c1 = norm_data + abs(diff(norm_data))
c1 <- na.fill(c1,"extend")

plot(c1)

eje_x <- Normalize_Data(log(data))

eje_x <- Normalize_Data(data)
eje_y <- Normalize_Data(c1)

agrupamientos <- as.data.frame(cbind(eje_x,eje_y))
#agrupamientos <- na.fill(agrupamientos,"extend")
agrupamientos <- na.omit(agrupamientos)
colnames(agrupamientos) <- c("log price", "C1 norm")
rownames(agrupamientos) <- format(index(eje_y),"%d/%b/%y")
head(agrupamientos)

write.csv(agrupamientos,"C1_entropy_norm.csv", row.names = T)
(cl <- kmeans(agrupamientos, 3))
cl$cluster

#plot(agrupamientos, col = cl$cluster)
fviz_cluster(cl, data = agrupamientos)

###Poner labels e indices para saber fechas

dates <- index(eje_y)
new_dates <- tail(dates, length(eje_y))
grups.xts <- xts(agrupamientos, order.by = new_dates)
final <- merge(data,grups.xts, all = FALSE)
final <- as.data.frame(final)
cluster_TDA <- cbind(final, cl$cluster)

head(cluster_TDA)
write.csv(cluster_TDA,"cluster_TDA.csv", row.names = T)


cluster_minicrash <- analisis_cluster(cl$cluster, eje_y)
print(cluster_minicrash)

#Escala original
plot(x=dates, y=final$BTC.USD.Open, col = cl$cluster, lwd = 3)
par(new=TRUE)
plot(x=dates, y=final$BTC.USD.Open, type = "l" )
legend("topleft",
       c("Alerta fuerte de caída"),
       #fill=cl$cluster[cluster_minicrash],
       fill="red",
       bty = "n"
)

#Escala logaritmica
plot(x=dates, y=log(final$BTC.USD.Open), col = cl$cluster, lwd = 3)
par(new=TRUE)
plot(x=dates, y=log(final$BTC.USD.Open), type = "l" )
legend("topleft",
       c("Alerta fuerte de caída"),
       #fill=cl$cluster[cluster_minicrash],
       fill="red",
       bty = "n"
)

#Escala original entropy
plot(x=dates, y=final$Close, col = cl$cluster, lwd = 3)
par(new=TRUE)
plot(x=dates, y=final$Close, type = "l" )
legend("topleft",
       c("Alerta fuerte de caída"),
       #fill=cl$cluster[cluster_minicrash],
       fill="red",
       bty = "n"
)
