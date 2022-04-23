# Se establece el directorio de trabajo para cargar el script TDA_Crash
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
getwd()

source("TDA_Crash.R")

# Load required packages
library(quantmod)   # for getSymbols
library(TDA)        # for ripsDiag
library(PKNCA)      # for pk.calc.auc.all
library(factoextra) # Para agrupamientos
library(ggplot2)    # Gráficas

#Función extra para normalizar ejes de para graficar el agrupamiento.
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
  return(as.numeric(as.character(res)))
}

BTC <- read.zoo('/Users/sacbe/Documents/Tesis licenciatura/Algoritmos/cleaning/BTC-USD.csv', sep = ",", header =TRUE)
data <- as.xts(BTC$Close)
data <- subset(data, index(data) > as.Date("2021-01-01") )
colnames(data) <- c("Close")
head(data)

data <- na.fill(data,"extend")
# La ventana de tiempo entre mayor mas datos generados

norm_data <- analyze_1d(data, returns = TRUE, K_max = 10, window = 50)

#Se grafican las normas
plot(norm_data, type = "l")
plot(log_r(data), type = "l" )
#autoplot(norm_data)

output(norm_data, "BTC.zoo")

# Cluster generados por el log(price) en el eje x y la norma C1 en el eje y.
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
tail(agrupamientos)

write.csv(agrupamientos,"/Users/sacbe/Documents/Tesis licenciatura/Algoritmos/C1_entropy_norm.csv", row.names = T)
(cl <- kmeans(agrupamientos, 3))
cl$cluster

#plot(agrupamientos, col = cl$cluster)
ggcluster <- fviz_cluster(cl, data = agrupamientos)
ggcluster

###Poner labels e indices para saber fechas

dates <- index(eje_y)
new_dates <- tail(dates, length(eje_y))
grups.xts <- xts(agrupamientos, order.by = new_dates)
final <- merge(data,grups.xts, all = FALSE)
final <- as.data.frame(final)
cluster_TDA <- cbind(final, cl$cluster)

tail(cluster_TDA)
write.csv(cluster_TDA,"cluster_TDA.csv", row.names = T)


cluster_minicrash <- analisis_cluster(cl$cluster, eje_y)
print(cluster_minicrash)


cluster_TDA$Dates <- as.Date(new_dates, "%m/%d/%Y")
print(cluster_TDA)

# Las alertas de caída se encuentran en las subidas repentinas
# del precio para luego caer en picada, es decir, no hay un
# pequeño valle despues de la caida 

ggplot(data = cluster_TDA, aes(Dates, Close, color = factor(cl$cluster))) +
  geom_point() +
  scale_x_date(date_labels = "%Y-%m-%d") +
  geom_line(data = cluster_TDA, color = "black", alpha = 0.3)+
  #+ theme(legend.position = "none")
  guides(fill = guide_legend(title = "Title"))

#Escala original
plot(x=dates, y=final$Close, col = cl$cluster, lwd = 1)
par(new=TRUE)
plot(x=dates, y=final$Close, type = "l" )
#axis(1,dates,format(dates, "%Y-%m-%d"))
#help(axis)
legend("topleft",
       c("Alerta fuerte de caída"),
       #fill=cl$cluster[cluster_minicrash],
       fill="red",
       bty = "n"
)

#Escala logaritmica
plot(x=dates, y=log(final$Close), col = cl$cluster, lwd = 3)
par(new=TRUE)
plot(x=dates, y=log(final$Close), type = "l" )
legend("topleft",
       c("Alerta fuerte de caída"),
       #fill=cl$cluster[cluster_minicrash],
       fill="red",
       bty = "n"
)
