#Libraries
###################################################################
library(xts)
library(fpp2)
library(tictoc)
library(forecast)
library(e1071)
library(ranger)
library(DescTools)

if (FALSE){
  library(keras)
  library(reticulate)
  library(tensorflow)
  install_tensorflow(
    method               = "conda", 
    version              = "2.0.0", # Installs TF 2.0.0 (as of May 15, 2020)
    envname              = "tf_r", 
    conda_python_version = "3.7.10" 
    #extra_packages       = c("matplotlib", "numpy", "pandas", "scikit-learn")
  )
  
  conda_list()
  use_condaenv("tf_r", required = TRUE)
  tf$constant("Hellow Tensorflow")
}

###################################################################
#Data processing
###################################################################
 
btc_median <- read.zoo('/Users/sacbe/Library/Mobile Documents/iCloud~md~obsidian/Documents/Tesis licenciatura/Algoritmos/cleaning/BTC-USD.csv', sep = ",", header =TRUE)
btc_median <- as.xts(btc_median)
btc_median <- btc_median['/2021-03-09']

#plot(btc_median$Close[index(btc_median) >= as.Date("2018-01-01 00:00:00.000")],xlab = "", ylab = "", axes = FALSE, bty="n")

btc_blockchain <- read.zoo('/Users/sacbe/Library/Mobile Documents/iCloud~md~obsidian/Documents/Tesis licenciatura/Algoritmos/cleaning/BTC - coinMetrics_marzo.csv', sep = ",", header =TRUE)
btc_blockchain <- as.xts(btc_blockchain)

btc_blockchain <- subset(btc_blockchain, select = -PriceUSD)
btc_blockchain <- btc_blockchain['/2021-03-09']

btc_median_cmetric <- merge(x = btc_median, y = btc_blockchain, join = 'inner')

s <- as.integer(nrow(btc_median_cmetric)*0.8)
train <- btc_median_cmetric[1:s,]
test <- btc_median_cmetric[(s+1):nrow(btc_median_cmetric),]

testc <- test$Close
trainc <- train$Close

models_names <- c('Naive', 'SES', 'Holt','ETS','ARIMA', 'Regresion', 'SVM','RF',"LSTM")
transform_names <- c('Original', 'log', 'BoxCox', 'diff', 'diff(log)') 
RMSE <- matrix(nrow = length(models_names), ncol = length(transform_names))
colnames(RMSE) <- transform_names
rownames(RMSE) <- models_names
ACC <- matrix(nrow = length(models_names), ncol = length(transform_names))
colnames(ACC) <- transform_names
rownames(ACC) <- models_names

naive_list <- list()
ses_list <- list()
holt_list <- list()
ets_list <- list()
arima_list <- list()
regresion_list <- list()
svm_list <- list()
rf_list <- list()
lstm_list <- list()
###################################################################
#Functions
###################################################################
rmse <- function(error){
  sqrt(mean(error^2))
}
lags <- function(x, k){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
normalize = function(train, test, feature_range = c(0, 1)) {
  #Normalizamos los datos con el maximo de los datos de entrenamiento
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((test - min(x) ) / (max(x) - min(x)  ))
  
  scaled_train = std_train *(fr_max -fr_min) + fr_min
  scaled_test = std_test *(fr_max -fr_min) + fr_min
  
  return(list(scaled_train = as.vector(scaled_train), 
              scaled_test = as.vector(scaled_test),
              scaler= c(min =min(x), max = max(x))))
}
inverter = function(scaled, scaler, feature_range = c(0, 1)){
  min = scaler[1]
  max = scaler[2]
  n = length(scaled)
  mins = feature_range[1]
  maxs = feature_range[2]
  inverted_dfs = numeric(n)
  
  for(i in 1:n){
    X = (scaled[i]- mins)/(maxs - mins)
    rawValues = X *(max - min) + min
    inverted_dfs[i] <- rawValues
  }
  return(inverted_dfs)
}
###################################################################
#Univariate time series
###################################################################
time_pred <- length(testc)
tic()
for (i in 1:5) {
  if(i == 1){
    t_trainData <- trainc
  } else if(i == 2){
    t_trainData <- log(trainc)
  } else if(i == 3){
    t_trainData <- BoxCox(trainc, lambda = BoxCox.lambda(trainc))
  } else if(i == 4){
    t_trainData <- na.fill(diff(trainc),"extend")
  } else{
    t_trainData <- na.fill(diff(log(trainc)),"extend")
  }
  #Naive
  naive.btc <- naive(t_trainData, h=time_pred)
  #SES
  ses.btc <- ses(t_trainData, h=time_pred)
  #Holt
  h.btc <- holt(t_trainData, h=time_pred, damped = TRUE)
  #ETS
  ets.btc <- ets(as.ts(t_trainData))
  etsf.btc <- predict(ets.btc, time_pred)
  #ARIMA
  arima.btc <- auto.arima(as.ts(t_trainData))
  arimaf.btc <- forecast(arima.btc, time_pred)
  
  if(i == 1){
    naive.btc$fitted = na.fill(naive.btc$fitted,"extend")
    ses.btc$fitted = ses.btc$fitted
    h.btc$fitted = h.btc$fitted
    etsf.btc$fitted = etsf.btc$fitted
    arimaf.btc$fitted = arimaf.btc$fitted
  }else if(i == 2){
    naive.btc$fitted = na.fill(exp(naive.btc$fitted),"extend")
    ses.btc$fitted = exp(ses.btc$fitted)
    h.btc$fitted = exp(h.btc$fitted)
    etsf.btc$fitted = exp(etsf.btc$fitted)
    arimaf.btc$fitted = exp(arimaf.btc$fitted)
  }else if(i == 3){
    naive.btc$fitted = na.fill(InvBoxCox(naive.btc$fitted, lambda = BoxCox.lambda(trainc)),"extend")
    ses.btc$fitted = InvBoxCox(ses.btc$fitted, lambda = BoxCox.lambda(trainc))
    h.btc$fitted = InvBoxCox(h.btc$fitted, lambda = BoxCox.lambda(trainc))
    etsf.btc$fitted = InvBoxCox(etsf.btc$fitted, lambda = BoxCox.lambda(trainc))
    arimaf.btc$fitted = InvBoxCox(arimaf.btc$fitted, lambda = BoxCox.lambda(trainc))
  }else{
    #i <- 4
    for (j in 1:(length(trainc))) {
      
      if(i == 4){
        naive.btc$fitted[j]  =  naive.btc$fitted[j] + btc_median_cmetric$Close[(1+j)]
        ses.btc$fitted[j]  = ses.btc$fitted[j] + btc_median_cmetric$Close[(1+j)]
        h.btc$fitted[j]  = h.btc$fitted[j] + btc_median_cmetric$Close[(1+j)]
        etsf.btc$fitted[j]  = etsf.btc$fitted[j] + btc_median_cmetric$Close[(1+j)]
        arimaf.btc$fitted[j]  = arimaf.btc$fitted[j] + btc_median_cmetric$Close[(1+j)]
        
      }else{
        naive.btc$fitted[j]  = exp(naive.btc$fitted[j]) + btc_median_cmetric$Close[(1+j)]
        ses.btc$fitted[j]  = exp(ses.btc$fitted[j]) + btc_median_cmetric$Close[(1+j)]
        h.btc$fitted[j]  = exp(h.btc$fitted[j]) + btc_median_cmetric$Close[(1+j)]
        etsf.btc$fitted[j]  = exp(etsf.btc$fitted[j]) + btc_median_cmetric$Close[(1+j)]
        arimaf.btc$fitted[j]  = exp(arimaf.btc$fitted[j]) + btc_median_cmetric$Close[(1+j)]
      }
    }
    
  }
  if(i == 4 | i == 5){
    naive.btc$fitted = na.fill(naive.btc$fitted,"extend")
  }
  naive_list[[i]] <- naive.btc$fitted
  RMSE[1,i] <- rmse(trainc - as.vector(naive_list[[i]]))
  ACC[1,i] <- TheilU(trainc, as.vector(naive_list[[i]]))
  ses_list[[i]] <- ses.btc$fitted
  RMSE[2,i] <- rmse(trainc - as.vector(ses_list[[i]]))
  ACC[2,i] <- TheilU(trainc, as.vector(ses_list[[i]]))
  holt_list[[i]] <- h.btc$fitted
  RMSE[3,i] <- rmse(trainc - as.vector(holt_list[[i]]))
  ACC[3,i] <- TheilU(trainc, as.vector(holt_list[[i]]))
  ets_list[[i]] <- etsf.btc$fitted
  RMSE[4,i] <- rmse(trainc - as.vector(ets_list[[i]]))
  ACC[4,i] <- TheilU(trainc, as.vector(ets_list[[i]]))
  arima_list[[i]] <- arimaf.btc$fitted  
  RMSE[5,i] <- rmse(trainc - as.vector(arima_list[[i]]))
  ACC[5,i] <- TheilU(trainc, as.vector(arima_list[[i]]))
  trainc <- train$Close
}

#### PRUEBAS #####

fit <- nnetar(trainc)
autoplot(forecast(fit,h=time_pred))

test_plot <- window(as.ts(btc_median_cmetric$Close), start=s+1)
train_plot <- window(as.ts(btc_median_cmetric$Close), end=s)


(ets.btc <- ets(train_plot))
checkresiduals(ets.btc)
etsf.btc <- predict(ets.btc, time_pred)
#ARIMA
(arima.btc <- auto.arima(train_plot))
checkresiduals(arima.btc)
arimaf.btc <- forecast(arima.btc, time_pred)

train_plot %>% ets() %>% forecast(h=30) %>% autoplot()

tail(trainc,5)
head(test_plot,50)
head(variable$fitted,50)
plot(test_plot, ylab = NULL, xlab = NULL, main = 'Naive',lwd=0.5)
lines(naive.btc$fitted, col = "red", type= "l",lty=2)
###################################################################
#Machine learning
###################################################################
for (i in 1:5) {
  if(i == 1){
    t_trainData <- train
  } else if(i == 2){
    t_trainData <- log(train)
  } else if(i == 3){
    t_trainData <- BoxCox(train, lambda = BoxCox.lambda(trainc))
  } else if(i == 4){
    t_trainData <- na.approx(diff(train))
  } else{
    t_trainData <- na.approx(diff(log(train)))
  }
  Scaled = normalize(as.data.frame(t_trainData[,c(1,4,5,6,7,8,9,10,11,12,13,14,15)]),as.data.frame(t_trainData[,c(1,4,5,6,7,8,9,10,11,12,13,14,15)]))
  scaler = Scaled$scaler
  
  t_trainData = as.ts(Scaled$scaled_train)

  #AQUI AGREAGAR OTRAS VARIABLES PARA PREDICCIÓN
  
  regresion.btc <- tslm(Close ~ Open + Volume_Currency + AdrActCnt + DiffMean + FeeTotNtv + SplyCur + SplyFF + TxTfrCnt + TxTfrValUSD, data = t_trainData, lambda = NULL)
  #regresion.btc <- tslm(Close ~ Open + Volume_Currency + AdrActCnt + BlkSizeByte + BlkSizeMeanByte + CapRealUSD + DiffMean + HashRate + SplyCur + SplyFF + TxCnt, data = t_trainData, lambda = NULL)
  #regresion.btc <- tslm(Close ~ Open + Volume_Currency, data = t_trainData, lambda = NULL)
  #regresion.btc <- tslm(Close ~ Open + Volume_Currency + CapRealUSD + DiffMean + HashRate, data = t_trainData, lambda = NULL)
  regresionf.btc <- predict(regresion.btc,as.data.frame(t_trainData))
  
  svm.btc <- svm(Close ~ Open + Volume_Currency + AdrActCnt + DiffMean + FeeTotNtv + SplyCur + SplyFF + TxTfrCnt + TxTfrValUSD, data = as.data.frame(t_trainData))
  svmf.btc <- predict(svm.btc,as.data.frame(t_trainData))
  
  rf.btc <- ranger(Close ~ Open + Volume_Currency + AdrActCnt + DiffMean + FeeTotNtv + SplyCur + SplyFF + TxTfrCnt + TxTfrValUSD, data = as.data.frame(t_trainData), num.trees = 500, seed = 123)
  rff.btc <- predict(rf.btc,as.data.frame(t_trainData))$predictions
  
  regresionf.btc = inverter(regresionf.btc,scaler)
  svmf.btc = inverter(svmf.btc,scaler)
  rff.btc = inverter(rff.btc,scaler)
  
  if(i == 1){
    svmf.btc = svmf.btc
    rff.btc = rff.btc
    regresionf.btc = regresionf.btc
  }else if(i == 2){
    svmf.btc = exp(svmf.btc)
    rff.btc = exp(rff.btc)
    regresionf.btc = exp(regresionf.btc)
  }else if(i == 3){
    svmf.btc = InvBoxCox(svmf.btc, lambda = BoxCox.lambda(trainc))
    rff.btc = InvBoxCox(rff.btc, lambda = BoxCox.lambda(trainc))
    regresionf.btc = InvBoxCox(regresionf.btc, lambda = BoxCox.lambda(trainc)) 
  }else{
    for (j in 1:length(trainc)) {
      if(i == 4){
        svmf.btc[j]  = svmf.btc[j] + btc_median_cmetric$Close[(1+j)]
        rff.btc[j]  = rff.btc[j] + btc_median_cmetric$Close[(1+j)]
        regresionf.btc[j]  = regresionf.btc[j] + btc_median_cmetric$Close[(1+j)]
      }else{
        svmf.btc[j]  = exp(svmf.btc[j]) + btc_median_cmetric$Close[(1+j)]
        rff.btc[j]  = exp(rff.btc[j]) + btc_median_cmetric$Close[(1+j)]
        regresionf.btc[j]  = exp(regresionf.btc[j]) + btc_median_cmetric$Close[(1+j)]
      }
    }
  }
  
  regresion_list[[i]] <- regresionf.btc
  RMSE[6,i] <- rmse(na.approx(trainc - regresion_list[[i]]))
  ACC[6,i] <- TheilU(trainc, regresion_list[[i]], na.rm = TRUE)
  svm_list[[i]] <- svmf.btc
  RMSE[7,i] <- rmse(na.approx(trainc - svm_list[[i]]))
  ACC[7,i] <- TheilU(trainc, svm_list[[i]], na.rm = TRUE)
  rf_list[[i]] <- rff.btc
  RMSE[8,i] <- rmse(na.approx(trainc - rf_list[[i]]))
  ACC[8,i] <- TheilU(trainc, rf_list[[i]],na.rm = TRUE)

}

library(kernlab)
(rmse(na.approx(trainc - svmf.btc)))
model_svm <- ksvm(Close ~ Open + Volume_Currency + AdrActCnt + DiffMean + FeeTotNtv + SplyCur + SplyFF + TxTfrCnt + TxTfrValUSD, data = as.data.frame(t_trainData), kernel = "vanilladot")
model_svm_f <- predict(model_svm,as.data.frame(t_trainData))
model_svm_f = inverter(model_svm_f,scaler)

(rmse(na.approx(trainc - model_svm_f)))



RMSE
ACC
###################################################################
#Deep learning
###################################################################
for (i in 1:5) {
  if(i == 1){
    transf = btc_median_cmetric$Close
  }else if(i == 2){
    transf = log(btc_median_cmetric$Close)
  }else if(i == 3){
    transf = BoxCox(btc_median_cmetric$Close, lambda = BoxCox.lambda(btc_median_cmetric$Close))
  }else if(i == 4){
    transf = diff(btc_median_cmetric$Close, differences = 1)
  }else{
    transf = diff(log(btc_median_cmetric$Close), differences = 1)
  }
  supervised = lags(transf, 1)
  #tail(supervised)
  train_lstm = supervised[1:s, ]
  test_lstm  = supervised[(s+1):nrow(btc_median_cmetric), ]

  Scaled = normalize(train_lstm, test_lstm, c(-1, 1))
  y_train = Scaled$scaled_train[, 2]
  x_train = Scaled$scaled_train[, 1]
  
  y_test = Scaled$scaled_test[, 2]
  x_test = Scaled$scaled_test[, 1]
  
  # Reshape the input to 3-dim
  dim(x_train) <- c(length(x_train), 1, 1)
  
  # specify required arguments
  X_shape2 = dim(x_train)[2]
  X_shape3 = dim(x_train)[3]
  batch_size = 1
  units = 2
  
  model <- NULL
  model <- keras_model_sequential() 
  model%>%
    layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer_adam( lr= 0.02 , decay = 1e-6 ),  
    metrics = c('accuracy')
  )
  
  #summary(model)
  
  nb_epoch = 50   
  for(j in 1:nb_epoch ){
    model %>% fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
    model %>% reset_states()
  }
  
  L = length(x_train)
  dim(x_train) = c(length(x_train), 1, 1)
  scaler = Scaled$scaler
  predictions = numeric(L)
  
  for(k in 1:L){
    X = x_train[k,,]
    dim(X) = c(1,1,1)
    # forecast
    yhat = model %>% predict(X, batch_size=batch_size)
    # invert scaling
    yhat = inverter(yhat, scaler,  c(-1, 1))
    # invert transforms
    if(i==1){
      yhat = yhat
    }else if(i==2){
      yhat = exp(yhat)
    }else if(i==3){
      yhat = InvBoxCox(yhat, lambda = BoxCox.lambda(btc_median_cmetric$Close)) 
    }else if(i==4){
      yhat  = yhat + btc_median_cmetric$Close[(1+k)]  
    }else{
      yhat  = exp(yhat) + btc_median_cmetric$Close[(1+k)]
    }
    # save prediction
    predictions[k] <- yhat
  }
  lstm_list[[i]] <- predictions
  RMSE[9,i] <- rmse(trainc - lstm_list[[i]])
  ACC[9,i] <- TheilU(trainc, lstm_list[[i]])
}
toc()

###################################################################
#Chose model
###################################################################
min <- apply( RMSE, 2, which.min)
best_model <-rownames(RMSE[min,])
bestRMSE <- data.frame('Transformation' = transform_names,'Best.model.RMSE' = best_model)
min <- apply( ACC, 2, which.min)
best_model <-rownames(ACC[min,])
bestACC <- data.frame('Transformation' = transform_names,'Best.model.RMSE' = best_model)
RMSE
ACC
#bestRMSE
#bestACC

inx <- which(RMSE == min(RMSE), arr.ind = TRUE)
iny <- which(ACC == min(ACC), arr.ind = TRUE)

bestModel <- data.frame("Best RMSE"=RMSE[inx],"Best Theils U" = ACC[iny]) 
bestModel <- cbind("Modelo" = models_names[inx[1]], bestModel)
bestModel

###################################################################
#Plots
###################################################################
test_plot <- window(as.ts(btc_median_cmetric$Close), start=s+1)
train_plot <- window(as.ts(btc_median_cmetric$Close), end=s)

#Original
par(mfrow=c(3,3),mex=0.6,cex=0.8)
plot(train_plot, ylab = NULL, xlab = NULL, main = 'Naive')
lines(naive_list[[1]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'SES')
lines(ses_list[[1]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'Holt')
lines(holt_list[[1]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'ETS')
lines(ets_list[[1]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'ARIMA')
lines(arima_list[[1]], col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'RTS')
lines(as.ts(regresion_list[[1]]), col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'SVM')
lines(as.ts(svm_list[[1]]), col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'Random Forest')
lines(as.ts(rf_list[[1]]), col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'LSTM')
lines(as.ts(lstm_list[[1]]), col = "red")

#Logaritm
par(mfrow=c(3,3),mex=0.6,cex=0.8)
plot(train_plot, ylab = NULL, xlab = NULL, main = 'Naive')
lines(naive_list[[2]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'SES')
lines(ses_list[[2]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'Holt')
lines(holt_list[[2]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'ETS')
lines(ets_list[[2]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'ARIMA')
lines(arima_list[[2]], col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'RTS')
lines(as.ts(regresion_list[[2]]), col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'SVM')
lines(as.ts(svm_list[[2]]), col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'Random Forest')
lines(as.ts(rf_list[[2]]), col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'LSTM')
lines(as.ts(lstm_list[[2]]), col = "red")

#BoxCox
par(mfrow=c(3,3),mex=0.6,cex=0.8)
plot(train_plot, ylab = NULL, xlab = NULL, main = 'Naive')
lines(naive_list[[3]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'SES')
lines(ses_list[[3]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'Holt')
lines(holt_list[[3]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'ETS')
lines(ets_list[[3]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'ARIMA')
lines(arima_list[[3]], col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'RTS')
lines(as.ts(regresion_list[[3]]), col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'SVM')
lines(as.ts(svm_list[[3]]), col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'Random Forest')
lines(as.ts(rf_list[[3]]), col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'LSTM')
lines(as.ts(lstm_list[[3]]), col = "red")

#Diff
par(mfrow=c(3,3),mex=0.6,cex=0.8)
plot(train_plot, ylab = NULL, xlab = NULL, main = 'Naive')
lines(naive_list[[4]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'SES')
lines(ses_list[[4]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'Holt')
lines(holt_list[[4]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'ETS')
lines(ets_list[[4]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'ARIMA')
lines(arima_list[[4]], col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'RTS')
lines(as.ts(regresion_list[[1]]), col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'SVM')
lines(as.ts(svm_list[[4]]), col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'Random Forest')
lines(as.ts(rf_list[[4]]), col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'LSTM')
lines(as.ts(lstm_list[[4]]), col = "red")

#Diff(log)
par(mfrow=c(3,3),mex=0.6,cex=0.8)
plot(train_plot, ylab = NULL, xlab = NULL, main = 'Naive')
lines(naive_list[[5]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'SES')
lines(ses_list[[5]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'Holt')
lines(holt_list[[5]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'ETS')
lines(ets_list[[5]], col = "red")
plot(train_plot, ylab = NULL, xlab = NULL, main = 'ARIMA')
lines(arima_list[[5]], col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'RTS')
lines(as.ts(regresion_list[[5]]), col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'SVM')
lines(as.ts(svm_list[[5]]), col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'Random Forest')
lines(as.ts(rf_list[[5]]), col = "red")
plot(as.ts(trainc), ylab = NULL, xlab = NULL, main = 'LSTM')
lines(as.ts(lstm_list[[5]]), col = "red")

###################################################################
#Plot selected model
###################################################################
par(mfrow=c(1,1),mex=0.6,cex=0.8)
t_trainData <- na.approx(diff(train))
t_testData <- na.approx(diff(test))
Scaled = normalize(as.data.frame(t_trainData[,c(1,4,5,6,7,8,9,10,11,12,13,14,15)]), as.data.frame(t_testData[,c(1,4,5,6,7,8,9,10,11,12,13,14,15)]))
scaler = Scaled$scaler
t_trainData = as.ts(Scaled$scaled_train)
t_testData = as.ts(Scaled$scaled_test)
rf.btc <- ranger(Close ~ Open + Volume_Currency + AdrActCnt + CapMrktCurUSD + DiffMean + FeeTotNtv + SplyCur + SplyFF + TxTfrCnt + TxTfrValMeanUSD + TxTfrValUSD + VtyDayRet30d, data = as.data.frame(t_trainData), num.trees = 500, seed = 123)
rff.btc <- predict(rf.btc,as.data.frame(t_testData))$predictions
rff.btc = inverter(rff.btc,scaler)
for (j in 1:length(testc)) {
    rff.btc[j]  = rff.btc[j] + btc_median_cmetric$Close[(s+j)]
}
plot(as.ts(testc), ylab = NULL, xlab = NULL, main = 'Random Forest')
lines(as.ts(rff.btc), col = "red")

#######################################################################
par(mfrow=c(1,1),mex=0.6,cex=0.8)
t_trainData <- BoxCox(train, lambda = BoxCox.lambda(trainc))
t_testData <- BoxCox(test, lambda = BoxCox.lambda(trainc))
#t_trainData <- log(train)
#t_testData <- log(test)
Scaled = normalize(as.data.frame(t_trainData[,c(1,4,5,6,7,8,9,10,11,12,13,14,15)]),as.data.frame(t_testData[,c(1,4,5,6,7,8,9,10,11,12,13,14,15)]))
scaler = Scaled$scaler
t_trainData = as.ts(Scaled$scaled_train)
t_testData = as.ts(Scaled$scaled_test)

rf.btc <- ranger(Close ~ Open + Volume_Currency + AdrActCnt + DiffMean + FeeTotNtv + SplyCur + SplyFF + TxTfrCnt + TxTfrValMeanUSD + TxTfrValUSD, data = as.data.frame(t_trainData), num.trees = 500, seed = 123)
rff.btc <- predict(rf.btc,as.data.frame(t_testData))$predictions
rff.btc = inverter(rff.btc,scaler)
rff.btc = InvBoxCox(rff.btc, lambda = BoxCox.lambda(trainc))
#rff.btc = exp(rff.btc)
plot(as.ts(testc), ylab = NULL, xlab = NULL, main = 'Random Forest',type = "o")
lines(as.ts(rff.btc), col = "red")
legend("bottomright", legend = c("Original", "Predicción"),
       lwd = 3, col = c("black", "red"))
print(rmse(na.approx(testc - rff.btc)))

#######################################################################
par(mfrow=c(1,1),mex=0.6,cex=0.8)
#t_trainData <- BoxCox(train, lambda = BoxCox.lambda(trainc))
#t_testData <- BoxCox(test, lambda = BoxCox.lambda(trainc))
t_trainData <- train
t_testData <- test
Scaled = normalize(as.data.frame(t_trainData[,c(1,4,5,6,7,8,9,10,11,12,13,14,15)]),as.data.frame(t_testData[,c(1,4,5,6,7,8,9,10,11,12,13,14,15)]))
scaler = Scaled$scaler
t_trainData = as.ts(Scaled$scaled_train)
t_testData = as.ts(Scaled$scaled_test)

#rf.btc <- ranger(Close ~ Open, data = as.data.frame(t_trainData), num.trees = 500, seed = 123)
#rf.btc <- ranger(Close ~ Open + AdrActCnt + BlkSizeByte + BlkSizeMeanByte + CapMrktCurUSD + CapRealUSD + DiffMean + HashRate + SplyCur + SplyFF + TxCnt, data = as.data.frame(t_trainData), num.trees = 500, seed = 123)
rf.btc <- ranger(Close ~ Open + Volume_Currency + AdrActCnt + DiffMean + FeeTotNtv + SplyCur + SplyFF + TxTfrCnt + TxTfrValUSD, data = as.data.frame(t_trainData), num.trees = 500, seed = 123)
rff.btc <- predict(rf.btc,as.data.frame(t_testData))$predictions
rff.btc = inverter(rff.btc,scaler)
#rff.btc = InvBoxCox(rff.btc, lambda = BoxCox.lambda(trainc))
plot(as.ts(testc), ylab = NULL, xlab = NULL, main = 'Random Forest',type = "o")
lines(as.ts(rff.btc), col = "red")
legend("bottomright", legend = c("Original", "Predicción"),
       lwd = 3, col = c("black", "red"))
