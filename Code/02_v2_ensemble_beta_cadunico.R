## ----
## Estudo sobre métodos 
## ensemble sobre a Regressão Beta
## ----

## Dados: Taxa de atualização no sistema CadÚnico, em 2015.

## Bibliotecas

if(!require(pacman)) install.packages("pacman"); library(pacman)
p_load(ggplot2,dplyr, betareg, beepr,betaboost,e1071,caret,compiler,gridExtra)


## Leitura do banco de dados

setwd("C:/Users/dioni/OneDrive - University of São Paulo/Doutorado em Estatística/2023.2/2 - Aprendizado de Máquina Estatístico/artigo_publicar/github_content/Data")

cadu = read.csv2("final_data.csv")
head(cadu)

## ---
## Separação das covariaveis e da resposta
## ---

y = cadu %>% subset(select=c(tax))
X = cadu %>% subset(select=c(ESPVIDA,FECTOT, RAZDEP,
                             E_ANOSESTUDO, T_ANALF18M,
                             T_FBBAS, T_FBFUND,
                             T_FBMED, T_FBSUPER,T_MED18M, T_SUPER25M,
                             GINI, PIND, PPOB, RDPC1, RDPCT,
                             THEIL, T_BANAGUA, T_DENS,
                             T_LIXO, T_LUZ,AGUA_ESGOTO,
                             T_M10A14CF, T_M15A17CF,
                             I_ESCOLARIDADE,  IDHM, IDHM_L, IDHM_R))

## Realizando um pequena mudança na resposta
## para que o velopr não seja 1
## 1 munícipios teve todos os seus candidatos escolhidos

datacadu = cbind(X,y)
dim(datacadu)


## Separação entre treino e teste
set.seed(10)

# Proporção treino
prop_treino = 0.8
n_treino = round(nrow(datacadu) * prop_treino)

ind_treino = sample(1:nrow(datacadu), n_treino)

# Criar conjuntos de treino e teste 
treinocadu = datacadu[ind_treino, ]
testecadu = datacadu[-ind_treino, ]

dim(treinocadu);dim(testecadu)


## -----
## 0. Métricas de ajuste
## -----

MSE = function(y,ypred){mean((y - ypred)^2)}
MAE = function(y,ypred){mean(abs(y - ypred))}
R2 = function(y,ypred){
  num = sum((y-ypred)^2)
  dem =  sum((y-mean(y))^2)
  r = 1 - (num/dem)
  return(r)
}



## -------
## 1. Modelo de Regressão Beta
## -------
## A função log-log apresentou melhor performance


timebreg_inic <- Sys.time()

breg = betareg(tax ~.,
               data=treinocadu,
               link="loglog") 

timebreg_end <- Sys.time()

## Tempo de treinamento
paste("Tempo de treinamento: ", timebreg_end-timebreg_inic)

## Performance

ypredbreg = predict(breg,testecadu)

MAE(y=testecadu$tax,ypred=ypredbreg)
MSE(y=testecadu$tax,ypred=ypredbreg)
sqrt(MSE(y=testecadu$tax,ypred=ypredbreg))
R2(y=testecadu$tax,ypred=ypredbreg)

#confint(breg)



## ----
## 2. Algoritmo de Bagging para a Regressão Beta
## ----

bagging_betareg = function(xtreino,ytreino,xteste,n_estimadores,n_amostra,linkfun){
  
  ## predicoes do bagging em uma matriz
  matriz_bag = matrix(data=0,nrow=dim(xteste)[1],ncol=n_estimadores)
  
  enableJIT(3)
  for(preditor in 1:n_estimadores){
    random_id = sample(1:dim(xtreino)[1],replace=T)
    
    xtreino_bag = xtreino[random_id,]
    ytreino_bag = ytreino[random_id]
    
    dados_bag = cbind(ytreino_bag,xtreino_bag)
    colnames(dados_bag)[1] = "y"
    
    reg_bag = betareg(y ~ .,
                      data=dados_bag,
                      link=linkfun)
    
    y_pred_bag = predict(reg_bag, newdata=xteste)
    matriz_bag[,preditor] = y_pred_bag
  }
  
  predicoes_bag = rowSums(matriz_bag)/n_estimadores
  return(predicoes_bag)
}

timebag_inic <- Sys.time()

y_pred_bagging= bagging_betareg(xtreino=treinocadu[, -which(names(treinocadu) == "tax")],
                                ytreino=treinocadu[, "tax"],
                                xteste=testecadu[, -which(names(testecadu) == "tax")],
                                n_estimadores=100,
                                n_amostra=dim(treinocadu)[1],
                                linkfun = "loglog")

# Marcar o final do tempo
timebag_fim <- Sys.time()

paste("Tempo de treinamento: ", timebag_fim-timebag_inic)

MAE(y=testecadu$tax,ypred=y_pred_bagging)
MSE(y=testecadu$tax,ypred=y_pred_bagging)
sqrt(MSE(y=testecadu$tax,ypred=y_pred_bagging))


## ----
## grid search Bagging in Beta Regression
## ----

n_estimators = seq(10,800,by=10)

matrix_cv_bag = matrix(data=0,nrow=length(n_estimators),ncol=1)

timecvbag_inicio <- Sys.time()
enableJIT(3)
for(i in 1:length(n_estimators)){
  y_pred_bag = bagging_betareg(xtreino=treinocadu[, -which(names(treinocadu) == "tax")],
                               ytreino=treinocadu[, "tax"],
                               xteste=testecadu[, -which(names(testecadu) == "tax")],
                               n_estimadores=n_estimators[i],
                               n_amostra=dim(treinocadu)[1],
                               linkfun = "logit")
  
  matrix_cv_bag[i,1] = MSE(y=testecadu$tax,ypred=y_pred_bag)
  print(paste("estimators: ", n_estimators[i]))
  
}
timecvbag_fim <- Sys.time()

#plot(matrix_cv_bag,type='b')
cv_bag = as.data.frame(matrix_cv_bag)
cv_bag$ests = n_estimators 
colnames(cv_bag)[1] = "EQM"

ggplot(data=cv_bag, aes(x=n_estimators, y=EQM, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  theme_minimal()


## MSE
MSE(y=testecadu$tax,ypred=y_pred_bagging)
## RMSE
sqrt(MSE(y=testecadu$tax,ypred=y_pred_bagging))
## MAE
MAE(y=testecadu$tax,ypred=y_pred_bagging)
## R2: Coeficiente de Determinação
R2(y=testecadu$tax,ypred=y_pred_bagging) 


timecvbag_fim-timecvbag_inic


## ----
## 3. Algoritmo de Estimação Feature Bagging (RF)
## ----


rf_betareg = function(xtreino,ytreino,xteste,
                      n_estimadores,n_amostra,n_features){
  
  ## predicoes do Random Forest em uma matriz
  matriz_rf = matrix(data=0,nrow=dim(xteste)[1],ncol=n_estimadores)
  
  enableJIT(3)
  for(preditor in 1:n_estimadores){
    random_id = sample(1:dim(xtreino)[1],replace=T)
    random_idf = sample(1:dim(xtreino)[2],replace=F)[1:n_features]
    
    xtreino_rf = xtreino[random_id,random_idf]
    ytreino_rf = ytreino[random_id]
    
    dados_rf = cbind(ytreino_rf,xtreino_rf)
    colnames(dados_rf)[1] = "y"
    
    reg_rf = betareg(y ~ .,
                     data=dados_rf,
                     link='logit')
    
    y_pred_rf = predict(reg_rf, newdata=xteste)
    matriz_rf[,preditor] = y_pred_rf
  }
  
  predicoes_rf = rowSums(matriz_rf)/n_estimadores
  return(predicoes_rf)
}


timerfbeta_inic = Sys.time()
y_pred_rf1 = rf_betareg(xtreino=treinocadu[, -which(names(treinocadu) == "tax")],
                        ytreino=treinocadu[, "tax"],
                        xteste=testecadu[, -which(names(testecadu) == "tax")],
                        n_estimadores=100,
                        n_amostra=dim(treinocadu)[1],
                        n_features=25)
timerfbeta_fim = Sys.time()


paste("Tempo de treinamento: ", (timerfbeta_fim-timerfbeta_inic)*60)


## MAE
MAE(y=testecadu$tax,ypred=y_pred_rf1)
## MSE
MSE(y=testecadu$tax,ypred=y_pred_rf1)
## RMSE
sqrt(MSE(y=testecadu$tax,ypred=y_pred_rf1))
## R2: Coeficiente de Determinação
R2(y=testecadu$tax,ypred=y_pred_rf1) 



n_estimators = seq(50,500,by=50)
n_features = c(5,10,15,25)

matrix_cv_RF = matrix(data=0,nrow=length(n_estimators),ncol=length(n_features))


timecvfb_inic <- Sys.time()
enableJIT(3)
for(i in 1:length(n_estimators)){
  enableJIT(3)
  for(j in 1:length(n_features)){
    y_pred_rf = rf_betareg(xtreino=treinocadu[, -which(names(treinocadu) == "tax")],
                           ytreino=treinocadu[, "tax"],
                           xteste=testecadu[, -which(names(testecadu) == "tax")],
                           n_estimadores=n_estimators[i],
                           n_amostra=dim(treinocadu)[1],
                           n_features=n_features[j])
    
    matrix_cv_RF[i,j] = MSE(y=testecadu$tax,ypred=y_pred_rf)
    print(paste("par: (",n_estimators[i],",",n_features[j],")"))
  }
}
timecvfb_fim <- Sys.time()


timecvfb_inic-timecvfb_fim

cv_RF = as.data.frame(matrix_cv_RF)
colnames(cv_RF) = c("f5","f10","f15","f25")
cv_RF$estimators = seq(50,500,by=50)

gf5 = ggplot(data=cv_RF, aes(x=estimators, y=f5, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2.5) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  labs(title="k = 5") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

gf10 = ggplot(data=cv_RF, aes(x=estimators, y=f10, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2.5) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  labs(title="k = 10") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

gf15 = ggplot(data=cv_RF, aes(x=estimators, y=f15, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2.5) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  labs(title="k = 15") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

gf25 = ggplot(data=cv_RF, aes(x=estimators, y=f25, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2.5) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  labs(title="k = 25") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

grid.arrange(gf5,gf10,gf15,gf25,ncol=2)




## ----
## Algoritmo de Boosing para a Regressão Beta
## Gradiente Boosting
## ----

## biblioteca betaboost implementa
## tem que puxar do github

library("devtools")

#install_github("boost-R/betaboost")
library("betaboost")

timeboost_inic <- Sys.time()

betaboosting = betaboost(tax ~.,
                         data=treinocadu, iterations = 500,
                         form.type = "betaboost")

timeboost_fim <- Sys.time()


timeboost_fim-timeboost_inic


y_predbetaboost1 = predict(betaboosting,testecadu)

## nomalização da resposta para o intervalo (0,1)
y_predbetaboost1 = (y_predbetaboost1 - min(y_predbetaboost1))/(max(y_predbetaboost1) - min(y_predbetaboost1))

## MSE
MSE(y=testecadu$tax,ypred=y_predbetaboost1)
## RMSE
sqrt(MSE(y=testecadu$tax,ypred=y_predbetaboost1))
## MAE
MAE(y=testecadu$tax,ypred=y_predbetaboost1)
## R2: Coeficiente de determinação
R2(y=testecadu$tax,ypred=y_predbetaboost1)




## ---------------------------------------------------------

## Cross-validation Process in Boosting for Beta Regression

n_estimators = seq(10,800,by=10)

matrix_cv_boost = matrix(data=0,nrow=length(n_estimators),ncol=1)

timecvboost_inicio <- Sys.time()
enableJIT(3)
for(i in 1:length(n_estimators)){
  
  cv_bestaboost = betaboost(tax ~.,
            data=treinocadu, iterations = n_estimators[i],
            form.type = "betaboost")
  
  y_pred_boost = predict(cv_bestaboost,testecadu)
  y_pred_boost = (y_pred_boost - min(y_pred_boost))/(max(y_pred_boost) - min(y_pred_boost))
  
  matrix_cv_boost[i,1] = MSE(y=testecadu$tax,ypred=y_pred_boost)
  print(paste("estimators: ", n_estimators[i]))
}
timecvboost_fim <- Sys.time()


cv_boost = as.data.frame(matrix_cv_boost)
cv_boost$ests = n_estimators
colnames(cv_boost)[1] = "EQM"

ggplot(data=cv_boost, aes(x=n_estimators, y=EQM, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2) +
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  theme_minimal()


## Tempo para a validação cruzada
timecvboost_fim-timecvboost_inicio








