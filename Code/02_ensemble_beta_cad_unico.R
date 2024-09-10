## ----
## 02 - Training and Cross-validation for Ensemble methods 
## in Beta regression.
## ----

## Data: Demopgrafical covariates of Brazil's municipalities (2010)
## and the rate of update in the CadUnico system (2015) .

## Libraries

if(!require(pacman)) install.packages("pacman"); library(pacman)
p_load(ggplot2,dplyr, betareg, beepr,betaboost,e1071,caret,compiler,gridExtra)

## ---
## 1 - Reading data ====
## ---

cadu = read.csv2("https://raw.githubusercontent.com/Dionisioneto/Ensemble_Learning_Beta/master/Data/final_data.csv")
head(cadu)

## ---
## 2 - Selection of features and target ====
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


datacadu = cbind(X,y)
dim(datacadu)

## ---
## 3 - Slipt into training and test data ====
## ---

set.seed(10) ## for reproducibility

# Training proportion
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


## Verificando qual a melhor função de ligaçãoapresentou o melhor desempenho

linkfun = c("logit", "probit", "cloglog", "cauchit", "log", "loglog")
mse_linkfun = rep(0,length(linkfun))


enableJIT(3)
for(link in linkfun){
  breglin = betareg(tax ~.,
                 data=treinocadu,
                 link=link)
  
  ypredbreglin = predict(breglin,testecadu)
  

  mse_linkfun[which(linkfun==link)] = MSE(y=testecadu$tax,ypred=ypredbreglin)
  
}

cbind(linkfun,as.numeric(mse_linkfun))

## A função cauchit apresentou melhor performance (MSE)

timebreg_inic <- Sys.time()

breg = betareg(tax ~.,
               data=treinocadu,
               link="cauchit") 

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
#coef(breg)

#write.csv2(coef(breg),file="coef_breg.csv")
#write.csv2(confint(breg),file="ci_breg.csv")

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


## ----
## grid search Bagging in Beta Regression
## ----

n_estimators_bag = seq(10,800,by=10)

matrix_cv_bag = matrix(data=0,nrow=length(n_estimators_bag),ncol=1)

timecvbag_inicio <- Sys.time()
enableJIT(3)
for(i in 1:length(n_estimators_bag)){
  y_pred_bag = bagging_betareg(xtreino=treinocadu[, -which(names(treinocadu) == "tax")],
                               ytreino=treinocadu[, "tax"],
                               xteste=testecadu[, -which(names(testecadu) == "tax")],
                               n_estimadores=n_estimators_bag[i],
                               n_amostra=dim(treinocadu)[1],
                               linkfun = "cauchit")
  
  matrix_cv_bag[i,1] = MSE(y=testecadu$tax,ypred=y_pred_bag)
  print(paste("estimators: ", n_estimators_bag[i]))
  
}
timecvbag_fim <- Sys.time()


timecvbag_fim-timecvbag_inicio 


#plot(matrix_cv_bag,type='b')
cv_bag = as.data.frame(matrix_cv_bag)
cv_bag$ests = n_estimators_bag 
colnames(cv_bag)[1] = "EQM"

ggplot(data=cv_bag, aes(x=ests, y=EQM, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  theme_minimal()

rownames(cv_bag) = n_estimators_bag

#write.csv2(cv_bag, file = "cv_bag.csv")

## ---
## The best model (Bagging) ====
## ---

min_cv_bag = which(cv_bag$EQM==min(cv_bag$EQM))

cv_bag[min_cv_bag,] ## 30

## n_estimadores = 30

timebag_inic <- Sys.time()

y_pred_bagging= bagging_betareg(xtreino=treinocadu[, -which(names(treinocadu) == "tax")],
                                ytreino=treinocadu[, "tax"],
                                xteste=testecadu[, -which(names(testecadu) == "tax")],
                                n_estimadores=30,
                                n_amostra=dim(treinocadu)[1],
                                linkfun = "cauchit")

# Marcar o final do tempo
timebag_fim <- Sys.time()

timebag_fim-timebag_inic # tempo de excução

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
                      n_estimadores,n_amostra,n_features,linkfun){
  
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
                     link=linkfun)
    
    y_pred_rf = predict(reg_rf, newdata=xteste)
    matriz_rf[,preditor] = y_pred_rf
  }
  
  predicoes_rf = rowSums(matriz_rf)/n_estimadores
  return(predicoes_rf)
}




## ----
## grid search Bagging in Beta Regression
## ----

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
                           n_features=n_features[j],
                           linkfun="cauchit")
    
    matrix_cv_RF[i,j] = MSE(y=testecadu$tax,ypred=y_pred_rf)
    print(paste("par: (",n_estimators[i],",",n_features[j],")"))
  }
}
timecvfb_fim <- Sys.time()


timecvfb_fim-timecvfb_inic

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

cros_val_feabag_beta = cv_RF[,1:4]
colnames(cros_val_feabag_beta) = n_features
rownames(cros_val_feabag_beta) = n_estimators

#write.csv2(cros_val_feabag_beta, file = "cros_val_feabag_beta.csv")

## ---
## The best model (feature Bagging) ====
## ---

apply(cv_RF[,1:4], MARGIN=2, FUN = min)


min_cv_feature_bag = which(cv_RF$f25==min(apply(cv_RF[,1:4], MARGIN=2, FUN = min)))

cv_RF[min_cv_feature_bag,]

## n_estimators=50 and n_features=25

timerfbeta_inic = Sys.time()
y_pred_rf1 = rf_betareg(xtreino=treinocadu[, -which(names(treinocadu) == "tax")],
                        ytreino=treinocadu[, "tax"],
                        xteste=testecadu[, -which(names(testecadu) == "tax")],
                        n_estimadores=50,
                        n_amostra=dim(treinocadu)[1],
                        n_features=25,
                        linkfun="cauchit")
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



## ----
## Algoritmo de Boosting para a Regressão Beta (Gradiente Boosting) ====
## ----

## biblioteca betaboost implementa
## tem que puxar do github

install.packages("devtools");install.packages("shiny")
library("devtools")

#install_github("boost-R/betaboost")
library("betaboost")

## Estudo de validação cruzada

iteracoes = seq(10,500,10) # number of iterations
pas = seq(0.5,2,0.25) # learning rate

matrix_cv_Rboosting3 = matrix(data=0,nrow=length(iteracoes),ncol=length(pas))

timecvboost_inic3 <- Sys.time()
enableJIT(3)
for(j in 1:length(pas)){
  for(i in 1:length(iteracoes)){
    mod_pred_boost = betaboost(tax ~.,sl = pas[j],
                               data=treinocadu, iterations = iteracoes[i])
    
    y_pred_boost = predict(mod_pred_boost,testecadu)
    
    matrix_cv_Rboosting3[i,j] = MSE(y=testecadu$tax,ypred=y_pred_boost)
    print(paste("iteracoes: ", iteracoes[i], "step: ", pas[j]))
  }
}
timecvboost_fim3 <- Sys.time()


timecvboost_fim3- timecvboost_inic3

plot(matrix_cv_Rboosting3[,2],type='b')

colnames(matrix_cv_Rboosting3) = pas
rownames(matrix_cv_Rboosting3) = iteracoes
# write.csv2(matrix_cv_Rboosting3, file = "matrix_cv_Rboosting3.csv")


## Graphs

cv_boost = as.data.frame(matrix_cv_Rboosting3)
colnames(cv_boost) = c("lr05", "lr075", "lr1", "lr125", "lr15", "lr175", "lr2")
cv_boost$estimators = iteracoes

gf05 = ggplot(data=cv_boost, aes(x=estimators, y=lr05, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2.5) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  labs(title="Learning rate = 0.5") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

gf075 = ggplot(data=cv_boost, aes(x=estimators, y=lr075, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2.5) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  labs(title="Learning rate = 0.75") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


gf1 = ggplot(data=cv_boost, aes(x=estimators, y=lr1, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2.5) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  labs(title="Learning rate = 1.0") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))



gf125 = ggplot(data=cv_boost, aes(x=estimators, y=lr125, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2.5) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  labs(title="Learning rate = 1.25") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


gf15 = ggplot(data=cv_boost, aes(x=estimators, y=lr15, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2.5) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  labs(title="Learning rate = 1.5") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


gf175 = ggplot(data=cv_boost, aes(x=estimators, y=lr175, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2.5) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  labs(title="Learning rate = 1.75") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


gf2 = ggplot(data=cv_boost, aes(x=estimators, y=lr2, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2.5) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  labs(title="Learning rate = 2.0") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))



grid.arrange(gf05,gf075,gf1,gf15,gf175,gf2,ncol=2)



## ---
## treinamento do melhor modelo
## ---

which(matrix_cv_Rboosting3 == min(matrix_cv_Rboosting3), arr.ind = TRUE)

## numero de estimators = 10
## taxa de aprendizagem = 2

iteracoes[1]
pas[7]

timeboost_inic <- Sys.time()

betaboosting = betaboost(tax ~.,sl = 2,phi.formula=NULL,
                         data=treinocadu, iterations = 10)

timeboost_fim <- Sys.time()

timeboost_fim-timeboost_inic


y_predbetaboost1 = predict(betaboosting,testecadu)
hist(y_predbetaboost1)
summary(y_predbetaboost1)

## nomalização da resposta para o intervalo (0,1)
#y_predbetaboost1 = (y_predbetaboost1 - min(y_predbetaboost1))/(max(y_predbetaboost1) - min(y_predbetaboost1))

## MSE
MSE(y=testecadu$tax,ypred=y_predbetaboost1)
## RMSE
sqrt(MSE(y=testecadu$tax,ypred=y_predbetaboost1))
## MAE
MAE(y=testecadu$tax,ypred=y_predbetaboost1)
## R2: Coeficiente de determinação
R2(y=testecadu$tax,ypred=y_predbetaboost1)




















