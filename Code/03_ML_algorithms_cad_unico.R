## ----
## 02 - Training and Cross-validation Of Machine Learning 
## Algorithms and Linear Regression.
## ----

## Data: Demopgrafical covariates of Brazil's municipalities (2010)
## and the rate of update in the CadUnico system (2015) .

## Libraries

if(!require(pacman)) install.packages("pacman"); library(pacman)
p_load(ggplot2,dplyr, tidyverse,
       caret,   # for general data preparation and model fitting
       e1071) # for fitting the xgboost model

## Leitura do banco de dados



#setwd("C:/Users/dioni/OneDrive - University of São Paulo/Doutorado em Estatística/2023.2/2 - Aprendizado de Máquina Estatístico/artigo_publicar/github_content/Data")

cadu = read.csv2("https://raw.githubusercontent.com/Dionisioneto/Ensemble_Learning_Beta/master/Data/final_data.csv")
head(cadu)

## ---
## Separação das covariaveis e da resposta ====
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

## -------
## Machine Learning Regression Models
## -------

## Métricas de desempenho

MSE = function(y,ypred){mean((y - ypred)^2)}
MAE = function(y,ypred){mean(abs(y - ypred))}
R2 = function(y,ypred){
  num = sum((y-ypred)^2)
  dem =  sum((y-mean(y))^2)
  r = 1 - (num/dem)
  return(r)
}


## ----
## Suport Vector Machines (SVM)
## ----

## Tuning SVR model by varying values of maximum allowable error and cost parameter

timesvm_cv_inic <- Sys.time()
OptModelsvm=tune(svm, tax~.,
                 data = treinocadu,
                 ranges=list(gamma=c(1,0.1,0.01,0.001),
                             cost=c(0.1,1,10,100), kernel=c("linear", "polynomial",
                                                            "radial", "sigmoid")),
                 tunecontrol = tune.control(cross = 10))
timesvm_cv_end <- Sys.time()


timesvm_cv_end-timesvm_cv_inic # time of cross-valitdation

#Print optimum value of parameters
print(OptModelsvm)


## Training with the best model

svm_inic <- Sys.time()

modelsvm=svm(tax~.,
             data = treinocadu,
             gamma = 0.01,
             cost = 1,
             kernel = "radial")

svm_end <- Sys.time()

svm_end-svm_inic

ypredsvm=predict(modelsvm,data=testecadu )

MSE(y=testecadu $tax,ypred=ypredsvm)
sqrt(MSE(y=testecadu $tax,ypred=ypredsvm))
MAE(y=testecadu $tax,ypred=ypredsvm)
R2(y=testecadu $tax,ypred=ypredsvm)

paste("Tempo de treinamento: ", timesvm_end - timesvm_inic)


## ---
## Random Forest Regressor (RFR)
## ---

#Tune the Random Forest model
#install.packages("randomForest")
library(randomForest)


## Cross-validation in random Forest


timerf_cv_inic <- Sys.time()
OptModelRF=tune(randomForest, tax~.,
                data = treinocadu,
                ranges=list(ntree = c(25,50,100,500),
                            maxfeatures = c(5,10,20,25),
                            maxnodes = c(3,6,9),
                            tunecontrol = tune.control(cross = 10)))

timerf_cv_end <- Sys.time()

timerf_cv_end-timerf_cv_inic
  
#Print optimum value of parameters
print(OptModelRF)



## Training with the best model

RF_inic <- Sys.time()

mdRFtree = randomForest(tax~.,
                        data = treinocadu,
                        ntree = 25,
                        maxfeatures=25,
                        maxnodes = 9)
RF_end <- Sys.time()


RF_end-RF_inic

ypredRF = predict(mdRFtree,newdata=testecadu )

## Metricas
MAE(y=testecadu $tax,ypred=ypredRF)
MSE(y=testecadu $tax,ypred=ypredRF)
sqrt(MSE(y=testecadu $tax,ypred=ypredRF))
R2(y=testecadu $tax,ypred=ypredRF)


# feature importance
tab_feaimp = importance(mdRFtree)
tab_feaimp = data.frame(tab_feaimp)

tab_feaimp$var = paste("X", seq(1, dim(X)[2]), sep = "")


# Graph for feature importance
library(ggplot2)


ggplot(tab_feaimp, aes(x = reorder(var, IncNodePurity), y = IncNodePurity)) +
  geom_bar(stat = "identity",fill="steelblue") +
  xlab("Feature \n") + ylab("\n Mean Decrease Accuracy") +
  coord_flip() +
  theme(axis.text.x = element_text(angle = 90, vjust = 1, 
                                   size = 9, hjust = 1)) +
  theme_minimal() 



## ---
## Linear Regression performance
## ---


lmreg_inic <- Sys.time()

lmreg = lm(tax ~.,
           data=treinocadu) 

lmreg_end <- Sys.time()


lmreg_end-lmreg_inic

## Performance

ypredlmreg = predict(lmreg,testecadu)

MAE(y=testecadu$tax,ypred=ypredlmreg)
MSE(y=testecadu$tax,ypred=ypredlmreg)
sqrt(MSE(y=testecadu$tax,ypred=ypredlmreg))
R2(y=testecadu$tax,ypred=ypredlmreg)








