## -----
## Estudo do método ensemble Bagging
## no modelo de Regressão Beta
## -----


## Pacotes necessarios
if(!require("pacman")) install.packages("pacman"); library(pacman)
p_load(betareg,gamlss,gamlss.dist,
       caret)

## Simulacao de dados

set.seed(10)
beta <- c(4, -0.9, 1.5, -0.1, 0.2) ## coeficientes de regressao

n = 500
x1 = rnorm(n, 2, 2)
x2 = rnorm(n, 0, 2)
x3 = rnorm(n, 0, 3.4)
x4 = rnorm(n, -1, 0.2)

eta = cbind(1,x1,x2,x3,x4) %*% beta ## combinacao linear

## media modelada por um termo regressor
mu = binomial(link = logit)$linkinv(eta)

phi = 15  ## dispersao constante

## Variavel resposta
set.seed(12)
y = rbeta(n, mu * phi, (1 - mu) * phi)

dat <- data.frame(cbind(y, x1,x2,x3,x4))

hist(dat$y, col = "darkgrey", border = F,
     main = "Distribuição da Variável resposta")

summary(dat$y)



## -----
## Vamos realizar a regressão beta e estimar o erro quadrático médio
## -----
## temos que considerar que a nossa simulação tem betas
## inlflacionada


model <- gamlss(y ~ x1+x2+x3+x4, 
                family=BEOI, 
                data=dat)

summary(model)

## ----
## Separação dos dados em treino e teste
## ----

# semente
set.seed(123)

# Divida os dados em treinamento (70%) e teste (30%)
id_treino <- createDataPartition(dat$y, p = 0.7, list = FALSE)
treino <- dat[id_treino, ]
teste <- dat[-id_treino, ]


beta_inf1 <- gamlss(y ~ x1+x2+x3+x4, 
                family=BEOI, 
                data=treino)

y_pred=predict(beta_inf1,newdata=teste)

## Erro Quadrático Médio (EQM)

EQM = function(y_true, y_pred){
  eqm = sum((y_true-y_pred)^2)/length(y_true)
  return(eqm)
}

EQM(y_true=teste$y,y_pred=y_pred)

## ----
## 2. Algoritmo Bagging
## ----

bagging_betareg = function(dados,n_estimadores, n_amostra){
  
  # Divida os dados em treinamento (70%) e teste (30%)
  id_treino = sample(1:dim(dados)[1],0.7*dim(dados)[1])
  treino <- dat[id_treino, ]
  teste <- dat[-id_treino, ]
  
  ## predicoes do bagging em uma matriz
  matriz_bag = matrix(data=0,nrow=dim(teste)[1],ncol=n_estimadores)
  
  for(preditor in 1:n_estimadores){
    random_id = sample(1:dim(treino)[1],replace=F)[1:n_amostra]
    
    treino_bag = treino[random_id,]
    
    reg_bag = gamlss(y ~ x1+x2+x3+x4, 
                    family=BEOI, 
                    data=treino_bag)
    
    y_pred_bag = predict(reg_bag, newdata=teste)
    matriz_bag[,preditor] = y_pred_bag
  }
  
  predicoes_bag = rowSums(matriz_bag)/n_estimadores
  return(predicoes_bag)
}

y_pred_bag =bagging_betareg(dados=dat,n_estimadores=10,n_amostra=187)

length(y_pred_bag)


EQM(y_true=teste$y,y_pred=y_pred_bag)

## [Corrigir a parte do porque não est acompativel o lenght]


