### ----
### Método Bootstrap para a 
### Previsão do Algoritmo
### ----

### ----
### Algoritmo gerador de dados
### na regressão beta.
### ----

set.seed(1234)
eta = c(1, -0.2, 0.4, 0.1, 0.9)

n = 1000
x1 = rnorm(n, 10, 2)
x2 = rnorm(n, 0, 1)
x3 = rbinom(n,size=1,prob=0.5)
x4 = rbinom(n,size=1,prob=0.82)

ni = eta[1] + eta[2]*x1 + eta[3]*x2 + eta[4]*x3 + eta[5]*x4

mu = binomial(link = logit)$linkinv(ni)
phi = 100 ## Disperso, distribuição centralizada

y <- rbeta(n, mu * phi, (1 - mu) * phi)
dat <- data.frame(cbind(y,x1,x2,x3,x4))

hist(dat$y)

## ----
## Separação dos Dados em treino e teste
## ----

# Divida os dados em treinamento (70%) e teste (30%)

id_treino = sample(1:dim(dat)[1],0.7*dim(dat)[1],replace=F)
treino <- dat[id_treino, ]
teste <- dat[-id_treino, ]

## ----
## Algoritmo de Bagging para a Regressão Beta
## ----

bagging_betareg = function(xtreino,ytreino,xteste,n_estimadores,n_amostra){
  
  ## predicoes do bagging em uma matriz
  matriz_bag = matrix(data=0,nrow=dim(xteste)[1],ncol=n_estimadores)
  
  for(preditor in 1:n_estimadores){
    random_id = sample(1:dim(xtreino)[1],replace=T)
    
    xtreino_bag = xtreino[random_id,]
    ytreino_bag = ytreino[random_id]
    
    dados_bag = cbind(xtreino_bag,ytreino_bag)
    colnames(dados_bag) = c('x1','x2','x3','x4','y')
    
    reg_bag = betareg(y ~ x1+x2+x3+x4,
                      data=dados_bag,
                     link='logit')
    
    y_pred_bag = predict(reg_bag, newdata=xteste)
    matriz_bag[,preditor] = y_pred_bag
  }
  
  predicoes_bag = rowSums(matriz_bag)/n_estimadores
  return(predicoes_bag)
}

y_pred_bagging= bagging_betareg(xtreino=treino[,2:5],ytreino=treino[,1],
                                xteste=teste[,2:5],
                                n_estimadores=12,n_amostra=dim(treino)[1])

## formula do erro quadratico médio

EQM = function(y_true, y_pred){
  eqm = sum((y_true-y_pred)^2)/length(y_true)
  return(eqm)
}

EQM(y_true=teste[,1], y_pred=y_pred_bagging)

## ---
## Comparando o EQM com a estimação do convencional
## ---

regbeta = betareg(y ~ x1+x2+x3+x4,
                  data = treino,
                  link='logit')

y_pred_rbeta = predict(regbeta, newdata=teste)

EQM(y_true=teste[,1], y_pred=y_pred_rbeta)














