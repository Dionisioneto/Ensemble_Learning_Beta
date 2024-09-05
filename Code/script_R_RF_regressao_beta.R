## -----
## Estudo do método ensemble Random Forest
## no modelo de Regressão Beta
## -----

## Definir o tamnho amostral para as amostras bootstrap (nb)
## Definir a quantidade de preditores fracos (B)
## Definir a quantidade de sorteio de preditores (M)

boostbetareg = function(dados,B, nb, M){
  
  # Divida os dados em treinamento (70%) e teste (30%)
  id_treino = sample(1:dim(dados)[1],0.7*dim(dados)[1])
  treino <- dat[id_treino, ]
  teste <- dat[-id_treino, ]
  
  ## predicoes do boosting em uma matriz
  matriz_boost = matrix(data=0,nrow=dim(teste)[1],ncol=B)
  
  for(preditor in 1:B){
    random_id = sample(1:dim(treino)[1],replace=F)[1:nb]
    
    cols_pred=(1:dim(treino)[2])[-1]
      
    random_predictors = sample(cols_pred,replace=F)[1:M]
    
    treino_boost = treino[random_id,c(1,random_predictors)]
    
    reg_boost = gamlss(y ~ ., 
                     family=BEOI, 
                     data=treino_boost)
    
    y_pred_boost = predict(reg_boost, newdata=teste)
    matriz_boost[,preditor] = y_pred_boost
  }
  
  predicoes_boost = rowSums(matriz_boost)/B
  return(predicoes_boost)
}

y_pred_bag = boostbetareg(dados=dat,B=10, nb=180, M=2)

length(y_pred_bag)