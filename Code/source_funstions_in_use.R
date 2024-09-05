### ----
### Script de funções utilizadas durante a execução do projeto
### ----

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









