## ----
## Estudo descritivo
## ----

## Dados: Taxa de de atualização do CadUnico

## Bibliotecas

if(!require(pacman)) install.packages("pacman"); library(pacman)
p_load(ggplot2,dplyr,geobr, sf,rio,readr)

## 1st step: Loading data

cadu = read.csv2("https://raw.githubusercontent.com/Dionisioneto/Ensemble_Learning_Beta/master/Data/final_data.csv")
head(cadu)
dim(cadu)

## ---
## Investigação da matriz de correlações 
## das preditoras
## ---

preditoras = cadu %>% subset(select=c(ESPVIDA,FECTOT, RAZDEP,
                                     E_ANOSESTUDO, T_ANALF18M,
                                     T_FBBAS, T_FBFUND,
                                     T_FBMED, T_FBSUPER,T_MED18M, T_SUPER25M,
                                     GINI, PIND, PPOB, RDPC1, RDPCT,
                                     THEIL, T_BANAGUA, T_DENS,
                                     T_LIXO, T_LUZ,AGUA_ESGOTO,
                                     T_M10A14CF, T_M15A17CF,
                                     I_ESCOLARIDADE,  IDHM, IDHM_L, IDHM_R))

head(preditoras)

## Armazenado as correlações

cormat = round(cor(preditoras),4)
head(cormat)

library(reshape2)
melted_cormat <- melt(cormat)
head(melted_cormat)

# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}
# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}

upper_tri <- get_upper_tri(cormat)
upper_tri

colnames(upper_tri) = paste("X", seq(1, dim(upper_tri)[1]), sep = "")
rownames(upper_tri) = paste("X", seq(1, dim(upper_tri)[1]), sep = "")

# Heatmap

library(ggplot2)

ggplot(data = melt(upper_tri, na.rm = TRUE), aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "#FF6787", high = "#7286D3", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson \nCorrelation") +
  theme_minimal()+ 
  xlab("")+
  ylab("")+
  theme(axis.text.x = element_text(angle = 90, vjust = 1, 
                                   size = 9, hjust = 1))+
  coord_fixed()


## ---
## Criação de um mapa 
## ---


## analise pelos municipios

brasil_muni = read_municipality(code_muni = "all",year=2010) %>% 
  select(-c("code_state"))

codigos_abrev <- substr(brasil_muni$code_muni, 1, nchar(brasil_muni$code_muni) - 1)

codigos_full = cbind(as.numeric(brasil_muni$code_muni), as.numeric(codigos_abrev))
colnames(codigos_full) = c("cod_ibge_muni", "shortibge")
codigos_full=as.data.frame(codigos_full)

cadu = left_join(cadu,codigos_full, by = c("codigo_ibge"="shortibge"))

## adicionando ao objeto geom
dfmapa = left_join(brasil_muni,cadu[,c("cod_ibge_muni", "Município","tax")], by = c("code_muni"="cod_ibge_muni"))

#st_as_sf(dfmapa, coords = "geometry")

#dfmapa=dfmapa[!is.na(dfmapa),]

ggplot() +
  geom_sf(data=dfmapa, aes(fill=tax), color= NA, size=.15)+
  labs(title="",
       size=8)+
  scale_fill_distiller(palette = "RdPu", limits=c(0.3, 1),
                       name="Update rate \n",direction = 1)+
  theme_minimal() 



## map to visualize the train and test data

set.seed(10) ## for reproducibility

# Training proportion
prop_treino = 0.8
n_treino = round(nrow(cadu) * prop_treino)

ind_treino = sample(1:nrow(cadu), n_treino)



maptrain = ggplot() +
  geom_sf(data=dfmapa,size=1.2,
          fill = "grey")+
  geom_sf(data=dfmapa[ind_treino,],color="grey",aes(size=0.2),
          fill = "steelblue")+
  labs(title="",
       size=1)+
  theme_minimal() 

maptest = ggplot() +
  geom_sf(data=dfmapa,size=.15,
          fill = "grey")+
  geom_sf(data=dfmapa[-ind_treino,],size=.05,
          fill = "green4")+
  labs(title="",
       size=8)+
  theme_minimal()

gridExtra::grid.arrange(maptrain,maptest,ncol=2)























