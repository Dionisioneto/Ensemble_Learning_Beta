## ---------
## Ensemble Learning for Beta Regression
## Joining data of census 2010 and Cad Unico System
## Date: 09/01/2024
## ---------

library(readxl)

## 1st step: Set your work directory.

## 2nd step: Load data.

cov = read_excel("covariates_Brazil_census2010.xlsx",sheet=3)
cadu = read.table("data_cad_unico_2015.txt",sep=",",header=TRUE)

## 3rd step: Filter the cadu data for anomes in 201512 (December of 2015).
## The update in the final of the 2015.

ind = which(cadu$anomes_s=="201512")
newcadu = cadu[ind,]

#dim(cov)

## Exclude covariates that contains no information.

indcov = which(cov$ANO=="2010")
cols = c("ANO","UF","Codmun6","Codmun7","Município","ESPVIDA",
        "FECTOT", "RAZDEP", "E_ANOSESTUDO", "T_ANALF18M",
         "T_FBBAS", "T_FBFUND", "T_FBMED", "T_FBSUPER", 
         "T_MED18M", "T_SUPER25M", "GINI", "PIND", "PPOB",
         "RDPC1", "RDPCT", "THEIL","T_BANAGUA",
         "T_DENS", "T_LIXO", "T_LUZ", "AGUA_ESGOTO",
         "T_M10A14CF", "T_M15A17CF", "I_ESCOLARIDADE", "IDHM",
         "IDHM_E","IDHM_L", "IDHM_R")

newcov = cov[indcov,cols]


## 4th step: Verify which municipalities is in 2015 are not in 2010.
## We need to exclude some municiplities because there were some
## changes in the geographical setting between 2010 and 2015.

mun_exc = setdiff(newcadu$codigo_ibge, newcov$Codmun6)

newcadu = newcadu[!(newcadu$codigo_ibge %in% mun_exc),]

#dim(newcov);dim(newcadu)

## for the target, we consider the covariate 
## "cadun_taxa_atualizacao_cadastral_d" in Cad Unico data (cadu).

cadu_uptax = newcadu[,c("codigo_ibge", "cadun_taxa_atualizacao_cadastral_d")]
colnames(cadu_uptax) = c("codigo_ibge","tax")

cadu_uptax$tax = cadu_uptax$tax/100

cadu_uptax$tax=replace(cadu_uptax$tax,
                       cadu_uptax$tax>0.99,0.9999)

summary(cadu_uptax$tax)

## --------------------------

## 5th step: Merge between two datasets.
library(dplyr)

colnames(newcov)[3] = "codigo_ibge" 

dcadund = left_join(cadu_uptax,newcov, 
                    by = "codigo_ibge", "Codmun6")

head(dcadund)


write.csv2(dcadund, file = "final_data.csv")


















