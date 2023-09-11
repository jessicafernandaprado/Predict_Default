#!/usr/bin/env python
# coding: utf-8

# ## CASE NEON- Prever probabilidade de inadimplência

# #### Probabilidade de inadimplência
#  Um dos assuntos que causam mais problemas para bancos, financiadoras, fintechs ou empresas sejam grandes ou pequenas, é a inadimplência dos clientes. Ao avaliar o comportamento do cliente se irá deixar de cumprir com as suas obrigações financeiras ajuda a saber o risco e implementar ações em clientes com maiores chances de serem não pagadores (risco de default) para que a companhia não sofra. 
#  
# Utilizando os dados referente ao históricos de pagamentos e extratos de contas de clientes de cartão de crédito em Taiwan, no período de Abril de 2005 até Setembro de 2005, disponível no Kaggle "Default of Credit Card Clients Dataset"
#  
# #### Objetivo: 
# Prever a probabilidade de inadimplência dos clientes
# 
# #### Variável resposta:
#  1 se é inadimplente;
#  0 caso contrário
#  

# ### Descrição das variáveis:
# 
# Variables
# ID: ID of each client
# 
# LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit)
# 
# SEX: Gender (1=male, 2=female)
# 
# EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# 
# MARRIAGE: Marital status (1=married, 2=single, 3=others)
# 
# AGE: Age in years
# 
# PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
# 
# PAY_2: Repayment status in August, 2005 (scale same as above)
# 
# PAY_3: Repayment status in July, 2005 (scale same as above)
# 
# PAY_4: Repayment status in June, 2005 (scale same as above)
# 
# PAY_5: Repayment status in May, 2005 (scale same as above)
# 
# PAY_6: Repayment status in April, 2005 (scale same as above)
# 
# BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
# 
# BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
# 
# BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
# 
# BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
# 
# BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
# 
# BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
# 
# PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
# 
# PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
# 
# PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
# 
# PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
# 
# PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
# 
# PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
# 
# default.payment.next.month: Default payment (1=yes, 0=no)

# In[259]:


##### Importação das LIBS
import numpy as np  
import pandas as pd 

# visualização dos dados
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)

# estilo dos gráficos gerais 
sns.set_style('dark')

# Biblioteca de Machine Learning (ML)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split

from sklearn.metrics import roc_auc_score ### métricas
from sklearn.model_selection import cross_validate

import shap

# filtrar mensagens de warning
import warnings
warnings.filterwarnings('ignore')


# In[ ]:





# ## Importação do dataset

# Será utilizado a função read_csv() da biblioteca pandas. Em que, será visualizado a 5 primeiras entradas do nosso dataset para que possa verificar a estrutura dos dados inicialmente.

# In[16]:


### Importando o dataset:
dados_df= pd.read_csv('C:/Users/jessi/Downloads/UCI_Credit_Card.csv')

## Visualizando as 5 primeiras entradas:

dados_df.head()


# 

# ### Análise exploratório dos dados:

# In[30]:


## Detectando se existem valores ausentes:
dados_df.info()

### Número total de valores ausentes:

print('Total de números ausentes:', dados_df.isnull().sum().sum())


# In[28]:


## Quantidade linhas x colunas
print(dados_df.shape)


# Aprofundamento nos dados: Dos 30.000 observações,cada linha representa uma 
# específica característica, que pode ter um unico número ID, e contém 24 diferentes informações sobre eles. 
# 

# In[311]:


dados_df.nunique()


# In[33]:


# check valores faltantes
dados_df.isna().sum()   


# ### Estatísticas descritivas no conjunto de dados:

# In[29]:


dados_df.describe().T


# Alguns pontos importantes:
# - A coluna ID: identifica o usuário, pode ser descartada pois não agrega na análise;
# - Default payment (1=yes, 0=no): é nossa variável alvo;
# - As colunas SEX, EDUCATION, MARRIAGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, default_payment_next_month são variáveis categóricas;
# 
# Além disso, ao olhar a tabela com as informações descritivas aparentemente não existem valores (infinitos) que possam interferir no modelo. 
# 

# In[34]:


### Tamanho do dataset:
print("TAMANHO DO DATASET")
print(f"OBSERVATIONS:\t {dados_df.shape[0]} ")
print(f"FEATURES:\t {dados_df.shape[1]} ")


# In[40]:


### renomeando essa coluna para facilitar as manipulações:
dados_df.rename(columns={'default.payment.next.month':'def_pay'}, inplace=True)


# 

# ### Descritiva por variáveis:

# In[54]:


#Descrição de variáveis categóricas:
dados_df[['SEX', 'EDUCATION', 'MARRIAGE']].describe()

# ### SEX: Gender (1=male, 2=female)
# #EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# MARRIAGE: Marital status (1=married, 2=single, 3=others)


## Estranho que para a target "Education" categoria 5 e 6 estão desconhecidas e a categoria 0 é desconhecida na documentação
## Similar a coluna "Marriage" que não está na documentação o label 0;


# In[47]:


# Descrição do atraso no pagamento:
dados_df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].describe()



### Todas as colunas apresentam um lab -2 e não está na documentação do dataset


# In[49]:


# Descrição do extrato na fatura
dados_df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].describe()


### Estranho esses valores negativos. Como pode-se considerar como crédito?


# In[51]:


# Descrição do pagamento anterior:
dados_df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].describe()


# ### Tratando os casos encontrados acima:

# In[57]:


### Renomeando a coluna PAY_0 para PAY_1
dados_df.rename(columns={'PAY_0':'PAY_1'}, inplace=True)
dados_df.head()



# Como existem algumas categorias que estão mal rotuladas ou o mesmo sem documentação, segue-se a nova arcação:
# - Considera-se 0 da coluna 'Marriage' como outro, sendo 3
# - Considera-se 0 não documentado e 5,6 (desconhecido na documentação) da coluna 'EDUCATION' como outro, sendo 4

# In[58]:


# Alterando as categorias 0,5,6 para 4 da target 'EDUCATION'
dados_df['EDUCATION'] = dados_df['EDUCATION'].apply(lambda x: 4 if x in [0, 5, 6] else x)


# In[59]:


# Alterando a categoria 0 para 3 da target 'MARRAIGE'
dados_df['MARRIAGE'] = dados_df['MARRIAGE'].replace(0, 3)


# Outro tratamento a ser realizado é com relação a variável PAY_N(1,2,3,4,5,6) que se refere ao atraso no pagamento, em que, na marcação existem -1 (ou seja, está dentro do prazo) e -2, 0 sem descrição do que possa ser, porém avaliando pode-se considerar como 0 que é um pagamento dentro do prazo em ambos os casos.
# 

# In[60]:


### Redefinindo marcações:

def replace_to_zero(col):
    troca = (dados_df[col] == -2) | (dados_df[col] == -1) | (dados_df[col] == 0)
    dados_df.loc[troca, col] = 0

for i in ['PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:
    replace_to_zero(i)


# In[63]:


dados_df[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].describe()


# ### Categorias finais no dataset após o tratamento:
# 
# - COLUNA SEX: 1 MASCULINO, 2 FEMININO;
# - EDUCATION: 1 graduate school, 2 university, 3 high school, 4 others;
# - MARRIAGE: 1 married, 2 single, 3 others;
# - PAY_1,2,3,4,5,6: 0 paid duly, 1 payment delay for one month, ..., 9 payment delay for nine months and above
# 

# In[ ]:





# ### Visualização da variável de interesse (target: default_payment)

# In[55]:


def_cnt = (dados_df.def_pay.value_counts(normalize=True)*100)
def_cnt.plot.bar(figsize=(6,6))
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12)
plt.title("Probability Of Defaulting Payment Next Month", fontsize=15)
for x,y in zip([0,1],def_cnt):
    plt.text(x,y,y,fontsize=12)
plt.show()


# In[69]:


dados_df['def_pay'].value_counts()


# Ou seja, existem cerca de 22% dos clientes em defaul nos dados.

# In[70]:


g = sns.FacetGrid(dados_df, row='def_pay', col='MARRIAGE')
g = g.map(plt.hist, 'AGE')

plt.show()


# In[71]:


g = sns.FacetGrid(dados_df, row='def_pay', col='SEX')
g = g.map(plt.hist, 'AGE')


# In[83]:


g = sns.FacetGrid(dados_df, row='def_pay', col='EDUCATION')
g = g.map(plt.hist, 'AGE')


# In[92]:


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(18, 15))

# Bar plot for EDUCATION
ax1 = sns.barplot(x="EDUCATION", y="def_pay", data=dados_df, palette='dark', errorbar=None, ax=axes[0])
ax1.set_ylabel("% of Default", fontsize=12)
ax1.set_ylim(0, 0.5)
ax1.set_xticks([0, 1, 2, 3])
ax1.set_xticklabels(['Grad School', 'University', 'High School', 'Others'], fontsize=13)
for p in ax1.patches:
    ax1.annotate("%.2f" % (p.get_height()), (p.get_x() + 0.30, p.get_height() + 0.03), fontsize=15)

# Bar plot for MARRIAGE
ax2 = sns.barplot(x="MARRIAGE", y="def_pay", data=dados_df, palette='dark', errorbar=None, ax=axes[1])
ax2.set_ylabel("% of Default", fontsize=12)
ax2.set_ylim(0, 0.5)
ax2.set_xticks([0,1,2])
ax2.set_xticklabels(['Married', 'Single', 'Others'], fontsize=13)
for p in ax2.patches:
    ax2.annotate("%.2f" % (p.get_height()), (p.get_x() + 0.30, p.get_height() + 0.03), fontsize=15)

# Bar plot for SEX
ax3 = sns.barplot(x="SEX", y="def_pay", data=dados_df, palette='dark', errorbar=None, ax=axes[2])
ax3.set_ylabel("% of Default", fontsize=12)
ax3.set_ylim(0, 0.5)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Male', 'Female'], fontsize=13)
for p in ax3.patches:
    ax3.annotate("%.2f" % (p.get_height()), (p.get_x() + 0.30, p.get_height() + 0.03), fontsize=15)

plt.tight_layout()
plt.show()


# - Olhando a variável nível de escolaridade com relação a variável dependente, percebe-se descritivamente que tende a diminuir a inadimplência conforme aumenta o nível de escolaridade.
# - Já estado cívil estão muito próximas as proporções de serem inadimplentes
# - Com relação ao sexo, mesmo a proporção de mulheres serem maiores ao homens no dataset, porém apresentam maior percentual de serem inadimplentes.

# In[149]:


def show_value_counts(col):
    print(col)
    value_counts = dados_df[col].value_counts()
    percentage = value_counts / len(dados_df) * 100
    result_df = pd.DataFrame({'Value': value_counts.index, 'Count': value_counts, 'Percentage': percentage})
    result_df = result_df.sort_values(by='Value')
    print('--------------------------')
    print(result_df)
    print('--------------------------')
    generate_pie_plot(result_df)
    
    
def generate_pie_plot(data_frame):
    plt.figure(figsize=(6, 4))
    plt.pie(data_frame['Count'], labels=data_frame['Value'], autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()
    

show_value_counts('SEX')
show_value_counts('MARRIAGE')
show_value_counts('EDUCATION') 


# ### TABELA CRUZADA:

# In[161]:


# Criar crosstab de contagem
tabela_contagem = pd.crosstab(dados_df['SEX'], dados_df['def_pay'])
 
# Criar crosstab de porcentagem
tabela_porcentagem = pd.crosstab(dados_df['SEX'], dados_df['def_pay'], normalize='index')
 
# Concatenar as duas tabelas
tabela_final = pd.concat([tabela_contagem, tabela_porcentagem], keys=['Contagem', 'Porcentagem'])
print(tabela_final)


# In[162]:


# Criar crosstab de contagem
tabela_contagem = pd.crosstab(dados_df['MARRIAGE'], dados_df['def_pay'])
 
# Criar crosstab de porcentagem
tabela_porcentagem = pd.crosstab(dados_df['MARRIAGE'], dados_df['def_pay'], normalize='index')
 
# Concatenar as duas tabelas
tabela_final = pd.concat([tabela_contagem, tabela_porcentagem], keys=['Contagem', 'Porcentagem'])
print(tabela_final)


# In[163]:


# Criar crosstab de contagem
tabela_contagem = pd.crosstab(dados_df['EDUCATION'], dados_df['def_pay'])
 
# Criar crosstab de porcentagem
tabela_porcentagem = pd.crosstab(dados_df['EDUCATION'], dados_df['def_pay'], normalize='index')
 
# Concatenar as duas tabelas
tabela_final = pd.concat([tabela_contagem, tabela_porcentagem], keys=['Contagem', 'Porcentagem'])
print(tabela_final)


# 

# In[107]:


dados_df['SEXO'] = dados_df['SEX'].map({2: 'Fem', 1: 'Masc'})


# In[116]:


sex_0 = (dados_df.SEXO[dados_df['def_pay'] == 0].value_counts())
sex_1 = (dados_df.SEXO[dados_df['def_pay'] == 1].value_counts())
names= ['Fem', 'Masc']
plt.subplots(figsize=(8,5))
plt.bar(sex_0.index, sex_0.values, label='0')
plt.bar(sex_1.index, sex_1.values, label='1')
for x,y in zip(names,sex_0):
    plt.text(x,y,y,fontsize=12)
for x,y in zip(names,sex_1):
    plt.text(x,y,y,fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Number of clients in each age group", fontsize=15)
plt.legend(loc='upper right', fontsize=15)

plt.show()


# In[122]:


dados_df['def_pay'].groupby(dados_df['SEXO']).value_counts()


# In[127]:


#Matriz de correlação

k = 10 ## número de variáveis no mapa calor
corrmat = dados_df.corr()
cols = corrmat.nlargest(k, 'def_pay')['def_pay'].index ### variável dependente

cm = np.corrcoef(dados_df[cols].values.T)
sns.set(font_scale=1.25)
plt.subplots(figsize=(10,10))
hm = sns.heatmap(cm, cbar=True, 
                 annot=True,
                 square=True,
                 fmt='.2f', 
                 
                 annot_kws={'size': 10},
                 yticklabels=cols.values, 
                 xticklabels=cols.values)
plt.show()


# É notável que existe uma correlação de inadimplência do próximo mês dependendo do status de pagamento dos últimos 6 meses. Ou seja, se o cliente atrasou no primeiro mês possivelmente tende mais chances de atrasar nos próximos. Se fizesse uma análise para ver o ponto de corte mais aprofundado que ocorre a estatbilização talvez seria nos últimos 2-3 meses para causar um default.

# 

# ### Aplicação da Metodologia

# Separando do dataset a variável alvo (dependente) das demais e dividir os dados entre dados de treino e teste com a funçõ trains_test_split.

# In[286]:


# separar as variáveis independentes da variável alvo (dependente)
X = dados_df.drop(['def_pay', 'ID'], axis=1).select_dtypes(exclude='object')
y = dados_df['def_pay']


# ### Separação entre treino e teste

# In[287]:


# dividir o dataset entre treino e teste 
### 80% Treino
### 20% Teste

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.2,
                                                    random_state=2,
                                                    stratify=y)


# In[288]:


### Percentual na base de dados (Treino e Teste)  variável resposta
print('Dados de teste: \n', y_test.value_counts(),'\n')

print('Dados de treino: \n', y_train.value_counts(),'\n')
## 22% default 1 
## 78% default 0


# ### Aplicando Modelos de ML
# 
# Para a resolução deste problema será utilizado alguns modelos para testar em ambos cenários de previsão.
# 
# Para avaliação dos algoritmos utilizados, será utilizado as seguintes métricas:
# - GINI;
# - KS;
# - Área sob a curva ROC;
# - Acurácia: indica uma perfromance geral do modelo. Ou seja, dentre todas as classificaçãoes, quantas no geral o modelo classificou corretamente
# 
# 

# In[289]:


### kfoldes (número de partições)

# armazenar os resultados
score = {}
kf = KFold(n_splits=5, random_state=0, shuffle=True)

xgb_model = XGBClassifier(learning_rate=0.01,   ## Taxa de aprendizado
                          max_depth=6,          ## Degraus das árvores
                          min_child_weight=3,
                     subsample=0.9, 
                          colsample_bylevel=0.85,
                          n_estimators=900)


# In[303]:


xgb_model= xgb_model.fit(X_train, y_train)


# In[305]:


y_predito= xgb_model.predict(X_test)


# ### Métricas

# In[291]:


xgb = cross_validate(xgb_model,
                     X_train, 
                     y_train,
                     cv=kf, 
                     scoring=['accuracy', 'precision', 'recall', 'roc_auc'])




# In[294]:


summary = pd.DataFrame({
            'labels': ['accuracy', 'precision', 'recall', 'roc_auc'],
            
            'xgb': [xgb['test_accuracy'].mean(), xgb['test_precision'].mean(), xgb['test_recall'].mean(), xgb['test_roc_auc'].mean()]           
}).set_index('labels')
summary.index.name=None
summary = summary.transpose()    
summary.style.applymap(lambda x: 'background-color: lightgreen' if x >= 0.5 else '')


# In[295]:


roc_ac= cross_val_score(xgb_model, X_train, y_train, n_jobs=1, cv=kf, scoring='roc_auc').mean()
roc_ac


# In[296]:


roc_ac_T= cross_val_score(xgb_model, X_test, y_test, n_jobs=1, cv=kf, scoring='roc_auc').mean()
roc_ac_T


# In[297]:


# previsões e probabilidades em cima do dataset de treino
xgb_gini= 2 * roc_ac-1
xgb_ks= 2 * roc_ac-1/1.3

xgb_gini1= 2 * roc_ac_T-1
xgb_ks1= 2 * roc_ac_T-1/1.3


# ver performance do algoritmo
print("\nGINI SCORE TREINO:")
print (xgb_gini)

print("\nKS SCORE TREINO:")
print (xgb_ks)

print("\nGINI SCORE TESTE:")
print (xgb_gini1)

print("\nKS SCORE TESTE:")
print (xgb_ks1)


# In[ ]:





# ### Utilizando o SHAP:
# para avaliar a importância das variáveis, para avaliar no conjunto de treinamento como cada valor de cada variável influenciou no resultado alcançado dado o modelo preditivo.
# Ou seja, qual a porcentagem X que o modelo apresentou para dizer se a classe correta é 0 (não inadimplente) ou 1 (inadimplente),
# 

# In[298]:


# Cálculo do SHAP - Definindo explainer com características desejadas
explainer = shap.TreeExplainer(model=xgb_model)

# Cálculo do SHAP
shap_values_train = explainer.shap_values(X_train, y_train)


# In[300]:


# summarize the effects of all the features
shap.summary_plot(shap_values_train, X_train)


# In[301]:


shap.summary_plot(shap_values_train, X_train, plot_type="bar")


# ## Probabilidade de inadimplência:

# In[307]:


dados_df['probab'] = xgb_model.predict_proba(X[X_train.columns])[:,1]
dados_df[['ID','probab']]


# Agora tem a probabilidade do cliente ser Inadimplente e pode-se criar ações com base nisso.

# ## Separando por decil:

# In[308]:


dados_df['QuantilRank'] = pd.qcut(dados_df['probab'], 10, labels=False)


# In[309]:


dados_df.head()


# In[310]:


dados_df['QuantilRank'].describe()

