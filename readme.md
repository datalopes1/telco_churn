# Predição de Churn com Machine Learning - Telco Customer Churn 📞
O dataset Telco Customer Churn contêm informações sobre uma empresa fictícia de telecomunicações que forneceu serviços de telefone residencial e Internet para 7043 clientes na Califórnia no terceiro trimestre. Eles indicam quais clientes deixaram, permaneceram ou se inscreveram para os seus serviços. Os dados podem ser encontrados no [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data) e foram 
disponibilizados por [BlastChar](https://www.kaggle.com/blastchar).

![img](https://i.imgur.com/n3hyBGx.jpeg)

## Features
|Coluna|Descrição|
|---|---|
|customerID|Identificador único dos clientes|
|gender|Gênero|
|SeniorCitizen|É idoso?|
|Partner|Possui parceiro?|
|Dependents|Possui dependentes?|
|tenure|Tempo de relacionamento (em meses)|
|PhoneService|Possui serviço telefonico?|
|MultipleLines|Possui multiplas linhas?(Sim, não, não possui serviço telefonico)|
|InternetService|Provedor de serviços de internet (DSL, Fibra ou não)|
|OnlineSecurity|Possui seguro online?|
|OnlineBackup|Possui backup online?|
|DeviceProtection|Possui proteção do dispositivo?|
|TechSupport|Tem suporte técnico?|
|StreamingTV|Possui streaming de TV?|
|StreamingMovies|Possui streaming de Filmes?|
|Contract|Tipo de contrato(mês-a-mês, anual ou bi-anual)|
|PaperlessBilling|Recebe boletos?|
|PaymentMethod|Método de pagamento|
|MonthlyCharges|Taxa de serviço|
|TotalCharges|Total pago pelo cliente|
|Churn|Churn?|

## Metas e objetivos

Este projeto tem o intuíto de realizar uma breve análise exploratória e construir um modelo de Machine Learning para predição de Churn.

### Resultados
#### Insights da Análise Exploratória

- Clientes com maior tempo de contrato tem maior probabilidade de permanecer utilizando os serviços da Telco, é interessante premiar ou realizar ações de marketing com estes.
- É necessário buscar melhorar os serviços para: (1) novos clientes, (2) clientes com maior valor de taxas de serviço.
- Oferer um plano semestral pode estimular clientes da modalidade mês-a-mês a fazerem contratos mais longos (que sãos os com menor probabilida de de Churn).
- Oferta de serviços como suporte técnico tem peso na permanência dos clientes, deve-se buscar formas de facilitar o acesso. 

#### O modelo escolhido

Entre os três modelos testados (RandomForestClassifier, LogisticRegression e XGBClassifier) o que apresentou melhor desempenho considerando as métricas de Acurácia e ROC-AUC e capacidade generalização (diferença menor que 0.05 entre resultados de treino e teste), com os seguintes resultados:

|Métrica|Resultado|
|---|---|
|Accuracy Train|0.8059|
|Accuracy Test|0.8055|
|ROC-AUC Train|0.8492|
|ROC-AUC Test|0.8418|

Ao fim do projeto foi gerado um arquivo .xlsx com as previsões e probabilidades de Churn calculadas pelo modelo.

### Ferramentas utilizadas
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

### Bibliotecas Python utilizadas
#### Manipulação de dados
- Pandas, NumPy.
#### EDA
- Seaborn, Matplotlib.
#### Machine Learning e Feature Engineering
- Scikit-learn, XGBoost, feature_engine.

# Exploratory Data Analysis
### Comportamento do target
![](https://github.com/datalopes1/telco_churn/blob/main/doc/img/plot1.png?raw=true)

Uma taxa de 26,54% de churn é bastante alta, é necessário buscar compreender os fatores que levam a isso.

### Target x Features

![](https://github.com/datalopes1/telco_churn/blob/main/doc/img/plot2.png?raw=true)

Clientes com maior tempo de relacionamento tendem a ficar mais tempo na Telco, é necessário investigar a qualidade do atendimento a novos contratantes.

![](https://github.com/datalopes1/telco_churn/blob/main/doc/img/plot3.png?raw=true)
![](https://github.com/datalopes1/telco_churn/blob/main/doc/img/plot4.png?raw=true)

Clientes com maior tempo de contrato, tem tendência a permanecer e portanto geral maior receita total, a tendência de Churn em clientes com maior média de taxa de serviço mostra também necessidade de melhora na prestação de serviço. Vamos observar agora em relação ao tipo de contrato.

![](https://github.com/datalopes1/telco_churn/blob/main/doc/img/plot5.png?raw=true)

Contratos de duração mais longa tem uma menor proporção de Churn, existe um volume alto em contratos de renovação mensal. Uma sugestão seria a disponibilização de contratos semestrais.

![](https://github.com/datalopes1/telco_churn/blob/main/doc/img/plot6.png?raw=true)

Clientes que optam por meios automaticos de pagamento tem menor tendência de deixar os serviços da Telco, esses meios devem ser incentivados. Métodos de pagamento com chekc-up, especialmente o Electronic check tem alta proporção de Churn.

![](https://github.com/datalopes1/telco_churn/blob/main/doc/img/plot7.png?raw=true)


A ausência de suporte técnico é um fator relevante para o Churn também.

# Modelo de Machine Learning

Entre os três modelos testados (RandomForestClassifier, LogisticRegression e XGBClassifier) o que apresentou melhor desempenho considerando as métricas de Acurácia e ROC-AUC e capacidade generalização (diferença menor que 0.05 entre resultados de treino e teste), com os seguintes resultados:

|Métrica|Resultado|
|---|---|
|Accuracy Train|0.8059|
|Accuracy Test|0.8055|
|ROC-AUC Train|0.8492|
|ROC-AUC Test|0.8418|

![](https://github.com/datalopes1/telco_churn/blob/main/doc/img/plot8.png?raw=true)