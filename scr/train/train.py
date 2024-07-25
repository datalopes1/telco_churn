#%% Importação das bibliotecas

# Manipulação dos dados
import pandas as pd
import numpy as np

# Visualizações
import matplotlib.pyplot as plt
import seaborn as sns

# Pré-processamento
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
# %%
df = pd.read_csv("../../data/raw/telcocustomerchurn.csv")
# %% Ajuste de TotalCharges
df.TotalCharges = df.TotalCharges.replace(' ', np.nan)
df.TotalCharges = df.TotalCharges.astype(float)
# %% Seleção das features
features = df.drop(columns = ['customerID', 'Churn'], axis = 1).columns.to_list()
target = 'Churn'

cat_features = df[features].select_dtypes(include = 'object').columns.to_list()
num_features = df[features].select_dtypes(include = 'number').columns.to_list()
# %% Encoding da variável target
le = LabelEncoder()
df[target] = le.fit_transform(df[target])
# %% Divisão em treino e teste
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42, stratify = y)
# %% Pré-processamento
num_transformer = Pipeline([
    ('imput', MeanMedianImputer(imputation_method='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imput', CategoricalImputer(imputation_method='frequent')),
    ('ohe', OneHotEncoder())
])

preprocessor = ColumnTransformer(
    transformers = [
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)
# %% Treinamento do modelo
model = LogisticRegression(max_iter=1000)

lr = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

lr.fit(X_train, y_train)
# %% Previsões
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

y_proba_train = lr.predict_proba(X_train)
y_proba_test = lr.predict_proba(X_test)
# %% Métricas
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)

auc_train = roc_auc_score(y_train, y_proba_train[:,1])
auc_test = roc_auc_score(y_test, y_proba_test[:,1])

print("Métricas do LogisticRegression")
print("=" * 45)
print("Acurácia")
print(f"Em treino: {acc_train}")
print(f"Em teste: {acc_test}")
print(f"Diferença Treino x Teste: {acc_train - acc_test}")
print("-" * 45)
print("ROC AUC")
print(f"Em treino: {auc_train}")
print(f"Em teste: {auc_test}")
print(f"Diferença Treino x Teste: {auc_train - auc_test}")

curve = roc_curve(y_test, y_proba_test[:,1])

fig, ax = plt.subplots(figsize = (12, 6))
plt.plot(curve[0], curve[1])
plt.plot([0, 1], [0, 1], '--')
ax.set_title('LogisticRegression ROC Curve', loc = 'left', fontsize = 18, pad = 12)
ax.set_xlabel('Falso positivo', fontsize = 8)
ax.set_ylabel('Verdadeiro positivo', fontsize = 8)
plt.show()
# %%
model_series = pd.Series({
    'model': lr,
    'features': features,
    'acc': acc_test
})

modelo_churn = model_series.to_pickle("../../models/model_churn.pkl")