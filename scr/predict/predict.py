# %% 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
# %%
data = pd.read_csv("../../data/raw/telcocustomerchurn.csv")
model = pd.read_pickle("../../models/model_churn.pkl")
# %%
data.TotalCharges = data.TotalCharges.replace(' ', np.nan)
data.TotalCharges = data.TotalCharges.astype(float)

le = LabelEncoder()
data['Churn'] = le.fit_transform(data['Churn'])
# %%
X = data[model['features']]
y = data['Churn']
# %%
y_pred = model['model'].predict(X)
y_proba = model['model'].predict_proba(X)
# %%
data['predChurn'] = y_pred
data['probChurn'] = y_proba[:,1]
data.to_excel("../../data/processed/churn_pred.xlsx", index = False)
# %%
acc = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_proba[:,1])

metricas = pd.Series({
    'Acurácia': acc,
    'ROC AUC': auc
})

print(metricas)