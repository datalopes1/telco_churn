# %% Importações
# Data manipulation
import pandas as pd
import numpy as np

# EDA
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, f1_score, make_scorer, precision_recall_curve, confusion_matrix, matthews_corrcoef

# Pre-processing
import optuna
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from sklearn.utils.class_weight import compute_class_weight
from category_encoders import TargetEncoder
# %% Carregamento dos dados
df = pd.read_csv("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = df['TotalCharges'].astype(float)

features = df.drop(columns = ['customerID', 'Churn'], axis = 1).columns.to_list()
target = 'Churn'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# %% Pipeline de pré-processamento
cat_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

num_features = ['tenure',  'MonthlyCharges', 'TotalCharges']

cat_transformer = Pipeline([
    ('cat_imput', CategoricalImputer(imputation_method = 'frequent')),
    ('cat_encoding', TargetEncoder())
])

num_transformer = Pipeline([
    ('num_imput', MeanMedianImputer(imputation_method='median'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_transformer, cat_features),
        ('num', num_transformer, num_features)
    ]
)
# %% Treinamento do modelo
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)

weights_dict = {class_label: weight for class_label, weight in zip(np.unique(y_train), class_weights)}

best_params = {'learning_rate': 0.008972199433787508, 
               'depth': 8, 
               'subsample': 0.06111651350523167, 
               'colsample_bylevel': 0.742008127122645, 
               'min_data_in_leaf': 98} 

clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', CatBoostClassifier(**best_params, 
                           verbose = 0, 
                           iterations = 1000, 
                           class_weights = weights_dict, 
                           random_state = 42))
])

clf.fit(X_train, y_train)
# %%
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

model_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'ROC AUC': roc_auc_score(y_test, y_proba),
    'MCC': matthews_corrcoef(y_test, y_pred)
}

print(model_metrics)
# %% Save model
model_series = pd.Series({
    'model': clf,
    'features': features,
    'metrics': model_metrics
})

model_series.to_pickle("../models/classifier.pkl")
# %% Dataset com tratado
df.to_csv("../data/processed/clean_dataset.csv", index = False)