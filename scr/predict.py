import os
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data_and_model(data_path, model_path):
    try:
        df = pd.read_csv(data_path)
        model_series = pd.read_pickle(model_path)
        logging.info("Dados e modelos carregados.")
        return df, model_series
    except Exception as e:
        logging.error(f"Erro: {e}")
        raise

def make_predictions(df, model_series):
    try:
        X = df[model_series['features']]
        y_proba = model_series['model'].predict_proba(X)[:,1]
        y_pred = (y_proba > 0.60).astype(int)
        df['churnPred'] = y_pred
        df['churnProb'] = y_proba
        logging.info("Predições salvas.")
        return df
    except Exception as e:
        logging.error(f"Erro: {e}")
        raise

def save_results(df, output_path):
    try:
        results = df[['customerID', 'Churn', 'churnPred', 'churnProb']].copy()
        results.to_excel(output_path, index=False)
        logging.info(f"Resultados salvos em {output_path}.")
    except Exception as e:
        logging.error(f"Erro: {e}")
        raise

def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data/processed/clean_dataset.csv"
    model_path = base_dir / "models/classifier.pkl"
    output_path = base_dir / "data/processed/model_predictions.xlsx"

    df, model_series = load_data_and_model(data_path, model_path)
    df = make_predictions(df, model_series)
    save_results(df, output_path)

if __name__ == "__main__":
    main()