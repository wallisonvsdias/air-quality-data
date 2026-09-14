"""
Módulo para Análise Exploratória de Dados (EDA).
Disciplina: DCT1401 - Inteligência Artificial (UFRN)
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, object]:
    """
    Retorna métricas gerais sobre a estrutura do dataset.
    """
    missing_series = df.isnull().sum()
    missing_pct = (missing_series / len(df)) * 100

    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "numeric_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_cols": df.select_dtypes(include=["object", "category", "string"]).columns.tolist(),
        "datetime_cols": df.select_dtypes(include=["datetime64"]).columns.tolist(),
        "missing_counts": missing_series[missing_series > 0].to_dict(),
        "missing_pct": missing_pct[missing_pct > 0].to_dict(),
        "total_missing": int(missing_series.sum()),
    }
    return summary


def get_numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula resumo estatístico completo das variáveis numéricas:
    média, mediana, desvio padrão, quartis, mínimo e máximo.
    """
    num_df = df.select_dtypes(include=[np.number])
    stats = pd.DataFrame({
        "Contagem": num_df.count(),
        "Média": num_df.mean(),
        "Desvio Padrão": num_df.std(),
        "Mínimo": num_df.min(),
        "Q1 (25%)": num_df.quantile(0.25),
        "Mediana (50%)": num_df.median(),
        "Q3 (75%)": num_df.quantile(0.75),
        "Máximo": num_df.max(),
        "IQR": num_df.quantile(0.75) - num_df.quantile(0.25),
    })
    return stats


def get_categorical_summary(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Retorna tabelas de frequências absolutas e relativas para colunas categóricas.
    """
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    summaries = {}
    for col in cat_cols:
        freq = df[col].value_counts(dropna=False)
        pct = df[col].value_counts(dropna=False, normalize=True) * 100
        summaries[col] = pd.DataFrame({"Frequência": freq, "Percentual (%)": pct})
    return summaries


def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    Detecta outliers usando o método do Intervalo Interquartil (IQR).

    Args:
        df: DataFrame com os dados.
        factor: Multiplicador do IQR (padrão 1.5 de Tukey).

    Returns:
        DataFrame com colunas: Q1, Q3, IQR, Limite Inferior, Limite Superior,
        Qtd Outliers e % Outliers.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    records = []

    for col in num_cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        outliers = series[(series < lower_bound) | (series > upper_bound)]
        count = len(outliers)
        pct = (count / len(df)) * 100

        records.append({
            "Variável": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Limite Inferior": lower_bound,
            "Limite Superior": upper_bound,
            "Qtd Outliers": count,
            "Percentual (%)": pct,
        })

    return pd.DataFrame(records).sort_values(by="Qtd Outliers", ascending=False).reset_index(drop=True)


def compute_correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Calcula a matriz de correlação entre as variáveis numéricas."""
    num_df = df.select_dtypes(include=[np.number])
    return num_df.corr(method=method)


def find_high_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
    """
    Identifica pares de variáveis com coeficiente de correlação absoluta acima do limiar,
    indicando potenciais riscos de multicolinearidade.
    """
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1, col2 = cols[i], cols[j]
            val = corr_matrix.loc[col1, col2]
            if abs(val) >= threshold:
                pairs.append((col1, col2, float(val)))
    return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)


def compute_target_correlations(df: pd.DataFrame, target_col: str) -> pd.Series:
    """Calcula a correlação de todas as variáveis numéricas com a variável alvo."""
    num_df = df.select_dtypes(include=[np.number])
    if target_col not in num_df.columns:
        raise ValueError(f"Target '{target_col}' não está presente nas colunas numéricas.")
    return num_df.corr()[target_col].sort_values(ascending=False)
