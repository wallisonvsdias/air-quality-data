"""
Módulo de carregamento de dados para o projeto de Qualidade do Ar Urbano da Grécia.
Disciplina: DCT1401 - Inteligência Artificial (UFRN)
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd


def get_project_root() -> Path:
    """Retorna o caminho raiz do repositório."""
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Retorna o caminho base do diretório de dados."""
    return get_project_root() / "data"


def load_raw_data(
    file_name: str = "greek_urban_air_quality_2020_2024.csv",
    parse_dates: bool = True,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Carrega o dataset bruto original da pasta data/raw.

    Args:
        file_name: Nome do arquivo CSV original.
        parse_dates: Se True, faz o parsing da coluna 'timestamp'.
        sample_frac: Fração amostral opcional (ex: 0.1 para 10%) para testes rápidos.
        random_state: Semente para reprodutibilidade ao amostrar.

    Returns:
        pd.DataFrame com os dados brutos.
    """
    raw_path = get_data_dir() / "raw" / file_name
    if not raw_path.exists():
        raise FileNotFoundError(f"Arquivo de dados brutos não encontrado em: {raw_path}")

    date_cols = ["timestamp"] if parse_dates else False
    df = pd.read_csv(raw_path, parse_dates=date_cols)

    if sample_frac is not None and 0.0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

    return df


def load_clean_data(file_name: str = "dataset_clean.csv") -> pd.DataFrame:
    """Carrega o dataset limpo da pasta data/processed."""
    clean_path = get_data_dir() / "processed" / file_name
    if not clean_path.exists():
        raise FileNotFoundError(f"Arquivo limpo não encontrado em: {clean_path}")
    return pd.read_csv(clean_path, parse_dates=["timestamp"])


def load_regression_data(file_name: str = "dataset_regression.csv") -> pd.DataFrame:
    """Carrega o dataset preparado para regressão da pasta data/processed."""
    reg_path = get_data_dir() / "processed" / file_name
    if not reg_path.exists():
        raise FileNotFoundError(f"Dataset de regressão não encontrado em: {reg_path}")
    return pd.read_csv(reg_path)


def load_classification_data(file_name: str = "dataset_classification.csv") -> pd.DataFrame:
    """Carrega o dataset preparado para classificação da pasta data/processed."""
    clf_path = get_data_dir() / "processed" / file_name
    if not clf_path.exists():
        raise FileNotFoundError(f"Dataset de classificação não encontrado em: {clf_path}")
    return pd.read_csv(clf_path)
