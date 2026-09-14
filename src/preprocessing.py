"""
Módulo para Limpeza, Transformação e Pré-processamento de Dados.
Disciplina: DCT1401 - Inteligência Artificial (UFRN)
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler


def get_processed_data_dir() -> Path:
    """Retorna o diretório de dados processados."""
    proc_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    return proc_dir


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa a limpeza inicial e engenharia de atributos temporais:
    - Garante conversão de datas;
    - Extrai 'month', 'hour', 'day_of_week', 'is_weekend';
    - Remove possíveis duplicatas.
    """
    data = df.copy()

    # Conversão de data/hora
    if not np.issubdtype(data["timestamp"].dtype, np.datetime64):
        data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Extração de features temporais
    data["hour"] = data["timestamp"].dt.hour
    data["month"] = data["timestamp"].dt.month
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    # Ordenação cronológica e remoção de duplicatas se existirem
    data = data.sort_values(by=["timestamp", "station"]).reset_index(drop=True)
    data = data.drop_duplicates().reset_index(drop=True)

    return data


def apply_capping(
    df: pd.DataFrame,
    columns: List[str],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Aplica Winsorização (capping) em quantis para mitigar o impacto de caudas extremas
    sem descartar observações ambientais reais valiosas.
    """
    data = df.copy()
    for col in columns:
        if col in data.columns:
            low_val = data[col].quantile(lower_quantile)
            high_val = data[col].quantile(upper_quantile)
            data[col] = np.clip(data[col], low_val, high_val)
    return data


def prepare_regression_dataset(
    df: pd.DataFrame,
    target_col: str = "AQI",
    apply_scaling: bool = True,
    scale_type: str = "standard",
) -> pd.DataFrame:
    """
    Prepara o dataset específico para a tarefa de Regressão.
    
    Target: 'AQI' (contínua)
    Features preditoras:
    - Meteorológicas: Temperature_C, Humidity_pct, Wind_Speed_ms, Wind_Direction_deg,
                      Pressure_hPa, Rainfall_mm, UV_Index
    - Urbanas/Antropogênicas: Traffic_Density_Index, population, industrial_index, Heating_Season
    - Eventos Críticos: Saharan_Dust_Event, Wildfire_Smoke_Event
    - Espaciais/Temporais: latitude, longitude, hour, month, is_weekend
    - Categóricas OHE: city (dummies)
    
    Nota: Excluímos 'AQI_Category', 'Outdoor_Activity_Recommendation' e métricas de risco
    derivadas para evitar vazamento de dados (data leakage). Os poluentes individuais
    (PM2.5, PM10) possuem correlação quase perfeita (>0.95) pois compõem diretamente a fórmula
    do AQI; prever o AQI a partir de condições meteorológicas e urbanas representa a modelagem
    preditiva real na ausência de sensores químicos densos.
    """
    clean_df = clean_dataset(df)

    feature_cols = [
        "latitude",
        "longitude",
        "population",
        "industrial_index",
        "Temperature_C",
        "Humidity_pct",
        "Wind_Speed_ms",
        "Wind_Direction_deg",
        "Pressure_hPa",
        "Rainfall_mm",
        "UV_Index",
        "Traffic_Density_Index",
        "Heating_Season",
        "Saharan_Dust_Event",
        "Wildfire_Smoke_Event",
        "hour",
        "month",
        "is_weekend",
    ]

    # Cria dummies para 'city'
    city_dummies = pd.get_dummies(clean_df["city"], prefix="city", drop_first=True, dtype=int)

    # Base de features
    X = clean_df[feature_cols].copy()
    y = clean_df[target_col].copy()

    # Normalização / Padronização
    if apply_scaling:
        num_to_scale = [
            "population",
            "industrial_index",
            "Temperature_C",
            "Humidity_pct",
            "Wind_Speed_ms",
            "Wind_Direction_deg",
            "Pressure_hPa",
            "Rainfall_mm",
            "UV_Index",
            "Traffic_Density_Index",
        ]
        scaler = StandardScaler() if scale_type == "standard" else RobustScaler()
        X[num_to_scale] = scaler.fit_transform(X[num_to_scale])

    reg_df = pd.concat([X, city_dummies, y], axis=1)
    return reg_df


def prepare_classification_dataset(
    df: pd.DataFrame,
    target_col: str = "Outdoor_Activity_Recommendation",
    apply_scaling: bool = True,
    scale_type: str = "standard",
) -> pd.DataFrame:
    """
    Prepara o dataset específico para a tarefa de Classificação.
    
    Target: 'Outdoor_Activity_Recommendation' (4 classes ordinais acionáveis)
    Features preditoras:
    - Poluentes atmosféricos principais: PM2_5_ugm3, PM10_ugm3, NO2_ugm3, O3_ugm3, SO2_ugm3, CO_mgm3
    - Meteorológicas: Temperature_C, Humidity_pct, Wind_Speed_ms, UV_Index
    - Antropogênicas/Eventos: Traffic_Density_Index, Saharan_Dust_Event, Wildfire_Smoke_Event
    - Categóricas OHE: city (dummies)
    
    Nota: Removemos 'AQI', 'AQI_Category' e índices de risco infantil/idosos por serem
    mapeamentos quase determinísticos da recomendação, focando nas medições observáveis.
    """
    clean_df = clean_dataset(df)

    feature_cols = [
        "PM2_5_ugm3",
        "PM10_ugm3",
        "NO2_ugm3",
        "O3_ugm3",
        "SO2_ugm3",
        "CO_mgm3",
        "Temperature_C",
        "Humidity_pct",
        "Wind_Speed_ms",
        "UV_Index",
        "Traffic_Density_Index",
        "Saharan_Dust_Event",
        "Wildfire_Smoke_Event",
    ]

    city_dummies = pd.get_dummies(clean_df["city"], prefix="city", drop_first=True, dtype=int)
    X = clean_df[feature_cols].copy()

    # Mapeamento ordinal para o target categórico
    target_mapping = {
        "All activities safe": 0,
        "Sensitive groups limit outdoor activity": 1,
        "Reduce prolonged outdoor activity": 2,
        "Avoid outdoor activity": 3,
    }
    y = clean_df[target_col].map(target_mapping)

    if apply_scaling:
        num_to_scale = [
            "PM2_5_ugm3",
            "PM10_ugm3",
            "NO2_ugm3",
            "O3_ugm3",
            "SO2_ugm3",
            "CO_mgm3",
            "Temperature_C",
            "Humidity_pct",
            "Wind_Speed_ms",
            "UV_Index",
            "Traffic_Density_Index",
        ]
        scaler = StandardScaler() if scale_type == "standard" else RobustScaler()
        X[num_to_scale] = scaler.fit_transform(X[num_to_scale])

    clf_df = pd.concat([X, city_dummies, y.rename("target_class")], axis=1)
    return clf_df


def save_all_processed_datasets(raw_df: pd.DataFrame) -> Tuple[Path, Path, Path]:
    """
    Gera e salva em disco os 3 datasets processados:
    1. dataset_clean.csv
    2. dataset_regression.csv
    3. dataset_classification.csv
    """
    proc_dir = get_processed_data_dir()

    clean_df = clean_dataset(raw_df)
    clean_path = proc_dir / "dataset_clean.csv"
    clean_df.to_csv(clean_path, index=False)

    reg_df = prepare_regression_dataset(raw_df)
    reg_path = proc_dir / "dataset_regression.csv"
    reg_df.to_csv(reg_path, index=False)

    clf_df = prepare_classification_dataset(raw_df)
    clf_path = proc_dir / "dataset_classification.csv"
    clf_df.to_csv(clf_path, index=False)

    return clean_path, reg_path, clf_path
