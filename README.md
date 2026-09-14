# Análise Exploratória e Preparação de Dados para Machine Learning

**Aluno:** Wallison Valdemiro Silvino Dias  
**Matrícula:** 20250023771  
**Disciplina:** DCT1401 - Inteligência Artificial (Período 2026.2)  
**Professor:** Thommas K. S. Flores  
**Instituição:** Universidade Federal do Rio Grande do Norte (UFRN) — CERES / DCT  
**Dataset Oficial:** [Greek Urban Air Quality and Health Impact Dataset (2020-2024)](https://www.kaggle.com/datasets/uniquetech/greek-urban-air-quality-and-health-impact-dataset)

---

## Descrição do Projeto

Este repositório contém a **Análise Exploratória de Dados (EDA)**, **Higienização e Pré-processamento** e a **Formulação de Percepções Analíticas** para a disciplina de Inteligência Artificial. O objetivo central é preparar um conjunto de dados atmosféricos de 12 cidades gregas (453.096 instâncias temporais) para dois paradigmas de aprendizado supervisionado:
1. **Regressão:** Previsão contínua do Índice de Qualidade do Ar (`AQI`).
2. **Classificação:** Previsão da recomendação de restrição de atividades ao ar livre (`Outdoor_Activity_Recommendation`).

---

## Estrutura do Repositório

```text
.
├── README.md                      # Descrição do projeto, instruções e resultados
├── requirements.txt               # Dependências Python do projeto
├── .gitattributes                 # Rastreamento de arquivos CSV grandes via Git LFS
├── .gitignore                     # Arquivos ignorados pelo controle de versão
├── data/
│   ├── raw/
│   │   └── greek_urban_air_quality_2020_2024.csv # Dataset original bruto (453k linhas)
│   └── processed/
│       ├── dataset_clean.csv                     # Dataset limpo com features temporais
│       ├── dataset_regression.csv                # Dataset preparado para regressão (AQI)
│       └── dataset_classification.csv            # Dataset preparado para classificação
├── notebooks/
│   ├── 01_eda.ipynb                              # Análise exploratória e diagnósticos
│   ├── 02_preprocessing.ipynb                    # Limpeza, tratamentos, OHE e escalas
│   ├── 03_regression_analysis.ipynb              # Análise preditiva para Regressão (AQI)
│   └── 04_classification_analysis.ipynb          # Análise preditiva para Classificação
├── src/
│   ├── __init__.py                               # Inicializador do pacote modular
│   ├── data_loader.py                            # Funções para carregamento de dados
│   ├── eda.py                                    # Funções analíticas e detecção de outliers
│   ├── preprocessing.py                          # Funções de higienização, codificação e scaling
│   └── visualization.py                          # Funções padronizadas para geração de gráficos
├── reports/
│   └── figures/                                  # Gráficos de alta resolução (300 DPI)
└── docs/
    └── relatorio_final.pdf                       # Relatório técnico completo da atividade
```

---

## Como Executar

### 1. Clonar o Repositório e Configurar o Git LFS
```bash
git clone https://github.com/wallisonvsdias/air-quality-data.git
cd air-quality-data
git lfs pull
```

### 2. Instalar as Dependências
Recomenda-se a utilização de um ambiente virtual Python (versão 3.10 ou superior):
```bash
python -m venv .venv
# No Windows:
.venv\Scripts\activate
# No Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Execução dos Notebooks
Os notebooks devem ser executados estritamente na ordem sequencial:
1. `notebooks/01_eda.ipynb` — Conduz a inspeção inicial, cálculo estatístico, diagnóstico de ausentes, outliers e correlações.
2. `notebooks/02_preprocessing.ipynb` — Executa a limpeza, tratamento de outliers, codificação (OHE) e padronização, gerando os arquivos em `data/processed/`.
3. `notebooks/03_regression_analysis.ipynb` — Analisa o problema de regressão com target `AQI`, avaliando correlações, multicolinearidade e modelos lineares/árvores.
4. `notebooks/04_classification_analysis.ipynb` — Analisa o problema de classificação com target `Outdoor_Activity_Recommendation`, testes de hipóteses (ANOVA, $\chi^2$) e modelos baselines.

---

## Principais Resultados e Percepções

### 1. Problema de Regressão
- **Variável Alvo:** `AQI` (Air Quality Index) — contínua, com média $93.42$ e intervalo de $12.10$ a $729.10$.
- **Justificativa:** Indicador ambiental internacional padronizado; prever o AQI com base em dados meteorológicos e fluxo de tráfego urbano permite a inferência e alerta em tempo real sem a dependência de estações químicas dispendiosas em cada localidade.
- **Features Mais Relevantes:** `Traffic_Density_Index`, `Wildfire_Smoke_Event`, `Saharan_Dust_Event`, `industrial_index`, `Wind_Speed_ms` e `Temperature_C`.
- **Desafios e Algoritmos:** Relações não-lineares e assimetria por eventos extremos de queimadas. Algoritmos recomendados: **XGBoost / LightGBM** e **Random Forest Regressor**, com **Ridge** como linha de base regularizada.

### 2. Problema de Classificação
- **Variável Alvo:** `Outdoor_Activity_Recommendation` — categórica com 4 classes ordinais acionáveis em saúde pública.
- **Distribuição de Classes:** 
  - *Sensitive groups limit outdoor activity* ($39.6\%$)
  - *Reduce prolonged outdoor activity* ($23.4\%$)
  - *All activities safe* ($19.7\%$)
  - *Avoid outdoor activity* ($17.3\%$)
- **Justificativa:** Evita o desbalanceamento severo verificado em `AQI_Category` (onde a classe *Hazardous* representa apenas $0.83\%$). Traduz o risco ambiental em ação sanitária direta.
- **Features Mais Relevantes:** $PM_{2.5}$, $PM_{10}$, $NO_2$, $CO$ e eventos de queimadas (testes ANOVA com $p < 0.0001$).
- **Algoritmos Recomendados:** **Random Forest Classifier**, **Gradient Boosting (LightGBM)** e **Regressão Logística Multinomial / Ordinal**.

---

## Contato
- **Email:** wallison.dias.711@ufrn.edu.br
