import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

plt.style.use("seaborn-v0_8")
nexos_palette = [
    "#6a0dad",
    "#ff69b4",
    "#9370db",
    "#ff1493",
]
sns.set_palette(nexos_palette)
pd.set_option("display.max_columns", None)

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.facecolor": "white",
        "axes.titlepad": 25,
        "axes.labelpad": 15,
        "figure.autolayout": True,
    }
)

df = pd.read_csv("data/nexosCsv.csv")

column_mapping = {
    "Carimbo de data/hora": "timestamp",
    "Os requisitos selecionados pelo NEXOS são importantes para o projeto?": "importancia_requisitos",
    "Qual seu nível de experiência com gestão de requisitos?": "experiencia",
    "Você selecionaria requisitos similares aos recomendados pelo NEXOS?": "selecionaria_similares",
    "Você faria modificações nas releases/sprints feitas pelo NEXOS?": "modificacoes",
    "O NEXOS conseguiu auxiliar com planning da sprint/release?": "auxiliou_planning",
    "Você recomendaria o uso do NEXOS para outras equipes?": "recomendaria",
}

df = df.rename(columns=column_mapping)

# Converter respostas numéricas para inteiros
df["importancia_requisitos"] = pd.to_numeric(
    df["importancia_requisitos"], errors="coerce"
)

# Classificar experiência em dois grupos
conditions = [
    df["experiencia"].str.contains("Nenhuma experiência|Pouca experiência"),
    df["experiencia"].str.contains("Média experiência|Muita experiência"),
]
choices = ["Menos experiente", "Mais experiente"]
df["grupo_experiencia"] = np.select(conditions, choices, default="Menos experiente")


# Função para salvar gráficos
def save_plot(fig, filename):
    fig.savefig(f"graph/{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# 1. Análise descritiva da importância dos requisitos
print("\n=== Análise Descritiva - Importância dos Requisitos ===")
desc_stats = df["importancia_requisitos"].describe()
print(desc_stats)

# Gráfico de distribuição
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=df, x="importancia_requisitos", bins=5, kde=True, ax=ax)
ax.axvline(
    desc_stats["mean"],
    color="r",
    linestyle="--",
    label=f'Média: {desc_stats["mean"]:.2f}',
)
ax.axvline(3, color="g", linestyle=":", label="Valor neutro (3)")
ax.set_title("Distribuição da Importância dos Requisitos Selecionados pelo NEXOS")
ax.set_xlabel("Nível de Importância (1-5)")
ax.set_ylabel("Contagem")
ax.legend()
save_plot(fig, "importancia_requisitos_distribuicao")

# Boxplot por grupo de experiência
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x="grupo_experiencia", y="importancia_requisitos", ax=ax)
ax.set_title("Importância dos Requisitos por Nível de Experiência")
ax.set_xlabel("Grupo de Experiência")
ax.set_ylabel("Nível de Importância (1-5)")
save_plot(fig, "importancia_por_experiencia")

# 2. Análise das respostas categóricas
categorical_cols = [
    "selecionaria_similares",
    "modificacoes",
    "auxiliou_planning",
    "recomendaria",
]

for col in categorical_cols:
    print(f"\n=== Análise Descritiva - {col} ===")

    # Estatísticas descritivas
    count_table = df[col].value_counts(normalize=True) * 100
    print("Distribuição de respostas (%):")
    print(count_table)

    # Gráfico de barras geral
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x=col, ax=ax, order=df[col].value_counts().index)
    ax.set_title(f"Distribuição de Respostas")
    ax.set_xlabel("Resposta")
    ax.set_ylabel("Contagem")
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height()}\n({p.get_height()/len(df)*100:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )
    save_plot(fig, f"{col}_distribuicao")

    # Gráfico por grupo de experiência
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=df,
        x=col,
        hue="grupo_experiencia",
        ax=ax,
        order=df[col].value_counts().index,
    )
    ax.set_title(f"Distribuição de Respostas - {col} por Experiência")
    ax.set_xlabel("Resposta")
    ax.set_ylabel("Contagem")
    ax.legend(title="Grupo de Experiência")
    save_plot(fig, f"{col}_por_experiencia")

# 3. Análise de correlação entre variáveis numéricas e categóricas
print("\n=== Correlação entre Variáveis ===")

# Matriz de correlação (para variáveis numéricas)
numeric_cols = ["importancia_requisitos"]
if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr()
    print(corr_matrix)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matriz de Correlação entre Variáveis Numéricas")
