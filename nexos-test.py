import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set plotting style and color palette
plt.style.use("seaborn-v0_8")
nexos_palette = ["#6a0dad", "#ff69b4", "#9370db", "#ff1493"]
sns.set_palette(nexos_palette)

# Show all columns when displaying DataFrames
pd.set_option("display.max_columns", None)

# Update matplotlib rcParams for consistent figure style
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.facecolor": "white",
    "axes.titlepad": 25,
    "axes.labelpad": 15,
    "figure.autolayout": True,
})

# Load dataset
df = pd.read_csv("data/nexosCsv.csv")

# Rename columns to English-friendly names
column_mapping = {
    "Carimbo de data/hora": "timestamp",
    "Os requisitos selecionados pelo NEXOS são importantes para o projeto?": "requirements_importance",
    "Qual seu nível de experiência com gestão de requisitos?": "experience_level",
    "Você selecionaria requisitos similares aos recomendados pelo NEXOS?": "would_select_similar",
    "Você faria modificações nas releases/sprints feitas pelo NEXOS?": "would_modify",
    "O NEXOS conseguiu auxiliar com planning da sprint/release?": "helped_planning",
    "Você recomendaria o uso do NEXOS para outras equipes?": "would_recommend",
}
df = df.rename(columns=column_mapping)

# Translate categorical responses from Portuguese to English
response_translations = {
    'Sim': 'Yes',
    'Não': 'No',
    'Sim, mas poucas modificações': 'Yes, but few changes',
    'Sim, faria muitas modificações': 'Yes, would make many changes',
    'Nenhuma experiência': 'No experience',
    'Pouca experiência': 'Little experience',
    'Média experiência': 'Some experience',
    'Muita experiência': 'Much experience'
}

# List of categorical columns to translate
categorical_cols = [
    "experience_level",
    "would_select_similar",
    "would_modify",
    "helped_planning",
    "would_recommend"
]

# Apply translations to categorical columns
for col in categorical_cols:
    df[col] = df[col].map(response_translations).fillna(df[col])

# Convert 'requirements_importance' column to numeric, coercing errors to NaN
df["requirements_importance"] = pd.to_numeric(df["requirements_importance"], errors="coerce")

# Categorize experience into two groups: Less experienced and More experienced
conditions = [
    (df['experience_level'] == 'No experience') | (df['experience_level'] == 'Little experience'),
    (df['experience_level'] == 'Some experience') | (df['experience_level'] == 'Much experience')
]
choices = ['Less experienced', 'More experienced']
df['experience_group'] = np.select(conditions, choices, default='Unknown')

# Print experience group distribution
print("\n=== Experience Group Distribution ===")
print(df['experience_group'].value_counts())

# Function to save plots in a folder named 'graph'
def save_plot(fig, filename):
    os.makedirs("graph", exist_ok=True)
    fig.savefig(f"graph/{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# 1. Descriptive statistics for requirements importance
print("\n=== Descriptive Analysis - Requirements Importance ===")
desc_stats = df["requirements_importance"].describe()
print(desc_stats)

# Plot distribution of requirements importance
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=df, x="requirements_importance", bins=5, kde=True, ax=ax)
ax.axvline(desc_stats["mean"], color="r", linestyle="--", label=f'Mean: {desc_stats["mean"]:.2f}')
ax.axvline(3, color="g", linestyle=":", label="Neutral value (3)")
ax.set_title("Distribution of Importance of Requirements Selected by NEXOS")
ax.set_xlabel("Importance Level (1-5)")
ax.set_ylabel("Count")
ax.legend()
save_plot(fig, "requirements_importance_distribution")

# Boxplot of requirements importance by experience group
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x="experience_group", y="requirements_importance", ax=ax)
ax.set_title("Requirements Importance by Experience Level")
ax.set_xlabel("Experience Group")
ax.set_ylabel("Importance Level (1-5)")
save_plot(fig, "importance_by_experience")

# Analysis of categorical responses
for col in categorical_cols[1:]:
    print(f"\n=== Descriptive Analysis - {col} ===")

    # Print percentage distribution of responses
    count_table = df[col].value_counts(normalize=True) * 100
    print("Response distribution (%):")
    print(count_table)

    # Bar plot of response counts
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x=col, ax=ax, order=df[col].value_counts().index)
    ax.set_title(f"Response Distribution - {col}")
    ax.set_xlabel("Response")
    ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height()}\n({p.get_height()/len(df)*100:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )
    save_plot(fig, f"{col}_distribution")

    # Bar plot of response counts grouped by experience group
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x=col, hue="experience_group", ax=ax, order=df[col].value_counts().index)
    ax.set_title(f"Response Distribution - {col} by Experience")
    ax.set_xlabel("Response")
    ax.set_ylabel("Count")
    ax.legend(title="Experience Group")
    save_plot(fig, f"{col}_by_experience")

# 3. Correlation analysis between numeric variables
print("\n=== Correlation Between Variables ===")
numeric_cols = ["requirements_importance"]
if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr()
    print(corr_matrix)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix Between Numeric Variables")
    save_plot(fig, "correlation_matrix")

# Map categorical answers to numeric values for ordinal analysis
ordinal_mapping = {
    'No': 1,
    'Yes, would make many changes': 1,
    'Yes, but few changes': 2,
    'Yes': 3,
}

# Copy DataFrame for numeric mapping
df_numeric = df.copy()

# Questions to analyze numerically (excluding experience)
numeric_questions = [
    "requirements_importance",
    "would_select_similar",
    "would_modify",
    "helped_planning",
    "would_recommend",
]

# Apply ordinal mapping to categorical responses
for col in numeric_questions:
    if col != "requirements_importance":
        df_numeric[col] = df_numeric[col].map(ordinal_mapping)

# Calculate standard deviation for each question
std_devs = df_numeric[numeric_questions].std()
print("\n=== Standard Deviation of Questions (excluding experience) ===")
print(std_devs)

# Line plot of standard deviation per question
fig, ax = plt.subplots(figsize=(10, 6))
std_devs.plot(kind="line", marker="o", ax=ax)
ax.set_title("Variation in Responses (Standard Deviation by Question)")
ax.set_ylabel("Standard Deviation")
ax.set_xlabel("Question")
ax.set_xticks(range(len(std_devs)))
ax.set_xticklabels([
    "Requirements Importance",
    "Would Select Similar Requirements",
    "Would Make Modifications",
    "Helped in Planning",
    "Would Recommend NEXOS"
], rotation=45, ha='right')
ax.grid(True)
save_plot(fig, "std_dev_per_question")
