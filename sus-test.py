import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/nexosCsv.csv')

sus_columns = df.columns[-10:]
sus_data = df[sus_columns]

def calculate_sus_score(row):
    score = 0
    for i, col in enumerate(sus_columns, start=1):
        value = row[col]
        if i % 2 == 1:
            score += (value - 1)
        else:
            score += (5 - value)
    return score * 2.5

# Calcular o score SUS para cada respondente
df['SUS_Score'] = sus_data.apply(calculate_sus_score, axis=1)

# Calcular estatísticas
average_sus = df['SUS_Score'].mean()
sus_scores = df['SUS_Score'].values

# Classificar os scores conforme os critérios de Brooke
def classify_sus(score):
    if score < 60:
        return "Inaceitável"
    elif 60 <= score < 70:
        return "Razoável"
    elif 70 <= score < 80:
        return "Bom"
    elif 80 <= score < 90:
        return "Excelente"
    else:
        return "Melhor Usabilidade Possível"

df['SUS_Class'] = df['SUS_Score'].apply(classify_sus)

# Exibir estatísticas
print("\nEstatísticas dos Scores SUS:")
print(f"Média: {average_sus:.1f}")
print(f"Mediana: {df['SUS_Score'].median():.1f}")
print(f"Mínimo: {df['SUS_Score'].min():.1f}")
print(f"Máximo: {df['SUS_Score'].max():.1f}")
print("\nClassificação dos Scores:")
print(df['SUS_Class'].value_counts())