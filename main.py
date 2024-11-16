import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'data/top_insta_influencers_data.csv'
data = pd.read_csv(file_path)

# Função para converter valores textuais de números em valores numéricos reais
def convert_to_numeric(value):
    if isinstance(value, str):
        value = value.lower().replace(',', '').strip()
        if 'k' in value:
            return float(value.replace('k', '')) * 1e3
        elif 'm' in value:
            return float(value.replace('m', '')) * 1e6
        elif 'b' in value:
            return float(value.replace('b', '')) * 1e9
        elif value.endswith('%'):
            return float(value.replace('%', '')) / 100  
        else:
            try:
                return float(value)
            except ValueError:
                return np.nan
    return value

# Aplicar conversões de texto para valores numéricos
columns_to_convert = ['posts', 'followers', 'avg_likes', '60_day_eng_rate', 'new_post_avg_like', 'total_likes']
for column in columns_to_convert:
    data[column] = data[column].apply(convert_to_numeric)

# Mapeamento de países para continentes
continent_map = {
    'United States': 20,  # América do Norte
    'Canada': 21,
    'Brazil': 1,          # América do Sul
    'Argentina': 2,
    'Spain': 40,          # Europa
    'France': 41,
    'Germany': 42,
    'India': 50,          # Ásia
    'China': 51,
    'Australia': 60,      # Oceania
    'South Africa': 70,   # África
}
data['continent'] = data['country'].map(continent_map)
data['continent'] = data['continent'].fillna(-1)  # Valores desconhecidos

# Seleção de variáveis
features = ['followers', 'avg_likes', '60_day_eng_rate', 'total_likes', 'continent']
target = 'influence_score'

# Remover valores desconhecidos (-1 em continent)
filtered_data = data[data['continent'] != -1]

# Divisão entre X e y
X = filtered_data[features]
y = filtered_data[target]

# Normalizar as variáveis
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Treinar modelo kNN inicial
knn = KNeighborsRegressor(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

# Fazer previsões
y_pred = knn.predict(X_test)

# Avaliação do modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")

# Otimização com GridSearchCV
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'metric': ['euclidean', 'manhattan']
}
grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Melhor modelo após otimização
best_knn = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Melhores Parâmetros:", best_params)

# Avaliação do modelo otimizado
y_pred_optimized = best_knn.predict(X_test)
mae_opt = mean_absolute_error(y_test, y_pred_optimized)
mse_opt = mean_squared_error(y_test, y_pred_optimized)
rmse_opt = mse_opt ** 0.5

print(f"MAE após otimização: {mae_opt:.2f}, MSE: {mse_opt:.2f}, RMSE: {rmse_opt:.2f}")

# Gráficos
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='followers', y='avg_likes', hue='continent', palette='viridis', alpha=0.7)
plt.title('Relação entre Seguidores e Curtidas Médias')
plt.xlabel('Seguidores')
plt.ylabel('Curtidas Médias')
plt.legend(title='Continente', bbox_to_anchor=(1, 1))
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=data.sort_values('rank'), x='rank', y='influence_score', palette='coolwarm')
plt.title('Relação entre Rank e Pontuação de Influência')
plt.xlabel('Rank')
plt.ylabel('Pontuação de Influência')
plt.xticks([])
plt.show()