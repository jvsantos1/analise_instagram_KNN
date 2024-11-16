
# Análise de Perfis do Instagram Usando k-Nearest Neighbors (kNN)

## Descrição do Projeto

Este projeto aplica o algoritmo **k-Nearest Neighbors (kNN)** para prever a pontuação de influência de perfis do Instagram com base em variáveis como número de seguidores, taxa de engajamento e curtidas médias. O modelo implementa validação cruzada, otimização de hiperparâmetros e visualizações gráficas para análise de desempenho. O relatório técnico em PDF também é fornecido, documentando o processo completo.

## Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/jvsantos1/analise_instagram_knn.git
   cd analise_instagram_regressaolinear
   ```

2. Crie um ambiente virtual:

   ```bash
   python -m venv venv
   ```

3. Ative o ambiente virtual:

   - No Windows:

     ```bash
     venv\Scripts\activate
     ```

   - No macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

4. Instale as dependências do projeto:

   ```bash
   pip install -r requirements.txt
   ```

## Como Executar

Para rodar o projeto, execute o script principal:

```bash
python main.py
```

O código irá:

- Carregar os dados do Instagram.
- Realizar a normalização das variáveis necessárias.
- Configurar e treinar o modelo kNN.
- Otimizar os hiperparâmetros com validação cruzada.
- Exibir gráficos de visualização e métricas de desempenho.

### Exemplo de Execução

- O modelo carregará automaticamente os dados de entrada e iniciará a execução.
- Gráficos como dispersão entre seguidores e curtidas médias, além de barras comparando o rank e a pontuação de influência, serão gerados.
- As métricas de desempenho (MAE, MSE e RMSE) serão exibidas no terminal.

## Estrutura dos Arquivos

- `/main.py`: Código principal para carregar, treinar e avaliar o modelo kNN.
- `/requirements.txt`: Lista de dependências necessárias para o projeto.
- `/data/`: Pasta onde os dados de entrada estão localizados.
- `/results/`: Pasta para salvar gráficos gerados e outros resultados.
- `/docs/`: Relatório técnico em PDF documentando todo o processo.

## Tecnologias Utilizadas

- **Python**: Linguagem de programação principal.
- **Scikit-Learn**: Para o modelo de regressão e validação cruzada.
- **Matplotlib**: Para a visualização de gráficos.
- **NumPy**: Para manipulação de arrays e dados numéricos.
- **Pandas**: Para manipulação de dados.

## Autores e Colaboradores

- **Arthur Lago Martins**
- **João Victor Oliveira Santos**