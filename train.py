import sys

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import mlflow

# Esta função tenta registrar o experimento no MLflow
def tentar_registrar_experimento(p_test_size, p_include_names, accuracy, dataset, model):
    with mlflow.start_run():
        # Vamos registrar as métricas
        mlflow.log_metric("acuracia", accuracy)
        # E o dataset (deve ser um caminho para um arquivo)
        mlflow.log_artifact(dataset)
        # E o modelo treinado
        mlflow.sklearn.log_model(model, "modelo")

if __name__ == "__main__":
    nltk.download('stopwords')

    # temos dois parâmetros agora
    # p_test_size: percentual de casos de teste, entre 0 e 1. Default é 0.2
    p_test_size = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    # p_include_names: se nomes devem ser incluídos no treinamento ou não, apenas descrição. Valor deve ser sim ou não. Default é sim
    p_include_names = sys.argv[2] if len(sys.argv) > 2 else 'sim'

    print("Treinando classificador de modelos...")
    print(f"Tamanho de testes={p_test_size}")
    print(f"Incluir nomes={p_include_names}")

    dataset = 'produtos.csv'

    products_data = pd.read_csv(dataset, delimiter=';', encoding='utf-8')

    if p_include_names == 'sim':
        # concatenando as colunas nome e descricao
        products_data['informacao'] = products_data['nome'] + products_data['descricao']
    else:
        # apenas a descricao
        products_data['informacao'] = products_data['descricao']

    # excluindo linhas com valor de informacao ou categoria NaN
    products_data.dropna(subset=['informacao', 'categoria'], inplace=True)
    products_data.drop(columns=['nome', 'descricao'], inplace=True)

    stop_words=set(stopwords.words("portuguese"))
    # transforma a string em caixa baixa e remove stopwords
    products_data['sem_stopwords'] = products_data['informacao'].str.lower().apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    products_data['tokens'] = products_data['sem_stopwords'].apply(tokenizer.tokenize) # aplica o regex tokenizer
    products_data.drop(columns=['sem_stopwords','informacao'],inplace=True) # Exclui as colunas antigas

    products_data["strings"]= products_data["tokens"].str.join(" ") # reunindo cada elemento da lista
    products_data.head()


    X_train,X_test,y_train,y_test = train_test_split( # Separação dos dados para teste e treino
        products_data["strings"], 
        products_data["categoria"], 
        test_size = p_test_size, 
        random_state = 10
    )
    pipe = Pipeline([('vetorizador', CountVectorizer()), ("classificador", MultinomialNB())]) # novo

    pipe.fit(X_train, y_train)

    y_prediction = pipe.predict(X_test)
    accuracy = accuracy_score(y_prediction, y_test)

    print(f"Acurácia={accuracy}")

    # Terminamos o treinamento, vamos tentar fazer o registro
    tentar_registrar_experimento(p_test_size, p_include_names, accuracy, dataset, pipe)