import sys

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import mlflow
from unidecode import unidecode
import spacy
import swifter

# Esta função tenta registrar o experimento no MLflow
def tentar_registrar_experimento(p_test_size, accuracy, dataset, model):
    with mlflow.start_run():
        # Vamos registrar as métricas
        mlflow.log_metric("acuracia", accuracy)
        # E o dataset (deve ser um caminho para um arquivo)
        mlflow.log_artifact(dataset)
        # E o modelo treinado
        mlflow.sklearn.log_model(model, "modelo")

if __name__ == "__main__":
    # p_test_size: percentual de casos de teste, entre 0 e 1. Default é 0.2
    p_test_size = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    
    print("Treinando classificador de modelos...")
    print(f"Tamanho de testes={p_test_size}")

    dataset = 'imdb-reviews-pt-br.csv.gz'

    products_data = pd.read_csv(dataset, index_col=0)
    
    products_data["sentiment_int"] = products_data["sentiment"].map({"pos": 0, "neg": 1})

    nlp = spacy.load('pt_core_news_sm')
    
    nltk.download('stopwords')
    nltk.download('rslp')

    stop_words = nltk.corpus.stopwords.words('portuguese')

    # determinando forma básica (lema) das palavras
    def lemmatizer(text):
        sent = []
        doc = nlp(text)
        for word in doc:
            if word.pos_ == "VERB":
                sent.append(word.lemma_)
            else:
                sent.append(word.orth_)
        return " ".join(sent)
    
    products_data['text'] = products_data.text_pt.swifter.apply(str.lower)
    products_data['text'] = products_data.text.swifter.apply(unidecode)
    products_data['text'] = products_data.text.swifter.apply(lemmatizer)

    X_train, X_test, y_train, y_test = train_test_split(products_data['text'], 
                                                        products_data["sentiment"], 
                                                        test_size = p_test_size, 
                                                        random_state = 42
                                                        )
    
    pipe = Pipeline(steps=[('vetorizador', TfidfVectorizer(ngram_range=(2,2), 
                                                       stop_words=stop_words,
                                                       use_idf=True, norm='l2'
                                                       )
                                                       ), 
                 ("cls", RandomForestClassifier(n_jobs=-1, random_state=42))
                 ]
                 )

    pipe.fit(X_train, y_train)

    y_prediction = pipe.predict(X_test)
    accuracy = accuracy_score(y_prediction, y_test)

    print(f"Acurácia={accuracy}")

    # Terminamos o treinamento, vamos tentar fazer o registro
    tentar_registrar_experimento(p_test_size, accuracy, dataset, pipe)