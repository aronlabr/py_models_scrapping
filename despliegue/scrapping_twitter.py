# !pip install git+https://github.com/JustAnotherArchivist/snscrape.git

# !pip install textblob emot nltk

"""### Obteniendo Tweets"""

import snscrape.modules.twitter as sntwitter
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

"""# Preprocesamiento"""
import nltk
try:
  nltk.data.find('corpora/stopwords')
except LookupError:
  nltk.download('popular', quiet=True)
# nltk.download('popular', quiet=True) #ejecútalo una vez y coméntalo para evitar que se descargue varias veces
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from emot.emo_unicode import UNICODE_EMOJI
import re
import string

"""# **Cargar las librerías estándar**"""
import pandas as pd
import numpy as np 

"""# **Carga de librerías especiales**"""
from textblob import TextBlob 
from wordcloud import WordCloud
import seaborn as sns

"""# **SKLEARN**"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

"""## **PRE-PROCESAMIENTO**"""
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

"""## ***MODELOS***"""
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

"""## **METRICAS DE EVALUACION DE MODELOS**"""
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
"""## **LIBRERIAS DE PARA SENTIMENT ANALYSIS (NLP)**"""
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import tensorflow as tf

import streamlit as st

def app():
  input_keywords = st.text_input('Introducir palabras claves' , 'IBM, #IBM, $IBM')
  input_start = st.date_input('Start' , value=pd.to_datetime('2014-1-1'))
  input_end = st.date_input('End' , value=pd.to_datetime('today'))
  input_lang = st.text_input('Introducir codigo de Idioma' , 'en')
  list_filter = ['verified', 'blue_verified', 'trusted', 'has_engagement']
  input_filters = st.multiselect('Filtros a aplicar', list_filter, [*list_filter])
  input_num = st.number_input('Cantidad de Tweets', value=10)

  if st.button('Empezar'):
    st.subheader('Extracción de Tweets') 
    # Created a list to append all tweet attributes(data)
    keywords= input_keywords.replace(",", " OR")
    start = input_start.strftime("%Y-%m-%d") # UTC
    end = input_end.strftime("%Y-%m-%d")
    lang = input_lang
    filters = " ".join(["filter:{}".format(e) for e in input_filters])
    query = f"({keywords}) since:{start} until:{end} lang:{lang} {filters}"
    attributes_container = []
    n_tweets = input_num

    st.write("Resultados encontrados")
    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
      if len(attributes_container)==n_tweets:
        break
      attributes_container.append([tweet.rawContent])
        
    # Creating a dataframe from the tweets list above 
    df = pd.DataFrame(attributes_container, columns=["tweet"])
    st.write(df)
    # df.to_csv('tweets_scrapped.csv', index=False)
    st.write("Dimension")
    st.write(df.shape)

    st.write("""#### Preprocesamiento de datos""")
    st.write("Esto implica estos pasos necesarios antes de llevar a cabo el análisis de sentimiento Para eliminar las palabras vacías Eliminación de etiquetas, enlaces de URL y otras palabras innecesarias Tokenización de las palabras Lemmitización de palabras")
    eng_stop_words = list(stopwords.words('english'))
    emoji = list(UNICODE_EMOJI.keys())

    # función para preprocesar tweet en preparación para análisis de sentimiento
    def ProcessedTweets(text):
        #cambiar el texto del tweet a letras pequeñas
        text = text.lower()
        # Eliminar @ y enlaces
        text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text).split())
        # eliminando caracteres repetidos
        text = re.sub(r'\@\w+|\#\w+|\d+', '', text)
        # eliminar caracteres que no sean letras en ingles
        text = re.sub(r'[^\x00-\x7f]',r'', text)
        # Eliminación de puntuación y números.
        punct = str.maketrans('', '', string.punctuation+string.digits)
        text = text.translate(punct)
        # tokenizar palabras y eliminar palabras vacías del texto del tweet
        tokens = word_tokenize(text)  
        filtered_words = [w for w in tokens if w not in eng_stop_words]
        filtered_words = [w for w in filtered_words if w not in emoji]
        # palabras lemetizantes
        lemmatizer = WordNetLemmatizer() 
        lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
        text = " ".join(lemma_words)
        return text

    # Genere una nueva columna llamada 'Tweets procesados' 
    # aplicando la función de tweets preprocesados ​​a la columna 'Tweet'.
    df['clean_tweet'] = df['tweet'].apply(ProcessedTweets)

    st.write(df.head(5))

    st.markdown("**Análisis de los sentimientos inicial**")
    st.write("Para llevar a cabo esto, la puntuación de polaridad se obtiene utilizando la biblioteca TextBlob, que generalmente se usa para procesos de PNL. El puntaje de polaridad indica el nivel de cuán buenas o malas son las palabras utilizadas en el tweet. Después de obtener la polaridad, se establece una condición para obtener los sentimientos.")
    st.markdown("El conjunto de datos tiene tres sentimientos: * negativo (-1), * neutral (0) y * positivo (+1).")

    # Función para puntaje de polaridad
    def polarity(tweet):
      polarity_m = TextBlob(tweet).sentiment.polarity
      if polarity_m < 0:
            return -1
      elif polarity_m == 0:
          return 0
      else:
          return 1

    # usando las funciones para obtener la polaridad y el sentimiento
    df['category'] = df['clean_tweet'].apply(polarity)
    st.write("""#### Tweets procesados sin originales""")
    df_clean = df.drop(columns=["tweet"])

    # df_clean.to_csv('tweets_data.csv', index=False)

    st.write(df_clean.head())

    st.subheader('Mineria de Textos')
    df = df_clean.copy()
    st.write("Tweets de categoria 1")
    text = ' '.join(list(df[df['category'] == 1.0]['clean_tweet']))
    st.write(df[df['category'] == 1.0]['clean_tweet'])

    st.write("""#### **PRIMER ANALISIS** EDA: Nube de Texto""")
    wordcloud = WordCloud(width=500, height=300, background_color='black', contour_width=3,
                          contour_color='steelblue',max_words=5000,
                          stopwords = set(nltk.corpus.stopwords.words("english")))
    wordcloud.generate(text)
    
    import matplotlib.pyplot as plt
    plt.style.use('fivethirtyeight')

    fig = plt.figure(figsize=(20,10), facecolor='k',edgecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot(fig)

    st.write("""##### Grafico de Palabras mas comunes""")
    #mostrar un grafico de barras con las palabras mas comunes
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')
    # Crear un dataframe de las palabras más comunes
    data_words = pd.DataFrame(wordcloud.words_.items(), columns=['word', 'count'])
    # Visualizar los datos
    fig = plt.figure(figsize=(18, 6))
    sns.barplot(x='word', y='count', data=data_words.head(8))
    plt.title("Palabras más comunes")
    st.pyplot(fig)

    st.markdown("""#### **SEGUNDO ANALISIS** Modelo de Clasificación""")
    st.write("CountVectorizer: Convierta una colección texto en una matriz de recuentos de tokens.")
    vec = CountVectorizer(max_features=10000)
    vec.fit(df['clean_tweet'])
    st.write("Separación para entrenamiento y prueba")
    trn, val = train_test_split(df, test_size=0.20, random_state=111) 
    trn_abs = vec.transform(trn['clean_tweet'])
    val_abs = vec.transform(val['clean_tweet'])
    st.write("Matriz de entrenamiento")
    st.write(trn_abs)
    st.write("Matriz de prueba")
    st.write(val_abs)

    #modelo de clasificación
    clf = OneVsRestClassifier(LogisticRegression(C = 10, n_jobs=-1))
    clf.fit(trn_abs, trn['category'])
    val_preds = clf.predict(val_abs)
    st.write(val['category'])

    st.markdown("""###### Evaluación de Modelo""")

    st.write(f"F1 Score: {f1_score(val['category'], val_preds, average='micro')}")
    st.write(f"Score del modelo:  {clf.score(val_abs, val['category'])}")
    st.write(f"Accuracy Score (Puntaje de Exactitud): {accuracy_score(val['category'], val_preds)}")
    st.write("Matriz de confusion:")
    st.write("Confusion Matrix:") 
    fig, ax = plt.subplots(figsize=(10,10))
    cm = confusion_matrix(val['category'], val_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    st.pyplot(fig)
    st.write("Reporte de Clasificación")
    report = pd.DataFrame(classification_report(val['category'], val_preds, output_dict=True)).transpose()
    st.write(report)
    
    st.markdown("#### **TERCER ANALISIS** Calculo de Subjetividad y polaridad")
    # from textblob import TextBlob
    def obtener_subjetividad(tweet):
      sub = TextBlob(tweet).sentiment.subjectivity
      return sub
    def obtener_polaridad(tweet):
      pol = TextBlob(tweet).sentiment.polarity
      return pol

    df['Subjectividad'] = df['clean_tweet'].apply(obtener_subjetividad)
    df['Polarity'] = df['clean_tweet'].apply(obtener_polaridad)

    st.markdown("""###### Grafico de Palabras mas comunes""")
    #graficamos los resultados
    fig = plt.figure(figsize=(10, 8))
    for i in range(0, df.shape[0]):
        plt.scatter(df["Polarity"][i], df["Subjectividad"][i], color='Blue')

    plt.title('Analisis de sentimiento', fontsize=20)
    plt.xlabel('<-- Negativo -------- Positivo -->', fontsize=15)
    plt.ylabel('<-- Hecho -------- Opinion -->', fontsize=15)
    st.pyplot(fig)

    ptweet = df[df.category == 1.0]
    porcentaje = round(ptweet.shape[0] / df.shape[0] * 100, 1)
    st.write(f"Porcentaje de Tweets Positivos: {porcentaje}")
    ptweet = df[df.category == -1.0]
    porcentaje = round(ptweet.shape[0] / df.shape[0] * 100, 1)
    st.write(f"Porcentaje de Tweets Negativos: {porcentaje}")

    fig = plt.figure(figsize=(10, 8))
    plt.title('Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Counts')
    df['category'].value_counts().plot(kind='bar')
    st.pyplot(fig)
