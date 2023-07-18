# -*- coding: utf-8 -*-
"""
### Importacion de Librerias
"""

import pandas as pd
import datetime
import numpy as np 
from matplotlib import style
from pandas_datareader.yahoo.daily import YahooDailyReader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

from sklearn import metrics
import plotly.express as px
# %matplotlib inline
# ignore warnings 
import warnings
warnings.filterwarnings('ignore')

def app():
    """### Obtener nuestros datos de stock:"""
    st.title('Model - LSTM')

    # Obtenga los datos de stock usando la API de Yahoo:
    style.use('ggplot')
    from pandas_datareader import data as pdr
    import yfinance as yfin
    yfin.pdr_override()

    start = st.date_input('Start Train' , value=pd.to_datetime('2014-1-1'))
    end = st.date_input('End Train' , value=pd.to_datetime('2018-12-30'))
    start_test = st.date_input('Start Test' , value=pd.to_datetime('2019-1-1'))
    end_test = st.date_input('End Test' , value=pd.to_datetime('today'))
    
    st.subheader('Predicción de tendencia de acciones')

    user_input = st.text_input('Introducir cotización bursátil' , 'AMZN')
    # obtener datos de 2014-2018 para entrenar nuestro modelo
    df = pdr.get_data_yahoo(user_input, start, end)
    # obtener datos de 2019 para probar nuestro modelo en
    test_df = pdr.get_data_yahoo(user_input, start_test, end_test)

    # Describiendo los datos

    """### Arreglando nuestros datos:"""

    # ordenar por fecha
    df = df.sort_values('Date')
    test_df = test_df.sort_values('Date')

    # fix the date 
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    test_df.reset_index(inplace=True)
    test_df.set_index("Date", inplace=True)

    """#### Graficando nuestros datos y la media móvil:"""

    # Commented out IPython magic to ensure Python compatibility.

    st.subheader('Datos del 2004 al 2022') 
    st.write(df.describe())

    # Visualice los datos del stock:

    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize = (12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    # Rolling mean
    close_px = df['Adj Close']
    mavg = close_px.rolling(window=100).mean()

    fig = plt.figure(figsize = (12,6))
    close_px.plot(label=user_input)
    mavg.plot(label='mavg')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)

    """#### Conversión de fechas:"""


    # cambiar las fechas en enteros para el entrenamiento
    dates_df = df.copy()
    dates_df = dates_df.reset_index()

    # Almacene las fechas originales para trazar las predicciones
    org_dates = dates_df['Date']

    # convert to ints
    dates_df['Date'] = dates_df['Date'].map(mdates.date2num)

    dates_df.tail()

    """### Normalizando nuestros datos:"""

    # Crear un conjunto de datos de entrenamiento de precios de 'Adj Close':
    train_data = df.loc[:,'Adj Close'].to_numpy()
    print(train_data.shape) # 1257 

    
    from sklearn.preprocessing import MinMaxScaler
    # Aplique la normalización antes de alimentar a LSTM usando sklearn:

    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1,1)

    scaler.fit(train_data)
    train_data = scaler.transform(train_data)

    """### Preparando nuestros datos para la red neuronal:"""

    '''Función para crear un conjunto de datos para alimentar un LSTM'''
    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)
        
        
    # Cree los datos para entrenar nuestro modelo en:
    time_steps = 36
    X_train, y_train = create_dataset(train_data, time_steps)

    # remodelarlo [muestras, pasos de tiempo, características]
    X_train = np.reshape(X_train, (X_train.shape[0], 36, 1))

    # print(X_train.shape)


    # Visualizando nuestros datos con impresiones:
    # print('X_train:')
    # print(str(scaler.inverse_transform(X_train[0])))
    # print("\n")
    # print('y_train: ' + str(scaler.inverse_transform(y_train[0].reshape(-1,1)))+'\n')

    """### Aplicando modelo LTSM:"""

    import keras
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout

    # Construye el modelo
    model = keras.Sequential()

    model.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 100))
    model.add(Dropout(0.2))

    # Capa de salida
    model.add(Dense(units = 1))

    # Compilando el modelo
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Ajuste del modelo al conjunto de entrenamiento
    history = model.fit(X_train, y_train, epochs = 10, batch_size = 10, validation_split=.30)

    """#### Grafica de la pérdida del modelo:"""

    # Plot training & validation loss values
    fig2=plt.figure(figsize=(12,6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    st.pyplot(fig2)

    """#### Haciendo la predicción:"""

    # Obtenga los precios de las acciones para 2019 para que nuestro modelo haga las predicciones
    test_data = test_df['Adj Close'].values
    test_data = test_data.reshape(-1,1)
    test_data = scaler.transform(test_data)

    # Cree los datos para probar nuestro modelo en:
    time_steps = 36
    X_test, y_test = create_dataset(test_data, time_steps)

    # almacenar los valores originales para trazar las predicciones
    y_test = y_test.reshape(-1,1)
    org_y = scaler.inverse_transform(y_test)

    # remodelarlo [muestras, pasos de tiempo, características]
    X_test = np.reshape(X_test, (X_test.shape[0], 36, 1))

    # Predecir los precios con el modelo.
    predicted_y = model.predict(X_test)
    predicted_y = scaler.inverse_transform(predicted_y)
    
    # Graficar los resultados 
    fig = plt.figure(figsize = (12,6))
    plt.plot(org_y, color = 'red', label = 'Real Stock Price')
    plt.plot(predicted_y, color = 'blue', label = 'Predicted Stock Price')
    plt.title(f'{user_input} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Tesla Stock Price')
    plt.legend()
    plt.show()
    st.pyplot(fig)
    
    # Visualizando datos
    valid = pd.concat([pd.DataFrame(org_y, columns=["Test"]), 
                        pd.DataFrame(predicted_y, columns=["Prediccion"])], axis=1)
    st.subheader('Mostrar los datos originales y predecidos') 
    st.write(valid)

##########PLANTILLA####################
    # Evaluación del modelo
    
    st.title('Evaluación del Modelo LSTM')
    ## Métricas
    MAE=metrics.mean_absolute_error(y_test, predicted_y)
    MSE=metrics.mean_squared_error(y_test, predicted_y)
    RMSE=np.sqrt(metrics.mean_squared_error(y_test, predicted_y))
    
    metricas = {
        'metrica' : ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error'],
        'valor': [MAE, MSE, RMSE]
    }
    metricas = pd.DataFrame(metricas)  
    ### Gráfica de las métricas
    st.subheader('Métricas de rendimiento') 
    fig = px.bar(        
        metricas,
        x = "metrica",
        y = "valor",
        title = "Métricas del Modelo LSTM",
        color="metrica"
    )
    st.plotly_chart(fig)