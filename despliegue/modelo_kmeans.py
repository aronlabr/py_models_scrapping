# install a library
# !pip install yfinance

from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import datetime
import numpy as np
import plotly.graph_objects as go
from pandas import read_csv
from numpy import arange
import streamlit as st

def app():
    companies_dict = {
    'Amazon':'AMZN',
    'Apple':'AAPL',
    'Walgreen':'WBA',
    'Northrop Grumman':'NOC',
    'Boeing':'BA',
    'Lockheed Martin':'LMT',
    'McDonalds':'MCD',
    'Intel':'INTC',
    #'Navistar':'NAV',#Navistar fue adquirida por Traton así que se desestimó su uso en este trabajo. 
    #Fuente: https://finance.yahoo.com/news/company-acquired-navistar-saw-sharp-035900178.html
    'IBM':'IBM',
    'Texas Instruments':'TXN',
    'MasterCard':'MA',
    'Microsoft':'MSFT',
    'General Electrics':'GE',
    'Symantec':'SYM.SG',#Cambiado el simbolo SYMC por el actual SYM.SG
    'American Express':'AXP',
    'Pepsi':'PEP',
    'Coca Cola':'KO',
    'Johnson & Johnson':'JNJ',
    'Toyota':'TM',
    'Honda':'HMC',
    'Mitsubishi':'8058.T',#Cambiado el simbolo MSBHY por el actual 8058.T
    'Sony':'SONY',#Cambiado el simbolo SNE por el actual SONY
    'Exxon':'XOM',
    'Chevron':'CVX',
    'Valero Energy':'VLO',
    'Ford':'F',
    'Bank of America':'BAC'}
    start = st.date_input('Start Train' , value=pd.to_datetime('2014-1-1'))
    end = st.date_input('End Train' , value=pd.to_datetime('2018-12-30'))

    df = yf.download(list(companies_dict.values()), start, end)
    st.title('Model - K-Means')
    st.subheader('Clasificación de acciones')

    df1 = df.dropna()
    st.write(df1.describe())

    stock_open = np.array(df1['Open']).T # stock_open is numpy array of transpose of df1['Open']
    stock_close = np.array(df1['Close']).T # stock_close is numpy array of transpose of df1['Close']

    movements = stock_close - stock_open

    sum_of_movement = np.sum(movements,1)

    for i in range(len(companies_dict)):
        st.write('Company:{}, Change:{}'.format(df1['High'].columns[i],sum_of_movement[i]))

    st.subheader('Visualizaciones de Acciones')

    fig = plt.figure(figsize = (12,6)) # Adjusting figure size
    plt.subplot(1,2,1) # Subplot 1
    plt.title('Company:Amazon',fontsize = 20)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 20)
    plt.xlabel('Date',fontsize = 20)
    plt.ylabel('Opening price',fontsize = 20)
    plt.plot(df['Open']['AMZN'])
    plt.subplot(1,2,2) # Subplot 2
    plt.title('Company:Apple',fontsize = 20) 
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 20)
    plt.xlabel('Date',fontsize = 20)
    plt.ylabel('Opening price',fontsize = 20)
    plt.plot(df['Open']['AAPL'])
    st.pyplot(fig)

    fig = plt.figure(figsize = (12,6)) # Adjusting figure size
    plt.title('Company:Amazon',fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 20)
    plt.xlabel('Date',fontsize = 20)
    plt.ylabel('Price',fontsize = 20)
    plt.plot(df.iloc[0:30]['Open']['AMZN'],label = 'Open') # Opening prices of first 30 days are plotted against date
    plt.plot(df.iloc[0:30]['Close']['AMZN'],label = 'Close') # Closing prices of first 30 days are plotted against date
    plt.legend(loc='upper left', frameon=False,framealpha=1,prop={'size': 22}) # Properties of legend box
    st.pyplot(fig)

    fig = plt.figure(figsize = (20,10)) 
    plt.title('Company:Amazon',fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 20)
    plt.xlabel('Date',fontsize = 20)
    plt.ylabel('Movement',fontsize = 20)
    plt.plot(movements[0][0:30])
    st.pyplot(fig)

    fig = plt.figure(figsize = (12,6)) # Adjusting figure size
    plt.title('Company:Amazon',fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 20)
    plt.xlabel('Date',fontsize = 20)
    plt.ylabel('Volume',fontsize = 20)
    plt.plot(df['Volume']['AMZN'],label = 'Open') # Volume prices of first 30 days are plotted against date
    st.pyplot(fig)

    fig = go.Figure(data=[go.Candlestick(x=df1.index,
                    open=df1.iloc[0:60]['Open']['AMZN'],
                    high=df1.iloc[0:60]['High']['AMZN'],
                    low=df1.iloc[0:60]['Low']['AMZN'],
                    close=df1.iloc[0:60]['Close']['AMZN'])])
    st.plotly_chart(fig)

    fig = plt.figure(figsize = (12,6)) 
    ax1 = plt.subplot(1,2,1)
    plt.title('Company:Amazon',fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 20)
    plt.xlabel('Date',fontsize = 20)
    plt.ylabel('Movement',fontsize = 20)
    plt.plot(movements[0]) 
    plt.subplot(1,2,2,sharey = ax1)
    plt.title('Company:Apple',fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 20)
    plt.xlabel('Date',fontsize = 20)
    plt.ylabel('Movement',fontsize = 20)
    plt.plot(movements[1])
    st.pyplot(fig)

    from sklearn.preprocessing import Normalizer
    normalizer = Normalizer() # Define a Normalizer
    norm_movements = normalizer.fit_transform(movements) # Fit and transform

    st.write(norm_movements.min())
    st.write(norm_movements.max())
    st.write(norm_movements.mean())

    fig = plt.figure(figsize = (12,6)) 
    ax1 = plt.subplot(1,2,1)
    plt.title('Company:Amazon',fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 20)
    plt.xlabel('Date',fontsize = 20)
    plt.ylabel('Normalized Movement',fontsize = 20)
    plt.plot(norm_movements[0]) 
    plt.subplot(1,2,2,sharey = ax1)
    plt.title('Company:Apple',fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 20)
    plt.xlabel('Date',fontsize = 20)
    plt.ylabel('Normalized Movement',fontsize = 20)
    plt.plot(norm_movements[1])
    st.pyplot(fig)
    # Import the necessary packages
    from sklearn.pipeline import make_pipeline
    from sklearn.cluster import KMeans

    # Define a normalizer
    normalizer = Normalizer()

    # Create Kmeans model
    kmeans = KMeans(n_clusters = 10,max_iter = 1000)

    # Make a pipeline chaining normalizer and kmeans
    pipeline = make_pipeline(normalizer,kmeans)

    # Fit pipeline to daily stock movements
    pipeline.fit(movements)

    labels = pipeline.predict(movements)

    df2 = pd.DataFrame({'labels':labels,'companies':list(companies_dict.keys())}).sort_values(by=['labels'],axis = 0)
    
    st.write("Data Frame de Compañias Ordenadas")
    st.write(df2)

    kmeans.inertia_

    from sklearn.decomposition import PCA

    # Define a normalizer
    normalizer = Normalizer()

    # Reduce the data
    reduced_data = PCA(n_components = 2)

    # Create Kmeans model
    kmeans = KMeans(n_clusters = 10,max_iter = 1000)

    # Make a pipeline chaining normalizer, pca and kmeans
    pipeline = make_pipeline(normalizer,reduced_data,kmeans)

    # Fit pipeline to daily stock movements
    pipeline.fit(movements)

    # Prediction
    labels = pipeline.predict(movements)

    # Create dataframe to store companies and predicted labels
    df3 = pd.DataFrame({'labels':labels,'companies':list(companies_dict.keys())}).sort_values(by=['labels'],axis = 0)
    st.write("Data Frame de Compañias Ordenadas")
    st.write(df3)

    # Reduce the data
    reduced_data = PCA(n_components = 2).fit_transform(norm_movements)

    # Define step size of mesh
    h = 0.01

    # Plot the decision boundary
    x_min,x_max = reduced_data[:,0].min()-1, reduced_data[:,0].max() + 1
    y_min,y_max = reduced_data[:,1].min()-1, reduced_data[:,1].max() + 1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

    # Obtain labels for each point in the mesh using our trained model
    Z = kmeans.predict(np.c_[xx.ravel(),yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    # Define color plot
    cmap = plt.cm.Paired

    # Plotting figure
    plt.clf()
    fig = plt.figure(figsize=(10,10))
    plt.imshow(Z,interpolation = 'nearest',extent=(xx.min(),xx.max(),yy.min(),yy.max()),cmap = cmap,aspect = 'auto',origin = 'lower')
    plt.plot(reduced_data[:,0],reduced_data[:,1],'k.',markersize = 5)

    # Plot the centroid of each cluster as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:,0],centroids[:,1],marker = 'x',s = 169,linewidths = 3,color = 'w',zorder = 10)

    plt.title('K-Means clustering on stock market movements (PCA-Reduced data)')
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    st.pyplot(fig)