# -*- coding: utf-8 -*-
"""
MÉTODOS DE CLUSTERING

CARGA DE LIBRERIAS
"""

from matplotlib import style
style
style.use("ggplot")

import seaborn as sns
sns
from scipy.stats import randint as sp_randint
sp_randint
from sklearn.decomposition import PCA
PCA
from pathlib import Path
Path
import pandas as pd
pd
import numpy as np
np
import matplotlib.pyplot as plt
import streamlit as st

def app():
    st.title('Model - Clustering Jerarquico')
    st.subheader("CARGAR LA DATA")

    df = pd.read_csv("https://raw.githubusercontent.com/tatsath/fin-ml/master/Chapter%208%20-%20Unsup.%20Learning%20-%20Clustering/Data_MasterTemplate.csv", parse_dates=True, index_col=0)
    st.write(df)

    st.subheader("TRANSFORMACION Y LIMPIEZA DE DATOS")

    # Para este paso debemos eliminar algunas columnas
    st.write("Eliminación de las columnas 'No es un número' para Dow Chemicals (DWDP) y Visa (V)")
    df.drop(['DWDP', 'V'], axis=1, inplace=True)
    df.head(2)

    st.write("Copiar el marco de datos para añadir características")
    data = pd.DataFrame(df.copy())
    data.head(2)

    # Rendimiento diario
    st.write("Rendimiento diario (%)")
    datareturns = np.log(data / data.shift(1))

    st.write(datareturns.head())

    datareturns_log = pd.DataFrame(datareturns.copy())

    """Cambio porcentual"""

    st.write("Daily Linear Returns (%)")
    datareturns = data.pct_change(1)

    st.write(datareturns.head(3))

    st.write("Dow Jones Equal Weighted RETURN")
    datareturns["DJIA"] = datareturns.mean(axis=1)
    st.write(datareturns["DJIA"].head(3))

    # Data Raw
    st.write("Data Raw")
    data_raw = datareturns
    data_raw.dropna(how='all', inplace=True)
    st.write(data_raw.head(3))

    # Normalizar los rendimientos
    st.write("Normalizar los rendimientos")
    data = (data_raw - data_raw.mean()) / data_raw.std()
    st.write(data.head(3))

    # Getting rid of the NaN values.
    data.dropna(how='any', inplace=True)
    data_raw.dropna(how='any', inplace=True)

    st.subheader("EDA: VISUALIZACION DE DATOS")

    # Visualizing Log Returns for the DJIA a
    st.write("Visualizing Log Returns for the DJIA a")

    fig=plt.figure(figsize=(16, 5))
    plt.title("Rendimiento lineal del Dow Jones Industrial Average (%)")
    data.DJIA.plot()
    plt.grid(True)
    plt.legend()
    st.pyplot(fig)

    """MODELAR: Investigating Hierarchical Clustering

    Investigación de la agrupación jerárquica

    Un gráfico desordenado de la data ("cambio porcentual diario normalizado") de todas las empresas y su promedio (DJIA)
    """
    st.subheader("MODELAR: Investigating Hierarchical Clustering")
    st.write("""Investigación de la agrupación jerárquica
    Un gráfico desordenado de la data ('cambio porcentual diario normalizado') de todas las empresas y su promedio (DJIA)""")
    # Commented out IPython magic to ensure Python compatibility.
    import matplotlib.cm as cm
    cm

    # %matplotlib inline
    fig=plt.figure(figsize=(16, 8))
    plt.plot(data)
    plt.xticks(rotation='vertical')
    st.pyplot(fig)

    """
    encontrar la matriz de correlación, es decir, las "distancias" entre cada acción
    """
    st.write("encontrar la matriz de correlación, es decir, las 'distancias' entre cada acción")

    corr = data.corr()
    size = 7
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr,cmap=cm.get_cmap('coolwarm'), vmin=0,vmax=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical', fontsize=8)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)
    st.write(fig)

    st.write("""Clusters of Correlation - Agglomerate

    Grupos de Correlación - Aglomerados

    El siguiente paso es buscar grupos de correlaciones mediante la técnica de agrupación jerárquica aglomerada. Su principal ventaja sobre otros métodos de agrupación es que no es necesario adivinar de antemano cuántos clusters puede haber. La agrupación jerárquica aglomerada asigna primero cada punto de datos a su propio clúster, y fusiona gradualmente los clústeres hasta que sólo queda uno. A continuación, el usuario debe elegir un umbral de corte y decidir cuántos clústeres hay.

    Linkage realiza la agrupación real en una línea de código, y devuelve una lista de los clusters unidos en el formato Z=[stock_1, stock_2, distancia, sample_count]

    También hay diferentes opciones para la medición de la distancia. La opción que elegiremos es la medición de la distancia media, pero son posibles otras (ward, single, centroid, etc.).
    """)
    st.write(corr)

    from scipy.cluster.hierarchy import dendrogram, linkage
    dendrogram
    linkage
    Z = linkage(corr, 'average')
    st.write(Z)

    type(Z)

    #Z[0]
    st.subheader("EVALUACION DE CALIDAD DEL MODELAMIENTO: Coeficiente de correlación cofénica")
    st.write("Es importante tener una idea de lo bien que funciona el clustering. Una medida es el Coeficiente de Correlación Cofénica, c. Éste compara (correlaciona) las distancias reales entre pares de todas sus muestras con las implícitas en la agrupación jerárquica. Cuanto más se acerque c a 1, mejor conservará el clustering las distancias originales. Por lo general, c > 0,7 se considera un buen ajuste del cluster. Por supuesto, es posible realizar otras comprobaciones de precisión.")

    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist
    import pylab
    c, coph_dists = cophenet(Z, pdist(corr))
    st.write(c)

    st.subheader("Dendogram")
    st.write("La mejor manera de visualizar un algoritmo de aglomeración es mediante un dendograma, que muestra un árbol de clusters, siendo las hojas las poblaciones individuales y la raíz el cluster único final. La 'distancia' entre cada conglomerado se muestra en el eje y, por tanto, cuanto más largas sean las ramas, menos correlacionados estarán dos conglomerados.")

    fig=plt.figure(figsize=(25, 10))
    labelsize=20
    ticksize=15
    plt.title('Hierarchical Clustering Dendrogram for '+"DJIA", fontsize=labelsize)
    plt.xlabel('stock', fontsize=labelsize)
    plt.ylabel('distance', fontsize=labelsize)
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        labels = corr.columns
    )
    pylab.yticks(fontsize=ticksize)
    pylab.xticks(rotation=-90, fontsize=ticksize)
    st.pyplot(fig)

    st.write("""Según el dendograma anterior, los dos valores más correlacionados son CVX y XOM. Es decir, Chevron Corporation y ExxonMobil. Ambas empresas son petroleras, por lo que es lógico que estén fuertemente correlacionadas. Vamos a trazarlas a continuación para ver visualmente lo bien que se correlacionan. Además, elijamos dos acciones que no estén bien correlacionadas en absoluto para compararlas con, por ejemplo, MCD y JPM.
    Plotear las correlaciones de las muestras
    """)

    #plot sample correlations
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8), sharey=True)
    plt.subplots_adjust(wspace=0.05)

    #high correlation
    sA = "CVX"
    sB = "XOM"
    ax1.plot(data[sA],label=sA)
    ax1.plot(data[sB],label=sB)
    ax1.set_title('Correlación de las Acciones = %.3f'%corr[sA][sB])
    ax1.set_ylabel('Variaciones de Precio de Cierre Normalizadas')
    ax1.legend(loc='upper left',prop={'size':8})
    plt.setp(ax1.get_xticklabels(), rotation=70)

    #low correlation
    sA = "MCD"
    sB = "JPM"
    ax2.plot(data[sA],label=sA)
    ax2.plot(data[sB],label=sB)
    ax2.set_title('Correlación de las Acciones = %.3f'%corr[sA][sB])
    ax2.legend(loc='upper left',prop={'size':8})
    plt.setp(ax2.get_xticklabels(), rotation=70)
    st.pyplot(f)