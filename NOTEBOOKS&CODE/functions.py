'''Mi script de funciones auxiliares'''

# Librerias
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.stats import pearsonr
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import statsmodels.api as sm


# Revision de columnas
def univar_analisis(data):

    results = []

    for feature in data.columns:
        data_type = data[feature].dtype  

        total = data.shape[0]
        nan_count = data[feature].isna().sum()
        no_missings = total - nan_count
        pct_missings = nan_count / total

        if pd.api.types.is_numeric_dtype(data[feature]):  
            promedio = data[feature].mean()
            desv_estandar = data[feature].std()
            varianza = data[feature].var()
            minimo = data[feature].min()
            p1 = data[feature].quantile(0.01)
            p5 = data[feature].quantile(0.05)
            p10 = data[feature].quantile(0.1)
            q1 = data[feature].quantile(0.25)
            mediana = data[feature].quantile(0.5)
            q3 = data[feature].quantile(0.75)
            p90 = data[feature].quantile(0.9)
            p95 = data[feature].quantile(0.95)
            p99 = data[feature].quantile(0.99)
            maximo = data[feature].max()

            inf_count = np.isinf(data[feature]).sum()
            neg_inf_count = (data[feature] == -np.inf).sum()
        else:
           
            promedio = None
            desv_estandar = None
            varianza = None
            minimo = None
            p1 = None
            p5 = None
            p10 = None
            q1 = None
            mediana = None
            q3 = None
            p90 = None
            p95 = None
            p99 = None
            maximo = None
            inf_count = 0
            neg_inf_count = 0

        results.append({
            'Variable': feature,
            'DataType':data_type,
            'Total': total,
            'No Missings': no_missings,
            'Missings': nan_count,
            '% Missings': pct_missings,
            'Inf Count': inf_count,
            '-Inf Count': neg_inf_count,
            'Promedio': promedio,
            'Desviación Estandar': desv_estandar,
            'Varianza': varianza,
            'Mínimo': minimo,
            'p1': p1,
            'p5': p5,
            'p10': p10,
            'q1': q1,
            'Mediana': mediana,
            'q3': q3,
            'p90': p90,
            'p95': p95,
            'p99': p99,
            'Máximo': maximo
        })

    return results

# Segmentacion por signo
def segment_by_sign(df, variable, positive_label='Positive', negative_label='Negative', zero_label=None):

    df = df.copy() 
    
    def assign_label(x):
        if pd.isnull(x):
            return 'Missing'  
        elif x > 0:
            return positive_label
        elif x < 0:
            return negative_label
        else:
            return zero_label if zero_label is not None else negative_label

    df['habito_pupil'] = df[variable].apply(assign_label)

    return df

# Para crear heatmap
def crear_heatmap(data, nombre_columna1, nombre_columna2, titulo, x_titulo, y_titulo, color, template):
    
    data_heatmap = data.dropna(subset=[nombre_columna1, nombre_columna2])

    # Crear tabla de frecuencia resumida (crosstab)
    heatmap_data = pd.crosstab(data_heatmap[nombre_columna1], data_heatmap[nombre_columna2])

    ## Definir los valores únicos de los años en la columna x
    unique_years = sorted(data[nombre_columna2].unique())

    # Crear el heatmap
    heatmap = go.Heatmap(
        z=heatmap_data.values,  
        x=heatmap_data.columns,  
        y=heatmap_data.index,    
        colorscale=color,     
        text=heatmap_data.values,  
        texttemplate="%{text}",   
        textfont={"size":12},     
        showscale=True,           
    )

    # Crear la figura
    fig = go.Figure(data=[heatmap])

    # Añadir título y etiquetas
    fig.update_layout(
        title=titulo,
        yaxis_title=y_titulo,
        template=template,
        xaxis=dict(
            tickmode='array',       
            tickvals=unique_years,   
            ticktext=unique_years,          
            title=x_titulo            
        )
    )

    # Mostrar la figura
    fig.show()

# Estandarizar nombres de columnas
def std_nombres_columnas(data):
    
    # Convertir nombres de las columnas a minúsculas y reemplazar espacios por guiones bajos
    data.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
    
    # Devolver los nombres de las columnas modificadas
    return data.columns


# Crear violines con boxplot
def crear_violin_boxplot(data, x_column, x_titulo, y_column,y_titulo, title, y_range, y_mean_annot):

    mean_values_A = data[data[y_column] > 0].groupby(x_column)[y_column].mean().reset_index()
    data = data.sort_values(x_column)
    mean_values_A = mean_values_A.sort_values(x_column)

    # Crear boxplot
    boxplot = go.Box(
        x=data[x_column],
        y=data[y_column],
        boxpoints=False,  
        opacity=0.2,
        name='Boxplot'
    )

    # Crear violin plot
    violin = go.Violin(
        x=data[x_column],
        y=data[y_column],
        line_color='gray',
        opacity=0.15,
        name='Violin',  
        width=0.8, 
        points=False  
    )

    # Crear puntos medios
    mean_points = go.Scatter(
        x=mean_values_A[x_column],
        y=mean_values_A[y_column],
        mode='markers',
        marker=dict(color='blue', size=10, symbol='diamond'),
        name='Mean Points'
    )

    # Crear anotaciones para los puntos medios
    annotations = [
        dict(
            x=mean_values_A[x_column].iloc[i],
            y=y_mean_annot,  
            text=str(round(mean_values_A[y_column].iloc[i], 2)),
            showarrow=False,
            font=dict(color='blue', size=12),
        ) for i in range(len(mean_values_A))
    ]

    # Línea horizontal de referencia en y=1
    hline = go.Scatter(
        x=data[x_column],
        y=[1] * len(data[x_column]),
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Reference Line'
    )

    ## Definir los valores únicos de los años en la columna x
    unique_years = sorted(data[x_column].unique())

    # Crear la figura
    fig = go.Figure(data=[boxplot, violin, mean_points])

    # Añadir las anotaciones y otras configuraciones
    fig.update_layout(annotations=annotations)

    # Ajustar el rango del eje y
    fig.update_yaxes(range=y_range)

    # Actualizar el layout con el título, etiquetas y configuración del gráfico
    fig.update_layout(
        title=title,
        xaxis_title=x_titulo,
        yaxis_title=y_titulo,
        template="simple_white",
        showlegend=False,
        width=1500,
        height=600,
        xaxis=dict(
            tickmode='array',       
            tickvals=unique_years,   
            ticktext=unique_years,          
            title='Año'              
        )
    )

    # Mostrar el gráfico
    fig.show()

# ICV a 180 dias
def grafico_icv(summary_data, x_column, y_column, title):

    # Constantes
    x_titulo="Monto Acumulativo de Incrementos"
    y_titulo="ICV"
    marker_color='orange'
    marker_size=8
    line_color='orange'
    y_hline=0.05

    # Crear la figura
    fig = go.Figure()

    # Línea de ICV con puntos marcados
    fig.add_trace(go.Scatter(
        x=summary_data[x_column],
        y=summary_data[y_column],
        mode='lines+markers',
        marker=dict(color=marker_color, size=marker_size),
        line=dict(color=line_color),
        name='ICV'
    ))

    # Línea horizontal en y=0.05
    fig.add_shape(type="line",
                  x0=0, x1=max(summary_data[x_column]),
                  y0=y_hline, y1=y_hline,
                  line=dict(color='red', dash='dash'),
                  name='Límite de referencia')

    # Títulos y etiquetas
    fig.update_layout(
        title=title,
        xaxis_title=x_titulo,
        yaxis_title=y_titulo,
        template='plotly_white',
        width=900,
        height=400
    )

    # Mostrar el gráfico
    fig.show()

# Frecuencia por incremento acumulado
def grafico_frecuencia(summary_data, x_column, y_column, titulo, x_titulo, y_titulo, color):

    # Constantes
    bar_color=color
    bar_opacity=0.5

    # Crear la figura
    fig = go.Figure()

    # Gráfico de barras
    fig.add_trace(go.Bar(
        x=summary_data[x_column],
        y=summary_data[y_column],
        marker=dict(color=bar_color, opacity=bar_opacity),
        name='Counter'
    ))

    # Título y etiquetas
    fig.update_layout(
        title=titulo,
        xaxis_title=x_titulo,
        yaxis_title=y_titulo,
        template='plotly_white',
        width=900,
        height=400
    )

    # Mostrar el gráfico
    fig.show()

def pca_elbow_plot(data, pct_varianza):

    # Ajustar PCA sin reducir el número de componentes
    pca = PCA()
    pca.fit(data)

    # Obtener la varianza explicada acumulada
    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Encontrar el número de componentes que explican al menos el 95% de la varianza
    num_components_x = np.argmax(explained_variance >=  pct_varianza) + 1

    # Gráfico elbow plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.axhline(y=pct_varianza, color='r', linestyle=':')  
    plt.axvline(x=num_components_x, color='r', linestyle=':')  
    plt.title('Elbow Plot - PCA')
    plt.xlabel('Número de componentes principales')
    plt.ylabel('Varianza explicada acumulada')
    plt.text(num_components_x, pct_varianza, f'{pct_varianza*100}% en componente {num_components_x}', color='red', fontsize=12)
    plt.grid(True)
    plt.show()


# Graficop de correlaciones
def correlation_matrix(data, title, x_title, y_title, template, colorscale='Magma'):
    # Calcular la matriz de correlación
    correlation_matrix = data.corr()

    # Invertir el orden de las etiquetas del eje y
    y_labels_inverted = correlation_matrix.index[::-1] 

    # Crear el heatmap usando Plotly
    heatmap = go.Heatmap(
        z=correlation_matrix.values[::-1],  
        x=correlation_matrix.columns,  
        y=y_labels_inverted,  
        colorscale=colorscale,  
        colorbar=dict(title='Correlación'), 
        text=correlation_matrix.values[::-1],  
        texttemplate="%{text:.2f}",  
    )

    # Crear la figura
    fig = go.Figure(data=[heatmap])

    # Añadir título y etiquetas
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template=template,
        width=800,
        height=800
    )

    # Mostrar la figura
    fig.show()


# Función para calcular la matriz de correlación y p-valores
def calculate_corr_and_pvalues(data):
    
    numeric_df = data.select_dtypes(include=[np.number])
    
    corr_matrix = numeric_df.corr()
    
    p_values_matrix = pd.DataFrame(np.zeros((numeric_df.shape[1], numeric_df.shape[1])), columns=numeric_df.columns, index=numeric_df.columns)

    # Calcular los coeficientes de correlación y los p-valores
    for col1 in numeric_df.columns:
        for col2 in numeric_df.columns:
            if col1 == col2:
                p_values_matrix.loc[col1, col2] = np.nan  
            else:
                _, p_value = pearsonr(numeric_df[col1], numeric_df[col2])
                p_values_matrix.loc[col1, col2] = p_value

    return corr_matrix, p_values_matrix

# Función para graficar la matriz de correlación y p-valores
def correlation_matrix_with_pvalues(data, title, x_title, y_title, template, colorscale1='Magma', colorscale2='Magma'):
   
    correlation_matrix, p_values_matrix = calculate_corr_and_pvalues(data)

    y_labels_inverted = correlation_matrix.index[::-1] 

    # Crear el heatmap de la matriz de correlación
    heatmap_corr = go.Heatmap(
        z=correlation_matrix.values[::-1],  
        x=correlation_matrix.columns,  
        y=y_labels_inverted,  
        colorscale=colorscale1,  
        colorbar=dict(title='Correlación', x=0.45), 
        text=correlation_matrix.values[::-1],  
        texttemplate="%{text:.2f}",  
    )

    # Crear el heatmap de los valores p
    heatmap_pvalues = go.Heatmap(
        z=p_values_matrix.values[::-1],  
        x=p_values_matrix.columns,  
        y=y_labels_inverted,  
        colorscale=colorscale2,  
        colorbar=dict(title='Valores p', x=1.0), 
        text=p_values_matrix.values[::-1],  
        texttemplate="%{text:.8f}",  
    )

    # Crear subplots para mostrar los dos heatmaps
    fig = make_subplots(
        rows=1, cols=2,  
        subplot_titles=("Matriz de correlación", "Valores p"),
        horizontal_spacing=0.1   
    )

    fig.add_trace(heatmap_corr, row=1, col=1)

    fig.add_trace(heatmap_pvalues, row=1, col=2)

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template=template,
        height= 600, 
        width = 1900 
    )

    # Título para el primer heatmap (correlación)
    fig.update_xaxes(title_text=x_title, row=1, col=1)

    # Título para el segundo heatmap (valores p)
    fig.update_xaxes(title_text=x_title, row=1, col=2)

    # Mostrar la figura
    fig.show()


# Distribuetion plot
def plot_distribution(df, column_name, nbins, hist_color, line_color, title, both = True, histograma = False, densidad = False):
    
    variable = df[column_name].dropna()  
    
    # Crear el histograma
    histogram = go.Histogram(
        x=variable,
        nbinsx=nbins,  
        opacity=0.7,
        name='Histograma', 
        marker=dict(color=hist_color)  
    )

    # Calcular la línea de densidad
    density = gaussian_kde(variable)
    x_vals = np.linspace(min(variable), max(variable), 1000)
    bin_width = (max(variable) - min(variable)) / nbins
    density_line = go.Scatter(
        x=x_vals,
        y=density(x_vals) * len(variable) * bin_width, 
        mode='lines',
        name='Densidad',
        line=dict(color=line_color, width=2)
    )

    # Condicionales para agregar histogramas y/o densidad
    if both:
        fig1 = go.Figure()
        fig1.add_trace(histogram)
        fig1.add_trace(density_line)
        fig1.update_layout(
            title=title,
            xaxis_title=column_name,
            yaxis_title='Frecuencia',
            template="plotly_white"
        )
        fig1.show()
    elif histograma:

        fig2 = go.Figure()
        fig2.add_trace(histogram)
        fig2.update_layout(
            title=title,
            xaxis_title=column_name,
            yaxis_title='Frecuencia',
            template="plotly_white"
        )
        fig2.show()
    elif densidad:

        fig3 = go.Figure()
        fig3.add_trace(density_line)
        fig3.update_layout(
            title=title,
            xaxis_title=column_name,
            yaxis_title='Frecuencia',
            template="plotly_white"
        )
        fig3.show()

# Regresion lineal con intervalo de confianza
def plot_regression_with_confidence(df, x_col, y_col, title, x_title, y_title, alp=0.05):
    
    # Eliminar filas con valores nulos en x_col o y_col
    df_clean = df[[x_col, y_col]].dropna().reset_index(drop=True)

    # Variables independientes y dependiente
    X = df_clean[x_col]
    y = df_clean[y_col]

    # Añadir constante para el intercepto
    X_with_const = sm.add_constant(X)

    # Ajustar el modelo de regresión lineal
    model = sm.OLS(y, X_with_const).fit()

    # Generar predicciones y el intervalo de confianza al 95%
    predictions = model.get_prediction(X_with_const)
    pred_summary = predictions.summary_frame(alpha=alp)  

    # Ordenar los valores para la gráfica
    sorted_idx = np.argsort(X.values)
    X_sorted = X.iloc[sorted_idx]
    y_sorted = y.iloc[sorted_idx]
    mean_pred = pred_summary['mean'].iloc[sorted_idx]
    ci_lower = pred_summary['mean_ci_lower'].iloc[sorted_idx]
    ci_upper = pred_summary['mean_ci_upper'].iloc[sorted_idx]

    # Crear el scatter plot de los datos originales
    scatter = go.Scatter(
        x=X_sorted,
        y=y_sorted,
        mode='markers',
        name='Datos',
        marker=dict(color='blue', opacity=0.6)
    )

    # Crear la línea de regresión
    regression_line = go.Scatter(
        x=X_sorted,
        y=mean_pred,
        mode='lines',
        name='Línea de Regresión',
        line=dict(color='red')
    )

    # Crear el intervalo de confianza
    ci_band = go.Scatter(
        x=pd.concat([X_sorted, X_sorted[::-1]]),  
        y=pd.concat([ci_upper, ci_lower[::-1]]),  
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255, 0, 0, 0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Intervalo de Confianza (95%)'
    )

    # Crear la figura
    fig = go.Figure(data=[scatter, regression_line, ci_band])

    # Actualizar el layout
    fig.update_layout(
        title=title,
        xaxis_title=x_title if x_title else x_col,
        yaxis_title=y_title if y_title else y_col,
        template='simple_white',
        legend=dict(title=''),
        hovermode='closest'
    )

    # Mostrar la figura
    fig.show()

