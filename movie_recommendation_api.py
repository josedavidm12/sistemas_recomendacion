import numpy as np
import pandas as pd
import sqlite3 as sql
import os
import random
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from google.colab import drive
from google.colab.output import eval_js
import nest_asyncio
import pyngrok
from fastapi import Body


# Aplicar nest_asyncio para permitir que el servidor FastAPI funcione en Colab
nest_asyncio.apply()

# Inicializar la API
app = FastAPI(
    title="API de Recomendación de Películas",
    description="Esta API proporciona recomendaciones de películas basadas en un sistema de filtrado colaborativo",
    version="1.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clase para la solicitud de recomendaciones
class UserRequest(BaseModel):
    user_ids: List[int]
    num_recommendations: Optional[int] = 5

# Clase para la respuesta de recomendaciones
class MovieRecommendation(BaseModel):
    user_ids: List[int]
    num_recommendations: int


# Montar Google Drive (si estamos en Colab)
try:
    drive.mount('/content/drive')
    print("Google Drive montado correctamente")
except:
    print("No se pudo montar Google Drive o no estamos en Colab")

def procesar():
    """Procesa los datos de la película desde la base de datos SQLite"""
    try:
        # Crear la conexión con la base de datos
        conn = sql.connect('/content/drive/MyDrive/analitica 3/sistemas_recomendacion/data/db_movies2')
        cur = conn.cursor()
        
        # Ejecutar el script SQL
        with open('/content/drive/MyDrive/analitica 3/sistemas_recomendacion/joins.sql', 'r') as file:
            sql_script = file.read()
        cur.executescript(sql_script)
        conn.commit()
        
        # Cargar los datos procesados
        df_final = pd.read_sql("SELECT * FROM df_final", conn)
        
        # DUMIZAR VARIABLE GENEROS
        # Separar columna de generos
        df_genres = pd.get_dummies(df_final['genres']).astype(int)
        # Tomar los datos únicos de cada película sin la columna de géneros
        df_movies = df_final.drop('genres', axis=1).drop_duplicates(subset=['user_id', 'movie_id', 'title'])
        # Agregar información de géneros agrupando por movie_id
        genres_by_movie = df_genres.groupby(df_final['movie_id']).max()
        # Unir los dataframes
        df_terminado = df_movies.set_index('movie_id').join(genres_by_movie).reset_index()
        
        # CAMBIO DE TIPO DE VARIABLE
        df_terminado['year_movies'] = df_terminado['year_movies'].astype('int64')
        df_terminado['movie_id'] = df_terminado['movie_id'].astype('object')
        df_terminado['user_id'] = df_terminado['user_id'].astype('object')
        
        # IMPUTAR ATIPICOS
        df_terminado = impute_outliers_with_mean(df_terminado, 'rating')
        
        # ESCALAR VARIABLES
        # Seleccion de las variables a escalar
        numcol = [col for col in df_terminado.columns if df_terminado[col].dtypes == "int64"]
        
        # Escalamiento MinMax
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        for col in numcol:
            df_terminado[[col]] = scaler.fit_transform(df_terminado[[col]])
        
        # Eliminar columnas innecesarias para el catálogo
        df_terminado2 = df_terminado.drop(columns=['user_id', 'rating'])
        df_terminado2 = df_terminado2.drop_duplicates(subset=['movie_id', 'title'])
        
        # Calcular promedio de rating por película
        promedios = df_terminado.groupby(['movie_id', 'title'])['rating'].mean().reset_index()
        promedios.rename(columns={'rating': 'promedio_rating'}, inplace=True)
        
        # Unir los promedios al dataframe filtrado
        df_catalogo = pd.merge(df_terminado2, promedios, on=['movie_id', 'title'], how='left')
        
        # Reordenar las columnas
        cols = df_catalogo.columns.tolist()
        cols_reordenadas = ['movie_id', 'title', 'promedio_rating'] + [col for col in cols if col not in ['movie_id', 'title', 'promedio_rating']]
        df_catalogo = df_catalogo[cols_reordenadas]
        
        return df_final, df_terminado, df_catalogo, conn, cur
    
    except Exception as e:
        print(f"Error al procesar los datos: {str(e)}")
        raise e

def impute_outliers_with_mean(df, column):
    """Imputar outliers con la media de la columna"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Calcular la media de los valores no outliers
    mean_value = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)][column].mean()
    
    # Reemplazar outliers con la media
    df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = mean_value
    
    return df

def recomendar():
    """Carga el modelo y genera todas las predicciones"""
    try:
        df_final, df_terminado, df_catalogo, conn, cur = procesar()
        
        # Cargar el modelo colaborativo
        modelo = joblib.load('/content/drive/MyDrive/analitica 3/sistemas_recomendacion/salidas/modelo_colaborativo.joblib')
        
        # Generar predicciones para todos los usuarios
        predset = modelo.trainset.build_anti_testset()
        predictions = modelo.test(predset)
        predictions_df = pd.DataFrame(predictions)
        
        return predictions_df, conn
    except Exception as e:
        print(f"Error al cargar el modelo o generar predicciones: {str(e)}")
        raise e

def recomendaciones(predictions_df, list_user, n_recomend=5):
    """Genera recomendaciones para una lista de usuarios"""
    try:
        # Obtener la conexión a la base de datos
        _, _, _, conn, _ = procesar()
        
        all_recommendations = []
        
        # Filtrar las predicciones para cada usuario y ordenar por la calificación estimada
        for user_id in list_user:
            predictions_userID = predictions_df[predictions_df['uid'] == user_id].\
                                sort_values(by="est", ascending=False).head(n_recomend)
            
            # Seleccionar las columnas necesarias y renombrarlas
            recomendados = predictions_userID[['uid', 'iid', 'r_ui', 'est']]
            recomendados.columns = ['user_id', 'movie_id', 'promedio_rating_real', 'estimacion_rating']
            
            all_recommendations.append(recomendados)
        
        # Concatenar todas las recomendaciones
        recomendaciones_df = pd.concat(all_recommendations, ignore_index=True)
        
        # Guardar las recomendaciones en la base de datos
        recomendaciones_df.to_sql('reco', conn, if_exists="replace", index=False)
        
        # Realizar la consulta SQL para obtener los títulos de las películas
        recomendaciones_df = pd.read_sql('''SELECT a.*, b.title
                                      FROM reco a
                                      LEFT JOIN df_final b
                                      ON a.movie_id = b.movie_id''', conn)
        
        # Eliminar filas duplicadas
        recomendaciones_df = recomendaciones_df.drop_duplicates(subset=['user_id', 'movie_id', 'title'])
        
        return recomendaciones_df
    
    except Exception as e:
        print(f"Error al generar recomendaciones: {str(e)}")
        raise e

# Cargar las predicciones al iniciar (se ejecutará una sola vez)
global predictions_df, conn
predictions_df = None
conn = None

@app.on_event("startup")
async def startup_event():
    global predictions_df, conn
    try:
        print("Cargando modelo y generando predicciones...")
        predictions_df, conn = recomendar()
        print("Modelo cargado y predicciones generadas correctamente")
    except Exception as e:
        print(f"Error al iniciar la API: {str(e)}")

@app.get("/")
def read_root():
    """Endpoint raíz para verificar que la API está funcionando"""
    return {"message": "API de Recomendación de Películas activa"}

@app.post("/recomendar", response_model=List[dict])
def get_recommendations(request: UserRequest):
    """Endpoint para obtener recomendaciones de películas para usuarios específicos"""
    global predictions_df
    
    if predictions_df is None:
        try:
            predictions_df, _ = recomendar()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al cargar el modelo: {str(e)}")
    
    try:
        # Validar que los user_ids estén dentro del rango válido (0-600)
        invalid_users = [user_id for user_id in request.user_ids if user_id < 0 or user_id > 600]
        if invalid_users:
            raise HTTPException(status_code=400, detail=f"IDs de usuario fuera de rango (0-600): {invalid_users}")
        
        # Generar recomendaciones
        recomendados = recomendaciones(predictions_df, request.user_ids, request.num_recommendations)
        
        # Convertir el resultado a un formato adecuado para la API
        result = recomendados.to_dict(orient='records')
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar recomendaciones: {str(e)}")


@app.get("/peliculas-populares/{n}")
def get_popular_movies(n: int):
    """Devuelve las películas más populares basadas en promedio de rating"""
    try:
        _, df_terminado, df_catalogo, _, _ = procesar()
        df_populares = df_catalogo.sort_values(by='promedio_rating', ascending=False).head(n)
        return df_populares[['movie_id', 'title', 'promedio_rating']].to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener películas populares: {str(e)}")
    

@app.post("/recomendar")
def recomendar_peliculas(request: MovieRecommendation = Body(...)):
    try:
        return recomendaciones(request.user_ids, request.num_recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar recomendaciones: {str(e)}")
    

# Función para configurar y ejecutar el servidor en Colab
def run_server_with_ngrok():
    from pyngrok import ngrok
    
    # Configurar pyngrok (puedes usar tu propio token si lo tienes)
    ngrok.set_auth_token("2uKEAm2ZsY5DntLont5EB2jM9dT_4U9WvspehV1cF7iUewXwv")
    
    # Crear un túnel HTTP para el puerto 8000
    public_url = ngrok.connect(8000)
    print(f"URL pública para acceder a la API: {public_url}")
    
    # Iniciar el servidor Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Para ejecutar en Colab
if __name__ == "__main__":
    run_server_with_ngrok()