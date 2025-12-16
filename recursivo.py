# %%
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import warnings
from datetime import datetime, timedelta
import shap

warnings.filterwarnings('ignore')

# --- Constantes Globales ---
RUTA_VENTAS = 'data_ventas_Ene2023_Nov2025.csv'
RUTA_VENTAS = 'ventas_consolidada_2023-01_2025-12-14.csv'
RUTA_CALENDARIO = 'calendario.csv'
# NOTA: Para forecast futuro, usamos todos los datos disponibles, no cortamos entrenamiento
COLUMNA_OBJETIVO = 'UNIDADES_VENDIDAS'
DIAS_A_PREDECIR = 20

print(f"Cargando datos de ventas desde {RUTA_VENTAS}...")
try:
    df_ventas = pd.read_csv(RUTA_VENTAS)
except FileNotFoundError:
    # Para propósitos de este script modificado, asumimos que existe o fallará
    raise FileNotFoundError(f"Error: No se encontró el archivo {RUTA_VENTAS}")

# 1. Limpieza y Conversión Inicial
df_ventas['FECHA'] = pd.to_datetime(df_ventas['FECHA']).dt.date
df_ventas = df_ventas[df_ventas['CATEGORIA']!='General']
df_ventas = df_ventas[df_ventas['CATEGORIA']!='Pedidos Especiales']

# 2. Agregación a nivel de Producto-Día
print("Agregando transacciones a nivel (Fecha, Categoria, Producto)...")
df_agg = df_ventas.groupby(
    [df_ventas['FECHA'], 'CATEGORIA', 'PRODUCTO']
).agg(
    TOTAL_VENTAS=('TOTAL', 'sum'),
    UNIDADES_VENDIDAS=('CANTIDAD', 'sum')
).reset_index()

df_agg.rename(columns={'FECHA': 'fecha'}, inplace=True)

# 3. Creación del Panel de Series Temporales (DataFrame Completo)
print("Creando panel de series temporales completo (llenando ceros)...")
items_validos = df_agg[['PRODUCTO','CATEGORIA']].drop_duplicates()
fecha_min = df_agg['fecha'].min()
fecha_max = df_agg['fecha'].max()
rango_fechas = pd.date_range(start=fecha_min, end=fecha_max, freq='D')

malla = pd.DataFrame({'fecha': rango_fechas}).merge(items_validos, how='cross')
indice_completo = pd.MultiIndex.from_frame(malla)

df_completo = (df_agg.set_index(['fecha', 'PRODUCTO','CATEGORIA'])
                 .reindex(indice_completo, fill_value=0)
                 .reset_index())

# --- FUNCIONES DE INGENIERÍA DE CARACTERÍSTICAS ---

def crear_caracteristicas_calendario(df):
    df_feat = df.copy()
    df_feat['fecha'] = pd.to_datetime(df_feat['fecha']) # Asegurar datetime
    df_feat['dia_de_la_semana'] = df_feat['fecha'].dt.dayofweek
    df_feat['dia_del_mes'] = df_feat['fecha'].dt.day
    df_feat['mes'] = df_feat['fecha'].dt.month
    df_feat['anio'] = df_feat['fecha'].dt.year
    return df_feat

def crear_caracteristicas_ciclicas(df):
    df_feat = df.copy()
    df_feat['dia_semana_sin'] = np.sin(2 * np.pi * df_feat['dia_de_la_semana'] / 7) 
    df_feat['dia_semana_cos'] = np.cos(2 * np.pi * df_feat['dia_de_la_semana'] / 7)
    df_feat['mes_sin'] = np.sin(2 * np.pi * df_feat['mes'] / 12) 
    df_feat['mes_cos'] = np.cos(2 * np.pi * df_feat['mes'] / 12)
    df_feat['dia_mes_sin'] = np.sin(2 * np.pi * df_feat['dia_del_mes'] / 31) 
    df_feat['dia_mes_cos'] = np.cos(2 * np.pi * df_feat['dia_del_mes'] / 31) 
    return df_feat

def _aplicar_ventana_ponderada(df_features, fechas_base, nombre_col, ponderaciones):
    if nombre_col not in df_features.columns:
        df_features[nombre_col] = 0.0
    fechas_base_dt = pd.to_datetime(fechas_base, format='%Y-%m-%d')
    for fecha_base in fechas_base_dt:
        for offset_dias, peso in ponderaciones.items():
            fecha_objetivo = fecha_base + pd.Timedelta(days=offset_dias)
            df_features.loc[df_features['fecha']==fecha_objetivo, nombre_col] = peso
    return df_features

def crear_caracteristicas_eventos(df, ruta_calendario):
    df_feat = df.copy()
    try:
        df_calendario = pd.read_csv(ruta_calendario) 
        df_calendario['Fecha'] = pd.to_datetime(df_calendario['Fecha'], format='%d/%m/%Y')
        # Crear conjuntos de fechas para búsqueda rápida
        fechas_puente = set(df_calendario[df_calendario['Tipo'] == 'Oficial - Puente']['Fecha'])
        fechas_nomina = set(df_calendario[df_calendario['Tipo'] == 'Nómina']['Fecha'])
    except:
        fechas_puente = set()
        fechas_nomina = set()

    df_feat['es_puente'] = df_feat['fecha'].isin(fechas_puente).astype(int)
    df_feat['es_nomina'] = df_feat['fecha'].isin(fechas_nomina).astype(int)

    # Eventos Clave (Misma lógica original)
    ponderaciones_clave = {-3: 0.3, -2: 0.6, -1: 0.9, 0: 1.0, 1: 0.5} 
    fechas_madre = ['2023-05-10', '2024-05-10', '2025-05-10', '2026-05-10'] # Añadido 2026 por si acaso
    df_feat = _aplicar_ventana_ponderada(df_feat, fechas_madre, 'impacto_dia_madre', ponderaciones_clave)
      # Día del Padre (3er Dom Junio)
    fechas_padre = ['2023-06-18', '2024-06-16', '2025-06-15','2026-06-21']
    df_feat = _aplicar_ventana_ponderada(df_feat, fechas_padre, 'impacto_dia_padre', ponderaciones_clave)
    
    # Día de los Abuelos (28 de Agosto)
    fechas_abuelos = ['2023-08-28', '2024-08-28', '2025-08-28', '2026-08-28']
    df_feat = _aplicar_ventana_ponderada(df_feat, fechas_abuelos, 'impacto_dia_abuelos', ponderaciones_clave)
    
     # Día de los Abuelos (28 de Agosto)
    fechas_navidad = ['2023-12-24', '2024-12-24', '2025-12-24', '2026-12-24']
    df_feat = _aplicar_ventana_ponderada(df_feat, fechas_navidad, 'impacto_dia_navidad', ponderaciones_clave)
    
    
    fechas_fin_anio = ['2023-12-31', '2024-12-31', '2025-12-31', '2026-12-31']
    df_feat = _aplicar_ventana_ponderada(df_feat, fechas_fin_anio, 'impacto_dia_fin', ponderaciones_clave)
    return df_feat

def crear_estadisticas_por_producto(df):
    # NOTA: Esta función es computacionalmente intensiva en el bucle recursivo
    # pero necesaria para actualizar los lags con las nuevas predicciones.
    df = df.sort_values(by=['PRODUCTO', 'fecha'])
    
    # Lags
    df['lag_7d_total'] = df.groupby(['PRODUCTO'])['TOTAL_VENTAS'].shift(7)
    df['lag_14d_total'] = df.groupby(['PRODUCTO'])['TOTAL_VENTAS'].shift(14)
    df['lag_28d_total'] = df.groupby(['PRODUCTO'])['TOTAL_VENTAS'].shift(21)
    df['lag_1d_total'] = df.groupby(['PRODUCTO'])['TOTAL_VENTAS'].shift(1)

    # Rolling
    # shift(1) es crucial para no usar el dato actual
    g = df.groupby(['PRODUCTO'])['TOTAL_VENTAS'].shift(1)
    df['rolling_mean_7d_total'] = g.rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
    df['rolling_mean_14d_total'] = g.rolling(window=14, min_periods=1).mean().reset_index(0, drop=True)
    df['rolling_mean_28d_total'] = g.rolling(window=28, min_periods=1).mean().reset_index(0, drop=True)
    df['rolling_std_7d_total'] = g.rolling(window=7, min_periods=1).std().reset_index(0, drop=True)
    
    df['ratio_momentum_7_28'] = df['rolling_mean_7d_total'] / df['rolling_mean_28d_total']
    
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    return df

# --- PREPARACIÓN INICIAL DE DATOS ---
df_features = crear_caracteristicas_calendario(df_completo)
df_features = crear_caracteristicas_ciclicas(df_features)
df_features = crear_caracteristicas_eventos(df_features, RUTA_CALENDARIO)
df_features = crear_estadisticas_por_producto(df_features)

# --- ENTRENAMIENTO DEL MODELO (CON TODOS LOS DATOS) ---

def entrenar_modelo_final(df_features, categoria):
    print(f"\n--- Entrenando Modelo Final para: {categoria} ---")
    
    df_cat = df_features[df_features['CATEGORIA'] == categoria].copy()
    
    # Manejo de categóricas
    categorical_cols = ['CATEGORIA','PRODUCTO']
    for col in categorical_cols:
        df_cat[col] = df_cat[col].astype('category')
        # Creamos los códigos explícitamente y los guardamos como numéricos o categoría
        # Para XGBoost enable_categorical=True, pasamos el tipo 'category'
    
    # Definir columnas
    cols_objetivo = [COLUMNA_OBJETIVO]
    cols_ignorar = ['prediccion', 'real', 'TOTAL_VENTAS'] 
    
    features_cols = [c for c in df_cat.columns if c not in cols_objetivo and c not in cols_ignorar and c != 'fecha']
    
    X = df_cat[features_cols]
    y = df_cat[cols_objetivo]

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=2000, 
        learning_rate=0.01, # LR un poco más alto para reentrenamiento completo
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        tree_method="approx", 
        enable_categorical=True,
        n_jobs=-1
    )
    
    model.fit(X, y, verbose=False)
    
    return model, features_cols

# Diccionario para guardar modelos y lista de features usada
modelos_por_categoria = {}
features_por_categoria = {}
categorias_principales = df_features['CATEGORIA'].unique()

for categoria in categorias_principales:
    if categoria == 0: continue
    model, feats = entrenar_modelo_final(df_features, categoria)
    modelos_por_categoria[categoria] = model
    features_por_categoria[categoria] = feats

print("\n--- Entrenamiento completado. Iniciando Forecast Recursivo ---")

# %%
# --- FORECAST RECURSIVO ---

# 1. Definir punto de partida
ultima_fecha = df_features['fecha'].max()
print(f"Última fecha real: {ultima_fecha}")

# Dataframe base para ir agregando el futuro
df_history = df_features.copy()

# Guardamos la lista de productos y categorías únicos para expandir el frame
items_unicos = df_history[['PRODUCTO', 'CATEGORIA']].drop_duplicates()

for i in range(1, DIAS_A_PREDECIR + 1):
    fecha_futura = ultima_fecha + timedelta(days=i)
    print(f"Prediciendo día {i}/{DIAS_A_PREDECIR}: {fecha_futura}")
    
    # 2. Crear el esqueleto para el nuevo día
    df_dia_futuro = items_unicos.copy()
    df_dia_futuro['fecha'] = fecha_futura
    
    # Inicializar columnas vacías necesarias para concatenar
    df_dia_futuro[COLUMNA_OBJETIVO] = 0 # Inicialmente 0, aquí pondremos la predicción
    df_dia_futuro['TOTAL_VENTAS'] = 0 # Asumiremos temporalmente que TOTAL = UNIDADES (o estimar precio promedio)
    
    # 3. Concatenar al historial (temporalmente con ventas 0)
    # Convertimos fecha a datetime para que coincida con el df principal
    df_dia_futuro['fecha'] = pd.to_datetime(df_dia_futuro['fecha'])
    df_history = pd.concat([df_history, df_dia_futuro], ignore_index=True)
    
    # 4. Recalcular características (Esto actualiza lags y rolling con las filas nuevas)
    # AVISO: Recalcular todo el DF es lento, pero seguro. 
    # Para optimizar, se podría calcular solo la cola, pero es complejo con rolling.
    df_history = crear_caracteristicas_calendario(df_history)
    df_history = crear_caracteristicas_ciclicas(df_history)
    df_history = crear_caracteristicas_eventos(df_history, RUTA_CALENDARIO)
    df_history = crear_estadisticas_por_producto(df_history)
    
    # 5. Predecir solo para la fecha futura
    # Filtramos solo las filas del día actual que estamos prediciendo
    mask_dia_actual = df_history['fecha'] == fecha_futura
    
    predicciones_dia = []
    
    for categoria in categorias_principales:
        if categoria not in modelos_por_categoria: continue
        
        # Filtrar por categoría y día
        mask_cat = (df_history['CATEGORIA'] == categoria) & mask_dia_actual
        df_subset = df_history[mask_cat].copy()
        
        if len(df_subset) == 0: continue
        
        # Preparar features
        # Importante: Asegurar tipos category para que coincida con entrenamiento
        for col in ['CATEGORIA', 'PRODUCTO']:
            df_subset[col] = df_subset[col].astype('category')
            
        cols_modelo = features_por_categoria[categoria]
        X_future = df_subset[cols_modelo]
        
        # Predecir
        model = modelos_por_categoria[categoria]
        preds = model.predict(X_future)
        
        # Guardar predicciones en el dataframe principal
        # Las predicciones negativas se convierten en 0
        preds = np.maximum(preds, 0)
        
        # Actualizamos df_history con la predicción
        df_history.loc[mask_cat & mask_dia_actual, COLUMNA_OBJETIVO] = preds
        # Asumimos que TOTAL_VENTAS = UNIDADES_VENDIDAS (o multiplicar por precio promedio si se tuviera)
        # Para mantener el rolling mean funcionando, necesitamos llenar TOTAL_VENTAS
        # Si tienes precio promedio por producto, úsalo aquí: preds * precio_promedio
        df_history.loc[mask_cat & mask_dia_actual, 'TOTAL_VENTAS'] = preds 

# --- RESULTADOS FINALES Y FORMATO PIVOTE ---
print("\nGenerando archivos de salida...")

# 1. Filtrar solo los días futuros y limpiar columnas
df_forecast_final = df_history[df_history['fecha'] > ultima_fecha].copy()
df_forecast_final = df_forecast_final[['fecha', 'CATEGORIA', 'PRODUCTO', COLUMNA_OBJETIVO]]
df_forecast_final.rename(columns={COLUMNA_OBJETIVO: 'PREDICCION_UNIDADES'}, inplace=True)

# 2. Generar archivo original (Formato largo: Base de datos)
nombre_archivo_largo = f'forecast_long_{datetime.now().strftime("%Y%m%d")}.csv'
df_forecast_final.to_csv(nombre_archivo_largo, index=False)
print(f"Archivo formato largo guardado: {nombre_archivo_largo}")

# 3. Generar archivo PIVOTE (Formato solicitado: Fechas como columnas)
# index=['CATEGORIA', 'PRODUCTO'] asegura que tengas un renglón por producto (manteniendo su categoría)
# columns='fecha' convierte cada fecha única en una columna nueva
# values='PREDICCION_UNIDADES' es lo que rellenará las celdas
df_pivot = df_forecast_final.pivot_table(
    index=['CATEGORIA', 'PRODUCTO'],
    columns='fecha',
    values='PREDICCION_UNIDADES',
    aggfunc='sum' # Suma por seguridad, aunque debería haber solo 1 dato por celda
).fillna(0)

# Formatear las columnas de fecha para que sean strings legibles (YYYY-MM-DD)
df_pivot.columns = [col.strftime('%Y-%m-%d') for col in df_pivot.columns]

# Resetear el índice para que CATEGORIA y PRODUCTO sean columnas normales en el Excel
df_pivot = df_pivot.reset_index()

# Guardar archivo pivote
nombre_archivo_pivot = f'forecast_pivot_por_fechas_{datetime.now().strftime("%Y%m%d")}.csv'
df_pivot.to_csv(nombre_archivo_pivot, index=False)

print(f"Archivo formato PIVOTE guardado: {nombre_archivo_pivot}")
print("\nVista previa del formato pivote:")
print(df_pivot.head())