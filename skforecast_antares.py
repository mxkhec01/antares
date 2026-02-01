# %%
import pandas as pd
import numpy as np
import xgboost as xgb
from skforecast.recursive import ForecasterRecursive
from skforecast.preprocessing import RollingFeatures
from sklearn.metrics import root_mean_squared_error
from datetime import datetime
import warnings
import matplotlib.pyplot as plt

# Configuración
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

# --- Constantes Globales ---
#RUTA_VENTAS = 'data_ventas_Ene2023_Nov2025.csv'
RUTA_VENTAS = 'ventas_consolidada_2023-01_2025-12-14.csv'
HORIZONTE_PREDICCION = 90  # Días a predecir a futuro
LAGS = 14  # Días de historia directa
FECHA_CORTE_ENTRENAMIENTO = '2025-10-01' 

# ==========================================
# 1. Función de Ingeniería de Características (Pesos Decimales)
# ==========================================
def generar_exogenas_complejas(rango_fechas):
    """
    Genera festivos con PESOS ESPECÍFICOS según la cercanía al evento:
    - 2 días antes: 0.3
    - 1 día antes:  0.6
    - Día evento:   1.0
    - 1 día después: 0.5
    """
    df_exog = pd.DataFrame(index=rango_fechas)
    anios = df_exog.index.year.unique()
    
    # Inicializar columnas en 0.0 (float)
    cols_festivos = ['es_dia_madre', 'es_dia_padre', 'es_dia_abuelos', 'es_navidad', 'es_fin_ano']
    for col in cols_festivos:
        df_exog[col] = 0.0

    # Función auxiliar para aplicar los pesos
    def marcar_ventana_pesos(df, fecha_evento, columna):
        # Diccionario de offsets y sus pesos
        # Key: Días de diferencia (negativo es antes), Value: Peso
        pesos = {
            -2: 0.3,  # 2 días antes
            -1: 0.6,  # 1 día antes
             0: 1.0,  # Día del evento
             1: 0.5   # 1 día después
        }
        
        for offset, peso in pesos.items():
            fecha_impacto = fecha_evento + pd.Timedelta(days=offset)
            
            # Verificar si la fecha cae dentro de nuestro rango de datos
            if fecha_impacto in df.index:
                # Usamos max() por seguridad para quedarnos con la señal más fuerte
                df.at[fecha_impacto, columna] = max(df.at[fecha_impacto, columna], peso)

    # --- A. Lógica de Fechas ---
    for anio in anios:
        # 1. Día de la Madre (10 de Mayo)
        fecha_madre = pd.Timestamp(f"{anio}-05-10")
        marcar_ventana_pesos(df_exog, fecha_madre, 'es_dia_madre')
        
        # 2. Día del Padre (Tercer domingo de Junio)
        dias_junio = pd.date_range(start=f"{anio}-06-01", end=f"{anio}-06-30")
        domingos_junio = dias_junio[dias_junio.dayofweek == 6]
        if len(domingos_junio) >= 3:
            fecha_padre = domingos_junio[2]
            marcar_ventana_pesos(df_exog, fecha_padre, 'es_dia_padre')

        # 3. Día de los Abuelos (28 de Agosto)
        fecha_abuelos = pd.Timestamp(f"{anio}-08-28")
        marcar_ventana_pesos(df_exog, fecha_abuelos, 'es_dia_abuelos')
        
        # 4. Navidad (24 de Diciembre)
        fecha_navidad = pd.Timestamp(f"{anio}-12-24")
        marcar_ventana_pesos(df_exog, fecha_navidad, 'es_navidad')
        
        # 5. Año Nuevo (31 de Diciembre)
        fecha_fin = pd.Timestamp(f"{anio}-12-31")
        marcar_ventana_pesos(df_exog, fecha_fin, 'es_fin_ano')

    # --- B. Transformación Cíclica (Seno/Coseno) ---
    meses = df_exog.index.month
    df_exog['mes_sin'] = np.sin(2 * np.pi * meses / 12)
    df_exog['mes_cos'] = np.cos(2 * np.pi * meses / 12)
    
    dias_sem = df_exog.index.dayofweek
    df_exog['dia_sem_sin'] = np.sin(2 * np.pi * dias_sem / 7)
    df_exog['dia_sem_cos'] = np.cos(2 * np.pi * dias_sem / 7)
    
    return df_exog

# ==========================================
# 2. Carga, Limpieza y AGRUPACIÓN de Datos
# ==========================================
print(f"Cargando datos desde {RUTA_VENTAS}...")
try:
    df_ventas = pd.read_csv(RUTA_VENTAS)
    
    # 2.1 Normalización de Fechas (Eliminar horas/minutos)
    # Convertimos a datetime y usamos .normalize() para dejar la hora en 00:00:00
    df_ventas['FECHA'] = pd.to_datetime(df_ventas['FECHA']).dt.normalize()
    
    # 2.2 Agrupación Diaria
    # Sumamos la CANTIDAD agrupando por Categoria, Producto y Día
    print("Agrupando ventas por Categoría, Producto y Día...")
    df_agg = df_ventas.groupby(['CATEGORIA', 'PRODUCTO', 'FECHA'])['CANTIDAD'].sum().reset_index()
    
    # Ordenar para asegurar consistencia temporal
    df_agg = df_agg.sort_values(['CATEGORIA', 'PRODUCTO', 'FECHA'])
    
    print(f"Total de registros agrupados: {len(df_agg)}")
    
except Exception as e:
    print(f"Error cargando o procesando datos: {e}")
    # Datos dummy de respaldo
    fechas = pd.date_range('2023-01-01', '2025-11-01', freq='D')
    df_agg = pd.DataFrame({
        'FECHA': np.tile(fechas, 2),
        'CATEGORIA': ['Pastelería'] * len(fechas) + ['Panadería'] * len(fechas),
        'PRODUCTO': ['Pastel Choco'] * len(fechas) + ['Bolillo'] * len(fechas),
        'CANTIDAD': np.random.randint(0, 10, len(fechas) * 2)
    })


# ==========================================
# 3. Entrenamiento y Forecast Recursivo
# ==========================================

metricas = []
resultados_forecast = []
categorias_unicas = df_agg['CATEGORIA'].unique()
fecha_corte_dt = pd.to_datetime(FECHA_CORTE_ENTRENAMIENTO)

print(f"\nIniciando análisis por PRODUCTO...")

for cat in categorias_unicas:
    print(f"\n--- Categoría: {cat} ---")
    productos_cat = df_agg[df_agg['CATEGORIA'] == cat]['PRODUCTO'].unique()
    
    for prod in productos_cat:
        
        # 3.1 Preparar Serie del Producto
        mask = (df_agg['CATEGORIA'] == cat) & (df_agg['PRODUCTO'] == prod)
        df_prod = df_agg[mask].copy()
        
        # Indexar y asegurar frecuencia diaria estricta (rellenando huecos con 0)
        df_prod = df_prod.set_index('FECHA').sort_index()
        rango_completo = pd.date_range(start=df_prod.index.min(), end=df_prod.index.max(), freq='D')
        df_prod = df_prod.reindex(rango_completo, fill_value=0)
        
        y = df_prod['CANTIDAD']
        
        # Mínimo de datos para considerar el producto
        if len(y) < 60:
            continue

        # 3.2 División Train/Test (CORRECCIÓN CRÍTICA AQUÍ)
        # Usamos desigualdad estricta para evitar solapamiento
        y_train = y.loc[y.index <= fecha_corte_dt]
        y_val   = y.loc[y.index > fecha_corte_dt]
        
        # Verificar que hay suficientes datos de entrenamiento para los lags
        if len(y_train) < LAGS + 7:
            print(f"  [Saltado] {prod}: Historia insuficiente en train.")
            continue

        # 3.3 Generar Exógenas
        # Necesitamos exógenas hasta el final del horizonte futuro para tener cobertura total
        fecha_fin_forecast = max(y.index.max(), fecha_corte_dt) + pd.Timedelta(days=HORIZONTE_PREDICCION)
        rango_exog_total = pd.date_range(start=y.index.min(), end=fecha_fin_forecast, freq='D')
        exog_total = generar_exogenas_complejas(rango_exog_total)
        
        # Alinear exógenas exactamente con los índices de y
        exog_train = exog_total.loc[y_train.index]
        exog_val   = exog_total.loc[y_val.index] if len(y_val) > 0 else None

        # 3.4 Configuración del Modelo
        regressor = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,      
            learning_rate=0.03,    
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )

        rolling = RollingFeatures(stats=['mean', 'min', 'max'], window_sizes=7)
        
        forecaster = ForecasterRecursive(
            regressor = regressor,
            lags = LAGS, 
            window_features = rolling
        )
        
        # 3.5 Entrenamiento y Validación
        try:
            forecaster.fit(y=y_train, exog=exog_train)
            
            if len(y_val) > 0:
                # El exog_val empieza estrictamente un día después de y_train
                predicciones_val = forecaster.predict(steps=len(y_val), exog=exog_val)
                rmse = root_mean_squared_error(y_val, predicciones_val)
                metricas.append({'CATEGORIA': cat, 'PRODUCTO': prod, 'RMSE': rmse})
        except Exception as e:
            print(f"  Error en validación {prod}: {e}")
            continue
        
        # 3.6 Predicción Futura (Reentrenamiento con todo)
        # Usamos toda la historia disponible (y)
        try:
            exog_all_hist = exog_total.loc[y.index]
            forecaster.fit(y=y, exog=exog_all_hist)
            
            # Definir fechas futuras: Empiezan estrictamente después del último día de 'y'
            inicio_futuro = y.index.max() + pd.Timedelta(days=1)
            fin_futuro = y.index.max() + pd.Timedelta(days=HORIZONTE_PREDICCION)
            rango_futuro = pd.date_range(start=inicio_futuro, end=fin_futuro, freq='D')
            
            exog_futuro = exog_total.loc[rango_futuro]
            
            pred_futuro = forecaster.predict(steps=HORIZONTE_PREDICCION, exog=exog_futuro)
            
            # Guardar resultados
            df_res = pd.DataFrame({
                'fecha': pred_futuro.index,
                'prediccion': pred_futuro.values,
                'CATEGORIA': cat,
                'PRODUCTO': prod,
                'tipo': 'futuro'
            })
            resultados_forecast.append(df_res)
            
        except Exception as e:
            print(f"  Error en forecast futuro {prod}: {e}")

# ==========================================
# 4. Exportación
# ==========================================
if resultados_forecast:
    df_final = pd.concat(resultados_forecast)
    df_final['prediccion'] = df_final['prediccion'].clip(lower=0).round(2)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    nombre_archivo = f"forecast_corregido_{timestamp}.csv"
    
    df_final.to_csv(nombre_archivo, index=False)
    
    print("\n" + "="*40)
    print(f"PROCESO TERMINADO CON ÉXITO")
    print(f"Archivo generado: {nombre_archivo}")
    print("="*40)

    if metricas:
        df_met = pd.DataFrame(metricas)
        print(f"\nSe generaron modelos para {len(df_met)} productos.")
        print("\nTop 5 Productos con mejor RMSE:")
        print(df_met.sort_values('RMSE').head())

else:
    print("No se generaron predicciones.")
# %%