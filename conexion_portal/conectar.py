import requests
from bs4 import BeautifulSoup
import json
import urllib.parse
import os
import pickle
from dotenv import load_dotenv

load_dotenv()


# Configuración de URLs y credenciales
BASE_URL = "https://laluz.antarespasteleria.com"
LOGIN_URL = f"{BASE_URL}/nova/login"
# Reemplaza con la URL específica del recurso JSON que deseas descargar
API_URL = f"{BASE_URL}/reportes/ventas/get"

EMAIL = os.getenv('ANTARES_EMAIL')
PASSWORD = os.getenv('ANTARES_PASSWORD')
COOKIE_FILE = 'session_cookies.pkl'

if not EMAIL or not PASSWORD:
    raise ValueError("Las credenciales (ANTARES_EMAIL, ANTARES_PASSWORD) no están configuradas en el archivo .env")


# 1. Iniciar una sesión persistente para mantener las cookies [1, 3]
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
})

def save_cookies(session):
    """Guarda las cookies de la sesión en un archivo."""
    try:
        with open(COOKIE_FILE, 'wb') as f:
            pickle.dump(session.cookies, f)
        # print("Cookies guardadas.")
    except Exception as e:
        print(f"Error guardando cookies: {e}")

def load_cookies(session):
    """Carga las cookies de la sesión desde un archivo si existe."""
    if os.path.exists(COOKIE_FILE):
        try:
            with open(COOKIE_FILE, 'rb') as f:
                session.cookies.update(pickle.load(f))
            return True
        except Exception as e:
            print(f"Error cargando cookies: {e}")
    return False

def download_nova_json():
    api_headers = {
        'X-Inertia': 'true',
        'X-Requested-With': 'XMLHttpRequest',
        'Accept': 'application/json'
    }
    
    params = {
        'fecha_inicio': '2026-01-01 06:00:00',
        'fecha_fin': '2026-02-02 05:59:00',
        'cliente': '',
        'order': '',
        'facturado': '',
        'categoria': '',
        'cliente_interno': '',
        'canal_venta_id': '',
        'uuid': '',
        'folio': '',
        'serie': '',
        'socio': '',
        'tarjeta': ''
    }

    # Intentar usar sesión en caché primero
    if load_cookies(session):
        print("Intentando usar sesión en caché...")
        try:
            # Intentar obtener el recurso directamente
            json_response = session.get(API_URL, params=params, headers=api_headers)
            
            # Verificar si el token sigue siendo válido (no redirección a login y status 200)
            if json_response.status_code == 200 and "login" not in json_response.url and json_response.headers.get('Content-Type', '').startswith('application/json'):
                print("Sesión en caché válida.")
                guardar_json(json_response.json())
                return
            else:
                print("Sesión en caché expirada o inválida. Re-autenticando...")
        except Exception as e:
            print(f"Error al probar sesión en caché: {e}. Re-autenticando...")
    
    # Si no hay caché o falló, procedemos con el login
    perform_login_and_download(params, api_headers)

def perform_login_and_download(params, api_headers):
    try:
        # 2. Primer paso: GET a la página de login para obtener el token CSRF inicial 
        print("Iniciando proceso de login...")
        response = session.get(LOGIN_URL)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extraer el valor del campo oculto '_token' [4, 5]
        csrf_token = soup.find('input', {'name': '_token'})
        if not csrf_token:
            print("No se pudo encontrar el token CSRF. Verifica la URL.")
            return
        
        token_value = csrf_token.get('value')
        
        # 3. Segundo paso: Ejecutar el Login (POST) [1, 6]
        payload = {
            'email': EMAIL,
            'password': PASSWORD,
            '_token': token_value
        }
        
        # Es vital enviar el token X-XSRF-TOKEN en la cabecera si el sitio usa Nova 4+ [6, 7]
        if 'XSRF-TOKEN' in session.cookies:
            # Bug corregido: Acceder específicamente a la cookie por su nombre
            decoded_token = urllib.parse.unquote(session.cookies['XSRF-TOKEN'])
            session.headers.update({'X-XSRF-TOKEN': decoded_token})

        login_response = session.post(LOGIN_URL, data=payload)
        
        # Verificar si el login fue exitoso (usualmente redirige al dashboard)
        if login_response.status_code == 200 and "login" not in login_response.url:
            print("Autenticación exitosa.")
            save_cookies(session) # Guardar cookies para la próxima vez
            
            json_response = session.get(API_URL, params=params, headers=api_headers)
            
            if json_response.status_code == 200:
                guardar_json(json_response.json())
            else:
                print(f"Error al descargar JSON: {json_response.status_code}")
        else:
            print("Fallo en el inicio de sesión. Revisa tus credenciales.")
            
    except Exception as e:
        print(f"Ocurrió un error: {e}")

def guardar_json(data):
    with open('datos_descargados.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("Archivo JSON descargado correctamente.")

if __name__ == "__main__":
    download_nova_json()