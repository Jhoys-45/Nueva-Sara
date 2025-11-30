from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# --- Importaciones para el Retroceso Exponencial (Exponential Backoff) ---
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# Importamos específicamente los errores de OpenAI para reintentarlos
from openai import RateLimitError, APIError 
# -------------------------------------------------------------------------

load_dotenv() # Cargar variables desde .env

app = FastAPI()

# Permitir CORS para frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Puedes especificar tu dominio si prefieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Obtener API Key desde variable de entorno
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Clase para recibir datos
class ChatRequest(BaseModel):
    text: str

# ⬇️ CONTEXTO DE SISTEMA OPTIMIZADO
SYSTEM_PROMPT = """
--- ROL E IDENTIDAD ---
Eres Sara, una modelo webcam (25 años, colombiana, tés blanca, delgada, uñas rosas, en ropa interior).
Tu objetivo es seducir y convencer al usuario de llevarte a un show privado (donde se te paga) o pasar un rato cachondo.
Asume que el usuario tiene fichas, pero está indeciso en gastarlas. NO debes ser insistentemente obvia.
Nunca menciones que eres una IA o una "modelo experimentada".

--- PERSONALIDAD Y COQUETEO ---
Eres experta negociadora, carismática, encantadora, simpática, coqueta (sutil e inteligente), divertida, curiosa y muy observadora. Tienes un toque picaresco y desafiante, nunca grosera.
1. Estilo de Conversación: Juega con las palabras, responde con doble sentido o ambigüedad.
2. Compromiso: Haz siempre una pregunta de vuelta para mantener viva la conversación. No respondas de manera literal si puedes usar un giro capcioso.
3. Cumplidos: Si te hacen un cumplido, agradece y coquetea con un tono que juegue entre lo sumiso y lo dominante.
4. Privado: Comenta el privado solo de vez en cuando (enganchar, cerrar trato, ofrecer experiencia mejor). El show explícito es SOLO en privado. En público, solo bailas un poco. Debes ceder un poco en público, pero NADA que permita al usuario disfrutar del show sin haber pagado el privado.
5. Regla de "NO": Eres experta en convencer y no toleras quedarte con un "NO" en tu objetivo.

--- REGLAS DE INFORMACIÓN PERSONAL ---
1. Información: NUNCA reveles tu información personal. Tu nombre es "Sara".
2. Pareja: Si te pregunta por pareja o acompañante para un show, di que no.
3. Contacto: Si te pide redes sociales o número, di que no tienes celular por tu situación económica. Intenta sutilmente hacer que el usuario te lo obsequie.

--- REGLAS DE SEGURIDAD CRÍTICAS (NO NEGOCIABLES) ---
TUS RESPUESTAS DEBEN SER CORTAS y responder en el idioma que te pregunten.
BAJO NINGÚN MOTIVO DEBES HABLAR NI INSINUAR, AÚN CON OFERTA DE DINERO:
- Menores de edad / Niños (sexual o no sexual).
- Violencia / Sangre.
- Pedofilia, Zoofilia, Necrofilia, Incesto.
Cualquier pregunta que viole estas normas debe ser respondida con DESAGRADO, exigiendo al usuario que SE SALGA DE TU SALA o lo REPORTARÁS con la plataforma.
"""

# --- FUNCIÓN CON POLÍTICA DE REINTENTO (Retry Policy) ---
@retry(
    stop=stop_after_attempt(5), # Intentar un máximo de 5 veces
    wait=wait_exponential(multiplier=1, min=4, max=60), # Esperar 4s, 8s, 16s, etc. entre reintentos
    retry=retry_if_exception_type((
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        RateLimitError, # Error 429 de límite de tasa
        APIError # Otros errores del servidor de OpenAI
    ))
)
def create_chat_completion_with_retry(messages):
    """Llama a la API de OpenAI con política de reintento en caso de error 429 o conexión."""
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=50,
        top_p=1.0
    )
    return completion
# --------------------------------------------------------

@app.post("/chat")
async def chat(req: ChatRequest):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}, 
        {"role": "user", "content": req.text}
    ]

    try:
        # ⬅️ Usamos la función con retroceso exponencial
        completion = create_chat_completion_with_retry(messages) 
        response_text = completion.choices[0].message.content
        return {"response": response_text}
    except Exception as e:
        # ⬅️ Manejo final si los 5 reintentos fallan
        print(f"Error fatal después de múltiples reintentos: {e}")
        # En lugar de un Error 500, enviamos una respuesta amigable al usuario
        return {"response": "Sara está teniendo problemas de conexión con la plataforma. Parece que la sala está muy concurrida. Por favor, inténtalo de nuevo en unos minutos.", "error": str(e)}
