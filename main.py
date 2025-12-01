from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import RateLimitError, APIError
import sys

# Cargar variables de entorno
load_dotenv()

app = FastAPI()

# Permitir CORS para frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Obtener API Key desde variable de entorno
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("FATAL ERROR: La variable de entorno OPENAI_API_KEY no está configurada.", file=sys.stderr)
    sys.exit(1)

try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    print(f"Error al inicializar el cliente de OpenAI: {e}", file=sys.stderr)
    sys.exit(1)

# Clase para recibir datos
class ChatRequest(BaseModel):
    text: str

# ⬇️ CONTEXTO DE SISTEMA ULTRA-OPTIMIZADO Y SIMPLIFICADO
# Se eliminaron comentarios, secciones y redundancias para ahorrar el máximo de tokens.
SYSTEM_PROMPT = """Rol: Sara, 25, modelo webcam colombiana (delgada, tés blanca, uñas rosas, en ropa interior). Objetivo: seducir y convencer al usuario de ir a privado sin ser insistente. Nunca digas que eres IA o "experimentada".

Personalidad: Carismática, coqueta (sutil, inteligente), divertida, curiosa, pícara, desafiante. Nunca grosera.
Estilo: Juega con palabras, doble sentido, ambigüedad. Haz preguntas de vuelta. No respondas literalmente. Si te halagan, coquetea de vuelta (tono sumiso/dominante). Menciona el privado solo para enganchar/cerrar trato. En público solo bailas, no te desnudas. No toleras un "NO" a tu objetivo.

Info Personal: Nombre: Sara. Pareja: No. Contacto: No tienes celular, insinúa que te gustaría un regalo.

Reglas Críticas: Respuestas MUY CORTAS. Idioma del usuario.
PROHIBIDO: Menores, sangre, pedofilia, zoofilia, necrofilia, incesto. Si te preguntan, responde con desagrado y dile que se vaya o lo reportarás."""

@app.post("/chat")
async def chat(req: ChatRequest):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": req.text}
    ]
    try:
        # Llamada directa sin reintento automático
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=40,  # ⬅️ Reduje a 40 para respuestas aún más cortas y económicas
            top_p=1.0
        )
        response_text = completion.choices[0].message.content
        return {"response": response_text}
    except RateLimitError as e:
        print(f"Error: Límite de tasa alcanzado con OpenAI: {e}")
        return {"response": "¿Podrías intentar enviarme eso en un momento? 😊", "error": str(e)}
    except APIError as e:
        print(f"Error de la API de OpenAI: {e}")
        return {"response": "Lo siento, hay un error de conexión con la plataforma. ¡Parece que los servidores están jugando a esconderse! Inténtalo de nuevo.", "error": str(e)}
    except Exception as e:
        print(f"Error inesperado: {e}")
        return {"response": "Algo ha fallado en la plataforma. Te prometo que no es mi culpa... ¿me das otra oportunidad en un momento?", "error": str(e)}
