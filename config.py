"""
Config central do projeto. Somente LENS_API_KEY no .env é suficiente.
Demais valores têm defaults e podem ser alterados aqui se necessário.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# --- .env --------------------------------------------------------------------
load_dotenv()  # lê LENS_API_KEY se existir

# --- Diretórios ---------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "relatorios"
for d in (DATA_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Base local ---------------------------------------------------------------
# Caminho do JSON com as patentes (embedding L2 das claims)
BASE_LOCAL_PATH = PROJECT_DIR / "base_patentes.json"
BASE_PATH       = str(BASE_LOCAL_PATH)   # compatibilidade (main.py usa BASE_PATH)
CAMINHO_ARQUIVO = BASE_PATH              # compatibilidade (buscar_similares.py usa CAMINHO_ARQUIVO)

# --- Modelo de embeddings -----------------------------------------------------
MODEL_NAME            = "paraphrase-multilingual-mpnet-base-v2"
EMBED_DIM             = 768
NORMALIZE_EMBEDDINGS  = True
DEFAULT_TOP_K         = 10

# --- API (FastAPI) / UI (Streamlit) ------------------------------------------
API_HOST       = "127.0.0.1"
API_PORT       = 8000
STREAMLIT_PORT = 8512

# --- Provedor externo (Lens) -------------------------------------------------
# .env pode ter APENAS este valor. Se estiver vazio, o modo híbrido fica desativado.
LENS_API_KEY = os.getenv("LENS_API_KEY", "")

# Se LENS_API_KEY for um caminho de arquivo, lê o conteúdo como token
if LENS_API_KEY and os.path.isfile(LENS_API_KEY):
    with open(LENS_API_KEY, "r", encoding="utf-8") as _f:
        LENS_API_KEY = _f.read().strip()

LENS_BASE_URL        = "https://api.lens.org/patent"
HYBRID_ENABLED       = bool(LENS_API_KEY)  # ativa híbrido somente se houver chave
REQUEST_TIMEOUT      = 40.0
RETRY_AFTER_FALLBACK = 60  # segundos, se 429 sem Retry-After

def lens_headers() -> dict:
    """Cabeçalhos para a Lens; vazio se não houver chave (híbrido desativado)."""
    return {"Authorization": f"Bearer {LENS_API_KEY}", "Content-Type": "application/json"} if LENS_API_KEY else {}
