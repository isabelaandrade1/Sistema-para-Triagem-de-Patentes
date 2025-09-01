# limpar_texto.py — limpeza multilíngue, Unicode-safe, sem downloads em import
import re
from typing import Set

# --------- detecção de idioma (opcional) ----------
try:
    from langdetect import detect  # pip install langdetect (opcional)
except Exception:
    detect = None  # sem detecção -> seguimos sem remover stopwords específicas

def _detect_lang(text: str):
    if detect is None:
        return None
    try:
        return detect(text)  # ex.: 'en', 'pt', 'es', 'zh-cn'...
    except Exception:
        return None

# --------- stopwords por idioma (via NLTK se disponível) ----------
def _stopwords_for(lang_code: str) -> Set[str]:
    """
    Tenta carregar stopwords do NLTK para o idioma detectado.
    Se não encontrar, retorna conjunto vazio (ou seja, não remove stopwords).
    """
    try:
        from nltk.corpus import stopwords
        # Mapa de códigos ISO -> nomes do NLTK
        lang_map = {
            "en": "english", "pt": "portuguese", "es": "spanish", "fr": "french",
            "de": "german", "it": "italian", "nl": "dutch", "sv": "swedish",
            "da": "danish", "no": "norwegian", "fi": "finnish", "ru": "russian",
            "ro": "romanian", "sl": "slovene", "tr": "turkish", "el": "greek",
            "hu": "hungarian", "ar": "arabic", "az": "azerbaijani", "kk": "kazakh",
            "ne": "nepali", "tg": "tajik"
        }
        # normaliza códigos chineses/variantes para pular stopwords (não úteis)
        if lang_code and lang_code.startswith("zh"):
            return set()  # chinês não usa espaços; não remover stopwords
        lang = lang_map.get(lang_code)
        if lang and lang in stopwords.fileids():
            return set(stopwords.words(lang))
    except Exception:
        pass
    return set()

# --------- regex Unicode-safe ----------
_URL_RE = re.compile(r"http[s]?://\S+|www\.\S+", re.IGNORECASE)
# Mantém: letras/dígitos Unicode (\w é Unicode-aware), espaços, hífen e barra
_KEEP_RE = re.compile(r"[^\w\s\-/]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)

# Detecta presença de caracteres CJK para evitar operações que dependem de espaços
_HAS_CJK_RE = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u30FF\uAC00-\uD7AF]")

def limpar_texto(texto) -> str:
    """
    Limpeza multilíngue:
      - Converte entrada para str
      - Remove URLs
      - Mantém apenas letras/dígitos Unicode, espaços, hífen e barra
      - Normaliza espaços
      - Remove stopwords específicas do idioma (se detectado e suportado)
    """
    if texto is None:
        return ""
    if isinstance(texto, list):
        texto = " ".join(map(str, texto))
    elif not isinstance(texto, str):
        texto = str(texto)

    # Normalização básica
    texto = texto.strip()
    if not texto:
        return ""

    # Lowercase (não afeta scripts como CJK; é seguro para maioria dos idiomas)
    texto = texto.lower()

    # Remoções genéricas
    texto = _URL_RE.sub(" ", texto)
    texto = _KEEP_RE.sub(" ", texto)
    texto = _WS_RE.sub(" ", texto).strip()

    if not texto:
        return ""

    # Se contém CJK, não aplicar stopwords (não há separação por espaços confiável)
    if _HAS_CJK_RE.search(texto):
        return texto

    # Detecta idioma e tenta aplicar stopwords
    lang = _detect_lang(texto)
    sw = _stopwords_for(lang) if lang else set()

    if not sw:
        # Sem stopwords disponíveis -> retorna texto como está
        return texto

    tokens = [t for t in texto.split() if t not in sw]
    return " ".join(tokens)
