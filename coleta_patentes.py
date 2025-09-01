import os
import json
import time
import random
import requests
from dotenv import load_dotenv
from limpar_texto import limpar_texto
from processamento_texto import gerar_embedding

load_dotenv()
LENS_API_KEY = os.getenv("LENS_API_KEY")
if not LENS_API_KEY:
    raise RuntimeError("Defina LENS_API_KEY no .env")

HEADERS = {
    "Authorization": f"Bearer {LENS_API_KEY}",
    "Content-Type": "application/json"
}

CAMINHO_ARQUIVO = "base_patentes.json"
TOTAL_DESEJADO = 10000
TAMANHO_PAGINA = 50

# Termos genéricos para variações na busca
TERMOS_ALEATORIOS = [
    "a", "the", "method", "system", "data", "device",
    "information", "network", "process", "input", "output"
]

def normalizar(texto):
    if isinstance(texto, list):
        return " ".join([str(item) for item in texto])
    return str(texto or "")

def buscar_ids_aleatorios(offset):
    termo_aleatorio = random.choice(TERMOS_ALEATORIOS)

    url = "https://api.lens.org/patent/search"
    query = {
        "query": {
            "simple_query_string": {
                "query": termo_aleatorio,
                "fields": ["claims^3", "description^2", "abstract"],
                "default_operator": "and"
            }
        },
        "size": TAMANHO_PAGINA,
        "from": offset,
        "include": ["lens_id"]
    }

    response = requests.post(url, headers=HEADERS, json=query)
    if response.status_code == 429:
        wait = int(response.headers.get("Retry-After", 60))
        print(f"⚠️ Rate limit. Aguardando {wait}s...")
        time.sleep(wait)
        return buscar_ids_aleatorios(offset)
    response.raise_for_status()
    data = response.json()
    return [r["lens_id"] for r in data.get("data", [])]

def buscar_detalhes_patente(lens_id):
    url = f"https://api.lens.org/patent/{lens_id}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 429:
        wait = int(response.headers.get("Retry-After", 60))
        print(f"⚠️ Rate limit (detalhes). Aguardando {wait}s...")
        time.sleep(wait)
        return buscar_detalhes_patente(lens_id)
    response.raise_for_status()
    return response.json()

def carregar_existente():
    if os.path.exists(CAMINHO_ARQUIVO):
        with open(CAMINHO_ARQUIVO, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def salvar_base(patentes):
    with open(CAMINHO_ARQUIVO, "w", encoding="utf-8") as f:
        json.dump(patentes, f, ensure_ascii=False, indent=2)

def coletar_patentes():
    print("🔄 Inicializando pipeline de embeddings...")
    _ = gerar_embedding(texto_claims="warmup", normalize=True)  # carrega modelo
    print("✅ Modelo pronto.")

    base_existente = carregar_existente()
    ids_existentes = set(p["lens_id"] for p in base_existente if "lens_id" in p)
    patentes_processadas = base_existente.copy()

    offset = 0
    while len(patentes_processadas) < TOTAL_DESEJADO:
        novos_ids = buscar_ids_aleatorios(offset)
        if not novos_ids:
            print("⚠️ Nenhum novo ID retornado. Finalizando.")
            break

        for lens_id in novos_ids:
            if lens_id in ids_existentes:
                continue

            try:
                detalhes = buscar_detalhes_patente(lens_id)

                titulo = limpar_texto(normalizar(detalhes.get("title", "")))
                resumo = limpar_texto(normalizar(detalhes.get("abstract", "")))
                claims = limpar_texto(normalizar(detalhes.get("claims", "")))
                descricao = limpar_texto(normalizar(detalhes.get("description", "")))

                if not claims.strip():
                    print(f"⚠️ Ignorando {lens_id} por falta de 'claims'")
                    continue

                # Embedding L2-normalizado das claims (fonte da verdade)
                embedding = gerar_embedding(texto_claims=claims, normalize=True).tolist()

                patente = {
                    "lens_id": lens_id,
                    "title": titulo,
                    "abstract": resumo,
                    "claims": claims,
                    "description": descricao,
                    "embedding": embedding
                }

                patentes_processadas.append(patente)
                ids_existentes.add(lens_id)

                print(f"✅ ({len(patentes_processadas)}/{TOTAL_DESEJADO}) {lens_id} coletado.")
                time.sleep(1.2)  # respeita rate

                if len(patentes_processadas) % 100 == 0:
                    salvar_base(patentes_processadas)

            except Exception as e:
                print(f"❌ Erro em {lens_id}: {e}")

        offset += TAMANHO_PAGINA

    salvar_base(patentes_processadas)
    print(f"\n📁 Coleta concluída. {len(patentes_processadas)} patentes salvas em {CAMINHO_ARQUIVO}.")

if __name__ == "__main__":
    coletar_patentes()
