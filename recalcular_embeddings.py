import json
import os
from tqdm import tqdm
from processamento_texto import gerar_embedding

CAMINHO_ARQUIVO = "base_patentes.json"

def atualizar_base(caminho=CAMINHO_ARQUIVO):
    if not os.path.exists(caminho):
        print("❌ Arquivo base_patentes.json não encontrado.")
        return

    with open(caminho, "r", encoding="utf-8") as f:
        base = json.load(f)

    if not base:
        print("⚠️ A base está vazia.")
        return

    nova_base = []
    print(f"🔄 Recalculando embeddings para {len(base)} patentes...")

    for patente in tqdm(base, desc="🔁 Processando"):
        claims = patente.get("claims", "")
        # Embedding somente das claims (padrão do projeto), normalizado
        novo_embedding = gerar_embedding(texto_claims=claims, normalize=True).tolist()
        patente["embedding"] = novo_embedding
        nova_base.append(patente)

    backup_path = caminho.replace(".json", "_backup.json")
    os.replace(caminho, backup_path)
    print(f"\n📦 Backup salvo em: {backup_path}")

    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(nova_base, f, ensure_ascii=False, indent=2)

    print(f"✅ Base atualizada: {caminho}")
    print(f"📈 Total de embeddings atualizados: {len(nova_base)}")

if __name__ == "__main__":
    atualizar_base()
