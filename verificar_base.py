import json
import os

ARQUIVO = "base_patentes.json"

def campo_preenchido(valor):
    if isinstance(valor, list):
        return any(str(item).strip() for item in valor)
    return bool(str(valor).strip())

def verificar_campos(arquivo_json=ARQUIVO):
    if not os.path.exists(arquivo_json):
        print(f"❌ Arquivo não encontrado: {arquivo_json}")
        return

    with open(arquivo_json, 'r', encoding='utf-8') as f:
        patentes = json.load(f)

    total = len(patentes)
    com_claims = com_descricao = com_resumo = 0
    nenhum = um = dois = tres = 0

    for p in patentes:
        preenchidos = 0
        if campo_preenchido(p.get("claims", "")):
            com_claims += 1; preenchidos += 1
        if campo_preenchido(p.get("description", "")):  # chave correta
            com_descricao += 1; preenchidos += 1
        if campo_preenchido(p.get("abstract", "")):     # chave correta
            com_resumo += 1; preenchidos += 1

        if preenchidos == 0: nenhum += 1
        elif preenchidos == 1: um += 1
        elif preenchidos == 2: dois += 1
        else: tres += 1

    print("📊 Verificação de Campos")
    print(f"Total: {total}")
    print(f"Claims: {com_claims} | Description: {com_descricao} | Abstract: {com_resumo}")
    print(f"Nenhum: {nenhum} | 1 campo: {um} | 2 campos: {dois} | 3 campos: {tres}")

if __name__ == "__main__":
    verificar_campos()
