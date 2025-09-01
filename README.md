# 🧭 README — Similaridade Semântica para Triagem de Patentes;

> **Objetivo**: dado o texto das **reivindicações (claims)** de uma invenção, o sistema retorna as **N patentes mais similares** por **similaridade de cosseno** entre **embeddings** (modelo multilíngue). Há três modos: **Local (JSON)**, **Híbrido (Local + APIs externas/Lens)** e **Via API FastAPI**.

> **Importante**: a ferramenta **não decide** concessão/validade. Ela **ranqueia** candidatos para **triagem**, **busca de anterioridade** e **apoio** à análise técnico‑jurídica.

---

## Índice

1. [Visão geral](#visão-geral)
2. [Arquitetura & Técnicas](#arquitetura--técnicas)
3. [Pré‑requisitos](#pré-requisitos)
4. [Instalação](#instalação)
5. [Configuração (.env)](#configuração-env)
6. [Estrutura do projeto](#estrutura-do-projeto)
7. [Base local (formato) & Regra de ouro](#base-local-formato--regra-de-ouro)
8. [Executando](#executando)
9. [Streamlit (UI)](#streamlit-ui)
10. [Modo Híbrido (Local + Lens/API externa)](#modo-híbrido-local--lensapi-externa)
11. [Coleta de patentes com a Lens API](#coleta-de-patentes-com-a-lens-api)
12. [Usar **outro** banco (JSON/CSV/SQL) ou **outra** API](#usar-outro-banco-jsoncsvsql-ou-outra-api)
13. [Reprocessar embeddings](#reprocessar-embeddings)
14. [Testes automatizados](#testes-automatizados)
15. [Antes de publicar no GitHub](#antes-de-publicar-no-github)
16. [Solução de problemas](#solução-de-problemas)
17. [Links úteis](#links-úteis)
18. [Licença, contato & contribuições](#licença-contato--contribuições)

---

## Visão geral

* **Modelo**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (dim=768, multilíngue), com **L2‑normalização** dos vetores.
* **Similaridade**: **cosseno** (produto escalar entre vetores L2). Retorno em **0–1** (métricas) e **%** (exibição).
* **Modos de uso**:

  * **Local (JSON)**: busca somente em `base_patentes.json`.
  * **Híbrido (Local + Lens)**: agrega candidatos da Lens (ou outros provedores), re‑embebe localmente e **re‑ranqueia**.
  * **Via API FastAPI**: cliente/serviço externo consulta a API HTTP local.

---

## Arquitetura & Técnicas

```
claims → limpeza → embedding (mpnet, L2) →
   ├─ busca LOCAL (cosine Top‑K)
   ├─ busca EXTERNA (Lens/…)* → re‑rank local
   └─ mescla + dedup (id/lens_id) → Top‑K final
* se LENS_API_KEY estiver configurada
```

**Técnicas/Ferramentas**

* **Transformers** (MPNet multilingue) via **Sentence‑Transformers**.
* **Embeddings** L2‑normalizados → cosseno por **produto escalar**.
* **Híbrido**: *candidate generation* externo + *re‑ranking* local.
* **Avaliação**: pares rotulados (0/1) para métricas; ranking com P\@K/R\@K/nDCG\@K.
* **Stack**: Python, FastAPI, Streamlit, scikit‑learn, NumPy, Torch (CPU por padrão).

---

## Pré‑requisitos

* **SO**: Windows 10/11 x64 (funciona também em Linux/macOS).
* **Python**: 3.10 ou 3.11 (64‑bits).
* **RAM**: 8 GB mínimo (12–16 GB recomendado p/ bases maiores).
* **Disco**: 2–5 GB (modelo + cache + base). Bases grandes pedem mais.
* **Rede**: para baixar o modelo (1ª execução) e para modo **Híbrido**.
* **Portas**: API `8000` (ajustável), Streamlit `8512` (ajustável).
* **GPU**: opcional. Projeto está configurado para **CPU**.

---

## Instalação

> Abaixo, instalação **CPU/Windows**. Em Linux/macOS, ajuste o índice do Torch conforme sua plataforma.

```bash
# 1) Ambiente virtual
python -m venv .venv
# Windows
. .venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

# 2) Primeiro, instale o Torch (CPU) pelo índice da PyTorch
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.2.2"

# 3) Demais dependências (versões estáveis entre si)
pip install --no-cache-dir -r requirements.txt
```

**Versões pinadas (sugestão)**

```
fastapi==0.111.0
starlette==0.37.2
uvicorn[standard]==0.30.1
streamlit==1.36.0
sentence-transformers==2.7.0
transformers==4.41.2
tokenizers==0.19.1
huggingface-hub==0.23.5
torch==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
scipy==1.16.1
pandas==2.3.2
matplotlib==3.9.0
seaborn==0.13.2
nltk==3.9.1
langdetect==1.0.9
PyPDF2==3.0.1
python-dotenv==1.0.1
requests==2.32.3
pydantic==2.11.7
```

---

## Configuração (.env)

Crie um arquivo `.env` **na raiz**:

```
# obrigatório para modo Híbrido/Lens
LENS_API_KEY=SEU_TOKEN_DA_LENS
```

> Se não existir `LENS_API_KEY`, o sistema funciona em **modo Local**.

---

## Estrutura do projeto

```
.
├── app.py                      # (opcional) servidor alternativo
├── main.py                     # FastAPI (endpoints /, /verificar, /buscar_similares)
├── streamlit_app.py            # UI (Local/Híbrido/API)
├── processamento_texto.py      # limpeza + gerar_embedding (modelo mpnet, L2)
├── similaridade.py             # funções de score (0–1 e %)
├── buscar_similares.py         # CLI de busca local (JSON)
├── buscar_hibrido.py           # busca híbrida (Local + Lens/outros)
├── lens_api.py                 # chamadas à Lens API (candidatos por claims)
├── coleta_patentes.py          # coleta/hidratação via Lens → base_patentes.json
├── recalcular_embeddings.py    # reprocessa embeddings da base local
├── base_patentes.json          # base local (N itens, com embeddings L2)
├── tests/                      # suíte de testes (unittest)
├── relatorios/                 # saídas de avaliação (opcional)
└── README.md                   # este arquivo
```

---

## Base local (formato) & Regra de ouro

Arquivo **`base_patentes.json`** contém **lista** de objetos no formatoArquivo: base_patentes.json — contém uma lista de objetos no formato abaixo. Baixar **`base_patentes.json`** em:
https://drive.google.com/file/d/144_DDZIJCmMNE82ZSqQF40R5IIkBxvwg/view?usp=sharing:

```json
{
  "lens_id": "...",
  "title": "...",
  "abstract": "...",
  "claims": "...",
  "description": "...",
  "embedding": [0.0123, -0.0456, ...]  
}
```

**Regra de ouro**: o campo `embedding` deve ser um **vetor de 768 floats L2‑normalizado** das **claims** usando **o mesmo modelo** da consulta. Caso troque o modelo, **recalcule toda a base** (ver [Reprocessar embeddings](#reprocessar-embeddings)).

---

## Executando

### 1) API (FastAPI)

```bash
python main.py --port 8000
# ou (se preferir uvicorn direto)
# uvicorn main:app --host 127.0.0.1 --port 8000
```

**Endpoints**

* `GET /` → status:

```json
{"status":"OK","itens_na_base":2250,"modo_hibrido_disponivel":true}
```

* `POST /buscar_similares` (usa apenas `claims`):

```json
{ "claims": "texto das claims...", "top_k": 10 }
```

* `POST /verificar` (payload completo; prioriza `claims` se houver):

```json
{ "titulo":"...", "resumo":"...", "claims":"...", "descricao":"...", "top_k": 10 }
```

**Resposta (exemplo)**

```json
{
  "resultados": [
    {
      "lens_id": "140-...-70X",
      "title": "...",
      "claims": "trecho...",
      "link": "https://www.lens.org/lens/patent/140-...-70X",
      "score": 0.8245,
      "similaridade_percentual": 82.45,
      "fonte": "local"
    }
  ]
}
```

### 2) CLI local

```bash
python buscar_similares.py
# cole o texto das claims quando solicitado
```

---

## Streamlit (UI)

```bash
streamlit run streamlit_app.py --server.port 8512
```

* **Local (JSON)**: consulta apenas `base_patentes.json`.
* **Híbrido (Local + Lens)**: requer `LENS_API_KEY` no `.env`.
* **Via API FastAPI**: a UI chama os endpoints HTTP locais.

---

## Modo Híbrido (Local + Lens/API externa)

1. Gere o **embedding L2** da **consulta** (claims) localmente.
2. Busque **candidatos externos** (Lens/outros provedores).
3. Para cada candidato, obtenha texto relevante (idealmente **claims**), gere **embedding L2** **local** e **re‑ranqueie** junto com a base local.
4. **Mescle** resultados (dedup por `lens_id`/`id`) e retorne **Top‑K**.

> Re‑ranquear localmente garante que todos os vetores estejam **no mesmo espaço** (mesmo modelo), evitando escalas incompatíveis.

---

## Coleta de patentes com a Lens API

### Solicitar acesso

1. Crie conta em **lens.org** e acesse a área de **API**.
2. Solicite um **API Key** (uso: pesquisa/educacional/protótipo).
3. Configure o `.env`:

```
LENS_API_KEY=SEU_TOKEN_DA_LENS
```

> Políticas e limites podem mudar. Consulte a documentação/painel da Lens.

### Rodar a coleta (script pronto)

```bash
python coleta_patentes.py
```

O script:

* Faz *search* com `simple_query_string` em `claims^3, description^2, abstract` (termos aleatórios p/ diversidade);
* Busca detalhes por `lens_id`;
* **Limpa** o texto e gera **embedding L2** das **claims**;
* Persiste incrementalmente em `base_patentes.json` (checkpoint);
* Respeita `429/Retry-After`.

Parâmetros no arquivo:

* `TOTAL_DESEJADO`, `TAMANHO_PAGINA`, `TERMOS_ALEATORIOS`.

> Se interromper, roda novamente e o script **retoma** sem duplicar.

---

## Usar **outro** banco (JSON/CSV/SQL) ou **outra** API

Você **não** está preso ao `base_patentes.json` original.

### A) Outro arquivo **JSON**

1. Converta sua base para o **formato** da seção [Base local](#base-local-formato--regra-de-ouro).
2. Salve como `base_meu_bd.json`.
3. Aponte o caminho no código:

   * `main.py`: `BASE_PATH = "base_meu_bd.json"`
   * `buscar_similares.py`: `CAMINHO_ARQUIVO = "base_meu_bd.json"`

### B) **CSV/SQL** → JSON (exemplo de adaptador)

```python
# adapters/csv_para_json.py
import pandas as pd, json
from processamento_texto import gerar_embedding

def csv_para_json(caminho_csv, saida_json):
    df = pd.read_csv(caminho_csv)
    itens = []
    for _, r in df.iterrows():
        claims = str(r.get('claims', '') or '').strip()
        if not claims:
            continue
        emb = gerar_embedding(texto_claims=claims, normalize=True).tolist()
        itens.append({
            "lens_id": str(r.get('id', '')),
            "title": str(r.get('title', '')),
            "abstract": str(r.get('abstract', '')),
            "claims": claims,
            "description": str(r.get('description', '')),
            "embedding": emb
        })
    with open(saida_json, 'w', encoding='utf-8') as f:
        json.dump(itens, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    csv_para_json("minha_base.csv", "base_patentes.json")
```

> Para **SQL**, leia com pandas/SQLAlchemy, gere os embeddings e grave o JSON no mesmo formato.

### C) **Outra API externa** (além da Lens)

Crie um provedor que retorne candidatos por **claims** e re‑ranqueie localmente:

```python
# providers/meu_provedor.py (exemplo)
import os, requests
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("MEU_API_KEY")

def candidatos_por_claims(texto_claims: str, top_n: int = 30):
    # 1) chamar a API e coletar candidatos (id, title, claims/trecho, link)
    # 2) normalizar e retornar lista: {"id":"...","title":"...","claims":"...","link":"..."}
    return []
```

Registrar no híbrido (`buscar_hibrido.py`):

```python
from providers.meu_provedor import candidatos_por_claims as prov_meu
PROVEDORES = [
    ("lens", candidatos_por_claims),
    ("meu",  prov_meu),
]
```

> Gere o **embedding L2 local** para cada candidato antes do re‑ranking.

**Atenção**: trocou de **modelo** de embedding? Recalcule **toda a base**.

---

## Reprocessar embeddings

Use quando alterar **modelo** ou **pré‑processamento**:

```bash
python recalcular_embeddings.py
```

O script abre `base_patentes.json`, recalcula `embedding` das claims (**L2**) e salva.

---

## Testes automatizados

Rodar todos os testes:

```bash
# da raiz do projeto
python -m unittest discover -s tests -p "test_*.py" -v
```

Se seu código estiver em `src/`, exponha o caminho:

```powershell
# Windows (PowerShell)
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -p "test_*.py" -v
```

> Na primeira execução, o modelo será baixado (testes de embedding podem demorar um pouco). As próximas rodam mais rápido.

---

## Antes de publicar no GitHub

Remova dados sensíveis/pesados e segredos:

```bash
# apagar base local
# Windows
del base_patentes.json 2>$null
# Linux/macOS
rm -f base_patentes.json
```

`.gitignore` sugerido:

```
.env
base_patentes.json
relatorios/
data/
*.jsonl
*.parquet
*.csv
```

> Se algo já foi versionado por engano, use `git rm --cached <arquivo>` e faça novo commit (ou ferramentas como `git filter-repo` para limpar histórico).

---

## Solução de problemas

* **Porta ocupada**: mude a porta (UI `--server.port`, API `--port`). Para matar processo (Windows PowerShell):

  ```powershell
  Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force
  ```
* **NumPy/Torch**: use `numpy==1.26.4` com `torch==2.2.2`. Evite `numpy 2.x` com libs que foram compiladas p/ 1.x.
* **Transformers/Tokenizers**: `transformers 4.41.x` requer `tokenizers >=0.19,<0.20` → use `0.19.1`.
* **Modelo não baixa**: verifique rede/proxy; rode `pip cache purge` e tente novamente.
* **Base vazia**: gere `base_patentes.json` (coleta Lens ou CSV/SQL → JSON).

---

## Links úteis

* Modelo (Hugging Face): [https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
* Sentence‑Transformers: [https://www.sbert.net/](https://www.sbert.net/)
* Transformers: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
* FastAPI: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
* Streamlit: [https://docs.streamlit.io/](https://docs.streamlit.io/)
* Lens API: [https://docs.lens.org/](https://docs.lens.org/)

---

## Licença, contato & contribuições
* **Licença**: MIT — uso livre para fins acadêmicos, educacionais e prototipagem.
* **Contato**: [isabela.andradeaguiar1@gmail.com](mailto:isabela.andradeaguiar1@gmail.com)
* **Contribuir**: faça *fork* → *branch* (`feat/minha-feature`) → *PR* com escopo claro e, se possível, testes.