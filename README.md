# 🧭 README — Similaridade Semântica para Triagem de Patentes;

## 1) O que é

\*\*Busca semântica Top‑K de patentes a partir das suas \*\****claims***. Você cola o texto das reivindicações e recebe as **N patentes mais similares** (cosine) — para triagem, busca de anterioridade e apoio à análise técnica.

> O sistema **não decide** concessão; ele **ranqueia** candidatos por proximidade semântica.

---

## 2) TL;DR

* **Modelo**: `paraphrase-multilingual-mpnet-base-v2` (768d, L2) — multilíngue.
* **Modos**: Local (JSON) · Híbrido (Local + Lens) · Via API FastAPI.
* **Base**: local (JSON), CSV/SQL convertido ou **APIs externas** (Lens, etc.).
* **Rodar rápido**:

  ```bash
  # venv + dependências (Windows/CPU)
  python -m venv .venv && .venv\Scripts\activate
  pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.2.2"
  pip install --no-cache-dir -r requirements.txt

  # backend (API)
  python main.py --port 8000

  # UI (opcional)
  streamlit run streamlit_app.py --server.port 8512
  ```

---

## 3) Como funciona (resumo)

```
claims → limpeza → embedding (mpnet) →
   ├─ busca LOCAL (base_patentes.json)
   ├─ busca EXTERNA (Lens)* → re‑rank local
   └─ mescla + dedup (lens_id) → Top‑K
* se LENS_API_KEY estiver configurada
```

## 4) Metodologia (reprodutível / passo a passo)

> Detalhes para qualquer pessoa reproduzir coleta, indexação, busca e avaliação.

### 4.1 Ambiente e versões

* **Python**: 3.10 ou 3.11 (x64)
* **Stack fixado**: `torch==2.2.2`, `numpy==1.26.4`, `transformers==4.41.2`, `tokenizers==0.19.1`, `huggingface-hub==0.23.5`, `sentence-transformers==2.7.0`, `fastapi==0.111.0`, `streamlit==1.36.0`.
* **Seeds**: 42 para `random`, `numpy` e `torch`.

Instalação (CPU, genérica):

```bash
python -m venv .venv
python -m pip install --upgrade pip
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2
pip install --no-cache-dir -r requirements.txt
```

### 4.2 Pré-processamento de texto

* Minúsculas, remoção de URLs, manter letras/dígitos/acentos, colapsar espaços.
* Detecção de idioma e *stopwords* multilíngues (opcional).
* Implementação em `processamento_texto.py` (`limpar_texto`, `gerar_embedding`).

### 4.3 Embeddings (modelo e normalização)

* Modelo: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (dim=768).
* `encode(texto, convert_to_numpy=True)`.
* L2: `v = e / ||e||2` (se norma>0). Persistir como lista de 768 floats.
* Comprimento: truncamento/padding interno do Sentence-Transformers; ajuste com `model.max_seq_length` se necessário.

### 4.4 Construção da base local

* **Via CSV/SQL** → converter para JSON (chaves: `lens_id/title/abstract/claims/description`), gerar embedding das claims e salvar em `base_patentes.json`.
* **Via Lens (opcional)** → `.env` com `LENS_API_KEY` e `python coleta_patentes.py`.
* Busca na Lens: `simple_query_string` em `claims^3, description^2, abstract`; respeitar 429/Retry-After; descartar itens sem claims.

**Esquema JSON do item**

```json
{
  "lens_id":"...",
  "title":"...",
  "abstract":"...",
  "claims":"...",
  "description":"...",
  "embedding":[0.0123, -0.0456, 0.001, ...]
}
```

### 4.5 Busca local (cosine Top-K)

* Carregar matriz `E` (n×768) com embeddings L2; consulta `q` (768, L2).
* Cosine: `sims = E @ q`; selecionar Top‑K; ordenar decrescente; retornar metadados + `score` (0–1) e `%`.
* Implementações: `main.py` (/verificar, /buscar\_similares) e `buscar_similares.py` (CLI local).

### 4.6 Busca híbrida (Local + API externa)

* Se `.env` tiver `LENS_API_KEY`, chamar `lens_api.candidatos_por_claims(texto, top_n≈3K)`.
* Para cada candidato, obter `claims`/trecho, gerar embedding L2 local; **mesclar** (dedup por `lens_id/id`), recomputar `sims` e **re‑ranquear**; devolver Top‑K único.

### 4.7 Avaliação (pares rotulados — protocolo)

* **Amostragem**: estratifique por jurisdição/ano/IPC (documente no artigo).
* **Rotuladores**: ≥2 avaliadores de PI; **adjudicação** por terceiro em caso de empate.
* **Concordância**: reporte **Cohen’s κ**/**Krippendorff’s α**.
* **Métricas**: Accuracy, Macro‑F1, MCC, Kappa, ROC‑AUC, AP; para ranking: P\@10, R\@10, nDCG\@10.
* **Scripts**: `avaliar_pares_manual.py`, `avaliar_pares_rotulados_sem_zona.py`, `avaliar_metricas.py`.

Exemplo:

```bash
python avaliar_metricas.py --arquivo rotulos.jsonl --k 10
```

### 4.8 Execução ponta‑a‑ponta

```bash
# gerar/atualizar base (opcional)
python coleta_patentes.py
# subir API
python main.py --port 8000
# testar
curl -s http://127.0.0.1:8000/
curl -s -X POST http://127.0.0.1:8000/buscar_similares -H "Content-Type: application/json" -d '{"claims":"Texto das claims","top_k":10}'
# UI
streamlit run streamlit_app.py --server.port 8512
```

---

## 4) Endpoints principais (API)

* **GET /** → status: itens na base e se híbrido está disponível.
* **POST /buscar\_similares**

  ```json
  { "claims": "texto das claims…", "top_k": 10 }
  ```
* **POST /verificar** (payload completo; usa `claims` se existir)

  ```json
  { "titulo":"...", "resumo":"...", "claims":"...", "descricao":"...", "top_k": 10 }
  ```

**Resposta (exemplo):**

```json
{
  "resultados": [
    {"lens_id":"140-...-70X","title":"...","link":"https://www.lens.org/lens/patent/140-...-70X","score":0.8245,"similaridade_percentual":82.45,"fonte":"local"}
  ]
}
```

---

## 5) Base local (formato e regra de ouro)

Arquivo **`base_patentes.json`** (lista de objetos):

```json
{
  "lens_id":"...",
  "title":"...",
  "abstract":"...",
  "claims":"...",
  "description":"...",
  "embedding":[0.0123, -0.0456, ...]  // vetor 768 **L2-normalizado** das claims
}
```

**Sempre salve o embedding das *********************************claims********************************* já L2‑normalizado.**

---

## 6) Híbrido (Local + Lens) — opcional

1. Crie `.env` na raiz:

   ```
   LENS_API_KEY=SEU_TOKEN_DA_LENS
   ```
2. Mantenha `lens_api.py` e `buscar_hibrido.py` na raiz. O servidor tenta híbrido e faz **fallback** para Local se falhar.

---

## 7) Requisitos (matriz estável — Windows/CPU)

* `torch==2.2.2`  +  `numpy==1.26.4`
* `transformers==4.41.2`  +  `tokenizers==0.19.1`  +  `huggingface-hub==0.23.5`
* `sentence-transformers==2.7.0`
* `fastapi==0.111.0` · `starlette==0.37.2` · `uvicorn[standard]==0.30.1`
* `streamlit==1.36.0` · `scikit-learn==1.5.0`

> Instale o Torch (CPU) **antes** do restante, pelo índice da PyTorch (comando acima).
> **Arquivo principal de instalação**: use `requirements.txt` deste repositório para garantir compatibilidade.

---

## 8) Como gerar a base rapidamente

* **CSV/SQL → JSON**: extraia `id/title/claims/abstract/description`, calcule `embedding` (claims) com `processamento_texto.gerar_embedding(..., normalize=True)` e salve no formato do item 5.
* **Via Lens**: `python coleta_patentes.py` (respeita `Retry-After`).
* **Reprocessar** embeddings: `python recalcular_embeddings.py`.

---

## 8.1) Antes de publicar no GitHub (limpar dados locais)

Antes de subir o repositório, **remova qualquer dado privado ou pesado**:

* Apague a base local e arquivos derivados:

  ```bash
  # Windows (PowerShell)
  del base_patentes.json 2>$null
  # Linux/macOS
  rm -f base_patentes.json
  ```
* Opcional: mantenha **apenas um exemplo mínimo** (ex.: `base_20_patentes.json`) para demonstração, mas **não o use em produção**. No código, altere `BASE_PATH`/`CAMINHO_ARQUIVO` para esse arquivo de amostra quando quiser rodar o projeto sem dados reais.
* Garanta que segredos e dumps **não** sejam versionados. Sugestão de `.gitignore`:

  ```gitignore
  # Segredos
  .env

  # Bases e artefatos locais
  base_patentes.json
  data/
  relatorios/
  *.jsonl
  *.parquet
  *.csv
  ```

> Dica: se você já versionou algum desses arquivos por engano, remova do histórico com `git rm --cached <arquivo>` e faça um novo commit.

---

## 8.2) Como solicitar acesso à **Lens API** (para busca/coleta externa)

1. **Crie uma conta** em *lens.org* (gratuita) e entre na área de **desenvolvedores / API**.
2. **Solicite uma chave de API** (token Bearer). A Lens pode exigir um formulário simples com o **uso pretendido** (pesquisa/ensino/protótipo) e **limites** aceitos de uso.
3. Após aprovado, você receberá um **API key/token**. Guarde-o em local seguro e **não versionado**.
4. Configure seu token no arquivo `.env` na raiz do projeto:

   ```env
   LENS_API_KEY=SEU_TOKEN_DA_LENS
   ```

> Observação: a política da Lens (modelos de acesso, limites, rotas) pode mudar ao longo do tempo. Verifique a documentação oficial e o painel da sua conta ao solicitar o acesso.

**Modelo de e‑mail (caso precise solicitar por suporte):**

```
Assunto: Solicitação de acesso à Lens Patent API

Olá, 
Sou [Seu Nome], [Instituição/Empresa]. Estou desenvolvendo um protótipo acadêmico de triagem de patentes por similaridade semântica e gostaria de solicitar acesso à Lens Patent API (apenas leitura), com uso limitado para pesquisa. 
Obrigado!
[Seu contato]
```

---

## 8.3) Coletar patentes via Lens (script incluso)

Com a chave configurada, use o script `coleta_patentes.py` para **hidratar** a base local:

```bash
# ativa o ambiente e roda a coleta
python coleta_patentes.py
```

O script:

* Faz buscas paginadas usando termos genéricos aleatórios (para diversidade),
* Busca detalhes por `lens_id`,
* Limpa texto e **gera embedding L2** das *claims*,
* Salva/atualiza `base_patentes.json` continuamente (checkpoint a cada \~100 itens),
* Respeita `429/Retry-After` com espera automática.

**Parâmetros editáveis no arquivo** (valores padrão já no código):

* `TOTAL_DESEJADO`: volume alvo de patentes (ex.: `10000`).
* `TAMANHO_PAGINA`: tamanho do lote por requisição (ex.: `50`).
* `TERMOS_ALEATORIOS`: lista de termos usados na *simple\_query\_string*.

> Se a execução for interrompida, o script **retoma** a partir do que já existe em `base_patentes.json`.

**Re-hidratação / reprocessamento**

* Caso mude o **modelo** de embedding, rode `recalcular_embeddings.py` para recalcular os vetores a partir das *claims* já armazenadas.

---

## 9) Streamlit (UI)

* Modo **Local (JSON)**, **Híbrido** (se `.env` presente) ou **Via API FastAPI** (aponta para `http://127.0.0.1:8000`).
* Mostra **Top‑K** com link da Lens e trecho das claims.

---

## 10) Troubleshooting (rápido)

* **Porta ocupada**: mude `--server.port` (UI) ou `--port` (API).
* **NumPy/Torch**: use `numpy==1.26.4` com `torch==2.2.2`.
* **Tokenizers/Transformers**: `transformers 4.41.x` requer `tokenizers >=0.19,<0.20` → use `0.19.1`.
* **Base vazia**: gere `base_patentes.json` (item 8).

---

## 11) Licença & aviso

* **Licença**: defina MIT/Apache‑2.0.
* **Aviso**: ferramenta de apoio à triagem; não substitui análise técnica/jurídica.

---

## 12) Usar outro banco de dados (local ou API)

Você **não** está preso ao `base_patentes.json`. Qualquer fonte pode ser usada, desde que entregue **texto de claims** e, idealmente, **embeddings L2** no mesmo espaço do modelo.

### A) Outro **arquivo local** (JSON)

1. Converta seu banco para o **esquema do item 5** (chaves e embedding L2 das *claims*).
2. Salve como `base_meu_bd.json`.
3. Altere o caminho no servidor/API:

   * Em `main.py`, troque:

     ```python
     BASE_PATH = "base_patentes.json"  # → "base_meu_bd.json"
     ```
   * (Se usar busca direta local no app) Em `buscar_similares.py`, troque `CAMINHO_ARQUIVO` para o novo JSON.

> Recomenda-se **exportar para JSON** no formato do item 5. É simples, rápido e reaproveita todo o pipeline.

### B) **CSV/SQL** → JSON (conversão rápida)

Use um adaptador simples para transformar suas tabelas em JSON compatível:

```python
# adapters/csv_para_json.py (exemplo)
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
            "title": r.get('title', ''),
            "abstract": r.get('abstract', ''),
            "claims": claims,
            "description": r.get('description', ''),
            "embedding": emb
        })
    with open(saida_json, 'w', encoding='utf-8') as f:
        json.dump(itens, f, ensure_ascii=False, indent=2)
```

Depois aponte `BASE_PATH`/`CAMINHO_ARQUIVO` para o JSON gerado.

### C) **Outra API externa** (além da Lens)

Adapte um provedor que retorne **candidatos por claims**, depois **re‑ranqueie localmente**:

1. Crie um módulo do provedor, e.g. `providers/meu_provedor.py`:

   ```python
   # providers/meu_provedor.py (exemplo)
   import os, requests
   from dotenv import load_dotenv
   load_dotenv()
   API_KEY = os.getenv("MEU_API_KEY")

   def candidatos_por_claims(texto_claims: str, top_n: int = 30):
       # 1) chame a API e traga candidatos (id, título, claims/trecho, link)
       # 2) normalize campos e retorne lista de dicts compatíveis
       # Estrutura mínima por item: {"id":"...","title":"...","claims":"...","link":"..."}
       return []
   ```
2. Inclua no **híbrido** (`buscar_hibrido.py`):

   ```python
   from providers.meu_provedor import candidatos_por_claims as prov_meu

   PROVEDORES = [
       ("lens", candidatos_por_claims),   # já existente
       ("meu",  prov_meu),                # novo provedor
   ]

   todos = []
   for nome, fn in PROVEDORES:
       cands = fn(texto_claims, top_n=top_k*3)
       # calcule embedding localmente e anexe ao pool
       for c in cands:
           emb = gerar_embedding(texto_claims=c.get("claims",""), normalize=True)
           todos.append({
               "lens_id": c.get("id",""),
               "title": c.get("title",""),
               "claims": c.get("claims",""),
               "link": c.get("link",""),
               "embedding": emb.tolist(),
               "fonte": "externa"
           })
   # depois: dedup por lens_id/id, cosine com a query e Top‑K
   ```
3. Configure a chave no `.env` (ex.: `MEU_API_KEY=...`).

> Dica: se a API externa não fornece **claims** completas, use **resumo/trecho** relevante. O re‑ranqueamento local via embedding alinha as escalas.

### D) Observações importantes

* **Consistência de espaço vetorial**: se trocar o **modelo de embedding**, **recalcule toda a base** para o mesmo espaço (caso contrário, cosine não é comparável).
* **Tamanho da base**: para bases muito grandes, considere indexadores ANN (FAISS/HNSW) no futuro.
* **Chaves/segurança**: mantenha credenciais apenas no back‑end e use HTTPS em produção.

---

## 13) Pré‑requisitos (ambiente)

* **Python**: 3.10 ou 3.11 recomendados (Windows 10/11 x64). Funciona em Linux/macOS; ajuste instalação do Torch conforme plataforma.
* **RAM**: 8 GB mínimo (12–16 GB recomendado) para trabalhar com bases médias.
* **Disco**: 2–5 GB livres (modelos + cache + base local). Bases grandes exigem mais.
* **Rede**: necessária para baixar o modelo na 1ª execução e para o modo **Híbrido** (Lens ou outras APIs).
* **Portas**: API padrão `8000`, Streamlit `8512` (ajustáveis).
* **GPU**: opcional; o projeto foi configurado para **CPU**. Para GPU, instale a variante adequada do Torch e teste as versões do stack.

---

## 14) Testes rápidos / verificação

### 14.0 Rodar toda a suíte

> Execute a partir da **raiz do projeto**. Se seu código está em `src/`, defina o `PYTHONPATH`.

**Windows (PowerShell):**

```powershell
# ativar venv
. .\.venv\Scripts\activate
# expor src para os testes
$env:PYTHONPATH = "$PWD\src"
# rodar todos os testes em tests/
python -m unittest discover -s tests -p "test_*.py" -v
```

> Dica: na 1ª execução o modelo é baixado; a segunda rodada fica bem mais rápida.
> Se não quiser mexer em `PYTHONPATH`, no topo de cada teste você pode inserir:
>
> ```python
> import os, sys
> ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
> SRC  = os.path.join(ROOT, "src")
> for p in (ROOT, SRC):
>     sys.path.insert(0, p) if p not in sys.path else None
> ```

### 14.1 Sanidade do modelo

```bash
python - << 'PY'
from sentence_transformers import SentenceTransformer
import numpy as np
m = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
e = m.encode("teste rápido", convert_to_numpy=True)
print("shape:", e.shape, "norm:", float(np.linalg.norm(e)))
PY
```

Saída esperada: `shape: (768,)` e `norm > 0`.

### 14.2 API online

```bash
# subir API
python main.py --port 8000
# status
curl -s http://127.0.0.1:8000/
# busca
curl -s -X POST http://127.0.0.1:8000/buscar_similares \
  -H "Content-Type: application/json" \
  -d '{"claims":"Um sistema para ...","top_k":10}'
```

> Se seu `main.py` estiver em `src/` e você preferir usar uvicorn direto:
> `uvicorn src.main:app --host 127.0.0.1 --port 8000`

### 14.3 UI (Streamlit)

```bash
streamlit run streamlit_app.py --server.port 8512
```

Escolha **Local**, **Híbrido** (se `.env`) ou **Via API FastAPI**.

### 14.4 Scripts de avaliação (se aplicável)

* `similaridade.py`: funções de score (0–1 para métricas; % para exibição).
* `avaliar_metricas.py`: gera métricas agregadas (precisão\@K, etc.).
* `avaliar_pares_manual.py` / `avaliar_pares_rotulados_sem_zona.py`: avaliam pares rotulados.

Exemplo:

```bash
python avaliar_metricas.py --arquivo rotulos.jsonl --k 10
```

> Observação: os testes e avaliações usam **apenas base local**; não chamam a Lens (sem rede).

---

## 15) Exemplo visual

Adicione imagens na pasta `assets/` e referencie aqui:

**UI (Streamlit)**

# 🧭 README — IA de Triagem de Patentes (versão enxuta)

## 1) O que é

\*\*Busca semântica Top‑K de patentes a partir das suas \*\****claims***. Você cola o texto das reivindicações e recebe as **N patentes mais similares** (cosine) — para triagem, busca de anterioridade e apoio à análise técnica.

> O sistema **não decide** concessão; ele **ranqueia** candidatos por proximidade semântica.

---

## 2) TL;DR

* **Modelo**: `paraphrase-multilingual-mpnet-base-v2` (768d, L2) — multilíngue.
* **Modos**: Local (JSON) · Híbrido (Local + Lens) · Via API FastAPI.
* **Base**: local (JSON), CSV/SQL convertido ou **APIs externas** (Lens, etc.).
* **Rodar rápido**:

  ```
  # venv + dependências (Windows/CPU)
  python -m venv .venv && .venv\Scripts\activate
  pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.2.2"
  pip install --no-cache-dir -r requirements.txt

  # backend (API)
  python main.py --port 8000

  # UI (opcional)
  streamlit run streamlit_app.py --server.port 8512

  ```

---

## 3) Como funciona (resumo)

```
claims → limpeza → embedding (mpnet) →
   ├─ busca LOCAL (base_patentes.json)
   ├─ busca EXTERNA (Lens)* → re‑rank local
   └─ mescla + dedup (lens_id) → Top‑K
* se LENS_API_KEY estiver configurada

```

## 4) Metodologia (reprodutível / passo a passo)

> Detalhes para qualquer pessoa reproduzir coleta, indexação, busca e avaliação.

### 4.1 Ambiente e versões

* **Python**: 3.10 ou 3.11 (x64)
* **Stack fixado**: `torch==2.2.2`, `numpy==1.26.4`, `transformers==4.41.2`, `tokenizers==0.19.1`, `huggingface-hub==0.23.5`, `sentence-transformers==2.7.0`, `fastapi==0.111.0`, `streamlit==1.36.0`.
* **Seeds**: 42 para `random`, `numpy` e `torch`.

Instalação (CPU, genérica):

```
python -m venv .venv
python -m pip install --upgrade pip
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2
pip install --no-cache-dir -r requirements.txt

```

### 4.2 Pré-processamento de texto

* Minúsculas, remoção de URLs, manter letras/dígitos/acentos, colapsar espaços.
* Detecção de idioma e *stopwords* multilíngues (opcional).
* Implementação em `processamento_texto.py` (`limpar_texto`, `gerar_embedding`).

### 4.3 Embeddings (modelo e normalização)

* Modelo: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (dim=768).
* `encode(texto, convert_to_numpy=True)`.
* L2: `v = e / ||e||2` (se norma>0). Persistir como lista de 768 floats.
* Comprimento: truncamento/padding interno do Sentence-Transformers; ajuste com `model.max_seq_length` se necessário.

### 4.4 Construção da base local

* **Via CSV/SQL** → converter para JSON (chaves: `lens_id/title/abstract/claims/description`), gerar embedding das claims e salvar em `base_patentes.json`.
* **Via Lens (opcional)** → `.env` com `LENS_API_KEY` e `python coleta_patentes.py`.
* Busca na Lens: `simple_query_string` em `claims^3, description^2, abstract`; respeitar 429/Retry-After; descartar itens sem claims.

**Esquema JSON do item**

```
{
  "lens_id":"...",
  "title":"...",
  "abstract":"...",
  "claims":"...",
  "description":"...",
  "embedding":[0.0123, -0.0456, 0.001, ...]
}

```

### 4.5 Busca local (cosine Top-K)

* Carregar matriz `E` (n×768) com embeddings L2; consulta `q` (768, L2).
* Cosine: `sims = E @ q`; selecionar Top‑K; ordenar decrescente; retornar metadados + `score` (0–1) e `%`.
* Implementações: `main.py` (/verificar, /buscar\_similares) e `buscar_similares.py` (CLI local).

### 4.6 Busca híbrida (Local + API externa)

* Se `.env` tiver `LENS_API_KEY`, chamar `lens_api.candidatos_por_claims(texto, top_n≈3K)`.
* Para cada candidato, obter `claims`/trecho, gerar embedding L2 local; **mesclar** (dedup por `lens_id/id`), recomputar `sims` e **re‑ranquear**; devolver Top‑K único.

### 4.7 Avaliação (pares rotulados — protocolo)

* **Amostragem**: estratifique por jurisdição/ano/IPC (documente no artigo).
* **Rotuladores**: ≥2 avaliadores de PI; **adjudicação** por terceiro em caso de empate.
* **Concordância**: reporte **Cohen’s κ**/**Krippendorff’s α**.
* **Métricas**: Accuracy, Macro‑F1, MCC, Kappa, ROC‑AUC, AP; para ranking: P\@10, R\@10, nDCG\@10.
* **Scripts**: `avaliar_pares_manual.py`, `avaliar_pares_rotulados_sem_zona.py`, `avaliar_metricas.py`.

Exemplo:

```
python avaliar_metricas.py --arquivo rotulos.jsonl --k 10

```

### 4.8 Execução ponta‑a‑ponta

```
# gerar/atualizar base (opcional)
python coleta_patentes.py
# subir API
python main.py --port 8000
# testar
curl -s http://127.0.0.1:8000/
curl -s -X POST http://127.0.0.1:8000/buscar_similares -H "Content-Type: application/json" -d '{"claims":"Texto das claims","top_k":10}'
# UI
streamlit run streamlit_app.py --server.port 8512

```

---

## 4) Endpoints principais (API)

* **GET /** → status: itens na base e se híbrido está disponível.
* **POST /buscar\_similares**

  ```
  { "claims": "texto das claims…", "top_k": 10 }

  ```
* **POST /verificar** (payload completo; usa `claims` se existir)

  ```
  { "titulo":"...", "resumo":"...", "claims":"...", "descricao":"...", "top_k": 10 }

  ```

**Resposta (exemplo):**

```
{
  "resultados": [
    {"lens_id":"140-...-70X","title":"...","link":"https://www.lens.org/lens/patent/140-...-70X","score":0.8245,"similaridade_percentual":82.45,"fonte":"local"}
  ]
}

```

---

## 5) Base local (formato e regra de ouro)

Arquivo \`\` (lista de objetos):

```
{
  "lens_id":"...",
  "title":"...",
  "abstract":"...",
  "claims":"...",
  "description":"...",
  "embedding":[0.0123, -0.0456, ...]  // vetor 768 **L2-normalizado** das claims
}

```

**Sempre salve o embedding das *********************claims********************* já L2‑normalizado.**

---

## 6) Híbrido (Local + Lens) — opcional

1. Crie `.env` na raiz:

   ```
   LENS_API_KEY=SEU_TOKEN_DA_LENS

   ```
2. Mantenha `lens_api.py` e `buscar_hibrido.py` na raiz. O servidor tenta híbrido e faz **fallback** para Local se falhar.

---

## 7) Requisitos (matriz estável — Windows/CPU)

* `torch==2.2.2` + `numpy==1.26.4`
* `transformers==4.41.2` + `tokenizers==0.19.1` + `huggingface-hub==0.23.5`
* `sentence-transformers==2.7.0`
* `fastapi==0.111.0` · `starlette==0.37.2` · `uvicorn[standard]==0.30.1`
* `streamlit==1.36.0` · `scikit-learn==1.5.0`

> Instale o Torch (CPU) **antes** do restante, pelo índice da PyTorch (comando acima).
> **Arquivo principal de instalação**: use `requirements.txt` deste repositório para garantir compatibilidade.

---

## 8) Como gerar a base rapidamente

* **CSV/SQL → JSON**: extraia `id/title/claims/abstract/description`, calcule `embedding` (claims) com `processamento_texto.gerar_embedding(..., normalize=True)` e salve no formato do item 5.
* **Via Lens**: `python coleta_patentes.py` (respeita `Retry-After`).
* **Reprocessar** embeddings: `python recalcular_embeddings.py`.

---

## 9) Streamlit (UI)

* Modo **Local (JSON)**, **Híbrido** (se `.env` presente) ou **Via API FastAPI** (aponta para `http://127.0.0.1:8000`).
* Mostra **Top‑K** com link da Lens e trecho das claims.

---

## 10) Troubleshooting (rápido)

* **Porta ocupada**: mude `--server.port` (UI) ou `--port` (API).
* **NumPy/Torch**: use `numpy==1.26.4` com `torch==2.2.2`.
* **Tokenizers/Transformers**: `transformers 4.41.x` requer `tokenizers >=0.19,<0.20` → use `0.19.1`.
* **Base vazia**: gere `base_patentes.json` (item 8).

---

## 11) Licença & aviso

* **Licença**: defina MIT/Apache‑2.0.
* **Aviso**: ferramenta de apoio à triagem; não substitui análise técnica/jurídica.

---

## 12) Usar outro banco de dados (local ou API)

Você **não** está preso ao `base_patentes.json`. Qualquer fonte pode ser usada, desde que entregue **texto de claims** e, idealmente, **embeddings L2** no mesmo espaço do modelo.

### A) Outro **arquivo local** (JSON)

1. Converta seu banco para o **esquema do item 5** (chaves e embedding L2 das *claims*).
2. Salve como `base_meu_bd.json`.
3. Altere o caminho no servidor/API:

   * Em `main.py`, troque:

     ```
     BASE_PATH = "base_patentes.json"  # → "base_meu_bd.json"

     ```
   * (Se usar busca direta local no app) Em `buscar_similares.py`, troque `CAMINHO_ARQUIVO` para o novo JSON.

> Recomenda-se **exportar para JSON** no formato do item 5. É simples, rápido e reaproveita todo o pipeline.

### B) **CSV/SQL** → JSON (conversão rápida)

Use um adaptador simples para transformar suas tabelas em JSON compatível:

```
# adapters/csv_para_json.py (exemplo)
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
            "title": r.get('title', ''),
            "abstract": r.get('abstract', ''),
            "claims": claims,
            "description": r.get('description', ''),
            "embedding": emb
        })
    with open(saida_json, 'w', encoding='utf-8') as f:
        json.dump(itens, f, ensure_ascii=False, indent=2)

```

Depois aponte `BASE_PATH`/`CAMINHO_ARQUIVO` para o JSON gerado.

### C) **Outra API externa** (além da Lens)

Adapte um provedor que retorne **candidatos por claims**, depois **re‑ranqueie localmente**:

1. Crie um módulo do provedor, e.g. `providers/meu_provedor.py`:

   ```
   # providers/meu_provedor.py (exemplo)
   import os, requests
   from dotenv import load_dotenv
   load_dotenv()
   API_KEY = os.getenv("MEU_API_KEY")

   def candidatos_por_claims(texto_claims: str, top_n: int = 30):
       # 1) chame a API e traga candidatos (id, título, claims/trecho, link)
       # 2) normalize campos e retorne lista de dicts compatíveis
       # Estrutura mínima por item: {"id":"...","title":"...","claims":"...","link":"..."}
       return []

   ```
2. Inclua no **híbrido** (`buscar_hibrido.py`):

   ```
   from providers.meu_provedor import candidatos_por_claims as prov_meu

   PROVEDORES = [
       ("lens", candidatos_por_claims),   # já existente
       ("meu",  prov_meu),                # novo provedor
   ]

   todos = []
   for nome, fn in PROVEDORES:
       cands = fn(texto_claims, top_n=top_k*3)
       # calcule embedding localmente e anexe ao pool
       for c in cands:
           emb = gerar_embedding(texto_claims=c.get("claims",""), normalize=True)
           todos.append({
               "lens_id": c.get("id",""),
               "title": c.get("title",""),
               "claims": c.get("claims",""),
               "link": c.get("link",""),
               "embedding": emb.tolist(),
               "fonte": "externa"
           })
   # depois: dedup por lens_id/id, cosine com a query e Top‑K

   ```
3. Configure a chave no `.env` (ex.: `MEU_API_KEY=...`).

> Dica: se a API externa não fornece **claims** completas, use **resumo/trecho** relevante. O re‑ranqueamento local via embedding alinha as escalas.

### D) Observações importantes

* **Consistência de espaço vetorial**: se trocar o **modelo de embedding**, **recalcule toda a base** para o mesmo espaço (caso contrário, cosine não é comparável).
* **Tamanho da base**: para bases muito grandes, considere indexadores ANN (FAISS/HNSW) no futuro.
* **Chaves/segurança**: mantenha credenciais apenas no back‑end e use HTTPS em produção.

---

## 13) Pré‑requisitos (ambiente)

* **Python**: 3.10 ou 3.11 recomendados (Windows 10/11 x64). Funciona em Linux/macOS; ajuste instalação do Torch conforme plataforma.
* **RAM**: 8 GB mínimo (12–16 GB recomendado) para trabalhar com bases médias.
* **Disco**: 2–5 GB livres (modelos + cache + base local). Bases grandes exigem mais.
* **Rede**: necessária para baixar o modelo na 1ª execução e para o modo **Híbrido** (Lens ou outras APIs).
* **Portas**: API padrão `8000`, Streamlit `8512` (ajustáveis).
* **GPU**: opcional; o projeto foi configurado para **CPU**. Para GPU, instale a variante adequada do Torch e teste as versões do stack.

---

## 14) Testes rápidos / verificação

### 14.0 Rodar toda a suíte

> Execute a partir da **raiz do projeto**. Se seu código está em `src/`, defina o `PYTHONPATH`.

**Windows (PowerShell):**

```
# ativar venv
. .\.venv\Scripts\activate
# expor src para os testes
$env:PYTHONPATH = "$PWD\src"
# rodar todos os testes em tests/
python -m unittest discover -s tests -p "test_*.py" -v

```

> Dica: na 1ª execução o modelo é baixado; a segunda rodada fica bem mais rápida.
> Se não quiser mexer em `PYTHONPATH`, no topo de cada teste você pode inserir:
>
> ```
> import os, sys
> ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
> SRC  = os.path.join(ROOT, "src")
> for p in (ROOT, SRC):
>     sys.path.insert(0, p) if p not in sys.path else None
>
> ```

### 14.1 Sanidade do modelo

```
python - << 'PY'
from sentence_transformers import SentenceTransformer
import numpy as np
m = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
e = m.encode("teste rápido", convert_to_numpy=True)
print("shape:", e.shape, "norm:", float(np.linalg.norm(e)))
PY

```

Saída esperada: `shape: (768,)` e `norm > 0`.

### 14.2 API online

```
# subir API
python main.py --port 8000
# status
curl -s http://127.0.0.1:8000/
# busca
curl -s -X POST http://127.0.0.1:8000/buscar_similares \
  -H "Content-Type: application/json" \
  -d '{"claims":"Um sistema para ...","top_k":10}'

```

> Se seu `main.py` estiver em `src/` e você preferir usar uvicorn direto:
> `uvicorn src.main:app --host 127.0.0.1 --port 8000`

### 14.3 UI (Streamlit)

```
streamlit run streamlit_app.py --server.port 8512

```

Escolha **Local**, **Híbrido** (se `.env`) ou **Via API FastAPI**.

### 14.4 Scripts de avaliação (se aplicável)

* `similaridade.py`: funções de score (0–1 para métricas; % para exibição).
* `avaliar_metricas.py`: gera métricas agregadas (precisão\@K, etc.).
* `avaliar_pares_manual.py` / `avaliar_pares_rotulados_sem_zona.py`: avaliam pares rotulados.

Exemplo:

```
python avaliar_metricas.py --arquivo rotulos.jsonl --k 10

```

> Observação: os testes e avaliações usam **apenas base local**; não chamam a Lens (sem rede).

---

## 15) Exemplo visual

Adicione imagens na pasta `assets/` e referencie aqui:

**UI (Streamlit)**

**Resposta da API**

> Substitua pelos seus prints. No GitHub, arraste as imagens para a pasta `assets/`.

---

## 16) Links úteis

* Modelo (Hugging Face): [https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
* [Sentence](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)-Transformers (docs): [https://www.sbert.net/](https://www.sbert.net/)
* [Transfor](https://www.sbert.net/)mers (docs): [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)
* [FastAPI ](https://huggingface.co/docs/transformers/index)(docs): [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
* [Streamli](https://fastapi.tiangolo.com/)t (docs): [https://docs.streamlit.io/](https://docs.streamlit.io/)
* [Lens API](https://docs.streamlit.io/) (docs): [https://docs.lens.org/](https://docs.lens.org/)

---

## [17) Con](https://docs.lens.org/)tato & contribuições

* **Dúvidas/Sugestões**: abra uma *issue* no repositório ou envie e‑mail para [**SEU\_EMAIL@exemplo.com**](mailto:SEU_EMAIL@exemplo.com).
* **Contribuir**: faça um *fork*, crie uma *branch* (`feat/minha-feature`), abra um *PR* com descrição objetiva e testes (quando aplicável).

**Resposta da API**

> Substitua pelos seus prints. No GitHub, arraste as imagens para a pasta `assets/`.

---

## 16) Links úteis

* Modelo (Hugging Face): [https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
* Sentence-Transformers (docs): [https://www.sbert.net/](https://www.sbert.net/)
* Transformers (docs): [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)
* FastAPI (docs): [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
* Streamlit (docs): [https://docs.streamlit.io/](https://docs.streamlit.io/)
* Lens API (docs): [https://docs.lens.org/](https://docs.lens.org/)

---

## 17) Contato & contribuições

* **Dúvidas/Sugestões**: abra uma *issue* no repositório ou envie e‑mail para [isabela.andradeaguiar1@gmail.com](mailto:isabela.andradeaguiar1@gmail.com)\*\*.
* **Contribuir**: faça um *fork*, crie uma *branch* (`feat/minha-feature`), abra um *PR* com descrição objetiva e testes (quando aplicável).
