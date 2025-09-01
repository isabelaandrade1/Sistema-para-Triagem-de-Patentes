# streamlit_app.py — UI para triagem de patentes (Local / Híbrido / API)
import os
import json
import pandas as pd
import streamlit as st

# --- modos de busca disponíveis (imports com fallback) -----------------------
_HAS_HYBRID = False
try:
    from buscar_hibrido import buscar_similares_hibrido  # opcional (Local + Lens)
    _HAS_HYBRID = True
except Exception:
    buscar_similares_hibrido = None  # type: ignore

try:
    from buscar_similares import buscar_similares  # busca local (base_patentes.json)
except Exception as e:
    st.error(f"Erro ao importar buscar_similares: {e}")
    st.stop()

# --- helpers -----------------------------------------------------------------
def _api_call(api_url: str, route: str, payload: dict):
    """Chama a API FastAPI (se você preferir usar backend)."""
    import requests
    url = api_url.rstrip("/") + route
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("resultados", [])
    except Exception as e:
        st.error(f"Falha ao chamar API {url}: {e}")
        return []

def _format_result_list(res):
    """Normaliza campos e prepara DataFrame para exibição."""
    rows = []
    for i, it in enumerate(res, 1):
        lens_id = it.get("lens_id", "")
        title = it.get("title") or it.get("titulo") or ""
        link = it.get("link") or (f"https://www.lens.org/lens/patent/{lens_id}" if lens_id else "")
        fonte = it.get("fonte", "local")
        # pega percentual se houver, senão usa score 0-1 e converte
        perc = it.get("similaridade_percentual")
        if perc is None:
            score = float(it.get("score", 0.0))
            perc = round(score * 100.0, 2)
        rows.append({
            "rank": i,
            "lens_id": lens_id,
            "similaridade_%": perc,
            "fonte": fonte,
            "title": title,
            "link": link,
            "claims_trecho": (it.get("claims") or "")[:300],
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["rank","lens_id","similaridade_%","fonte","title","link","claims_trecho"])
    return df

def _base_existe():
    return os.path.exists("base_patentes.json")

# --- página ------------------------------------------------------------------
st.set_page_config(page_title="Triagem de Patentes (Semântica)", layout="wide")
st.title("🔎 Triagem de Patentes por Similaridade Semântica")

with st.sidebar:
    st.header("Configurações")
    modo = st.radio(
        "Modo de busca",
        options=["Local (JSON)", "Híbrido (Local + Lens)", "Via API FastAPI"],
        index=0 if not _HAS_HYBRID else 1
    )

    top_k = st.slider("Top-K", min_value=1, max_value=50, value=10, step=1)

    api_url = st.text_input("API URL (para modo Via API)", value="http://127.0.0.1:8000")
    st.caption("Se marcar 'Via API FastAPI', a busca chama /buscar_similares.")

    st.markdown("---")
    st.caption(f"📦 base_patentes.json: {'✅' if _base_existe() else '❌ não encontrado'}")
    st.caption(f"🧠 Híbrido disponível (módulo buscar_hibrido): {'✅' if _HAS_HYBRID else '❌'}")

st.subheader("Cole as claims da sua invenção")
claims = st.text_area(
    "Claims (texto)",
    height=200,
    placeholder="Ex.: Um sistema para detectar anomalias em fluxos de eventos...",
)

col1, col2 = st.columns([1,3])
with col1:
    run = st.button("Buscar similares", use_container_width=True)

with col2:
    st.write("")

st.markdown("---")

if run:
    if not claims or not claims.strip():
        st.warning("Por favor, cole as *claims* antes de buscar.")
        st.stop()

    resultados = []

    # --- modo API ------------------------------------------------------------
    if modo == "Via API FastAPI":
        payload = {"claims": claims.strip(), "top_k": int(top_k)}
        resultados = _api_call(api_url, "/buscar_similares", payload)

    # --- modo Híbrido (Local + Lens) ----------------------------------------
    elif modo == "Híbrido (Local + Lens)":
        if not _HAS_HYBRID or not callable(buscar_similares_hibrido):
            st.info("Módulo híbrido indisponível. Usando **Local (JSON)** como fallback.")
            resultados = buscar_similares(claims, top_k=int(top_k))
        else:
            try:
                resultados = buscar_similares_hibrido(claims, top_k=int(top_k))
                if not resultados:
                    st.info("Sem resultados do híbrido. Usando **Local (JSON)** como fallback.")
                    resultados = buscar_similares(claims, top_k=int(top_k))
            except Exception as e:
                st.warning(f"Híbrido falhou: {e}. Usando **Local (JSON)** como fallback.")
                resultados = buscar_similares(claims, top_k=int(top_k))

    # --- modo Local ----------------------------------------------------------
    else:
        if not _base_existe():
            st.error("Arquivo base_patentes.json não encontrado na raiz do projeto.")
            st.stop()
        try:
            resultados = buscar_similares(claims, top_k=int(top_k))
        except Exception as e:
            st.error(f"Falha na busca local: {e}")
            st.stop()

    # --- exibição ------------------------------------------------------------
    if not resultados:
        st.info("Nenhum resultado retornado.")
        st.stop()

    df = _format_result_list(resultados)
    if df.empty:
        st.info("Nenhum resultado formatado.")
        st.stop()

    st.subheader(f"Top-{len(df)} mais similares")
    st.dataframe(
        df.drop(columns=["claims_trecho"]),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Ver trechos de claims (até 300 caracteres)"):
        for _, row in df.iterrows():
            link = row["link"]
            titulo = row["title"] or row["lens_id"]
            st.markdown(f"**{int(row['rank'])}. [{titulo}]({link})** — {row['similaridade_%']:.2f}%  \n"
                        f"<small>Fonte: {row['fonte']}</small>", unsafe_allow_html=True)
            st.write(row["claims_trecho"])
            st.markdown("---")
