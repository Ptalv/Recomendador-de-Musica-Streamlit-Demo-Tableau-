import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pyarrow.parquet as pq

st.set_page_config(page_title="KNN Music Recommender", layout="wide")

# ---------- Cargar artefactos ----------
@st.cache_data
def load_df():
    return pq.read_table("artifacts/df_app.parquet").to_pandas()

@st.cache_resource
def load_models():
    X_main = joblib.load("artifacts/X_main.joblib")
    knn_main = joblib.load("artifacts/knn_main.joblib")
    return X_main, knn_main

df = load_df()
X_main, knn_main = load_models()

# ---------- Session state ----------
if "liked_ids" not in st.session_state:
    st.session_state.liked_ids = []
if "liked_labels" not in st.session_state:
    st.session_state.liked_labels = []

# ---------- Recomendador (sin repetidos visibles) ----------
def recommend_for_user(liked_track_ids, top_n=10, extra_pool=300):
    liked_idx = df.index[df["track_id"].isin(liked_track_ids)].tolist()
    if not liked_idx:
        raise ValueError("No encontré esos likes en el catálogo.")

    # Vector usuario = promedio de los likes
    user_vec = X_main[liked_idx].mean(axis=0)
    user_vec = np.asarray(user_vec).reshape(1, -1)

    # Pedimos un pool amplio para poder quitar repetidos y rellenar
    n_neighbors = min(len(df), top_n + len(liked_idx) + extra_pool)
    distances, indices = knn_main.kneighbors(user_vec, n_neighbors=n_neighbors)

    # Filtrar: quitar los likes
    cand_idx = [i for i in indices[0] if i not in liked_idx]
    cand_dist = [d for i, d in zip(indices[0], distances[0]) if i not in liked_idx]

    out = df.loc[cand_idx, ["track_name", "artist_name", "artist_genres_key", "track_id"]].copy()
    out["distance"] = cand_dist

    # Quitar repetidos "visibles" (misma canción + artista)
    out = out.sort_values("distance").drop_duplicates(subset=["track_name", "artist_name"], keep="first")

   
    out = out.head(top_n).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)

    
    return out[["rank", "track_name", "artist_name","distance"]]

# ---------- UI ----------
st.title("🎧 Demo — KNN Recomendador de Música")

with st.sidebar:
    top_n = st.slider("Top-N recomendaciones", 5, 30, 10)
    st.caption("Tip: agrega 2–5 likes para mejores resultados.")

    if st.button("🧹 Limpiar likes"):
        st.session_state.liked_ids = []
        st.session_state.liked_labels = []

st.subheader("Tus likes actuales")
if st.session_state.liked_labels:
    st.dataframe(
        pd.DataFrame({"Likes": st.session_state.liked_labels}),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Aún no has agregado likes. Busca una canción o artista y agrégalos.")

st.subheader("1) Busca canciones para agregar")
q = st.text_input("Buscar por canción o artista", "")

view = df.copy()
if q.strip():
    mask = (
        view["track_name"].astype(str).str.contains(q, case=False, na=False)
        | view["artist_name"].astype(str).str.contains(q, case=False, na=False)
    )
    view = view[mask]

# Limitar resultados para que el selector sea rápido
view = view.head(250).copy()

# Label que ve el usuario (sin track_id)
view["label"] = view["track_name"].astype(str) + " — " + view["artist_name"].astype(str)

# Mapeo oculto label -> track_id (interno)
label_to_id = dict(zip(view["label"], view["track_id"]))

selected_labels = st.multiselect(
    "Selecciona canciones (se guardan aunque cambies la búsqueda)",
    options=view["label"].tolist(),
)

# Agregar al carrito de likes sin duplicar (por track_id)
for lab in selected_labels:
    tid = label_to_id.get(lab)
    if tid is None:
        continue
    if tid not in st.session_state.liked_ids:
        st.session_state.liked_ids.append(tid)
        st.session_state.liked_labels.append(lab)

st.divider()
st.subheader("2) Generar recomendaciones")

if st.button("✨ Recomendar", type="primary"):
    if not st.session_state.liked_ids:
        st.warning("Agrega al menos 1 like.")
    else:
        recs = recommend_for_user(st.session_state.liked_ids, top_n=top_n)
        st.success("Listo ✅")
        st.dataframe(recs, use_container_width=True)

        csv = recs.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar recomendaciones (CSV)", csv, "recomendaciones.csv", "text/csv")
