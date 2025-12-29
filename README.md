# Recomendador de Musica Demo – KNN + Streamlit + Tableau

Este proyecto es una demo ligera e interactiva de un **recomendador de música basado en K-Nearest Neighbors (KNN)** entrenado con metadata de Spotify.  
Incluye una aplicación en **Streamlit** para probar recomendaciones a partir de likes persistentes y un **dashboard analítico en Tableau** para storytelling del dataset.

---

## 🎯 Objetivo
Recomendar canciones similares a partir de una selección de tracks que el usuario marca como favoritos (“likes”), usando similitud por **distancia coseno** y evitando repeticiones visibles.

---

## ⚙️ Tecnologías utilizadas

- 🐍 **Python 3.10+**
- 📊 **pandas, numpy, matplotlib**
- 🤖 **scikit-learn**
  - `ColumnTransformer`
  - `OneHotEncoder`
  - `StandardScaler`
  - `TfidfVectorizer` para embeddings de géneros clave
  - `NearestNeighbors` para el modelo KNN
- 🧩 **Streamlit** (demo interactiva)
- 📦 **joblib** (serialización de artefactos)
- 📁 **Tableau** para análisis visual del catálogo
- 🧠 **KMeans** (clustering offline para el dashboard)

---

## ✨ Funcionalidades

- Búsqueda por canción o artista.
- Likes persistentes (no se pierden al filtrar).
- Control de **Top-N recomendaciones**.
- Eliminación de **repetidos visibles** por `track_name + artist_name`.
- Exportación de recomendaciones a CSV.
- Dashboard analítico en Tableau para portafolio.

---

## 📊 Dashboard de análisis en Tableau

Se desarrolló un **dashboard interactivo en Tableau** para explorar el dataset usado por el recomendador.  
El dashboard muestra:

- Popularidad de canciones y artistas
- Alcance del artista (seguidores)
- Géneros clave generados con TF-IDF
- Clusters de similitud (KMeans, calculado offline)

- 🔗 **Ver Dashboard en Tableau Public:**  
👉 https://public.tableau.com/app/profile/pedro.alvarez.martinez/viz/DashboardSpotifyPedroAlvarez/DASHBOARD

--

## 🚀 Correr la demo localmente

```bash
pip install -r requirements.txt
python -m streamlit run app.py --server.port 8502 --server.address 127.0.0.1


