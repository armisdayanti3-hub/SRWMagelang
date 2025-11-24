import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# ===============================
# 🔧 KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Rekomendasi Wisata Magelang",
    page_icon="🏞️",
    layout="wide"
)

# ===============================
# 📂 LOAD DATA
# ===============================
@st.cache_data
def load_data():
    try:
        rating_df = pd.read_csv("Dataset_Rating_Mgl.csv")
        place_df = pd.read_csv("Dataset_tourisMagelang.csv")
        user_df = pd.read_csv("Dataset_usermgl.csv")
    except FileNotFoundError:
        st.error("❌ File dataset tidak ditemukan.")
        return None, None, None
    return rating_df, place_df, user_df

# ===============================
# 📦 LOAD MODEL SVD
# ===============================
@st.cache_data
def load_model():
    try:
        model = joblib.load("mf_model.pkl")
    except FileNotFoundError:
        st.error("❌ File model SVD (mf_model.pkl) tidak ditemukan.")
        return None
    return model

rating_df, place_df, user_df = load_data()
model = load_model()

if rating_df is None or model is None:
    st.stop()

# ===============================
# 🧠 FUNGSI PREDIKSI (SVD)
# ===============================
def predict_rating(user_id, place_id):
    pred = model.predict(user_id, place_id)
    return pred.est

# ===============================
# ⭐ FUNGSI REKOMENDASI
# ===============================
def recommend_places(user_id, top_n=5):
    rated_places = rating_df[rating_df['User_Id'] == user_id]['Place_Id'].values
    all_places = place_df['Place_Id'].values
    unrated_places = [p for p in all_places if p not in rated_places]

    predictions = []
    for place in unrated_places:
        est = predict_rating(user_id, place)
        place_name = place_df[place_df['Place_Id'] == place]['Place_Name'].values[0]
        predictions.append((place_name, est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_n]

# ===============================
# 🔍 SEARCH WISATA
# ===============================
def search_place(keyword):
    keyword_lower = keyword.lower()

    name_match = place_df[place_df['Place_Name'].str.contains(keyword, case=False, na=False)].copy()
    desc_match = place_df[place_df['Description'].str.contains(keyword, case=False, na=False)].copy()

    results = pd.concat([name_match, desc_match]).drop_duplicates().reset_index(drop=True)
    if results.empty:
        return results

    def relevance_score(row):
        name_score = row['Place_Name'].lower().count(keyword_lower)
        desc_score = row['Description'].lower().count(keyword_lower)
        return name_score * 2 + desc_score

    results["Relevance"] = results.apply(relevance_score, axis=1)
    results = results.sort_values("Relevance", ascending=False)
    return results

# ===============================
# 🖥️ UI STREAMLIT
# ===============================
st.title("🏞️ Sistem Rekomendasi Wisata Magelang")
st.caption("Menggunakan Matrix Factorization (SVD)")

st.markdown("---")

# ===============================
# 🔍 SEARCH BAR
# ===============================
search_query = st.text_input("🔍 Cari Tempat Wisata", placeholder="Misal: Borobudur")

if search_query:
    results = search_place(search_query)

    if results.empty:
        st.warning("Tempat tidak ditemukan.")
    else:
        for idx, row in results.iterrows():
            st.subheader(f"📍 {row['Place_Name']}")

            desc = row['Description']
            highlighted_desc = re.sub(f"(?i)({search_query})", r"**\1**", desc)
            st.markdown(f"📝 {highlighted_desc}")

            avg = rating_df[rating_df['Place_Name'] == row['Place_Name']]['Place_Rating'].mean()
            st.write(f"⭐ Rata-rata Rating: {avg:.2f}/5.0")

            st.markdown("---")

else:
    st.info("Cari tempat wisata untuk melihat detailnya.")

# ===============================
# 🎯 REKOMENDASI
# ===============================
st.subheader("🎯 Rekomendasi Berdasarkan User ID")

selected_user = st.selectbox(
    "Pilih User ID:",
    sorted(user_df['User_Id'].unique())
)

if st.button("Tampilkan Rekomendasi"):
    rekom = recommend_places(selected_user, top_n=5)

    st.write(f"Top 5 rekomendasi untuk User {selected_user}:")
    for place, score in rekom:
        st.markdown(f"- **{place}** — Prediksi Rating: `{score:.2f}` ⭐")

st.sidebar.caption("Sistem Rekomendasi Wisata — SVD Version")
