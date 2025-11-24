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
# 📂 LOAD DATA & MODEL
# ===============================
@st.cache_data
def load_data():
    try:
        rating_df = pd.read_csv("Dataset_Rating_Mgl.csv")
        place_df = pd.read_csv("Dataset_tourisMagelang.csv")
        user_df = pd.read_csv("Dataset_usermgl.csv")
    except FileNotFoundError:
        st.error("❌ File dataset tidak ditemukan. Pastikan semua CSV sudah diunggah.")
        return None, None, None
    return rating_df, place_df, user_df

@st.cache_data
def load_model():
    try:
        model = joblib.load("mf_model.pkl")
    except FileNotFoundError:
        st.error("❌ File mf_model.pkl tidak ditemukan.")
        return None
    return model

rating_df, place_df, user_df = load_data()
model = load_model()

if rating_df is None or model is None:
    st.stop()

# ===============================
# 🧠 FUNGSI REKOMENDASI (MATRIX FACTORIZATION)
# ===============================
def predict_rating(user_id, place_id):
    """Prediksi rating menggunakan model Matrix Factorization (SVD)."""
    pred = model.predict(user_id, place_id)
    return pred.est

def recommend_places(user_id, top_n=5):
    """Rekomendasi berdasarkan prediksi rating tertinggi (SVD)."""
    rated_places = rating_df[rating_df['User_Id'] == user_id]['Place_Id'].values
    all_places = place_df['Place_Id'].values
    unrated = [p for p in all_places if p not in rated_places]

    predictions = []
    for place in unrated:
        est_rating = predict_rating(user_id, place)
        place_name = place_df[place_df['Place_Id'] == place]['Place_Name'].values[0]
        predictions.append((place, place_name, est_rating))

    predictions.sort(key=lambda x: x[2], reverse=True)
    return predictions[:top_n]

# ===============================
# 🔍 FUNGSI SEARCH CERDAS
# ===============================
def search_place(keyword):
    """Pencarian tempat berdasarkan nama atau deskripsi."""
    keyword_lower = keyword.lower()

    name_match = place_df[place_df['Place_Name'].str.contains(keyword, case=False, na=False)].copy()
    desc_match = place_df[place_df['Description'].str.contains(keyword, case=False, na=False)].copy()

    results = pd.concat([name_match, desc_match]).drop_duplicates().reset_index(drop=True)

    if results.empty:
        return results

    def relevance_score(row):
        name_score = row['Place_Name'].lower().count(keyword_lower)
        desc_score = row['Description'].lower().count(keyword_lower) if pd.notna(row['Description']) else 0
        return name_score * 2 + desc_score

    results["Relevance"] = results.apply(relevance_score, axis=1)
    return results.sort_values(by="Relevance", ascending=False).reset_index(drop=True)

# ===============================
# 💬 FUNGSI MENAMPILKAN ULASAN
# ===============================
def get_reviews_for_place(place_name):
    reviews = rating_df.merge(user_df, on="User_Id", how="left")
    return reviews[reviews['Place_Name'].str.lower() == place_name.lower()]

# ===============================
# 🖥️ ANTARMUKA STREAMLIT
# ===============================
st.title("🏞️ Sistem Rekomendasi Wisata Magelang")
st.caption("Menggunakan *Collaborative Filtering* dengan algoritma *Matrix Factorization (SVD)*")

st.markdown("---")

# 🔍 Input Search
search_query = st.text_input("🔍 Cari Tempat Wisata", placeholder="Misal: Borobudur atau Hutan Pinus")

if search_query:
    results = search_place(search_query)

    if results.empty:
        st.warning("Tempat tidak ditemukan. Coba kata lain.")
    else:
        for idx, row in results.iterrows():
            st.subheader(f"📍 {row['Place_Name']}")

            # Highlight kata kunci
            if pd.notna(row['Description']):
                desc = row['Description']
                highlighted = re.sub(f"(?i)({search_query})", r"**\1**", desc)
                st.markdown(f"📝 {highlighted}")
            else:
                st.info("Belum ada deskripsi.")

            # Rating rata-rata
            avg_rating = rating_df[rating_df['Place_Name'].str.lower() ==
                                   row['Place_Name'].lower()]['Place_Rating'].mean()

            if not np.isnan(avg_rating):
                st.write(f"⭐ **Rata-rata Rating:** {avg_rating:.2f}/5.0")
            else:
                st.write("⭐ Belum ada rating.")

            # Ulasan
            reviews = get_reviews_for_place(row['Place_Name'])
            if not reviews.empty:
                st.write("💬 **Ulasan Pengguna:**")
                for _, review in reviews.iterrows():
                    info = f"{review['Gender']}, {review['Age']} ({review['Regional']})"
                    st.markdown(f"- 🧍 **{info}** memberi rating `{int(review['Place_Rating'])}` ⭐")
            else:
                st.info("Belum ada ulasan.")

            st.markdown("---")

else:
    st.info("Masukkan kata kunci untuk mulai mencari tempat wisata.")

st.sidebar.success("✅ Sistem siap digunakan!")
st.sidebar.caption("Dibuat oleh Armis Dayanti ❤️")
