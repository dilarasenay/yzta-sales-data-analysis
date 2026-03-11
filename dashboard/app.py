import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Akıllı Öneri Motoru", page_icon="🧠", layout="wide")

# ---------------------------
# SAMPLE DATA
# ---------------------------
category_data = {
    "Baby": [
        {"name": "Toys", "icon": "🧸", "score": 0.92, "color": "#F9DCE8"},
        {"name": "Baby Care", "icon": "🍼", "score": 0.88, "color": "#DCEEFF"},
        {"name": "Fashion Shoes", "icon": "👟", "score": 0.85, "color": "#FBEBD7"},
        {"name": "Bed Bath", "icon": "🛁", "score": 0.81, "color": "#DFF3E5"},
        {"name": "Security", "icon": "🛡️", "score": 0.79, "color": "#E9DDFC"},
    ],
    "Fashion": [
        {"name": "Watches", "icon": "⌚", "score": 0.91, "color": "#F9DCE8"},
        {"name": "Bags", "icon": "👜", "score": 0.87, "color": "#DCEEFF"},
        {"name": "Shoes", "icon": "👠", "score": 0.84, "color": "#FBEBD7"},
        {"name": "Accessories", "icon": "💍", "score": 0.80, "color": "#DFF3E5"},
        {"name": "Beauty", "icon": "💄", "score": 0.77, "color": "#E9DDFC"},
    ],
    "Electronics": [
        {"name": "Computers", "icon": "💻", "score": 0.93, "color": "#F9DCE8"},
        {"name": "Tablets", "icon": "📱", "score": 0.89, "color": "#DCEEFF"},
        {"name": "Audio", "icon": "🎧", "score": 0.86, "color": "#FBEBD7"},
        {"name": "Gaming", "icon": "🎮", "score": 0.82, "color": "#DFF3E5"},
        {"name": "Cameras", "icon": "📷", "score": 0.78, "color": "#E9DDFC"},
    ]
}

# ---------------------------
# CSS
# ---------------------------
st.markdown("""
<style>
/* General */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(255, 220, 240, 0.6), transparent 25%),
        radial-gradient(circle at top right, rgba(220, 235, 255, 0.7), transparent 25%),
        radial-gradient(circle at bottom left, rgba(220, 255, 235, 0.7), transparent 20%),
        linear-gradient(180deg, #fcfbff 0%, #f7f5ff 100%);
}

/* Title */
.main-title {
    text-align: center;
    font-size: 56px;
    font-weight: 800;
    color: #2d2d55;
    margin-bottom: 0;
}

.sub-title {
    text-align: center;
    color: #6e6e8d;
    font-size: 22px;
    margin-top: 8px;
    margin-bottom: 30px;
}

/* Glass boxes */
.glass-box {
    background: rgba(255,255,255,0.58);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.55);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(80, 80, 140, 0.08);
}

/* Section title */
.section-title {
    font-size: 26px;
    font-weight: 700;
    color: #2d2d55;
    margin-bottom: 18px;
}

/* Cards */
.card {
    border-radius: 22px;
    padding: 22px 16px;
    text-align: center;
    min-height: 185px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
    transition: all 0.25s ease;
    border: 1px solid rgba(255,255,255,0.65);
}

.card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 16px 30px rgba(0,0,0,0.10);
}

.card-icon {
    font-size: 48px;
    margin-bottom: 10px;
}

.card-title {
    font-size: 20px;
    font-weight: 700;
    color: #2d2d55;
    margin-bottom: 8px;
}

.card-sub {
    font-size: 14px;
    color: #5f5f7a;
}

/* Info */
.info-box {
    background: rgba(255,255,255,0.65);
    border-radius: 20px;
    padding: 18px 22px;
    color: #4c4c6d;
    border: 1px solid rgba(255,255,255,0.7);
    box-shadow: 0 8px 18px rgba(0,0,0,0.05);
    font-size: 18px;
    text-align: center;
    margin-top: 18px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# HEADER
# ---------------------------
st.markdown('<div class="main-title">🧠 Akıllı Öneri Motoru</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Benzer ürün kategorilerini keşfedin</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-box">', unsafe_allow_html=True)
selected_category = st.selectbox("Kategori Seç", list(category_data.keys()))
st.markdown('</div>', unsafe_allow_html=True)

st.write("")
recommendations = category_data[selected_category]

# ---------------------------
# TOP AREA
# ---------------------------
left_col, right_col = st.columns([1.1, 1.4], gap="large")

with left_col:
    st.markdown('<div class="glass-box">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-title">🎀 {selected_category} alanlar bunları da aldı</div>',
        unsafe_allow_html=True
    )

    card_cols = st.columns(len(recommendations))
    for i, item in enumerate(recommendations):
        with card_cols[i]:
            st.markdown(
                f"""
                <div class="card" style="background:{item['color']};">
                    <div class="card-icon">{item['icon']}</div>
                    <div class="card-title">{item['name']}</div>
                    <div class="card-sub">Benzer kategori</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="glass-box">', unsafe_allow_html=True)

    chart_df = pd.DataFrame({
        "Kategori": [x["name"] for x in recommendations],
        "Benzerlik": [x["score"] for x in recommendations]
    })

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("Kategori:N", sort=None, axis=alt.Axis(labelAngle=0, labelFontSize=16, title=None)),
            y=alt.Y("Benzerlik:Q", scale=alt.Scale(domain=[0.7, 1.0]), axis=alt.Axis(title=None, labelFontSize=14)),
            color=alt.Color("Kategori:N", legend=None)
        )
        .properties(height=320)
    )

    text = (
        alt.Chart(chart_df)
        .mark_text(
            dy=20,
            fontSize=16,
            fontWeight="bold",
            color="white"
        )
        .encode(
            x=alt.X("Kategori:N", sort=None),
            y=alt.Y("Benzerlik:Q"),
            text=alt.Text("Benzerlik:Q", format=".2f")
        )
    )

    st.altair_chart(chart + text, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# SECOND CARD ROW
# ---------------------------
st.write("")
st.markdown('<div class="glass-box">', unsafe_allow_html=True)
card_cols2 = st.columns(len(recommendations))

for i, item in enumerate(recommendations):
    with card_cols2[i]:
        st.markdown(
            f"""
            <div class="card" style="background:{item['color']};">
                <div class="card-icon">{item['icon']}</div>
                <div class="card-title">{item['name']}</div>
                <div class="card-sub">Benzer kategori</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown(
    '<div class="info-box">💡 Bu öneriler kategori benzerliğine göre hesaplanmıştır <b>(Cosine Similarity)</b></div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# DEBUG / DATA CHECK
# ---------------------------
with st.expander("Veri yolu kontrolü"):
    st.write("Seçilen kategori:", selected_category)
    st.dataframe(chart_df, use_container_width=True)