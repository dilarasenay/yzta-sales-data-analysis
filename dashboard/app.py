import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Brazil E-Commerce Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================
# CUSTOM STYLING
# ============================================

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING
# ============================================

@st.cache_data
def load_data():
    """Veri dosyalarını yükle"""
    try:
        df = pd.read_parquet("data/processed/fact_orders.parquet")
        payments = pd.read_csv("data/raw/olist_order_payments_dataset.csv")
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
        return df, payments
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        return None, None

# ============================================
# RECOMMENDATION ENGINE
# ============================================

@st.cache_data
def build_recommendation_engine(df):
    """Recommendation engine'i oluştur"""
    try:
        # Kategori kolonunu belirle
        category_col = (
            "product_category_name_english"
            if "product_category_name_english" in df.columns
            else "product_category_name"
        )
        
        # Veri hazırlığı
        df_rec = df[["order_id", category_col]].dropna().drop_duplicates().copy()
        
        # Kategori isimlerini temizle
        df_rec[category_col] = (
            df_rec[category_col]
            .astype(str)
            .str.strip()
            .str.replace("_", " ")
            .str.title()
        )
        
        # Sadece multi-kategori siparişleri al
        order_counts = df_rec.groupby("order_id")[category_col].nunique()
        valid_orders = order_counts[order_counts > 1].index
        df_rec = df_rec[df_rec["order_id"].isin(valid_orders)]
        
        if len(df_rec) == 0:
            return None, None
        
        # Categorical dönüşüm
        df_rec["order_id"] = df_rec["order_id"].astype("category")
        df_rec[category_col] = df_rec[category_col].astype("category")
        
        # Matrix oluştur
        row = df_rec[category_col].cat.codes
        col = df_rec["order_id"].cat.codes
        n_categories = len(df_rec[category_col].cat.categories)
        n_orders = len(df_rec["order_id"].cat.categories)
        
        category_order_matrix = csr_matrix(
            ([1] * len(df_rec), (row, col)),
            shape=(n_categories, n_orders)
        )
        
        # Similarity hesapla
        similarity = cosine_similarity(category_order_matrix, dense_output=False)
        category_names = df_rec[category_col].cat.categories
        
        return category_names, similarity
    except Exception as e:
        st.warning(f"Recommendation engine oluşturma hatası: {e}")
        return None, None

def recommend_categories(category_name, category_names, similarity, top_n=5):
    """Benzer kategorileri öner"""
    if category_name not in category_names:
        return pd.DataFrame()
    
    idx = category_names.get_loc(category_name)
    sim_scores = similarity[idx].toarray().flatten()
    
    similar_idx = sim_scores.argsort()[::-1]
    
    recommendations = pd.DataFrame({
        "recommended_category": category_names[similar_idx],
        "similarity_score": sim_scores[similar_idx]
    })
    
    # Kendisini çıkar
    recommendations = recommendations[
        recommendations["recommended_category"] != category_name
    ]
    
    # 0 similarity çıkar
    recommendations = recommendations[recommendations["similarity_score"] > 0]
    
    # Duplicates temizle
    recommendations = recommendations.drop_duplicates("recommended_category")
    
    return recommendations.head(top_n).reset_index(drop=True)

# ============================================
# LOAD DATA
# ============================================

df, payments = load_data()

if df is None:
    st.error("Veri yüklenemedi! Lütfen data/processed/fact_orders.parquet dosyasının var olduğunu kontrol edin.")
    st.stop()

# ============================================
# BUILD RECOMMENDATION ENGINE
# ============================================

category_names, similarity = build_recommendation_engine(df)

# ============================================
# HEADER
# ============================================

st.markdown("""
# 📊 Brazil E-Commerce Analytics

Satış, müşteri davranışı, lojistik ve ödemeler analizi
""")

st.markdown("---")

# ============================================
# TABS
# ============================================

tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Akıllı Öneri Motoru"])

# ============================================
# TAB 1: DASHBOARD
# ============================================

with tab1:
    
    # ---- SIDEBAR FILTERS FOR DASHBOARD ----
    st.sidebar.title("🔎 Dashboard Filtreri")
    
    # Şehir seçimi
    cities = ["Tüm Şehirler"] + sorted(df["customer_city"].dropna().unique().tolist())
    selected_city = st.sidebar.selectbox("Şehir Seçin", cities, key="city_dashboard")
    
    # Kategori seçimi
    if "product_category_name_english" in df.columns:
        categories = ["Tüm Kategoriler"] + sorted(df["product_category_name_english"].dropna().unique().tolist())
    else:
        categories = ["Tüm Kategoriler"] + sorted(df["product_category_name"].dropna().unique().tolist())
    
    selected_category = st.sidebar.selectbox("Kategori Seçin", categories, key="category_dashboard")
    
    # ---- FILTER DATA FOR DASHBOARD ----
    df_filtered = df.copy()
    
    if selected_city != "Tüm Şehirler":
        df_filtered = df_filtered[df_filtered["customer_city"] == selected_city]
    
    if selected_category != "Tüm Kategoriler":
        if "product_category_name_english" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["product_category_name_english"] == selected_category]
        else:
            df_filtered = df_filtered[df_filtered["product_category_name"] == selected_category]

    # ============================================
    # KPI METRICS
    # ============================================

    orders_unique = df_filtered.drop_duplicates("order_id")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_sales = orders_unique["total_item_value"].sum()
        st.metric("💰 Toplam Satış", f"${total_sales:,.0f}")

    with col2:
        total_orders = len(orders_unique)
        st.metric("📦 Toplam Sipariş", f"{total_orders:,}")

    with col3:
        avg_review = orders_unique["review_score"].mean()
        st.metric("⭐ Ort. Puan", f"{avg_review:.2f}")

    with col4:
        avg_order_value = total_sales / total_orders if total_orders > 0 else 0
        st.metric("🛒 Ort. Sipariş Değeri", f"${avg_order_value:,.2f}")

    st.markdown("---")

    # ============================================
    # SALES TRENDS
    # ============================================

    st.header("📈 Satış Trendleri")

    # Aylık satış
    monthly_sales = (
        orders_unique
        .groupby(orders_unique["order_purchase_timestamp"].dt.to_period("M"))["total_item_value"]
        .sum()
        .reset_index()
    )
    monthly_sales["order_purchase_timestamp"] = monthly_sales["order_purchase_timestamp"].astype(str)

    fig_sales = px.line(
        monthly_sales,
        x="order_purchase_timestamp",
        y="total_item_value",
        title="Zamanla Gelir Trendi",
        labels={"total_item_value": "Gelir ($)", "order_purchase_timestamp": "Ay"}
    )
    fig_sales.update_traces(line=dict(width=3, color="#2980b9"))

    # Aylık sipariş sayısı
    orders_month = (
        orders_unique
        .groupby(orders_unique["order_purchase_timestamp"].dt.to_period("M"))
        .size()
        .reset_index(name="orders")
    )
    orders_month["order_purchase_timestamp"] = orders_month["order_purchase_timestamp"].astype(str)

    fig_orders = px.line(
        orders_month,
        x="order_purchase_timestamp",
        y="orders",
        title="Sipariş Sayısı Trendi",
        labels={"orders": "Sipariş Sayısı", "order_purchase_timestamp": "Ay"}
    )
    fig_orders.update_traces(line=dict(width=3, color="#16a085"))

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_sales, use_container_width=True)
    with col2:
        st.plotly_chart(fig_orders, use_container_width=True)

    st.markdown("---")

    # ============================================
    # PRODUCT ANALYSIS
    # ============================================

    st.header("🛍️ Ürün Performansı")

    # En iyi kategoriler
    if "product_category_name_english" in df_filtered.columns:
        top_categories = (
            df_filtered
            .groupby("product_category_name_english")["total_item_value"]
            .sum()
            .nlargest(10)
            .reset_index()
        )
        top_categories.rename(columns={"product_category_name_english": "category"}, inplace=True)
    else:
        top_categories = (
            df_filtered
            .groupby("product_category_name")["total_item_value"]
            .sum()
            .nlargest(10)
            .reset_index()
        )
        top_categories.rename(columns={"product_category_name": "category"}, inplace=True)

    fig_category = px.bar(
        top_categories,
        x="category",
        y="total_item_value",
        color="total_item_value",
        color_continuous_scale="Blues",
        title="En İyi 10 Kategori"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_category, use_container_width=True)

    # En iyi ürünler
    top_products = (
        df_filtered
        .groupby("product_id")["total_item_value"]
        .sum()
        .nlargest(10)
        .reset_index()
    )

    fig_products = px.bar(
        top_products,
        x="product_id",
        y="total_item_value",
        color="total_item_value",
        color_continuous_scale="Greens",
        title="En İyi 10 Ürün"
    )

    with col2:
        st.plotly_chart(fig_products, use_container_width=True)

    st.markdown("---")

    # ============================================
    # GEOGRAPHIC ANALYSIS
    # ============================================

    st.header("🌎 Coğrafi Analiz")

    # En iyi şehirler
    city_sales = (
        orders_unique
        .groupby("customer_city")["total_item_value"]
        .sum()
        .nlargest(10)
        .reset_index()
    )

    fig_city = px.bar(
        city_sales,
        x="customer_city",
        y="total_item_value",
        color="total_item_value",
        color_continuous_scale="Purples",
        title="En İyi 10 Şehir"
    )

    # Eyaletler
    state_sales = (
        orders_unique
        .groupby("customer_state")["total_item_value"]
        .sum()
        .reset_index()
    )

    fig_state = px.bar(
        state_sales,
        x="customer_state",
        y="total_item_value",
        color="total_item_value",
        color_continuous_scale="Sunset",
        title="Eyalet Bazında Satış"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_city, use_container_width=True)
    with col2:
        st.plotly_chart(fig_state, use_container_width=True)

    st.markdown("---")

    # ============================================
    # CUSTOMER ANALYSIS
    # ============================================

    st.header("👥 Müşteri Davranışı")

    cust_orders = orders_unique.groupby("customer_unique_id").size().reset_index(name="order_count")
    repeat_rate = (cust_orders["order_count"] >= 2).mean() * 100

    st.metric("🔁 Tekrar Müşteri Oranı", f"{repeat_rate:.2f}%")

    cust_orders["segment"] = cust_orders["order_count"].apply(
        lambda x: "Yeni" if x == 1 else ("Dönen" if x <= 3 else "Sadık")
    )

    segment_counts = cust_orders["segment"].value_counts().reset_index()
    segment_counts.columns = ["segment", "count"]

    fig_segment = px.pie(
        segment_counts,
        names="segment",
        values="count",
        title="Müşteri Segmentasyonu",
        color_discrete_map={"Yeni": "#3498db", "Dönen": "#f39c12", "Sadık": "#27ae60"}
    )

    st.plotly_chart(fig_segment, use_container_width=True)

    st.markdown("---")

    # ============================================
    # DELIVERY ANALYSIS
    # ============================================

    st.header("🚚 Lojistik Performansı")

    df_delivered = df_filtered[df_filtered["order_status"] == "delivered"].copy()
    df_delivered["order_delivered_customer_date"] = pd.to_datetime(df_delivered["order_delivered_customer_date"])
    df_delivered["delivery_days"] = (
        df_delivered["order_delivered_customer_date"] - df_delivered["order_purchase_timestamp"]
    ).dt.days
    df_delivered = df_delivered[(df_delivered["delivery_days"] > 0) & (df_delivered["delivery_days"] < 60)]

    if len(df_delivered) > 0:
        avg_delivery = df_delivered["delivery_days"].mean()
        st.metric("📦 Ort. Teslimat Süresi", f"{avg_delivery:.0f} gün")
        
        fig_delivery = px.histogram(
            df_delivered,
            x="delivery_days",
            nbins=30,
            color_discrete_sequence=["#f39c12"],
            title="Teslimat Süresi Dağılımı"
        )
        st.plotly_chart(fig_delivery, use_container_width=True)
    else:
        st.info("Teslimat verisi bulunamadı.")

    st.markdown("---")

    # ============================================
    # PAYMENT ANALYSIS
    # ============================================

    st.header("💳 Ödeme Analizi")

    if len(payments) > 0:
        payment_dist = payments["payment_type"].value_counts().reset_index()
        payment_dist.columns = ["payment_type", "count"]
        
        fig_payment = px.pie(
            payment_dist,
            names="payment_type",
            values="count",
            title="Ödeme Yöntemi Dağılımı"
        )
        st.plotly_chart(fig_payment, use_container_width=True)
    else:
        st.info("Ödeme verisi bulunamadı.")

    st.markdown("---")

    # ============================================
    # SEASONALITY
    # ============================================

    st.header("📅 Mevsimsel Desenler")

    df_filtered_copy = df_filtered.copy()
    df_filtered_copy["month"] = df_filtered_copy["order_purchase_timestamp"].dt.month
    df_filtered_copy["month_name"] = df_filtered_copy["month"].map({
        1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
        7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
    })

    month_orders = (
        df_filtered_copy
        .groupby(["month", "month_name"])
        .size()
        .reset_index(name="orders")
        .sort_values("month")
    )

    fig_season = px.bar(
        month_orders,
        x="month_name",
        y="orders",
        color="orders",
        color_continuous_scale="IceFire",
        title="Aylık Sipariş Dağılımı"
    )

    st.plotly_chart(fig_season, use_container_width=True)

# ============================================
# TAB 2: SMART RECOMMENDATION ENGINE
# ============================================

with tab2:
    
    st.markdown("""
    <style>
        .rec-title {
            text-align: center;
            font-size: 48px;
            font-weight: 800;
            color: #2d2d55;
            margin-bottom: 8px;
        }
        
        .rec-subtitle {
            text-align: center;
            color: #6e6e8d;
            font-size: 18px;
            margin-bottom: 30px;
        }
        
        .rec-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            color: white;
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .rec-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.15);
        }
        
        .rec-card-icon {
            font-size: 40px;
            margin-bottom: 10px;
        }
        
        .rec-card-title {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .rec-card-score {
            font-size: 14px;
            opacity: 0.9;
        }
        
        .rec-info {
            background: rgba(102, 126, 234, 0.1);
            border-left: 4px solid #667eea;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="rec-title">🧠 Akıllı Öneri Motoru</div>', unsafe_allow_html=True)
    st.markdown('<div class="rec-subtitle">Benzer kategorileri keşfedin ve satış fırsatlarını yakalamış</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Kategori seçimi
    if category_names is not None:
        available_categories = sorted(category_names.tolist())
        selected_rec_category = st.selectbox(
            "📌 Kategori Seçin",
            available_categories,
            key="category_recommendation"
        )
        
        # Önerileri al
        recommendations = recommend_categories(selected_rec_category, category_names, similarity, top_n=5)
        
        if not recommendations.empty:
            st.markdown(f"### Müşteriler **{selected_rec_category}** kategorisini satın aldığında...")
            st.markdown("Aşağıdaki kategorileri de tercih etme eğilimindedirler:")
            
            st.markdown("---")
            
            # Önerileri kartlar halinde göster
            cols = st.columns(len(recommendations))
            
            for idx, (col, (_, row)) in enumerate(zip(cols, recommendations.iterrows())):
                with col:
                    score_percent = row["similarity_score"] * 100
                    st.markdown(f"""
                    <div class="rec-card">
                        <div class="rec-card-icon">🏷️</div>
                        <div class="rec-card-title">{row['recommended_category']}</div>
                        <div class="rec-card-score">Benzerlik: {score_percent:.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Detaylı tablo
            st.subheader("📊 Detaylı Analiz")
            
            rec_display = recommendations.copy()
            rec_display["similarity_score"] = rec_display["similarity_score"].apply(lambda x: f"{x:.1%}")
            rec_display.index = range(1, len(rec_display) + 1)
            
            st.dataframe(
                rec_display,
                column_config={
                    "recommended_category": st.column_config.TextColumn("Kategori"),
                    "similarity_score": st.column_config.TextColumn("Benzerlik Skoru"),
                },
                use_container_width=True
            )
            
            # Bilgi kutusu
            st.markdown(f"""
            <div class="rec-info">
                <strong>💡 Insight:</strong> Bu öneriler, "{selected_rec_category}" kategorisini satın alan müşterilerin 
                diğer satın alma davranışlarına dayanarak Cosine Similarity algoritması kullanılarak hesaplanmıştır.
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.warning(f"'{selected_rec_category}' kategorisi için yeterli veriye sahip değiliz.")
    else:
        st.error("Recommendation Engine oluşturulamadı. Lütfen verinizi kontrol edin.")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.caption(
    """
    📊 **Brazil E-Commerce Dashboard** | Veri Kaynağı: Olist E-Commerce Dataset  
    🛠️ Araçlar: Streamlit • Pandas • Plotly • Scikit-learn • SciPy
    """
)