import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

px.defaults.template = "plotly"
px.defaults.color_discrete_sequence = [
    "#2563eb", "#06b6d4", "#8b5cf6", "#f59e0b", "#ef4444",
    "#10b981", "#ec4899", "#14b8a6", "#f97316", "#6366f1"
]

st.set_page_config(
    page_title="Brazil E-Commerce Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CATEGORY_SYMBOLS = {
    "baby": "🧸",
    "stationery": "✏️",
    "cool_stuff": "✨",
    "consoles_games": "🎮",
    "arts_and_craftmanship": "🎨",
    "sports_leisure": "🏀",
    "toys": "🪀",
    "watches_gifts": "⌚",
    "electronics": "📱",
    "fashion_shoes": "👟",
    "bed_bath_table": "🛏️",
    "signaling_and_security": "🔐",
    "fashion_bags_accessories": "👜",
    "home_confort": "🏡",
    "furniture_decor": "🪑",
    "housewares": "🍽️",
    "books": "📚",
    "health_beauty": "💄",
    "computers_accessories": "💻",
    "telephony": "☎️",
    "garden_tools": "🌿",
    "pet_shop": "🐾",
    "auto": "🚗",
    "office_furniture": "🗄️",
    "musical_instruments": "🎵",
    "construction_tools_construction": "🧱",
    "small_appliances": "🔌",
    "food_drink": "🥤",
    "drinks": "🥂",
    "perfumery": "🌸",
    "luggage_accessories": "🧳",
    "default": "🛍️",
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 8%, rgba(37, 99, 235, 0.22), transparent 22%),
            radial-gradient(circle at 92% 10%, rgba(236, 72, 153, 0.18), transparent 22%),
            radial-gradient(circle at 80% 80%, rgba(6, 182, 212, 0.16), transparent 24%),
            linear-gradient(135deg, #edf4ff 0%, #f8fbff 40%, #f2fffb 100%);
    }

    header {
        background: transparent !important;
    }

    [data-testid="stToolbar"] {
        visibility: visible !important;
        display: block !important;
        right: 1rem;
        top: 0.5rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 0.45rem;
        padding-bottom: 1rem;
        max-width: 1380px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #06152b 0%, #0f2f63 45%, #0b74c8 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(241,247,255,0.98));
        border: 1px solid rgba(37, 99, 235, 0.10);
        padding: 18px 16px;
        border-radius: 24px;
        box-shadow: 0 14px 36px rgba(15, 23, 42, 0.10);
        margin-top: -8px;
    }

    [data-testid="stMetricLabel"] {
        font-weight: 700;
        color: #274060;
    }

    [data-testid="stMetricValue"] {
        color: #0f172a;
    }

    .hero-card {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #07162c 0%, #1646a0 48%, #06b6d4 100%);
        padding: 32px 30px 28px 30px;
        border-radius: 30px;
        color: white;
        box-shadow: 0 20px 48px rgba(7, 22, 44, 0.24);
        margin-bottom: 18px;
        border: 1px solid rgba(255,255,255,0.10);
    }

    .hero-card::before {
        content: '';
        position: absolute;
        width: 300px;
        height: 300px;
        border-radius: 50%;
        right: -80px;
        top: -130px;
        background: rgba(255,255,255,0.12);
    }

    .hero-card::after {
        content: '';
        position: absolute;
        width: 240px;
        height: 240px;
        border-radius: 50%;
        right: 110px;
        bottom: -130px;
        background: rgba(255,255,255,0.08);
    }

    .hero-badges {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 16px;
    }

    .hero-badge {
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.18);
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 0.88rem;
        font-weight: 600;
        backdrop-filter: blur(8px);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }

    .section-title {
        font-size: 1.28rem;
        font-weight: 800;
        color: #0b1f3a;
        margin-top: 10px;
        margin-bottom: 10px;
        letter-spacing: 0.2px;
    }

    .info-box {
        background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(240,249,255,0.98));
        border-left: 6px solid #06b6d4;
        padding: 16px 18px;
        border-radius: 18px;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
        border-top: 1px solid rgba(6, 182, 212, 0.12);
        border-right: 1px solid rgba(6, 182, 212, 0.12);
        border-bottom: 1px solid rgba(6, 182, 212, 0.12);
    }

    div[data-testid="stPlotlyChart"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(245,250,255,0.98));
        border: 1px solid rgba(37, 99, 235, 0.08);
        border-radius: 24px;
        padding: 10px 10px 2px 10px;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
    }

    div[data-testid="stDataFrame"] {
        background: rgba(255,255,255,0.92);
        border-radius: 18px;
        padding: 8px;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
    }

    .symbol-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(245,250,255,0.98));
        border-radius: 22px;
        padding: 18px 16px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(37, 99, 235, 0.08);
        margin-bottom: 10px;
        min-height: 220px;
    }

    .symbol-icon {
        height: 95px;
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        background: linear-gradient(135deg, #eff6ff, #ecfeff);
        border: 1px solid rgba(37, 99, 235, 0.08);
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def pick_first_existing(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None

def safe_to_datetime(df: pd.DataFrame, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def pretty_category_name(name: str) -> str:
    return str(name).replace("_", " ").title()

def shorten_text(text, max_len=18):
    text = str(text)
    return text if len(text) <= max_len else text[:max_len] + "..."

def format_axis_as_month_year(fig):
    fig.update_xaxes(
        tickformat="%b %Y",
        title_text="Month",
        tickangle=-35
    )

def create_monthly_series(df, date_col, value_col=None, count_mode=False):
    monthly = df.copy()
    monthly = monthly.dropna(subset=[date_col])
    monthly["month_start"] = monthly[date_col].dt.to_period("M").dt.to_timestamp()

    if count_mode:
        result = monthly.groupby("month_start").size().reset_index(name="value")
    else:
        result = monthly.groupby("month_start")[value_col].sum().reset_index(name="value")

    if result.empty:
        return result

    full_range = pd.date_range(
        start=result["month_start"].min(),
        end=result["month_start"].max(),
        freq="MS"
    )

    result = (
        result.set_index("month_start")
        .reindex(full_range, fill_value=0)
        .rename_axis("month_start")
        .reset_index()
    )

    return result

def find_category_symbol(category_name: str) -> str:
    key = str(category_name).strip().lower().replace(" ", "_")
    return CATEGORY_SYMBOLS.get(key, CATEGORY_SYMBOLS["default"])

@st.cache_data
def load_data():
    try:
        df = pd.read_parquet("data/processed/fact_orders.parquet")
    except Exception:
        try:
            df = pd.read_csv("data/processed/fact_orders.csv")
        except Exception as e:
            st.error(f"Processed veri yüklenemedi: {e}")
            return None, None

    try:
        payments = pd.read_csv("data/raw/olist_order_payments_dataset.csv")
    except Exception:
        payments = pd.DataFrame()

    df = safe_to_datetime(
        df,
        [
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )

    return df, payments

@st.cache_data
def build_recommendation_engine(df: pd.DataFrame):
    category_col = pick_first_existing(
        df, ["product_category_name_english", "product_category_name"]
    )

    if category_col is None or "order_id" not in df.columns:
        return None, None, None

    df_rec = df[["order_id", category_col]].dropna().drop_duplicates().copy()
    if df_rec.empty:
        return None, None, None

    df_rec[category_col] = df_rec[category_col].astype(str).str.strip().str.lower()

    order_counts = df_rec.groupby("order_id")[category_col].nunique()
    valid_orders = order_counts[order_counts > 1].index
    df_rec = df_rec[df_rec["order_id"].isin(valid_orders)]

    if df_rec.empty:
        return None, None, None

    df_rec["order_id"] = df_rec["order_id"].astype("category")
    df_rec[category_col] = df_rec[category_col].astype("category")

    row = df_rec[category_col].cat.codes
    col = df_rec["order_id"].cat.codes

    n_categories = len(df_rec[category_col].cat.categories)
    n_orders = len(df_rec["order_id"].cat.categories)

    matrix = csr_matrix(([1] * len(df_rec), (row, col)), shape=(n_categories, n_orders))
    similarity = cosine_similarity(matrix, dense_output=False)
    category_names = df_rec[category_col].cat.categories

    pretty_to_raw = {pretty_category_name(cat): cat for cat in category_names.tolist()}

    return category_names, similarity, pretty_to_raw

def recommend_categories(category_name, category_names, similarity, top_n=6):
    if category_names is None or similarity is None or category_name not in category_names:
        return pd.DataFrame()

    idx = category_names.get_loc(category_name)
    scores = similarity[idx].toarray().flatten()
    similar_idx = scores.argsort()[::-1]

    recs = pd.DataFrame(
        {
            "recommended_category": category_names[similar_idx],
            "similarity_score": scores[similar_idx],
        }
    )

    recs = recs[recs["recommended_category"] != category_name]
    recs = recs[recs["similarity_score"] > 0]
    recs = recs.drop_duplicates("recommended_category")
    return recs.head(top_n).reset_index(drop=True)

df, payments = load_data()
if df is None or df.empty:
    st.stop()

category_names, similarity, pretty_to_raw = build_recommendation_engine(df)

category_col = pick_first_existing(df, ["product_category_name_english", "product_category_name"])
value_col = pick_first_existing(df, ["total_item_value", "price"])
city_col = pick_first_existing(df, ["customer_city"])
review_col = pick_first_existing(df, ["review_score"])
status_col = pick_first_existing(df, ["order_status"])
customer_col = pick_first_existing(df, ["customer_unique_id", "customer_id"])
product_col = pick_first_existing(df, ["product_id"])
date_col = pick_first_existing(df, ["order_purchase_timestamp"])

if value_col is None or date_col is None:
    st.error("Dashboard için gerekli kolonlar eksik: satış değeri veya tarih kolonu bulunamadı.")
    st.stop()

st.sidebar.markdown("## 🛍️ Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Trend Analysis",
        "Sales Analysis",
        "Customer Insights",
        "Logistics & Payments",
        "Recommendation Engine",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎛️ Filters")

filtered_df = df.copy()

if city_col:
    city_options = ["All Cities"] + sorted(filtered_df[city_col].dropna().astype(str).unique().tolist())
    selected_city = st.sidebar.selectbox("City", city_options)
    if selected_city != "All Cities":
        filtered_df = filtered_df[filtered_df[city_col].astype(str) == selected_city]

if category_col:
    category_options = ["All Categories"] + sorted(
        filtered_df[category_col].dropna().astype(str).unique().tolist()
    )
    selected_category = st.sidebar.selectbox("Category", category_options)
    if selected_category != "All Categories":
        filtered_df = filtered_df[filtered_df[category_col].astype(str) == selected_category]

if date_col and pd.api.types.is_datetime64_any_dtype(filtered_df[date_col]):
    min_date = filtered_df[date_col].min()
    max_date = filtered_df[date_col].max()
    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df[date_col].dt.date >= start_date)
                & (filtered_df[date_col].dt.date <= end_date)
            ]

orders_unique = (
    filtered_df.drop_duplicates("order_id")
    if "order_id" in filtered_df.columns
    else filtered_df.copy()
)

# Eğer veri satır bazında ürün seviyesindeyse toplam satışı filtered_df üzerinden toplamak daha mantıklı
total_sales = float(filtered_df[value_col].fillna(0).sum()) if value_col in filtered_df.columns else 0.0
total_orders = orders_unique["order_id"].nunique() if "order_id" in orders_unique.columns else len(orders_unique)
avg_order_value = total_sales / total_orders if total_orders > 0 else 0.0
avg_review = (
    float(orders_unique[review_col].dropna().mean())
    if review_col and review_col in orders_unique.columns and not orders_unique[review_col].dropna().empty
    else 0.0
)

st.markdown(
    """
    <div class="hero-card">
        <div style="font-size: 0.9rem; opacity: 0.86; font-weight: 600; letter-spacing: 0.3px;">
            YZTA Sales Data Analysis • Executive Commerce Intelligence
        </div>
        <div style="font-size: 2.2rem; font-weight: 800; margin-top: 8px; line-height: 1.15; max-width: 760px; position: relative; z-index: 2;">
            Brazil E-Commerce Performance Dashboard
        </div>
        <div style="margin-top: 12px; font-size: 1rem; opacity: 0.93; max-width: 780px; position: relative; z-index: 2;">
            Revenue performance, category momentum, customer behavior, logistics quality and cross-sell opportunities are presented in a single executive view designed for fast and confident decision-making.
        </div>
        <div class="hero-badges" style="position: relative; z-index: 2;">
            <div class="hero-badge">📊 Revenue & Trend Monitoring</div>
            <div class="hero-badge">🎯 Cross-Sell Intelligence</div>
            <div class="hero-badge">🚚 Fulfillment Visibility</div>
            <div class="hero-badge">👥 Customer Insights</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Sales", f"${total_sales:,.2f}")
with m2:
    st.metric("Total Orders", f"{total_orders:,}")
with m3:
    st.metric("Avg Review", f"{avg_review:.2f} ⭐")
with m4:
    st.metric("Avg Order Value", f"${avg_order_value:,.2f}")

st.write("")

if page == "Overview":
    st.markdown('<div class="section-title">📈 Executive Overview</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.4, 1])

    with c1:
        monthly_sales = create_monthly_series(filtered_df, date_col, value_col=value_col, count_mode=False)
        if not monthly_sales.empty:
            fig = px.line(
                monthly_sales,
                x="month_start",
                y="value",
                markers=True,
                title="Monthly Revenue Trend",
            )
            fig.update_traces(line=dict(width=4))
            fig.update_layout(height=420, yaxis_title="Revenue")
            format_axis_as_month_year(fig)
            st.plotly_chart(fig)

    with c2:
        if category_col and category_col in filtered_df.columns:
            top_categories = (
                filtered_df.groupby(category_col)[value_col]
                .sum()
                .nlargest(8)
                .reset_index()
            )
            top_categories[category_col] = top_categories[category_col].apply(pretty_category_name)

            fig = px.bar(
                top_categories,
                x=value_col,
                y=category_col,
                orientation="h",
                title="Top Categories by Revenue",
                color=category_col,
            )
            fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"}, showlegend=False)
            st.plotly_chart(fig)

    c3, c4 = st.columns(2)

    with c3:
        if city_col and city_col in orders_unique.columns:
            city_sales = (
                filtered_df.groupby(city_col)[value_col]
                .sum()
                .nlargest(10)
                .reset_index()
            )
            fig = px.bar(
                city_sales,
                x=city_col,
                y=value_col,
                title="Top 10 Cities by Revenue",
                color=city_col,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)

    with c4:
        if review_col and review_col in orders_unique.columns:
            review_dist = orders_unique[review_col].dropna().round(2).astype(str).value_counts().reset_index()
            review_dist.columns = ["review_score", "count"]

            fig = px.pie(
                review_dist,
                names="review_score",
                values="count",
                title="Review Score Distribution",
                hole=0.45,
            )
            st.plotly_chart(fig)

    st.markdown(
        """
        <div class="info-box">
        <b>Executive summary:</b> This landing view brings together the most decision-critical signals of the business. Revenue direction, category contribution, geographic concentration and customer satisfaction can be reviewed at a glance before moving into detailed analysis pages.
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "Trend Analysis":
    st.markdown('<div class="section-title">📊 Trend Analysis</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        monthly_sales = create_monthly_series(filtered_df, date_col, value_col=value_col, count_mode=False)
        if not monthly_sales.empty:
            fig = px.line(
                monthly_sales,
                x="month_start",
                y="value",
                markers=True,
                title="Monthly Sales Trend"
            )
            fig.update_traces(line=dict(width=4))
            fig.update_layout(yaxis_title="Revenue")
            format_axis_as_month_year(fig)
            st.plotly_chart(fig)

    with c2:
        monthly_orders = create_monthly_series(orders_unique, date_col, count_mode=True)
        if not monthly_orders.empty:
            fig = px.bar(
                monthly_orders,
                x="month_start",
                y="value",
                title="Monthly Order Volume",
                color="value"
            )
            fig.update_layout(yaxis_title="Orders", coloraxis_showscale=False)
            format_axis_as_month_year(fig)
            st.plotly_chart(fig)

    if category_col:
        cat_trend = filtered_df.dropna(subset=[date_col, category_col]).copy()
        cat_trend["month_start"] = cat_trend[date_col].dt.to_period("M").dt.to_timestamp()
        cat_trend = (
            cat_trend.groupby(["month_start", category_col])[value_col]
            .sum()
            .reset_index()
        )
        cat_trend[category_col] = cat_trend[category_col].apply(pretty_category_name)

        top_cat_names = (
            cat_trend.groupby(category_col)[value_col]
            .sum()
            .nlargest(6)
            .index
            .tolist()
        )
        cat_trend = cat_trend[cat_trend[category_col].isin(top_cat_names)]

        fig = px.line(
            cat_trend,
            x="month_start",
            y=value_col,
            color=category_col,
            title="Category Sales Trend"
        )
        fig.update_layout(yaxis_title="Revenue")
        format_axis_as_month_year(fig)
        st.plotly_chart(fig)

    month_df = filtered_df.dropna(subset=[date_col]).copy()
    month_df["month"] = month_df[date_col].dt.month
    month_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    month_df["month_name"] = month_df["month"].map(month_map)

    seasonality = (
        month_df.groupby(["month", "month_name"])[value_col]
        .sum()
        .reset_index()
        .sort_values("month")
    )

    fig = px.area(
        seasonality,
        x="month_name",
        y=value_col,
        title="Seasonality Pattern"
    )
    fig.update_layout(yaxis_title="Revenue")
    st.plotly_chart(fig)

elif page == "Sales Analysis":
    st.markdown('<div class="section-title">💰 Sales Analysis</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        monthly_orders = create_monthly_series(orders_unique, date_col, count_mode=True)
        if not monthly_orders.empty:
            fig = px.line(
                monthly_orders,
                x="month_start",
                y="value",
                markers=True,
                title="Monthly Order Count",
            )
            fig.update_traces(line=dict(width=4))
            fig.update_layout(yaxis_title="Orders")
            format_axis_as_month_year(fig)
            st.plotly_chart(fig)

    with c2:
        month_df = filtered_df.dropna(subset=[date_col]).copy()
        month_df["month"] = month_df[date_col].dt.month
        month_map = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        month_df["month_name"] = month_df["month"].map(month_map)
        seasonality = (
            month_df.groupby(["month", "month_name"])[value_col]
            .sum()
            .reset_index()
            .sort_values("month")
        )

        fig = px.bar(
            seasonality,
            x="month_name",
            y=value_col,
            title="Seasonality by Month",
            color="month_name"
        )
        st.plotly_chart(fig)

    if product_col and product_col in filtered_df.columns:
        top_products = (
            filtered_df.groupby(product_col)[value_col]
            .sum()
            .nlargest(10)
            .reset_index()
        )
        top_products["product_short"] = top_products[product_col].apply(lambda x: shorten_text(x, 14))

        fig = px.bar(
            top_products,
            x="product_short",
            y=value_col,
            title="Top 10 Products by Revenue",
            color=value_col,
            hover_data={product_col: True, "product_short": False}
        )
        fig.update_xaxes(title_text="Product ID")
        st.plotly_chart(fig)

elif page == "Customer Insights":
    st.markdown('<div class="section-title">👥 Customer Insights</div>', unsafe_allow_html=True)

    if customer_col and customer_col in orders_unique.columns:
        customer_orders = orders_unique.groupby(customer_col).size().reset_index(name="order_count")
        repeat_rate = (customer_orders["order_count"] >= 2).mean() * 100 if not customer_orders.empty else 0.0

        a, b = st.columns([0.8, 1.2])

        with a:
            st.metric("Repeat Customer Rate", f"{repeat_rate:.2f}%")
            customer_orders["segment"] = customer_orders["order_count"].apply(
                lambda x: "New" if x == 1 else ("Returning" if x <= 3 else "Loyal")
            )
            seg = customer_orders["segment"].value_counts().reset_index()
            seg.columns = ["segment", "count"]

            fig = px.pie(
                seg,
                names="segment",
                values="count",
                title="Customer Segments",
                hole=0.45
            )
            st.plotly_chart(fig)

        with b:
            fig = px.histogram(
                customer_orders,
                x="order_count",
                nbins=20,
                title="Orders per Customer Distribution",
            )
            st.plotly_chart(fig)
    else:
        st.warning("Customer insight için gerekli müşteri kolonu bulunamadı.")

elif page == "Logistics & Payments":
    st.markdown('<div class="section-title">🚚 Logistics & Payments</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        if status_col and status_col in filtered_df.columns:
            status_df = filtered_df[status_col].dropna().astype(str).value_counts().reset_index()
            status_df.columns = ["order_status", "count"]

            fig = px.pie(
                status_df,
                names="order_status",
                values="count",
                title="Order Status Distribution",
                hole=0.45
            )
            st.plotly_chart(fig)

        if {
            "order_purchase_timestamp",
            "order_delivered_customer_date",
        }.issubset(filtered_df.columns):
            delivery_df = filtered_df.copy()
            delivery_df["delivery_days"] = (
                delivery_df["order_delivered_customer_date"]
                - delivery_df["order_purchase_timestamp"]
            ).dt.days

            delivery_df = delivery_df[
                (delivery_df["delivery_days"].notna())
                & (delivery_df["delivery_days"] > 0)
                & (delivery_df["delivery_days"] < 90)
            ]

            if not delivery_df.empty:
                avg_delivery = delivery_df["delivery_days"].mean()
                st.metric("Average Delivery Time", f"{avg_delivery:.2f} days")

                fig = px.histogram(
                    delivery_df,
                    x="delivery_days",
                    nbins=30,
                    title="Delivery Days Distribution",
                )
                st.plotly_chart(fig)

    with c2:
        if not payments.empty and "payment_type" in payments.columns:
            pay_dist = payments["payment_type"].astype(str).value_counts().reset_index()
            pay_dist.columns = ["payment_type", "count"]

            fig = px.bar(
                pay_dist,
                x="payment_type",
                y="count",
                title="Payment Methods",
                color="payment_type"
            )
            st.plotly_chart(fig)

            if "payment_installments" in payments.columns:
                inst = payments["payment_installments"].value_counts().sort_index().reset_index()
                inst.columns = ["payment_installments", "count"]

                fig = px.line(
                    inst,
                    x="payment_installments",
                    y="count",
                    markers=True,
                    title="Installment Usage"
                )
                fig.update_traces(line=dict(width=4))
                st.plotly_chart(fig)
        else:
            st.info("Ödeme dosyası bulunamadı ya da payment_type kolonu eksik.")

elif page == "Recommendation Engine":
    st.markdown('<div class="section-title">🎯 Recommendation Intelligence</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
        This module identifies product categories that tend to be purchased together and surfaces the strongest cross-sell opportunities. Similarity scores are calculated with cosine similarity to highlight the most relevant category pairings for recommendation and basket expansion.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if category_names is not None and similarity is not None and pretty_to_raw is not None:
        selected = st.selectbox(
            "Select a category",
            sorted(pretty_to_raw.keys())
        )

        selected_raw = pretty_to_raw[selected]
        recs = recommend_categories(selected_raw, category_names, similarity, top_n=6)

        if recs.empty:
            st.warning("Bu kategori için yeterli ortak satın alma verisi bulunamadı.")
        else:
            cards = st.columns(3)

            for i, (_, row) in enumerate(recs.head(3).iterrows()):
                with cards[i]:
                    display_name = pretty_category_name(row["recommended_category"])
                    symbol = find_category_symbol(row["recommended_category"])

                    st.markdown(
                        f"""
                        <div class="symbol-card">
                            <div class="symbol-icon">{symbol}</div>
                            <div style="font-size: 1.05rem; font-weight: 800; color: #0b1f3a;">
                                {display_name}
                            </div>
                            <div style="margin-top: 8px; font-size: 1rem; font-weight: 700; color: #2563eb;">
                                Similarity Score: {row['similarity_score']:.2%}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            recs_display = recs.copy()
            recs_display["recommended_category"] = recs_display["recommended_category"].apply(pretty_category_name)
            recs_display["similarity_score"] = recs_display["similarity_score"].map(lambda x: f"{x:.2%}")
            st.dataframe(recs_display)
    else:
        st.error("Recommendation engine oluşturulamadı. Kategori ve sipariş verisini kontrol et.")

st.markdown("---")
st.caption("Brazil E-Commerce Dashboard • Dilara • Emir Can • Ahmet • Betül")