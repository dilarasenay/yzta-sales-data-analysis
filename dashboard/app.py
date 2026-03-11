import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Brazil E-Commerce Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM STYLE ----------------

st.markdown("""
<style>

/* SIDEBAR BACKGROUND */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f2027,#203a43,#2c5364);
}

/* SIDEBAR TITLES */

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label {
    color: white !important;
}

/* DROPDOWN TEXT */

section[data-testid="stSidebar"] .stSelectbox div {
    color: black !important;
}

section[data-testid="stSidebar"] .stSelectbox div {
    color: white !important;

</style>
""", unsafe_allow_html=True)

# ---------------- DATA LOAD ----------------

@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/fact_orders.parquet")
    payments = pd.read_csv("data/raw/olist_order_payments_dataset.csv")

    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

    return df, payments


df, payments = load_data()

# ---------------- SIDEBAR ----------------

st.sidebar.title("🔎 Dashboard Filters")

city_list = ["All Cities"] + sorted(df["customer_city"].dropna().unique())
selected_city = st.sidebar.selectbox("Select City", city_list)

category_list = ["All Categories"] + sorted(df["product_category_name_english"].dropna().unique())
selected_category = st.sidebar.selectbox("Select Category", category_list)

# ---------------- FILTER DATA ----------------

df_filtered = df.copy()

if selected_city != "All Cities":
    df_filtered = df_filtered[df_filtered["customer_city"] == selected_city]

if selected_category != "All Categories":
    df_filtered = df_filtered[df_filtered["product_category_name_english"] == selected_category]

orders_unique = df_filtered.drop_duplicates("order_id")

# ---------------- TITLE ----------------

st.markdown("""
# 📊 Brazil E-Commerce Analytics Dashboard

### 🛍️ Sales • Customers • Delivery • Payments

This interactive dashboard explores **sales performance, customer behavior,
logistics efficiency and payment trends** using the **Brazilian Olist dataset**.
""")

st.markdown("---")

# ---------------- KPI ----------------

total_sales = orders_unique["total_item_value"].sum()
total_orders = orders_unique["order_id"].nunique()
avg_review = orders_unique["review_score"].mean()
avg_order_value = total_sales / total_orders if total_orders > 0 else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
col2.metric("📦 Total Orders", total_orders)
col3.metric("⭐ Avg Review Score", round(avg_review, 2))
col4.metric("🛒 Avg Order Value", f"${avg_order_value:,.2f}")

st.markdown("---")

# ---------------- SALES TRENDS ----------------

st.header("📈 Sales Trends")

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
    title="Revenue Over Time"
)

fig_sales.update_traces(line=dict(width=4, color="#2980b9"))

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
    title="Orders Trend Over Time"
)

fig_orders.update_traces(line=dict(width=4, color="#16a085"))

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_sales, use_container_width=True)

with col2:
    st.plotly_chart(fig_orders, use_container_width=True)

st.markdown("---")

# ---------------- PRODUCT ANALYSIS ----------------

st.header("🛍️ Product Performance")

category_sales = (
    df_filtered
    .groupby("product_category_name_english")["total_item_value"]
    .sum()
    .nlargest(10)
    .reset_index()
)

fig_category = px.bar(
    category_sales,
    x="product_category_name_english",
    y="total_item_value",
    color="total_item_value",
    color_continuous_scale="Blues",
    title="Top Product Categories"
)

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
    color_continuous_scale="Teal",
    title="Top Products by Revenue"
)

col3, col4 = st.columns(2)

with col3:
    st.plotly_chart(fig_category, use_container_width=True)

with col4:
    st.plotly_chart(fig_products, use_container_width=True)

# ---- Top Category Trend ----

st.subheader("📈 Top Category Revenue Trend")

top_cats = (
    df_filtered
    .groupby("product_category_name_english")["total_item_value"]
    .sum()
    .nlargest(5)
    .index
)

trend = df_filtered[df_filtered["product_category_name_english"].isin(top_cats)]

trend_month = (
    trend
    .groupby([
        trend["order_purchase_timestamp"].dt.to_period("M"),
        "product_category_name_english"
    ])["total_item_value"]
    .sum()
    .reset_index()
)

trend_month["order_purchase_timestamp"] = trend_month["order_purchase_timestamp"].astype(str)

fig_trend = px.line(
    trend_month,
    x="order_purchase_timestamp",
    y="total_item_value",
    color="product_category_name_english"
)

st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# ---------------- GEOGRAPHIC ----------------

st.header("🌎 Geographic Insights")

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
    color_continuous_scale="Purples"
)

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
    color_continuous_scale="Sunset"
)

col5, col6 = st.columns(2)

with col5:
    st.plotly_chart(fig_city, use_container_width=True)

with col6:
    st.plotly_chart(fig_state, use_container_width=True)

st.markdown("---")

# ---------------- CUSTOMER ANALYSIS ----------------

st.header("👥 Customer Behavior")

cust_orders = (
    orders_unique
    .groupby("customer_unique_id")
    .agg(order_count=("order_id", "count"))
    .reset_index()
)

repeat_rate = (cust_orders["order_count"] >= 2).mean() * 100

st.metric("🔁 Repeat Customer Rate", f"{repeat_rate:.2f}%")

cust_orders["segment"] = cust_orders["order_count"].apply(
    lambda x: "New" if x == 1 else ("Returning" if x <= 3 else "Loyal")
)

segment_counts = cust_orders["segment"].value_counts().reset_index()
segment_counts.columns = ["segment", "count"]

fig_segment = px.pie(
    segment_counts,
    names="segment",
    values="count"
)

st.plotly_chart(fig_segment, use_container_width=True)

st.markdown("---")

# ---------------- DELIVERY ----------------

st.header("🚚 Logistics Performance")

df_delivered = df_filtered[df_filtered["order_status"] == "delivered"].copy()

df_delivered["order_delivered_customer_date"] = pd.to_datetime(df_delivered["order_delivered_customer_date"])

df_delivered["delivery_days"] = (
    df_delivered["order_delivered_customer_date"]
    - df_delivered["order_purchase_timestamp"]
).dt.days

df_delivered = df_delivered[df_delivered["delivery_days"] < 60]

fig_delivery = px.histogram(
    df_delivered,
    x="delivery_days",
    nbins=30,
    color_discrete_sequence=["#f39c12"],
    title="Delivery Time Distribution"
)

st.plotly_chart(fig_delivery, use_container_width=True)

st.markdown("---")

# ---------------- ORDER BEHAVIOR ----------------

st.header("📅 Order Behavior")

orders_unique["weekday"] = orders_unique["order_purchase_timestamp"].dt.day_name()

weekday_orders = (
    orders_unique
    .groupby("weekday")
    .size()
    .reset_index(name="orders")
)

fig_weekday = px.bar(
    weekday_orders,
    x="weekday",
    y="orders",
    color="orders",
    color_continuous_scale="Viridis"
)

st.plotly_chart(fig_weekday, use_container_width=True)

st.markdown("---")

# ---------------- PAYMENTS ----------------

st.header("💳 Payment Analytics")

top_payment = payments["payment_type"].value_counts().idxmax()

st.metric("Most Used Payment Method", top_payment)

pay_counts = payments["payment_type"].value_counts().reset_index()
pay_counts.columns = ["payment_type", "count"]

fig_payment = px.pie(
    pay_counts,
    names="payment_type",
    values="count"
)

st.plotly_chart(fig_payment, use_container_width=True)

st.markdown("---")

# ---------------- SEASONALITY ----------------

st.header("📅 Seasonal Patterns")

df_filtered["month"] = df_filtered["order_purchase_timestamp"].dt.month

month_orders = (
    df_filtered
    .groupby("month")
    .size()
    .reset_index(name="orders")
)

fig_season = px.bar(
    month_orders,
    x="month",
    y="orders",
    color="orders",
    color_continuous_scale="IceFire"
)

st.plotly_chart(fig_season, use_container_width=True)

# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
"""
Data Source: Brazilian Olist E-Commerce Dataset  
Built with **Streamlit • Pandas • Plotly**
"""
)