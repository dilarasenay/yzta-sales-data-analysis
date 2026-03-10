import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Brazil E-Commerce Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- DATA LOAD ----------------

@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/fact_orders.parquet")
    payments = pd.read_csv("data/raw/olist_order_payments_dataset.csv")

    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    
    return df, payments


df, payments = load_data()

# ---------------- SIDEBAR ----------------

st.sidebar.title("🔎 Filters")

city_list = ["All Cities"] + sorted(df["customer_city"].dropna().unique())

selected_city = st.sidebar.selectbox(
    "Select City",
    city_list
)

category_list = ["All Categories"] + sorted(df["product_category_name_english"].dropna().unique())

selected_category = st.sidebar.selectbox(
    "Select Category",
    category_list
)

# ---------------- FILTER DATA ----------------

df_filtered = df.copy()

if selected_city != "All Cities":
    df_filtered = df_filtered[df_filtered["customer_city"] == selected_city]

if selected_category != "All Categories":
    df_filtered = df_filtered[df_filtered["product_category_name_english"] == selected_category]

orders_unique = df_filtered.drop_duplicates("order_id")

# ---------------- TITLE ----------------

st.title("📊 Brazil E-Commerce Analytics Dashboard")

st.markdown(
"""
This dashboard analyzes **sales performance, customer behavior, delivery metrics, and payment trends**
using the Brazilian **Olist E-Commerce Dataset**.
"""
)

st.markdown("---")

# ---------------- KPI ----------------

total_sales = orders_unique["total_item_value"].sum()
total_orders = orders_unique["order_id"].nunique()
avg_review = orders_unique["review_score"].mean()
avg_order_value = total_sales / total_orders if total_orders > 0 else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
col2.metric("📦 Total Orders", total_orders)
col3.metric("⭐ Avg Review Score", round(avg_review,2))
col4.metric("🛒 Avg Order Value", f"${avg_order_value:,.2f}")

st.markdown("---")

# ---------------- CATEGORY SALES ----------------

st.subheader("🛍️ Top Product Categories")

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
    title="Top 10 Product Categories by Sales",
    color="total_item_value",
    color_continuous_scale="Blues"
)

fig_category.update_layout(xaxis_tickangle=-40)

# ---------------- TOP PRODUCTS ----------------

st.subheader("🏆 Top Selling Products")

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
    title="Top 10 Products by Revenue",
    color="total_item_value",
    color_continuous_scale="Teal"
)

st.plotly_chart(fig_products, use_container_width=True)

# ---------------- MONTHLY SALES ----------------

st.subheader("📈 Monthly Sales Trend")

monthly_sales = (
    orders_unique
    .groupby(orders_unique["order_purchase_timestamp"].dt.to_period("M"))
    ["total_item_value"]
    .sum()
    .reset_index()
)

monthly_sales["order_purchase_timestamp"] = monthly_sales["order_purchase_timestamp"].astype(str)

fig_month = px.line(
    monthly_sales,
    x="order_purchase_timestamp",
    y="total_item_value",
    title="Revenue Over Time"
)

fig_month.update_traces(line=dict(width=4,color="#1976D2"))

# ---------------- MONTHLY ORDERS ----------------

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
    title="📦 Orders Trend Over Time"
)

fig_orders.update_traces(line=dict(width=4,color="#2E7D32"))

# ---------------- CITY SALES ----------------

st.subheader("🏙️ Top Cities by Sales")

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
    title="Top 10 Cities by Revenue",
    color="total_item_value",
    color_continuous_scale="Purples"
)

fig_city.update_layout(xaxis_tickangle=-40)

# ---------------- STATE SALES ----------------

st.subheader("🌎 Sales by State")

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
    title="Revenue by State",
    color="total_item_value",
    color_continuous_scale="Sunset"
)

st.plotly_chart(fig_state, use_container_width=True)

# ---------------- REVIEW DISTRIBUTION ----------------

st.subheader("⭐ Customer Review Distribution")

fig_reviews = px.histogram(
    orders_unique,
    x="review_score",
    nbins=5,
    title="Review Score Distribution",
    color_discrete_sequence=["#66BB6A"]
)

# ---------------- GRAPH LAYOUT ----------------

col5, col6 = st.columns(2)

st.plotly_chart(fig_orders, use_container_width=True)

with col5:
    st.plotly_chart(fig_category, use_container_width=True)

with col6:
    st.plotly_chart(fig_month, use_container_width=True)

col7, col8 = st.columns(2)

with col7:
    st.plotly_chart(fig_city, use_container_width=True)

with col8:
    st.plotly_chart(fig_reviews, use_container_width=True)

st.markdown("---")

# ---------------- REPEAT CUSTOMER RATE ----------------

st.subheader("🔁 Repeat Customer Analysis")

cust_orders = (
    orders_unique
    .groupby("customer_unique_id")
    .agg(order_count=("order_id","count"))
    .reset_index()
)

repeat_rate = (cust_orders["order_count"] >= 2).mean() * 100

st.metric("Repeat Customer Rate", f"{repeat_rate:.2f}%")

st.markdown("---")

# ---------------- CUSTOMER SEGMENTATION ----------------

st.subheader("👥 Customer Segmentation")

cust_orders["segment"] = cust_orders["order_count"].apply(
    lambda x: "New" if x == 1 else ("Returning" if x <= 3 else "Loyal")
)

segment_counts = cust_orders["segment"].value_counts().reset_index()
segment_counts.columns = ["segment","count"]

fig_segment = px.pie(
    segment_counts,
    names="segment",
    values="count",
    title="Customer Segments"
)

st.plotly_chart(fig_segment, use_container_width=True)

# ---------------- DELIVERY TIME ----------------

st.subheader("🚚 Delivery Time Distribution")

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
    title="Delivery Time (Days)",
    color_discrete_sequence=["#FB8C00"]
)

# ---------------- ORDERS BY WEEKDAY ----------------

st.subheader("📅 Orders by Weekday")

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
    title="Orders by Day of Week",
    color="orders",
    color_continuous_scale="Viridis"
)

st.plotly_chart(fig_weekday, use_container_width=True)

# ---------------- PAYMENT METHODS ----------------

top_payment = payments["payment_type"].value_counts().idxmax()

st.metric("💳 Most Used Payment Method", top_payment)

st.subheader("💳 Payment Method Distribution")

pay_counts = payments["payment_type"].value_counts().reset_index()
pay_counts.columns = ["payment_type","count"]

fig_payment = px.pie(
    pay_counts,
    names="payment_type",
    values="count",
    title="Payment Method Share"
)

col9, col10 = st.columns(2)

with col9:
    st.plotly_chart(fig_delivery, use_container_width=True)

with col10:
    st.plotly_chart(fig_payment, use_container_width=True)

# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
"""
Data Source: Brazilian Olist E-Commerce Dataset  
Built with **Streamlit, Pandas, and Plotly**
"""
)