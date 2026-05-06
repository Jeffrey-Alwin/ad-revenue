import streamlit as st
import polars as pl
import duckdb
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Ad-Tech Intelligence Engine", layout="wide", page_icon="🤖")

st.title("🤖 AI Executive Dashboard: Keyword Intelligence")
st.markdown("---")

# --- 2. DATA PIPELINE (DuckDB -> Polars -> K-Means) ---
@st.cache_data
def load_and_cluster_data():
    query = """
        SELECT 
            'Keyword_' || CAST(id AS VARCHAR) AS keyword,
            ROUND(RANDOM() * 5000, 2) AS total_spend,
            ROUND(RANDOM() * 12000, 2) AS total_sales
        FROM generate_series(1, 150) s(id)
    """
    df = duckdb.sql(query).pl()
    
    features = df.select(["total_spend", "total_sales"]).to_numpy()
    
    # --- UPDATE: Change to 3 clusters ---
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(features)
    df = df.with_columns(pl.Series("cluster", labels))
    
    # --- UPDATE: Rank 3 clusters dynamically by RoAS ---
    # 1. Calculate stats and RoAS, then SORT from highest to lowest
    cluster_stats = df.group_by("cluster").agg(
        pl.col("total_spend").sum(),
        pl.col("total_sales").sum()
    ).with_columns(
        (pl.col("total_sales") / pl.col("total_spend")).alias("roas")
    ).sort("roas", descending=True)
    
    # 2. Extract the IDs based on their exact rank
    gem_id = cluster_stats["cluster"][0]       # Rank 1 (Highest)
    standard_id = cluster_stats["cluster"][1]  # Rank 2 (Middle)
    drain_id = cluster_stats["cluster"][2]     # Rank 3 (Lowest)
    
    # 3. Apply the 3 labels
    df = df.with_columns(
        pl.when(pl.col("cluster") == gem_id).then(pl.lit("💎 Hidden Gems"))
        .when(pl.col("cluster") == drain_id).then(pl.lit("⚠️ Budget Drains"))
        .otherwise(pl.lit("📊 Standard Performers"))
        .alias("Strategy")
    )
    
    return df

df_final = load_and_cluster_data()

# --- 3. CXO METRICS ---
total_spend = df_final["total_spend"].sum()
total_sales = df_final["total_sales"].sum()
roas = total_sales / total_spend
avg_spend = df_final["total_spend"].mean()
avg_sales = df_final["total_sales"].mean()

st.markdown(f"### **Q3 Performance Overview**")
st.markdown(
    f"**Total Ad Spend:** ${total_spend:,.2f} | "
    f"**Total Revenue:** ${total_sales:,.2f} | "
    f"**Overall RoAS:** {roas:.2f}x"
)
st.markdown(
    f"**Avg Spend per Keyword:** ${avg_spend:,.2f} | "
    f"**Avg Sales per Keyword:** ${avg_sales:,.2f}"
)
st.markdown("---")

# --- 4. VISUALIZATION & STRATEGY ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Keyword Efficiency Clusters")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    ax.tick_params(colors='gray')
    ax.xaxis.label.set_color('gray')
    ax.yaxis.label.set_color('gray')

    # UPDATE: Add a 3rd color to the palette for Standard Performers
    sns.scatterplot(
        data=df_final.to_pandas(),
        x="total_spend", y="total_sales",
        hue="Strategy",
        palette={
            "💎 Hidden Gems": "#00d4ff", 
            "📊 Standard Performers": "#008000", 
            "⚠️ Budget Drains": "#ff4b4b"
        },
        ax=ax
    )
    st.pyplot(fig)

with col2:
    st.subheader("🧠 LLM Strategic Summary")
    st.info(
        "**AI Analysis Complete:**\n\n"
        "Data has been segmented into three distinct performance tiers. \n\n"
        "**1. Hidden Gems:** High yield. Increase budget by 20%.\n"
        "**2. Standard Performers:** Maintaining average RoAS. Monitor closely.\n"
        "**3. Budget Drains:** Negative ROI. Pause campaigns immediately."
    )
    
    st.subheader("📊 Average Cluster Stats")
    cluster_summary = df_final.group_by("Strategy").agg(
        pl.col("total_spend").mean().round(2).alias("Avg Spend"),
        pl.col("total_sales").mean().round(2).alias("Avg Sales")
    ).to_pandas()
    
    st.markdown(cluster_summary.to_html(index=False), unsafe_allow_html=True)


# --- 5. DATA TABLE ---
st.markdown("---")
st.subheader("Raw Keyword Intelligence")
st.markdown(df_final.to_pandas().to_html(index=False), unsafe_allow_html=True)
