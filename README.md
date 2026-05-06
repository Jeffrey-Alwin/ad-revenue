# 🤖 Ad-Tech Intelligence Engine

An end-to-end modern data pipeline that ingests simulated advertising data, processes it using an embedded analytical database, applies machine learning for keyword optimization, and serves insights via a CXO-ready dashboard.

## 🛠️ The Modern Data Stack Used
* **Storage & SQL Engine:** DuckDB (Zero-copy evaluation)
* **Data Transformation:** Polars (High-speed, multi-threaded processing)
* **Machine Learning:** Scikit-Learn (K-Means Clustering)
* **Frontend UI:** Streamlit & Matplotlib

## 🧠 Business Logic
The pipeline automatically analyzes ad keywords and clusters them based on Return on Ad Spend (RoAS):
1. **💎 Hidden Gems:** High sales, low spend. (Recommend budget increase)
2. **📊 Standard Performers:** Average yield. (Monitor)
3. **⚠️ Budget Drains:** High spend, no conversion. (Recommend immediate pause)

## 🚀 How to Run Locally
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the dashboard: `streamlit run app.py`
