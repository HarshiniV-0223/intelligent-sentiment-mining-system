import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import joblib

try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    model_loaded = True
except:
    model_loaded = False
if model_loaded:
    st.success("✅ ML Model Loaded Successfully!")
else:
    st.error("❌ Model not loaded. Check file name!")

from utils import (
    detect_columns,
    clean_text,
    sentiment_from_rating,
    sentiment_from_text,
    extract_issue_keywords,
    extract_positive_keywords,
    generate_improvement_suggestion
)

st.set_page_config(
    page_title="Intelligent Sentiment Mining System for E-Commerce Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# PREMIUM CSS
# --------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0b1020 0%, #111827 100%);
        color: white;
    }

    .main-title {
        background: linear-gradient(90deg, #1d4ed8, #7c3aed);
        padding: 24px;
        border-radius: 18px;
        text-align: center;
        color: white;
        font-size: 2.2rem;
        font-weight: 800;
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
        margin-bottom: 10px;
    }

    .sub-title {
        text-align: center;
        color: #dbeafe;
        font-size: 1.1rem;
        margin-bottom: 20px;
        font-weight: 500;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255,255,255,0.15);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        padding: 18px;
        color: #f8fafc !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        margin-bottom: 18px;
    }

    .glass-card * {
        color: #f8fafc !important;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #93c5fd;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(51,65,85,0.95));
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        text-align: center;
    }

    .metric-label {
        color: #cbd5e1;
        font-size: 1rem;
        font-weight: 600;
    }

    .metric-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin-top: 8px;
    }

    .small-note {
        color: #cbd5e1;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# HEADER
# --------------------------
st.markdown("""
<div class="main-title">🛒 Intelligent Sentiment Mining System for E-Commerce Platform</div>
<div class="sub-title">AI-Powered Customer Review Analysis for Business Intelligence & Decision Support</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <div class="section-title">📌 Project Objective</div>
    <p class="small-note">
        This intelligent system analyzes e-commerce customer reviews and converts them into actionable business intelligence.
        It helps identify sentiment trends, product-level weaknesses, customer pain points, strengths, and strategic improvement recommendations.
    </p>
    <ul>
        <li>Sentiment Classification (Positive / Negative / Neutral)</li>
        <li>Product-wise Performance Evaluation</li>
        <li>Department / Division / Category Analysis</li>
        <li>Complaint Pattern Detection</li>
        <li>Strength & Weakness Mining</li>
        <li>Business Improvement Recommendations</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# --------------------------
# SIDEBAR
# --------------------------
st.sidebar.title("⚙️ Analysis Control Panel")
st.sidebar.markdown("Upload your dataset and configure columns for analysis.")

uploaded_file = st.file_uploader("📂 Upload E-Commerce Dataset (CSV only)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset uploaded successfully!")
        # 🔥 CREATE PRODUCT COLUMN (VERY IMPORTANT)
        # SAFE PRODUCT CREATION
        if 'brand' in df.columns and 'title' in df.columns:
            df['product'] = df['brand'].astype(str) + " - " + df['title'].astype(str)
        else:
            df['product'] = "General Product"



        with st.expander("📋 Preview Uploaded Dataset", expanded=False):
            st.dataframe(df.head())

        detected = detect_columns(df)

        st.markdown("### 🔍 Column Configuration")

        all_cols = ["None"] + list(df.columns)
        # 🔥 Ensure 'product' is included
        if 'product' not in all_cols:
            all_cols.append('product')

        product_col = st.selectbox(
            "🛍 Select Product Column",
            all_cols,
            index=(all_cols.index("product") if "product" in all_cols else 0)
        )

        # AUTO DETECT REVIEW COLUMN SAFELY
        possible_review_cols = ["review", "text", "Text", "Review", "ReviewText"]

        review_col_default = "None"
        for col in possible_review_cols:
            if col in df.columns:
                review_col_default = col
                break

        review_col = st.selectbox(
            "📝 Select Review/Text Column",
            ["None"] + list(df.columns),
            index=(["None"] + list(df.columns)).index(review_col_default) if review_col_default != "None" else 0
        )

        rating_col = st.selectbox(
            "⭐ Select Rating/Score Column",
            all_cols,
            index=(all_cols.index(detected["rating"]) if detected["rating"] in all_cols else 0)
        )

        seller_col = st.selectbox(
            "🏪 Select Division Column (Optional)",
            all_cols,
            index=(all_cols.index(detected["seller"]) if detected["seller"] in all_cols else 0)
        )

        brand_col = st.selectbox(
            "🏷 Select Department Column (Optional)",
            all_cols,
            index=(all_cols.index(detected["brand"]) if detected["brand"] in all_cols else 0)
        )

        title_col = st.selectbox(
            "📰 Select Title Column (Optional)",
            all_cols,
            index=(all_cols.index(detected["title"]) if detected["title"] in all_cols else 0)
        )

        st.info("""
**Recommended for your clothing dataset:**
- Product → Class Name
- Review → Review Text
- Rating → Rating
- Division → Division Name
- Department → Department Name
- Title → Title
""")

        if review_col == "None":
            st.error("❌ Please select a valid Review/Text column to continue.")
            st.stop()

        run_analysis = st.button("🚀 Run Intelligent Analysis", use_container_width=True)

        if run_analysis or "analysis_done" in st.session_state:
            st.session_state.analysis_done = True
            st.write("Selected Review Column:", review_col)
            st.write("Sample Data:", df[review_col].head())
            st.write("Product Column Selected:", product_col)
            st.write("Review Column Selected:", review_col)

            st.write("Sample Product Values:")
            st.write(df[product_col].head())

            st.write("Sample Review Values:")
            st.write(df[review_col].head())
            analysis_df = df.copy()
            st.write("Analysis DF Product Preview:")
            st.write(analysis_df[product_col].head())
            # Build review text
            if title_col != "None":
                analysis_df["final_review"] = (
                    analysis_df[title_col].fillna("").astype(str) + " " +
                    analysis_df[review_col].fillna("").astype(str)
                )
            else:
                analysis_df["final_review"] = analysis_df[review_col].fillna("").astype(str)

            analysis_df["clean_review"] = analysis_df["final_review"].apply(clean_text)
            st.write("Final Review Sample:")
            st.write(analysis_df["final_review"].head())
            # Sentiment
            if rating_col != "None":
                analysis_df["Sentiment"] = analysis_df[rating_col].apply(sentiment_from_rating)
                analysis_df["Sentiment"] = analysis_df.apply(
                    lambda row: sentiment_from_text(row["clean_review"]) if row["Sentiment"] is None else row["Sentiment"],
                    axis=1
                )
            else:
                if model_loaded:
                    vectors = vectorizer.transform(analysis_df["clean_review"])
                    analysis_df["Sentiment"] = model.predict(vectors)
                else:
                    analysis_df["Sentiment"] = analysis_df["clean_review"].apply(sentiment_from_text)
                
                


            # Handle missing selections
            if product_col == "None":
                analysis_df["__Product__"] = "Unknown Product"
                product_col = "__Product__"

            if seller_col == "None":
                analysis_df["__Seller__"] = "Unknown Division"
                seller_col = "__Seller__"

            if brand_col == "None":
                analysis_df["__Brand__"] = "Unknown Department"
                brand_col = "__Brand__"

            # --------------------------
            # KPI Metrics
            # --------------------------
            total_reviews = len(analysis_df)
            pos_count = (analysis_df["Sentiment"] == "Positive").sum()
            neg_count = (analysis_df["Sentiment"] == "Negative").sum()
            neu_count = (analysis_df["Sentiment"] == "Neutral").sum()

            st.markdown("## 📊 Executive KPI Dashboard")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Reviews</div>
                    <div class="metric-value">{total_reviews}</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Positive Reviews</div>
                    <div class="metric-value">{pos_count}</div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Negative Reviews</div>
                    <div class="metric-value">{neg_count}</div>
                </div>
                """, unsafe_allow_html=True)

            with c4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Neutral Reviews</div>
                    <div class="metric-value">{neu_count}</div>
                </div>
                """, unsafe_allow_html=True)

            # Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Sentiment Overview",
                "🛍 Product Insights",
                "🏷 Segment Insights",
                "☁️ Keyword & WordCloud",
                "🧠 Recommendations"
            ])
            
            # --------------------------
            # TAB 1 - Sentiment Overview
            # --------------------------
            with tab1:
                st.markdown("### 📈 Sentiment Distribution Overview")

                sentiment_counts = analysis_df["Sentiment"].value_counts().reset_index()
                sentiment_counts.columns = ["Sentiment", "Count"]

                col1, col2 = st.columns(2)

                with col1:
                    fig_pie = px.pie(
                        sentiment_counts,
                        names="Sentiment",
                        values="Count",
                        title="Sentiment Distribution"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    fig_bar = px.bar(
                        sentiment_counts,
                        x="Sentiment",
                        y="Count",
                        color="Sentiment",
                        title="Sentiment Count Comparison"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            # --------------------------
            # TAB 2 - Product Insights
            # --------------------------
            with tab2:
                st.markdown("### 🛍 Product / Class-wise Sentiment Analysis")

                product_sentiment = analysis_df.groupby([product_col, "Sentiment"]).size().reset_index(name="Count")
                st.dataframe(product_sentiment.head(25), use_container_width=True)

                product_total = analysis_df.groupby(product_col).size().reset_index(name="TotalReviews")
                product_negative = analysis_df[analysis_df["Sentiment"] == "Negative"].groupby(product_col).size().reset_index(name="NegativeReviews")

                product_merged = pd.merge(product_total, product_negative, on=product_col, how="left").fillna(0)
                product_merged["NegativeRatio"] = (product_merged["NegativeReviews"] / product_merged["TotalReviews"]) * 100

                top_problem_products = product_merged.sort_values(by="NegativeRatio", ascending=False).head(10)
                top_best_products = product_merged.sort_values(by="NegativeRatio", ascending=True).head(10)

                st.markdown("#### ⚠️ Top 10 Product Categories Needing Improvement")
                st.dataframe(top_problem_products, use_container_width=True)

                fig_problem = px.bar(
                    top_problem_products,
                    x=product_col,
                    y="NegativeRatio",
                    color="NegativeRatio",
                    title="Highest Negative Review Ratio"
                )
                st.plotly_chart(fig_problem, use_container_width=True)

                st.markdown("#### 🌟 Top 10 Best Performing Product Categories")
                st.dataframe(top_best_products, use_container_width=True)

                fig_best = px.bar(
                    top_best_products,
                    x=product_col,
                    y="NegativeRatio",
                    color="NegativeRatio",
                    title="Lowest Negative Review Ratio"
                )
                st.plotly_chart(fig_best, use_container_width=True)

            # --------------------------
            # TAB 3 - Segment Insights
            # --------------------------
            with tab3:
                st.markdown("### 🏷 Department & Division Insights")

                colA, colB = st.columns(2)

                with colA:
                    st.markdown("#### 🏷 Department-wise Sentiment")
                    brand_summary = analysis_df.groupby([brand_col, "Sentiment"]).size().reset_index(name="Count")
                    st.dataframe(brand_summary.head(25), use_container_width=True)

                    fig_brand = px.bar(
                        brand_summary,
                        x=brand_col,
                        y="Count",
                        color="Sentiment",
                        barmode="group",
                        title="Department-wise Sentiment Distribution"
                    )
                    st.plotly_chart(fig_brand, use_container_width=True)

                with colB:
                    st.markdown("#### 🏪 Division-wise Sentiment")
                    seller_summary = analysis_df.groupby([seller_col, "Sentiment"]).size().reset_index(name="Count")
                    st.dataframe(seller_summary.head(25), use_container_width=True)

                    fig_seller = px.bar(
                        seller_summary,
                        x=seller_col,
                        y="Count",
                        color="Sentiment",
                        barmode="group",
                        title="Division-wise Sentiment Distribution"
                    )
                    st.plotly_chart(fig_seller, use_container_width=True)

            # --------------------------
            # TAB 4 - Keyword & WordCloud
            # --------------------------
            with tab4:
                st.markdown("### ☁️ Keyword Mining & Word Cloud Analysis")

                negative_reviews = analysis_df[analysis_df["Sentiment"] == "Negative"]
                positive_reviews = analysis_df[analysis_df["Sentiment"] == "Positive"]

                issue_keywords = extract_issue_keywords(negative_reviews["clean_review"], top_n=15)
                positive_keywords = extract_positive_keywords(positive_reviews["clean_review"], top_n=15)

                issue_df = pd.DataFrame(issue_keywords, columns=["Issue Keyword", "Frequency"])
                positive_df = pd.DataFrame(positive_keywords, columns=["Positive Keyword", "Frequency"])

                k1, k2 = st.columns(2)

                with k1:
                    st.markdown("#### ❌ Top Complaint Keywords")
                    st.dataframe(issue_df, use_container_width=True)
                    if not issue_df.empty:
                        fig_issue = px.bar(
                            issue_df,
                            x="Issue Keyword",
                            y="Frequency",
                            color="Frequency",
                            title="Most Frequent Complaint Keywords"
                        )
                        st.plotly_chart(fig_issue, use_container_width=True)

                with k2:
                    st.markdown("#### ✅ Top Appreciation Keywords")
                    st.dataframe(positive_df, use_container_width=True)
                    if not positive_df.empty:
                        fig_positive = px.bar(
                            positive_df,
                            x="Positive Keyword",
                            y="Frequency",
                            color="Frequency",
                            title="Most Frequent Positive Keywords"
                        )
                        st.plotly_chart(fig_positive, use_container_width=True)

                wc1, wc2 = st.columns(2)

                with wc1:
                    st.markdown("#### ☁️ Negative Review Word Cloud")
                    neg_text = " ".join(negative_reviews["clean_review"].dropna().astype(str))
                    if neg_text.strip():
                        wordcloud_neg = WordCloud(width=800, height=400, background_color="white").generate(neg_text)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud_neg, interpolation="bilinear")
                        ax.axis("off")
                        st.pyplot(fig)

                with wc2:
                    st.markdown("#### ☁️ Positive Review Word Cloud")
                    pos_text = " ".join(positive_reviews["clean_review"].dropna().astype(str))
                    if pos_text.strip():
                        wordcloud_pos = WordCloud(width=800, height=400, background_color="white").generate(pos_text)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud_pos, interpolation="bilinear")
                        ax.axis("off")
                        st.pyplot(fig)

            # --------------------------
            # TAB 5 - Recommendations
            # --------------------------
                           # --------------------------
            # TAB 5 - Recommendations
            # --------------------------
            with tab5:

                st.markdown("### 🧠 AI-Powered Recommendation Engine")

                categories = sorted(
                    analysis_df[product_col].dropna().astype(str).unique().tolist()
                )

                if "selected_product" not in st.session_state:
                    st.session_state.selected_product = categories[0]

                selected_product = st.selectbox(
                    "Select Product Category / Class for Deep Analysis",
                    categories,
                    key="selected_product"
                )

                product_df = analysis_df[
                    analysis_df[product_col].astype(str) == selected_product
                ]

                product_neg = product_df[
                    product_df["Sentiment"] == "Negative"
                ]

                product_pos = product_df[
                    product_df["Sentiment"] == "Positive"
                ]

                st.markdown(f"#### 🔍 Deep Analysis for: **{selected_product}**")

                st.write(f"**Total Reviews:** {len(product_df)}")
                st.write(f"**Positive Reviews:** {(product_df['Sentiment'] == 'Positive').sum()}")
                st.write(f"**Negative Reviews:** {(product_df['Sentiment'] == 'Negative').sum()}")
                st.write(f"**Neutral Reviews:** {(product_df['Sentiment'] == 'Neutral').sum()}")

                product_issue_keywords = extract_issue_keywords(
                    product_neg["clean_review"],
                    top_n=10
                )

                product_positive_keywords = extract_positive_keywords(
                    product_pos["clean_review"],
                    top_n=10
                )

                r1, r2 = st.columns(2)

                with r1:
                    st.markdown("#### ❌ Top Product Complaint Keywords")
                    st.dataframe(
                        pd.DataFrame(
                            product_issue_keywords,
                            columns=["Issue Keyword", "Frequency"]
                        ),
                        use_container_width=True
                    )

                with r2:
                    st.markdown("#### ✅ Top Product Strength Keywords")
                    st.dataframe(
                        pd.DataFrame(
                            product_positive_keywords,
                            columns=["Positive Keyword", "Frequency"]
                        ),
                        use_container_width=True
                    )

                suggestions = generate_improvement_suggestion(
                    product_issue_keywords
                )

                st.markdown("#### 💡 Recommended Improvements")

                for i, suggestion in enumerate(suggestions, start=1):
                    st.success(f"{i}. {suggestion}")

                st.markdown("## 📌 Executive Business Insight Summary")

                all_negative_reviews = analysis_df[
                    analysis_df["Sentiment"] == "Negative"
                ]

                overall_issue_keywords = extract_issue_keywords(
                    all_negative_reviews["clean_review"],
                    top_n=20
                )

                overall_suggestions = generate_improvement_suggestion(
                    overall_issue_keywords
                )

                if overall_suggestions:
                    for i, suggestion in enumerate(overall_suggestions, start=1):
                        st.info(f"{i}. {suggestion}")
                else:
                    st.success("✅ No major issues detected from customer reviews.")

                st.markdown("---")

                st.subheader("⬇️ Download Processed Results")

                csv = analysis_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="📥 Download Analyzed Dataset",
                    data=csv,
                    file_name="analyzed_ecommerce_sentiment.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    except Exception as e:
        st.error(f"❌ Error while processing dataset: {e}")
        st.warning("Please check whether the CSV file format is correct.")
else:
    st.warning("👆 Please upload a CSV file to begin analysis.")
    


