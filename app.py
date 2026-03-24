import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="CSV Dashboard Generator", layout="wide")

# ---------------------------
# HEADER
# ---------------------------
st.title("📊 CSV Dashboard Generator")
st.success("NEW VERSION RUNNING - TEST 123")
st.caption("AI-powered automated business reporting tool")
st.write(
    "Upload your CSV file and get instant insights, visualizations, "
    "data quality checks, and downloadable data."
)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"],accept_multiple_files=False)
st.write("Uploaded file object:",uploaded_file)


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def detect_date_columns(df):
    date_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_datetime(df[col], errors="coerce")
            if converted.notna().mean() > 0.7:
                date_cols.append(col)
        elif "datetime" in str(df[col].dtype):
            date_cols.append(col)
    return date_cols


def data_quality_checks(df):
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]

    checks = {
        "empty_cols": empty_cols,
        "constant_cols": constant_cols,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return checks


def detect_business_columns(df, numeric_cols, categorical_cols):
    business_hints = {
        "sales": None,
        "amount": None,
        "revenue": None,
        "profit": None,
        "quantity": None,
        "customer": None,
        "product": None,
        "category": None,
        "date": None
    }

    for col in df.columns:
        col_lower = col.lower()

        if business_hints["sales"] is None and "sale" in col_lower:
            business_hints["sales"] = col
        if business_hints["amount"] is None and "amount" in col_lower:
            business_hints["amount"] = col
        if business_hints["revenue"] is None and "revenue" in col_lower:
            business_hints["revenue"] = col
        if business_hints["profit"] is None and "profit" in col_lower:
            business_hints["profit"] = col
        if business_hints["quantity"] is None and ("qty" in col_lower or "quantity" in col_lower):
            business_hints["quantity"] = col
        if business_hints["customer"] is None and "customer" in col_lower:
            business_hints["customer"] = col
        if business_hints["product"] is None and "product" in col_lower:
            business_hints["product"] = col
        if business_hints["category"] is None and "category" in col_lower:
            business_hints["category"] = col
        if business_hints["date"] is None and "date" in col_lower:
            business_hints["date"] = col

    if business_hints["amount"] is None and numeric_cols:
        business_hints["amount"] = numeric_cols[0]

    if business_hints["category"] is None and categorical_cols:
        business_hints["category"] = categorical_cols[0]

    return business_hints


def get_outlier_info(series):
    clean_series = series.dropna()
    if len(clean_series) < 5:
        return None

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return None

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = clean_series[(clean_series < lower) | (clean_series > upper)]

    return {
        "count": len(outliers),
        "pct": round((len(outliers) / len(clean_series)) * 100, 2)
    }


def generate_smart_insights(df, numeric_cols, categorical_cols, date_cols, quality_checks):
    insights = []
    business_cols = detect_business_columns(df, numeric_cols, categorical_cols)

    if quality_checks["missing_values"] > 0:
        insights.append({
            "priority": 100,
            "text": f"Dataset contains **{quality_checks['missing_values']} missing values**, so some summaries may be affected."
        })

    if quality_checks["duplicate_rows"] > 0:
        insights.append({
            "priority": 95,
            "text": f"Dataset contains **{quality_checks['duplicate_rows']} duplicate rows**, which may inflate totals and averages."
        })

    if quality_checks["empty_cols"]:
        insights.append({
            "priority": 92,
            "text": f"These columns are completely empty: **{', '.join(quality_checks['empty_cols'])}**."
        })

    if quality_checks["constant_cols"]:
        insights.append({
            "priority": 90,
            "text": f"These columns have only one value throughout the dataset: **{', '.join(quality_checks['constant_cols'])}**."
        })

    for col in numeric_cols[:6]:
        clean_series = df[col].dropna()
        if clean_series.empty:
            continue

        mean_val = clean_series.mean()
        median_val = clean_series.median()

        if mean_val > median_val * 1.2 and median_val != 0:
            insights.append({
                "priority": 80,
                "text": f"**{col}** appears right-skewed because mean (**{mean_val:.2f}**) is higher than median (**{median_val:.2f}**)."
            })

        outlier_info = get_outlier_info(df[col])
        if outlier_info and outlier_info["count"] > 0:
            insights.append({
                "priority": 85,
                "text": f"**{col}** has **{outlier_info['count']} potential outliers** ({outlier_info['pct']}% of non-null values)."
            })

        if clean_series.nunique() > 1:
            cv = clean_series.std() / clean_series.mean() if clean_series.mean() != 0 else np.nan
            if pd.notna(cv) and cv > 1:
                insights.append({
                    "priority": 75,
                    "text": f"**{col}** is highly variable relative to its mean, indicating wide dispersion."
                })

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        corr_pairs = []

        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                value = corr.iloc[i, j]
                if not pd.isna(value):
                    corr_pairs.append((corr.columns[i], corr.columns[j], value, abs(value)))

        if corr_pairs:
            best_pair = sorted(corr_pairs, key=lambda x: x[3], reverse=True)[0]
            direction = "positive" if best_pair[2] > 0 else "negative"
            if best_pair[3] >= 0.5:
                insights.append({
                    "priority": 88,
                    "text": f"Strongest relationship found between **{best_pair[0]}** and **{best_pair[1]}** with **{direction} correlation = {best_pair[2]:.2f}**."
                })

    for col in categorical_cols[:4]:
        vc = df[col].value_counts(dropna=False)
        if not vc.empty:
            top_value = vc.index[0]
            top_count = vc.iloc[0]
            top_pct = round((top_count / len(df)) * 100, 2)

            if top_pct >= 50:
                insights.append({
                    "priority": 82,
                    "text": f"**{col}** is highly concentrated: top value **{top_value}** represents **{top_pct}%** of rows."
                })
            else:
                insights.append({
                    "priority": 60,
                    "text": f"Most frequent value in **{col}** is **{top_value}** ({top_pct}% of rows)."
                })

    chosen_date_col = None
    if business_cols["date"] in df.columns:
        chosen_date_col = business_cols["date"]
    elif date_cols:
        chosen_date_col = date_cols[0]

    value_col = None
    for candidate in [
        business_cols["sales"],
        business_cols["revenue"],
        business_cols["amount"],
        business_cols["profit"],
        business_cols["quantity"]
    ]:
        if candidate in numeric_cols:
            value_col = candidate
            break

    if chosen_date_col and value_col:
        temp = df[[chosen_date_col, value_col]].copy()
        temp[chosen_date_col] = pd.to_datetime(temp[chosen_date_col], errors="coerce")
        temp = temp.dropna(subset=[chosen_date_col, value_col])

        if len(temp) >= 3:
            temp = temp.sort_values(chosen_date_col)
            grouped = temp.groupby(temp[chosen_date_col].dt.to_period("M"))[value_col].sum()

            if len(grouped) >= 2:
                first_val = grouped.iloc[0]
                last_val = grouped.iloc[-1]

                if first_val != 0:
                    change_pct = ((last_val - first_val) / abs(first_val)) * 100
                    if change_pct > 10:
                        insights.append({
                            "priority": 87,
                            "text": f"**{value_col}** shows an increasing trend over time, rising by approximately **{change_pct:.2f}%**."
                        })
                    elif change_pct < -10:
                        insights.append({
                            "priority": 87,
                            "text": f"**{value_col}** shows a declining trend over time, decreasing by approximately **{abs(change_pct):.2f}%**."
                        })

    if value_col and business_cols["category"] in df.columns:
        grp = df.groupby(business_cols["category"])[value_col].sum().sort_values(ascending=False)
        if len(grp) > 0:
            top_cat = grp.index[0]
            top_val = grp.iloc[0]
            total_val = grp.sum()
            if total_val != 0:
                pct = round((top_val / total_val) * 100, 2)
                insights.append({
                    "priority": 86,
                    "text": f"Top contributing **{business_cols['category']}** is **{top_cat}**, contributing **{pct}%** of total **{value_col}**."
                })

    insights = sorted(insights, key=lambda x: x["priority"], reverse=True)

    final_texts = []
    seen = set()
    for item in insights:
        text = item["text"]
        if text not in seen:
            final_texts.append(text)
            seen.add(text)

    return final_texts[:8]


def build_summary_text(df, numeric_cols, categorical_cols, quality_checks, smart_insights):
    summary = []
    summary.append("CSV Dashboard Summary")
    summary.append("=" * 40)
    summary.append(f"Rows: {df.shape[0]}")
    summary.append(f"Columns: {df.shape[1]}")
    summary.append(f"Missing Values: {quality_checks['missing_values']}")
    summary.append(f"Duplicate Rows: {quality_checks['duplicate_rows']}")
    summary.append("")

    summary.append("Top Insights:")
    for item in smart_insights[:5]:
        summary.append(f"- {item}")
    summary.append("")

    if numeric_cols:
        summary.append("Numeric Columns Overview:")
        for col in numeric_cols[:5]:
            summary.append(
                f"- {col}: mean={df[col].mean():.2f}, median={df[col].median():.2f}, "
                f"min={df[col].min():.2f}, max={df[col].max():.2f}"
            )
        summary.append("")

    if categorical_cols:
        summary.append("Categorical Columns Overview:")
        for col in categorical_cols[:5]:
            mode_vals = df[col].mode()
            top_value = mode_vals.iloc[0] if not mode_vals.empty else "N/A"
            summary.append(f"- {col}: most frequent value={top_value}")
        summary.append("")

    if quality_checks["empty_cols"]:
        summary.append(f"Empty Columns: {', '.join(quality_checks['empty_cols'])}")

    if quality_checks["constant_cols"]:
        summary.append(f"Constant Columns: {', '.join(quality_checks['constant_cols'])}")

    return "\n".join(summary)


# ---------------------------
# MAIN APP
# ---------------------------
if uploaded_file is not None:
    if uploaded_file is not None:
    with st.spinner("Reading and analyzing your CSV file..."):
        try:
            uploaded_file.seek(0)

            try:
                df = pd.read_csv(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin1")

            st.success("File loaded successfully")
            st.write(df.head())

            st.sidebar.header("🔎 Filters")
            selected_columns = st.sidebar.multiselect(
                "Select columns to view",
                options=df.columns.tolist(),
                default=df.columns.tolist()
            )

            if not selected_columns:
                st.warning("Please select at least one column from the sidebar.")
                st.stop()

            df = df[selected_columns]

            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            date_cols = detect_date_columns(df)
            quality_checks = data_quality_checks(df)
            smart_insights = generate_smart_insights(
                df, numeric_cols, categorical_cols, date_cols, quality_checks
            )

            # rest of your tabs code stays same here

        except Exception as e:
            st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a CSV file to begin.")
            st.write("File loaded successfully:",uploaded_file)
            st.write(df.head())

            st.sidebar.header("🔎 Filters")
            selected_columns = st.sidebar.multiselect(
                "Select columns to view",
                options=df.columns.tolist(),
                default=df.columns.tolist()
            )

            if not selected_columns:
                st.warning("Please select at least one column from the sidebar.")
                st.stop()

            df = df[selected_columns]

            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            date_cols = detect_date_columns(df)
            quality_checks = data_quality_checks(df)
            smart_insights = generate_smart_insights(
                df, numeric_cols, categorical_cols, date_cols, quality_checks
            )

            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
                [
                    "📊 Overview",
                    "📈 Trends",
                    "📉 Distribution",
                    "🔗 Correlation",
                    "🛡️ Data Quality",
                    "🧠 Insights & Downloads"
                ]
            )

            # ---------------------------
            # TAB 1: OVERVIEW
            # ---------------------------
            with tab1:
                st.subheader("📌 Key Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Rows", df.shape[0])
                col2.metric("Columns", df.shape[1])
                col3.metric("Missing Values", int(df.isnull().sum().sum()))
                col4.metric("Duplicate Rows", int(df.duplicated().sum()))

                st.subheader("📂 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                st.subheader("🧾 Column Data Types")
                dtype_df = pd.DataFrame({
                    "Column": df.columns,
                    "Data Type": df.dtypes.astype(str)
                })
                st.dataframe(dtype_df, use_container_width=True)

                if categorical_cols:
                    st.subheader("🥧 Category Distribution")
                    overview_cat = st.selectbox(
                        "Select categorical column",
                        categorical_cols,
                        key="overview_cat"
                    )
                    pie_fig = px.pie(df, names=overview_cat, title=f"{overview_cat} Distribution")
                    st.plotly_chart(pie_fig, use_container_width=True)

                st.subheader("🔝 Top 5 Records")
                st.dataframe(df.head(5), use_container_width=True)

                st.subheader("🔚 Bottom 5 Records")
                st.dataframe(df.tail(5), use_container_width=True)

            # ---------------------------
            # TAB 2: TRENDS
            # ---------------------------
            with tab2:
                st.subheader("📈 Trend Analysis")

                if numeric_cols:
                    trend_col = st.selectbox(
                        "Select numeric column for trend analysis",
                        numeric_cols,
                        key="trend_col"
                    )

                    line_fig = px.line(df, y=trend_col, title=f"{trend_col} Trend")
                    st.plotly_chart(line_fig, use_container_width=True)

                    area_fig = px.area(df, y=trend_col, title=f"{trend_col} Area Trend")
                    st.plotly_chart(area_fig, use_container_width=True)

                    scatter_trend_fig = px.scatter(df, y=trend_col, title=f"{trend_col} Scatter Trend")
                    st.plotly_chart(scatter_trend_fig, use_container_width=True)

                    if date_cols:
                        trend_date_col = st.selectbox(
                            "Select date column",
                            date_cols,
                            key="trend_date_col"
                        )
                        temp = df[[trend_date_col, trend_col]].copy()
                        temp[trend_date_col] = pd.to_datetime(temp[trend_date_col], errors="coerce")
                        temp = temp.dropna(subset=[trend_date_col, trend_col]).sort_values(trend_date_col)

                        if not temp.empty:
                            dated_fig = px.line(
                                temp,
                                x=trend_date_col,
                                y=trend_col,
                                title=f"{trend_col} over {trend_date_col}"
                            )
                            st.plotly_chart(dated_fig, use_container_width=True)
                else:
                    st.info("No numeric columns found for trend analysis.")

            # ---------------------------
            # TAB 3: DISTRIBUTION
            # ---------------------------
            with tab3:
                st.subheader("📉 Distribution Analysis")

                if numeric_cols:
                    dist_col = st.selectbox(
                        "Select numeric column for distribution",
                        numeric_cols,
                        key="dist_col"
                    )

                    hist_fig = px.histogram(df, x=dist_col, nbins=30, title=f"Histogram of {dist_col}")
                    st.plotly_chart(hist_fig, use_container_width=True)

                    box_fig = px.box(df, y=dist_col, title=f"Box Plot of {dist_col}")
                    st.plotly_chart(box_fig, use_container_width=True)

                    violin_fig = px.violin(df, y=dist_col, title=f"Violin Plot of {dist_col}")
                    st.plotly_chart(violin_fig, use_container_width=True)

                    if categorical_cols:
                        group_col = st.selectbox(
                            "Select category for grouped box plot",
                            categorical_cols,
                            key="group_box_cat"
                        )
                        grouped_box = px.box(
                            df,
                            x=group_col,
                            y=dist_col,
                            title=f"{dist_col} by {group_col}"
                        )
                        st.plotly_chart(grouped_box, use_container_width=True)

                if categorical_cols:
                    st.subheader("📊 Categorical Analysis")
                    cat_col = st.selectbox(
                        "Select categorical column for bar chart",
                        categorical_cols,
                        key="cat_col_distribution"
                    )
                    cat_counts = df[cat_col].value_counts().head(10).reset_index()
                    cat_counts.columns = [cat_col, "Count"]

                    bar_fig = px.bar(
                        cat_counts,
                        x=cat_col,
                        y="Count",
                        title=f"Top Categories in {cat_col}"
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)

            # ---------------------------
            # TAB 4: CORRELATION
            # ---------------------------
            with tab4:
                st.subheader("🔗 Correlation Analysis")

                if len(numeric_cols) > 1:
                    corr = df[numeric_cols].corr()

                    heatmap_fig = px.imshow(
                        corr,
                        text_auto=True,
                        title="Correlation Heatmap"
                    )
                    st.plotly_chart(heatmap_fig, use_container_width=True)

                    x_axis = st.selectbox("Select X-axis", numeric_cols, key="corr_x")
                    y_axis = st.selectbox("Select Y-axis", numeric_cols, key="corr_y")

                    scatter_fig = px.scatter(
                        df,
                        x=x_axis,
                        y=y_axis,
                        title=f"{x_axis} vs {y_axis}"
                    )
                    st.plotly_chart(scatter_fig, use_container_width=True)

                    bubble_size = st.selectbox(
                        "Select bubble size column",
                        numeric_cols,
                        key="bubble_size"
                    )
                    bubble_fig = px.scatter(
                        df,
                        x=x_axis,
                        y=y_axis,
                        size=bubble_size,
                        title=f"Bubble Chart: {x_axis} vs {y_axis}"
                    )
                    st.plotly_chart(bubble_fig, use_container_width=True)
                else:
                    st.info("At least two numeric columns are needed for correlation analysis.")

            # ---------------------------
            # TAB 5: DATA QUALITY
            # ---------------------------
            with tab5:
                st.subheader("⚠️ Missing Values Summary")
                missing_df = pd.DataFrame({
                    "Column": df.columns,
                    "Missing Count": df.isnull().sum().values,
                    "Missing %": ((df.isnull().sum().values / len(df)) * 100).round(2)
                }).sort_values(by="Missing Count", ascending=False)
                st.dataframe(missing_df, use_container_width=True)

                st.subheader("🛡️ Data Quality Checks")

                if quality_checks["missing_values"] > 0:
                    st.warning(f"Dataset contains {quality_checks['missing_values']} missing values.")

                if quality_checks["duplicate_rows"] > 0:
                    st.warning(f"Dataset contains {quality_checks['duplicate_rows']} duplicate rows.")

                if quality_checks["empty_cols"]:
                    st.warning(f"Empty columns found: {', '.join(quality_checks['empty_cols'])}")

                if quality_checks["constant_cols"]:
                    st.warning(f"Constant columns found: {', '.join(quality_checks['constant_cols'])}")

                if (
                    quality_checks["missing_values"] == 0
                    and quality_checks["duplicate_rows"] == 0
                    and not quality_checks["empty_cols"]
                    and not quality_checks["constant_cols"]
                ):
                    st.success("No major data quality issues found.")

                if numeric_cols:
                    st.subheader("📈 Numeric Summary")
                    st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

                    summary_num_col = st.selectbox(
                        "Select numeric column for advanced stats",
                        numeric_cols,
                        key="summary_num_col"
                    )

                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Mean", round(df[summary_num_col].mean(), 2))
                    col_b.metric("Median", round(df[summary_num_col].median(), 2))
                    col_c.metric("Min", round(df[summary_num_col].min(), 2))
                    col_d.metric("Max", round(df[summary_num_col].max(), 2))

            # ---------------------------
            # TAB 6: INSIGHTS & DOWNLOADS
            # ---------------------------
            with tab6:
                st.subheader("🤖 Smart Insights")
                if smart_insights:
                    for insight in smart_insights:
                        st.markdown(f"- {insight}")
                else:
                    st.info("No strong insights detected for this dataset.")

                st.subheader("🧠 Advanced Insights")
                if numeric_cols:
                    for col in numeric_cols[:4]:
                        st.markdown(f"### {col}")
                        col_a, col_b, col_c, col_d = st.columns(4)
                        col_a.metric("Mean", round(df[col].mean(), 2))
                        col_b.metric("Median", round(df[col].median(), 2))
                        col_c.metric("Min", round(df[col].min(), 2))
                        col_d.metric("Max", round(df[col].max(), 2))
                else:
                    st.info("No numeric columns available for advanced insights.")

                st.subheader("⬇️ Download Processed Data")
                csv_data = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="processed_data.csv",
                    mime="text/csv"
                )

                summary_text = build_summary_text(
                    df,
                    numeric_cols,
                    categorical_cols,
                    quality_checks,
                    smart_insights
                )

                st.download_button(
                    label="Download Summary Report",
                    data=summary_text,
                    file_name="summary_report.txt",
                    mime="text/plain"
                )

                st.subheader("💎 Premium Features")
                st.info(
                    "Upgrade to unlock PDF reports, AI-generated business summary, "
                    "anomaly detection, saved dashboards, and multi-file comparison."
                )

        except Exception as e:
            st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a CSV file to begin.")
