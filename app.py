import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="CSV Dashboard Generator", layout="wide")

# ---------------------------
# HEADER
# ---------------------------
st.title("📊 CSV Dashboard Generator")
st.caption("AI-powered automated business reporting tool")
st.write("Upload your CSV file and get instant insights, visualizations, data quality checks, and downloadable data.")

# ---------------------------
# FILE UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def generate_auto_insights(df, numeric_cols, categorical_cols):
    insights = []

    if numeric_cols:
        for col in numeric_cols[:3]:
            insights.append(
                f"**{col}** → Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}, Min: {df[col].min():.2f}, Max: {df[col].max():.2f}"
            )

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().abs()
            corr_pairs = []

            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

            if corr_pairs:
                top_corr = sorted(corr_pairs, key=lambda x: x[2], reverse=True)[0]
                insights.append(
                    f"Strongest relationship found between **{top_corr[0]}** and **{top_corr[1]}** with correlation **{top_corr[2]:.2f}**."
                )

    if categorical_cols:
        for col in categorical_cols[:2]:
            if not df[col].mode().empty:
                top_item = df[col].mode().iloc[0]
                insights.append(f"Most frequent value in **{col}** is **{top_item}**.")

    missing_count = int(df.isnull().sum().sum())
    if missing_count > 0:
        insights.append(f"Dataset contains **{missing_count} missing values**. Data cleaning may be needed.")

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        insights.append(f"Dataset contains **{duplicate_count} duplicate rows**.")

    return insights


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


def build_summary_text(df, numeric_cols, categorical_cols, quality_checks):
    summary = []
    summary.append("CSV Dashboard Summary")
    summary.append("=" * 30)
    summary.append(f"Rows: {df.shape[0]}")
    summary.append(f"Columns: {df.shape[1]}")
    summary.append(f"Missing Values: {quality_checks['missing_values']}")
    summary.append(f"Duplicate Rows: {quality_checks['duplicate_rows']}")
    summary.append("")

    if numeric_cols:
        summary.append("Numeric Columns Overview:")
        for col in numeric_cols[:5]:
            summary.append(
                f"- {col}: mean={df[col].mean():.2f}, median={df[col].median():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}"
            )
        summary.append("")

    if categorical_cols:
        summary.append("Categorical Columns Overview:")
        for col in categorical_cols[:5]:
            if not df[col].mode().empty:
                summary.append(f"- {col}: most frequent value={df[col].mode().iloc[0]}")
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
    with st.spinner("Reading and analyzing your CSV file..."):
        try:
            df = pd.read_csv(uploaded_file)

            # Sidebar filters
            st.sidebar.header("🔎 Filters")
            selected_columns = st.sidebar.multiselect(
                "Select columns to view",
                options=df.columns.tolist(),
                default=df.columns.tolist()
            )

            if selected_columns:
                df = df[selected_columns]
            else:
                st.warning("Please select at least one column from the sidebar.")
                st.stop()

            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            # Metrics
            st.subheader("📌 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing Values", int(df.isnull().sum().sum()))
            col4.metric("Duplicate Rows", int(df.duplicated().sum()))

            # Data preview
            st.subheader("📂 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Data types
            st.subheader("🧾 Column Data Types")
            dtype_df = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str)
            })
            st.dataframe(dtype_df, use_container_width=True)

            # Missing values summary
            st.subheader("⚠️ Missing Values Summary")
            missing_df = pd.DataFrame({
                "Column": df.columns,
                "Missing Count": df.isnull().sum().values,
                "Missing %": ((df.isnull().sum().values / len(df)) * 100).round(2)
            }).sort_values(by="Missing Count", ascending=False)
            st.dataframe(missing_df, use_container_width=True)

            # Numeric summary
            if numeric_cols:
                st.subheader("📈 Numeric Summary")
                st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

                selected_num_col = st.selectbox("Select numeric column for histogram", numeric_cols)

                fig, ax = plt.subplots()
                counts, bins, patches = ax.hist(df[selected_num_col].dropna(), bins=20, edgecolor="black")

                for i in range(len(counts)):
                    if counts[i] > 0:
                        ax.text(
                            bins[i],
                            counts[i],
                            str(int(counts[i])),
                            fontsize=8,
                            ha="left",
                            va="bottom"
                        )

                ax.set_title(f"Histogram of {selected_num_col}")
                ax.set_xlabel(selected_num_col)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

            else:
                st.info("No numeric columns found in the uploaded dataset.")

            # Scatter plot
            if len(numeric_cols) >= 2:
                st.subheader("🔵 Scatter Plot")
                x_axis = st.selectbox("Select X-axis", numeric_cols, key="x_axis")
                y_axis = st.selectbox("Select Y-axis", numeric_cols, key="y_axis")

                fig2, ax2 = plt.subplots()
                ax2.scatter(df[x_axis], df[y_axis], label="Data Points")
                ax2.set_title(f"{x_axis} vs {y_axis}")
                ax2.set_xlabel(x_axis)
                ax2.set_ylabel(y_axis)
                ax2.legend()
                st.pyplot(fig2)

            # Categorical analysis
            if categorical_cols:
                st.subheader("📊 Categorical Analysis")
                selected_cat_col = st.selectbox("Select categorical column", categorical_cols)

                cat_counts = df[selected_cat_col].value_counts().head(10)

                fig3, ax3 = plt.subplots()
                bars = ax3.bar(cat_counts.index.astype(str), cat_counts.values, label="Category Count")

                for bar in bars:
                    height = bar.get_height()
                    ax3.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        str(int(height)),
                        ha="center",
                        va="bottom",
                        fontsize=8
                    )

                ax3.set_title(f"Top Categories in {selected_cat_col}")
                ax3.set_xlabel(selected_cat_col)
                ax3.set_ylabel("Count")
                ax3.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig3)

            else:
                st.info("No categorical columns found in the uploaded dataset.")

            # Correlation heatmap
            if len(numeric_cols) > 1:
                st.subheader("🔥 Correlation Heatmap")
                corr = df[numeric_cols].corr()

                fig4, ax4 = plt.subplots()
                cax = ax4.matshow(corr)
                fig4.colorbar(cax)

                ax4.set_xticks(range(len(corr.columns)))
                ax4.set_yticks(range(len(corr.columns)))
                ax4.set_xticklabels(corr.columns, rotation=90)
                ax4.set_yticklabels(corr.columns)

                st.pyplot(fig4)

            # Data quality checks
            st.subheader("🛡️ Data Quality Checks")
            quality_checks = data_quality_checks(df)

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

            # Auto insights
            st.subheader("🤖 Auto Insights")
            insights = generate_auto_insights(df, numeric_cols, categorical_cols)

            if insights:
                for insight in insights:
                    st.markdown(f"- {insight}")
            else:
                st.info("No auto insights available for this dataset.")

            # Advanced insights
            st.subheader("🧠 Advanced Insights")
            if numeric_cols:
                for col in numeric_cols[:4]:
                    st.markdown(f"### {col}")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Mean", round(df[col].mean(), 2))
                    col_b.metric("Median", round(df[col].median(), 2))
                    col_c.metric("Min", round(df[col].min(), 2))
                    col_d.metric("Max", round(df[col].max(), 2))

            # Record preview
            st.subheader("🔝 Top 5 Records")
            st.dataframe(df.head(5), use_container_width=True)

            st.subheader("🔚 Bottom 5 Records")
            st.dataframe(df.tail(5), use_container_width=True)

            # Download options
            st.subheader("⬇️ Download Processed Data")
            csv_data = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="processed_data.csv",
                mime="text/csv"
            )

            # Summary download
            summary_text = build_summary_text(df, numeric_cols, categorical_cols, quality_checks)
            st.download_button(
                label="Download Summary Report",
                data=summary_text,
                file_name="summary_report.txt",
                mime="text/plain"
            )

            # Premium section
            st.subheader("💎 Premium Features")
            st.info(
                "Upgrade to unlock PDF reports, AI-generated business summary, anomaly detection, saved dashboards, and multi-file comparison."
            )

        except Exception as e:
            st.error(f"Error reading file: {e}")

else:
    st.info("Please upload a CSV file to begin.")
