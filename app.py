import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="CSV Dashboard Generator", layout="wide")

st.title("📊 CSV Dashboard Generator")
st.write("Upload your CSV file and get instant insights, visualizations, and downloadable data.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

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

        st.subheader("⚠️ Missing Values Summary")
        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Count": df.isnull().sum().values,
            "Missing %": ((df.isnull().sum().values / len(df)) * 100).round(2)
        }).sort_values(by="Missing Count", ascending=False)
        st.dataframe(missing_df, use_container_width=True)

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

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

        st.subheader("🧠 Advanced Insights")

        if numeric_cols:
            for col in numeric_cols:
                st.markdown(f"### {col}")
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Mean", round(df[col].mean(), 2))
                col_b.metric("Median", round(df[col].median(), 2))
                col_c.metric("Min", round(df[col].min(), 2))
                col_d.metric("Max", round(df[col].max(), 2))

        if df.isnull().sum().sum() > 0:
            st.warning("Dataset contains missing values. Cleaning may be required.")

        if df.duplicated().sum() > 0:
            st.warning("Dataset contains duplicate rows.")

        st.subheader("🔝 Top 5 Records")
        st.dataframe(df.head(5), use_container_width=True)

        st.subheader("🔚 Bottom 5 Records")
        st.dataframe(df.tail(5), use_container_width=True)

        st.subheader("⬇️ Download Processed Data")
        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="processed_data.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.info("Please upload a CSV file to begin.")