import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from openai import OpenAI

st.set_page_config(page_title="CSV Dashboard Generator", layout="wide")


def load_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .hero-card {
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.92;
    }
    .section-card {
        background: white;
        padding: 1rem 1rem 0.8rem 1rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        margin-bottom: 1rem;
    }
    .small-note {
        color: #475569;
        font-size: 0.92rem;
        margin-top: 0.2rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 14px;
        border-radius: 16px;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    .ai-box {
        background: #f8fafc;
        border: 1px solid #dbeafe;
        border-radius: 14px;
        padding: 1rem;
        margin-top: 0.6rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)


load_custom_css()

st.markdown("""
<div class="hero-card">
    <div class="hero-title">📊 CSV Dashboard Generator</div>
    <div class="hero-subtitle">
        Upload CSV/XLSX files, explore KPIs, create polished visuals, and generate AI-powered business insights.
    </div>
</div>
""", unsafe_allow_html=True)

PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": False,
    "doubleClick": "reset",
    "responsive": True
}

CHART_HEIGHT = 380


# ---------------------------
# SAMPLE DATA
# ---------------------------
def get_sample_data():
    data = {
        "Transaction_ID": ["T001", "T002", "T003", "T004", "T005", "T006", "T007", "T008", "T009", "T010"],
        "Customer_ID": ["C001", "C002", "C003", "C004", "C005", "C001", "C006", "C007", "C008", "C009"],
        "Transaction_Date": ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05",
                             "2026-03-06", "2026-03-07", "2026-03-08", "2026-03-09", "2026-03-10"],
        "Product_Category": ["Electronics", "Clothing", "Grocery", "Electronics", "Clothing",
                             "Grocery", "Electronics", "Grocery", "Clothing", "Electronics"],
        "Product_Name": ["Headphones", "T-Shirt", "Rice Bag", "Smartphone", "Jeans",
                         "Milk Pack", "Keyboard", "Vegetables", "Jacket", "Mouse"],
        "Quantity": [2, 3, 1, 1, 2, 5, 1, 3, 1, 2],
        "Unit_Price": [1500, 500, 1200, 20000, 1200, 50, 800, 200, 3000, 400],
        "Total_Amount": [3000, 1500, 1200, 20000, 2400, 250, 800, 600, 3000, 800],
        "Payment_Method": ["Card", "Cash", "UPI", "Card", "UPI", "Cash", "Card", "UPI", "Card", "Cash"],
        "Store_Location": ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai",
                           "Bangalore", "Pune", "Kolkata", "Delhi", "Mumbai"]
    }
    return pd.DataFrame(data)


# ---------------------------
# HELPERS
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
    return {
        "empty_cols": empty_cols,
        "constant_cols": constant_cols,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def detect_business_columns(df, numeric_cols, categorical_cols):
    business_hints = {"amount": None, "category": None, "date": None}
    for col in df.columns:
        c = col.lower()
        if business_hints["amount"] is None and ("amount" in c or "sales" in c or "revenue" in c):
            business_hints["amount"] = col
        if business_hints["category"] is None and "category" in c:
            business_hints["category"] = col
        if business_hints["date"] is None and "date" in c:
            business_hints["date"] = col

    if business_hints["amount"] is None and numeric_cols:
        business_hints["amount"] = numeric_cols[0]
    if business_hints["category"] is None and categorical_cols:
        business_hints["category"] = categorical_cols[0]

    return business_hints


def get_outlier_info(series):
    clean = series.dropna()
    if len(clean) < 5:
        return None
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return None
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = clean[(clean < lower) | (clean > upper)]
    return {"count": len(outliers), "pct": round(len(outliers) / len(clean) * 100, 2)}


def generate_smart_insights(df, numeric_cols, categorical_cols, date_cols, quality_checks):
    insights = []
    business_cols = detect_business_columns(df, numeric_cols, categorical_cols)

    if quality_checks["missing_values"] > 0:
        insights.append(f"Dataset contains {quality_checks['missing_values']} missing values, so some results may be affected.")

    if quality_checks["duplicate_rows"] > 0:
        insights.append(f"Dataset contains {quality_checks['duplicate_rows']} duplicate rows, which may inflate totals and counts.")

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                value = corr.iloc[i, j]
                if not pd.isna(value):
                    corr_pairs.append((corr.columns[i], corr.columns[j], value, abs(value)))
        if corr_pairs:
            best = sorted(corr_pairs, key=lambda x: x[3], reverse=True)[0]
            direction = "positive" if best[2] > 0 else "negative"
            insights.append(
                f"Strongest relationship found between {best[0]} and {best[1]} with {direction} correlation of {best[2]:.2f}."
            )

    for col in numeric_cols[:3]:
        outlier = get_outlier_info(df[col])
        if outlier and outlier["count"] > 0:
            insights.append(f"{col} has {outlier['count']} potential outliers ({outlier['pct']}% of non-null values).")

        mean_val = df[col].dropna().mean()
        median_val = df[col].dropna().median()
        if median_val != 0 and mean_val > median_val * 1.2:
            insights.append(
                f"{col} appears right-skewed because the average ({mean_val:.2f}) is higher than the median ({median_val:.2f})."
            )

    if business_cols["amount"] in df.columns and business_cols["category"] in df.columns:
        grp = df.groupby(business_cols["category"])[business_cols["amount"]].sum().sort_values(ascending=False)
        if len(grp) > 0 and grp.sum() != 0:
            top_cat = grp.index[0]
            pct = grp.iloc[0] / grp.sum() * 100
            insights.append(
                f"Top contributing {business_cols['category']} is {top_cat}, contributing {pct:.2f}% of total {business_cols['amount']}."
            )

    return insights[:8]


def build_summary_text(df, numeric_cols, categorical_cols, quality_checks, smart_insights):
    lines = [
        "CSV Dashboard Summary",
        "=" * 30,
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        f"Missing Values: {quality_checks['missing_values']}",
        f"Duplicate Rows: {quality_checks['duplicate_rows']}",
        ""
    ]

    if smart_insights:
        lines.append("Top Insights:")
        for item in smart_insights[:5]:
            lines.append(f"- {item}")
        lines.append("")

    if numeric_cols:
        lines.append("Numeric Summary:")
        for col in numeric_cols[:5]:
            lines.append(
                f"- {col}: mean={df[col].mean():.2f}, median={df[col].median():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}"
            )

    return "\n".join(lines)


def create_pdf_report(df, quality_checks, smart_insights, ai_summary_text=None):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "CSV Dashboard Report")

    y -= 30
    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, f"Rows: {df.shape[0]}")
    y -= 18
    pdf.drawString(50, y, f"Columns: {df.shape[1]}")
    y -= 18
    pdf.drawString(50, y, f"Missing Values: {quality_checks['missing_values']}")
    y -= 18
    pdf.drawString(50, y, f"Duplicate Rows: {quality_checks['duplicate_rows']}")

    if ai_summary_text:
        y -= 30
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(50, y, "AI Executive Summary")
        y -= 20
        pdf.setFont("Helvetica", 10)
        for line in ai_summary_text.split("\n"):
            if not line.strip():
                continue
            if y < 80:
                pdf.showPage()
                y = height - 50
                pdf.setFont("Helvetica", 10)
            pdf.drawString(50, y, line[:110])
            y -= 16

    y -= 20
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Smart Insights")
    y -= 20
    pdf.setFont("Helvetica", 10)

    if smart_insights:
        for insight in smart_insights[:6]:
            if y < 80:
                pdf.showPage()
                y = height - 50
                pdf.setFont("Helvetica", 10)
            pdf.drawString(50, y, f"- {insight[:100]}")
            y -= 16
    else:
        pdf.drawString(50, y, "No major insights detected.")
        y -= 16

    y -= 20
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Top 5 Records")
    y -= 20
    pdf.setFont("Helvetica", 9)

    preview = df.head(5).astype(str)
    for _, row in preview.iterrows():
        line = " | ".join(row.values[:4])
        if y < 80:
            pdf.showPage()
            y = height - 50
            pdf.setFont("Helvetica", 9)
        pdf.drawString(50, y, line[:110])
        y -= 14

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def load_uploaded_file(uploaded_file, selected_sheet=None):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file), "Uploaded CSV"

    if file_name.endswith(".xlsx"):
        if selected_sheet is None:
            excel_file = pd.ExcelFile(uploaded_file)
            return excel_file.sheet_names, "excel_sheets"
        return pd.read_excel(uploaded_file, sheet_name=selected_sheet), f"Uploaded Excel File - {selected_sheet}"

    return None, None


# ---------------------------
# AI HELPERS
# ---------------------------
def get_openai_api_key():
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return os.getenv("OPENAI_API_KEY")


def get_ai_client():
    api_key = get_openai_api_key()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def convert_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    return value


def build_dataset_context(df, numeric_cols, categorical_cols, date_cols, quality_checks, smart_insights):
    sample_rows = df.head(8).copy()
    for col in sample_rows.columns:
        sample_rows[col] = sample_rows[col].apply(convert_value)

    numeric_summary = {}
    for col in numeric_cols[:8]:
        clean = df[col].dropna()
        if len(clean) == 0:
            continue
        numeric_summary[col] = {
            "mean": round(float(clean.mean()), 2),
            "median": round(float(clean.median()), 2),
            "min": round(float(clean.min()), 2),
            "max": round(float(clean.max()), 2)
        }

    category_summary = {}
    for col in categorical_cols[:5]:
        vc = df[col].astype(str).value_counts(dropna=False).head(5)
        category_summary[col] = {str(k): int(v) for k, v in vc.items()}

    context = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "date_columns": date_cols,
        "quality_checks": quality_checks,
        "smart_insights": smart_insights,
        "numeric_summary": numeric_summary,
        "top_categories": category_summary,
        "sample_rows": sample_rows.to_dict(orient="records")
    }
    return json.dumps(context, indent=2)


def ask_openai_for_summary(dataset_context, audience="business"):
    client = get_ai_client()
    if client is None:
        return None, "OpenAI API key not found. Add OPENAI_API_KEY in Streamlit secrets or environment variables."

    audience_map = {
        "business": "business user or manager",
        "technical": "data analyst or data scientist",
        "executive": "leadership audience"
    }

    prompt = f"""
You are an expert data analyst.

You are given a compact JSON summary of a dataset.
Write a clear executive summary for a {audience_map.get(audience, 'business user')}.

Requirements:
- Keep it concise and useful.
- Mention the dataset size and structure.
- Mention key trends or relationships.
- Mention data quality issues if any.
- Mention 3 to 5 actionable recommendations.
- Do not invent facts beyond the provided context.
- Use simple business-friendly language.
- Format the answer with short headings and bullet points.

Dataset context:
{dataset_context}
"""

    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt
        )
        return response.output_text, None
    except Exception as e:
        return None, f"AI summary failed: {str(e)}"


def ask_openai_about_data(dataset_context, user_question):
    client = get_ai_client()
    if client is None:
        return None, "OpenAI API key not found. Add OPENAI_API_KEY in Streamlit secrets or environment variables."

    prompt = f"""
You are helping answer questions about a tabular dataset.

Rules:
- Answer ONLY using the supplied dataset context.
- If the answer is not supported by the context, say that more rows or calculations are needed.
- Be clear, direct, and concise.
- If useful, provide a short reasoning section.
- Do not claim you inspected rows beyond the provided sample and summaries.

Dataset context:
{dataset_context}

User question:
{user_question}
"""

    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt
        )
        return response.output_text, None
    except Exception as e:
        return None, f"AI question answering failed: {str(e)}"


# ---------------------------
# DATA INPUT
# ---------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("🚀 Get Started")

c1, c2, c3 = st.columns([2.2, 1, 1])

with c1:
    uploaded_file = st.file_uploader(
        "Upload your CSV or Excel file",
        type=["csv", "xlsx"],
        key="main_file_uploader",
        help="Supported formats: .csv and .xlsx"
    )

with c2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    use_sample = st.button("Try Sample Dataset", use_container_width=True, key="sample_dataset_button")

with c3:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    st.download_button(
        "Download Demo CSV Format",
        data=get_sample_data().to_csv(index=False).encode("utf-8"),
        file_name="sample_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown(
    '<p class="small-note">Tip: Start with the sample dataset to test all charts and insights before uploading your own file.</p>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

df = None
source_label = None

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        df, source_label = load_uploaded_file(uploaded_file)

    elif file_name.endswith(".xlsx"):
        sheet_result, result_type = load_uploaded_file(uploaded_file)
        if result_type == "excel_sheets":
            selected_sheet = st.selectbox("Select Excel sheet", sheet_result, key="excel_sheet_select")
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            source_label = f"Uploaded Excel File - {selected_sheet}"
    else:
        st.error("Unsupported file format. Please upload a CSV or XLSX file.")
        st.stop()

elif use_sample:
    df = get_sample_data()
    source_label = "Sample Dataset"

if df is not None:
    st.success(f"Loaded successfully: {source_label}")

    with st.spinner("Analyzing your data..."):
        st.sidebar.markdown("## ⚙️ Control Panel")
        st.sidebar.caption("Customize columns, charts, and analysis view")

        selected_columns = st.sidebar.multiselect(
            "Select columns to include",
            options=df.columns.tolist(),
            default=df.columns.tolist(),
            key="sidebar_selected_columns"
        )

        show_preview_rows = st.sidebar.slider(
            "Rows to preview",
            min_value=5,
            max_value=50,
            value=10,
            step=5
        )

        chart_template = st.sidebar.selectbox(
            "Chart style",
            ["plotly", "plotly_white", "ggplot2", "seaborn", "simple_white"],
            index=1,
            key="chart_template"
        )

        ai_mode_enabled = st.sidebar.toggle("Enable AI features", value=True)
        ai_audience = st.sidebar.selectbox(
            "AI summary audience",
            ["business", "executive", "technical"],
            index=0
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
        dataset_context = build_dataset_context(
            df, numeric_cols, categorical_cols, date_cols, quality_checks, smart_insights
        )

    def apply_chart_style(fig):
        fig.update_layout(
            template=chart_template,
            height=CHART_HEIGHT,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        return fig

    if "ai_summary_text" not in st.session_state:
        st.session_state.ai_summary_text = None

    if "ai_answer_text" not in st.session_state:
        st.session_state.ai_answer_text = None

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    s1, s2, s3 = st.columns([1.2, 1, 1])

    with s1:
        st.markdown(f"**Source:** {source_label}")
    with s2:
        st.markdown(f"**Numeric columns:** {len(numeric_cols)}")
    with s3:
        st.markdown(f"**Categorical columns:** {len(categorical_cols)}")

    if date_cols:
        st.markdown(f"**Detected date columns:** {', '.join(date_cols[:5])}")
    else:
        st.markdown("**Detected date columns:** None")

    st.markdown('</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📊 Overview", "📈 Trends", "📉 Distribution", "🔗 Correlation", "🛡️ Data Quality", "🧠 Insights & AI"]
    )

    with tab1:
        st.subheader("📌 Key Metrics")
        a, b, c, d = st.columns(4)
        a.metric("Rows", df.shape[0])
        b.metric("Columns", df.shape[1])
        c.metric("Missing Values", int(df.isnull().sum().sum()))
        d.metric("Duplicate Rows", int(df.duplicated().sum()))

        st.subheader("📂 Data Preview")
        st.dataframe(df.head(show_preview_rows), use_container_width=True)

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
            apply_chart_style(pie_fig)
            st.plotly_chart(
                pie_fig,
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="overview_pie_chart"
            )

        st.subheader("🔝 Top 5 Records")
        st.dataframe(df.head(5), use_container_width=True)

        st.subheader("🔚 Bottom 5 Records")
        st.dataframe(df.tail(5), use_container_width=True)

    with tab2:
        st.subheader("📈 Trend Analysis")
        if numeric_cols:
            trend_col = st.selectbox(
                "Select numeric column for trend analysis",
                numeric_cols,
                key="trend_col"
            )

            line_fig = px.line(df, y=trend_col, title=f"{trend_col} Trend")
            apply_chart_style(line_fig)
            st.plotly_chart(
                line_fig,
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="trend_line_chart"
            )

            area_fig = px.area(df, y=trend_col, title=f"{trend_col} Area Trend")
            apply_chart_style(area_fig)
            st.plotly_chart(
                area_fig,
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="trend_area_chart"
            )

            if date_cols:
                date_col = st.selectbox("Select date column", date_cols, key="date_col")
                temp = df[[date_col, trend_col]].copy()
                temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
                temp = temp.dropna().sort_values(date_col)

                if not temp.empty:
                    dated_fig = px.line(
                        temp,
                        x=date_col,
                        y=trend_col,
                        title=f"{trend_col} over time"
                    )
                    apply_chart_style(dated_fig)
                    st.plotly_chart(
                        dated_fig,
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                        key="dated_trend_chart"
                    )
        else:
            st.info("No numeric columns found.")

    with tab3:
        st.subheader("📉 Distribution Analysis")

        if numeric_cols:
            dist_col = st.selectbox(
                "Select numeric column for distribution",
                numeric_cols,
                key="dist_col"
            )

            hist_fig = px.histogram(
                df,
                x=dist_col,
                nbins=30,
                title=f"Histogram of {dist_col}"
            )
            apply_chart_style(hist_fig)
            st.plotly_chart(
                hist_fig,
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="distribution_histogram"
            )

            box_fig = px.box(df, y=dist_col, title=f"Box Plot of {dist_col}")
            apply_chart_style(box_fig)
            st.plotly_chart(
                box_fig,
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="distribution_boxplot"
            )

            violin_fig = px.violin(df, y=dist_col, title=f"Violin Plot of {dist_col}")
            apply_chart_style(violin_fig)
            st.plotly_chart(
                violin_fig,
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="distribution_violin"
            )

        if categorical_cols:
            st.subheader("🍰 Category Distribution")
            cat_col = st.selectbox(
                "Select categorical column",
                categorical_cols,
                key="cat_dist"
            )

            pie_fig = px.pie(df, names=cat_col, title=f"{cat_col} Distribution")
            apply_chart_style(pie_fig)
            st.plotly_chart(
                pie_fig,
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="distribution_pie_chart"
            )

            cat_counts = df[cat_col].value_counts().head(10).reset_index()
            cat_counts.columns = [cat_col, "Count"]
            bar_fig = px.bar(
                cat_counts,
                x=cat_col,
                y="Count",
                title=f"Top Categories in {cat_col}"
            )
            apply_chart_style(bar_fig)
            st.plotly_chart(
                bar_fig,
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="distribution_bar_chart"
            )

    with tab4:
        st.subheader("🔗 Correlation Analysis")

        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            heatmap_fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
            apply_chart_style(heatmap_fig)
            st.plotly_chart(
                heatmap_fig,
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="correlation_heatmap"
            )

            x_axis = st.selectbox("Select X-axis", numeric_cols, key="corr_x")
            y_axis = st.selectbox("Select Y-axis", numeric_cols, key="corr_y")

            scatter_fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
            apply_chart_style(scatter_fig)
            st.plotly_chart(
                scatter_fig,
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="correlation_scatter"
            )

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
            apply_chart_style(bubble_fig)
            st.plotly_chart(
                bubble_fig,
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="correlation_bubble"
            )
        else:
            st.info("At least two numeric columns are needed.")

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

    with tab6:
        st.subheader("🤖 Smart Insights")
        if smart_insights:
            for insight in smart_insights:
                st.markdown(f"- {insight}")
        else:
            st.info("No strong insights detected.")

        st.subheader("🧠 Advanced Insights")
        if numeric_cols:
            for col in numeric_cols[:4]:
                st.markdown(f"### {col}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean", round(df[col].mean(), 2))
                c2.metric("Median", round(df[col].median(), 2))
                c3.metric("Min", round(df[col].min(), 2))
                c4.metric("Max", round(df[col].max(), 2))

        st.subheader("✨ AI Executive Summary")
        if ai_mode_enabled:
            col_ai_1, col_ai_2 = st.columns([1, 2])
            with col_ai_1:
                generate_ai_summary = st.button("Generate AI Summary", use_container_width=True)
            with col_ai_2:
                st.caption("Creates a business-friendly interpretation from dataset structure, summaries, and sample rows.")

            if generate_ai_summary:
                with st.spinner("Generating AI summary..."):
                    ai_summary_text, ai_error = ask_openai_for_summary(dataset_context, audience=ai_audience)
                    if ai_error:
                        st.error(ai_error)
                    else:
                        st.session_state.ai_summary_text = ai_summary_text

            if st.session_state.ai_summary_text:
                st.markdown('<div class="ai-box">', unsafe_allow_html=True)
                st.markdown(st.session_state.ai_summary_text)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Enable AI features from the sidebar to use AI summary and Q&A.")

        st.subheader("💬 Ask Your Data")
        if ai_mode_enabled:
            user_question = st.text_input(
                "Ask a question about this dataset",
                placeholder="Example: Which category looks strongest, and what data quality issue should I watch?"
            )
            ask_button = st.button("Ask AI")

            if ask_button:
                if not user_question.strip():
                    st.warning("Please enter a question first.")
                else:
                    with st.spinner("Analyzing your question..."):
                        ai_answer_text, ai_error = ask_openai_about_data(dataset_context, user_question.strip())
                        if ai_error:
                            st.error(ai_error)
                        else:
                            st.session_state.ai_answer_text = ai_answer_text

            if st.session_state.ai_answer_text:
                st.markdown('<div class="ai-box">', unsafe_allow_html=True)
                st.markdown(st.session_state.ai_answer_text)
                st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("⬇️ Download Options")
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Processed CSV",
            data=csv_data,
            file_name="processed_data.csv",
            mime="text/csv",
            key="download_processed_csv"
        )

        summary_text = build_summary_text(
            df, numeric_cols, categorical_cols, quality_checks, smart_insights
        )
        st.download_button(
            label="Download Summary Report (.txt)",
            data=summary_text.encode("utf-8"),
            file_name="summary_report.txt",
            mime="text/plain",
            key="download_summary_txt"
        )

        pdf_bytes = create_pdf_report(
            df,
            quality_checks,
            smart_insights,
            ai_summary_text=st.session_state.ai_summary_text
        )
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="csv_dashboard_report.pdf",
            mime="application/pdf",
            key="download_pdf_report"
        )

        st.subheader("💎 Premium Features")
        st.info(
            "Upgrade to unlock advanced AI explanations, saved dashboards, multi-file comparison, and premium PDF reports."
        )

else:
    st.info("Upload a CSV or XLSX file or click 'Try Sample Dataset' to begin.")
