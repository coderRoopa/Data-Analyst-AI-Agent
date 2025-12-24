import streamlit as st
import pandas as pd
import duckdb
from google import genai
import re

st.set_page_config(page_title="AI SQL Data Analyst", layout="wide")
st.title("📊 AI SQL Data Analyst (Gemini + DuckDB)")

with st.sidebar:
    st.header("Configuration")
    gemini_key = st.text_input("Gemini API Key", type="password")
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx"],
    )

if not gemini_key:
    st.info("Please enter your Gemini API key.")
    st.stop()

if not uploaded_file:
    st.info("Please upload a dataset.")
    st.stop()


client = genai.Client(api_key=gemini_key)


if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

st.subheader("📄 Dataset Preview")
st.write(f"Dataset shape: {df.shape}")
st.dataframe(df.head())
st.write("Columns and types:")
st.dataframe(df.dtypes)

# Register dataframe in DuckDB
con = duckdb.connect()
con.register("data", df)

user_query = st.text_area(
    "Ask a question about your data (e.g. total sales by month)"
)

if st.button("Run Query"):
    if not user_query.strip():
        st.warning("Please enter a question.")
        st.stop()

    schema = "\n".join([f"- {col} ({df[col].dtype})" for col in df.columns])

    sql_prompt = f"""
You are an expert SQL analyst using DuckDB.

Table name: data
Columns:
{schema}

Rules:
- Generate ONLY a valid DuckDB SQL query
- Use SELECT only
- Do NOT include markdown
- Do NOT include explanations

User question:
{user_query}
"""


    with st.spinner("Generating SQL..."):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=sql_prompt
        )
        sql = response.text.strip()

    st.subheader("🧠 Generated SQL")
    st.code(sql, language="sql")

    # Safety check
    if not re.match(r"(?i)^select", sql) or ";" in sql:
        st.error("Only single SELECT queries are allowed.")
        st.stop()


    try:
        result_df = con.execute(sql).df()
    except Exception as e:
        st.error(f"SQL execution error: {e}")
        st.stop()

    if result_df.empty:
        st.warning("Query returned no results.")
    else:
        st.subheader("📊 Query Result (Top 100 Rows)")
        st.dataframe(result_df.head(100))
        st.write(f"Showing 100 of {len(result_df)} rows" if len(result_df) > 100 else f"Total rows: {len(result_df)}")

        # Optional download
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="query_results.csv",
            mime="text/csv"
        )


    explain_prompt = f"""
You are a data analyst.

User question:
{user_query}

SQL used:
{sql}

Columns:
{schema}

Result (first 20 rows):
{result_df.head(20).to_markdown()}

Explain the results in clear, simple language.
Highlight any trends or insights.
"""


    with st.spinner("Generating explanation..."):
        explanation_resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=explain_prompt
        )
        explanation = explanation_resp.text.strip()

    st.subheader("📝 Explanation")
    st.markdown(explanation)
