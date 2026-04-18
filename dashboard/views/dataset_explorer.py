import streamlit as st
import pandas as pd
import os

BASE = os.path.join(os.path.dirname(__file__), "..", "..")
DATA_PATH = os.path.join(BASE, "data", "processed", "multimodal_dataset.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def run():
    st.title("Dataset Explorer")

    if not os.path.exists(DATA_PATH):
        st.error("Dataset not found. Run the preprocessing pipeline first.")
        return

    df = load_data()

    # ---- Overview ----
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Numeric Features", f"{len(df.select_dtypes('number').columns):,}")
    c4.metric("Missing Values", f"{int(df.isnull().sum().sum()):,}")

    # ---- Target Distribution ----
    target_col = "Class/ASD"
    if target_col in df.columns:
        st.markdown("---")
        st.subheader("Target Distribution")
        counts = df[target_col].value_counts().sort_index()
        tc1, tc2 = st.columns([1, 2])
        with tc1:
            for val, cnt in counts.items():
                pct = cnt / len(df) * 100
                st.metric(f"Class {int(val)}", f"{cnt}  ({pct:.1f}%)")
        with tc2:
            st.bar_chart(counts, height=250)

    # ---- Data Preview ----
    st.markdown("---")
    st.subheader("Data Preview")
    tab_head, tab_stats, tab_dtypes = st.tabs(["First Rows", "Statistics", "Column Types"])

    with tab_head:
        n_rows = st.slider("Rows to display", 5, 50, 10, key="preview_rows")
        st.dataframe(df.head(n_rows), use_container_width=True)

    with tab_stats:
        st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

    with tab_dtypes:
        type_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Missing": df.isnull().sum().values,
        })
        st.dataframe(type_df, use_container_width=True, hide_index=True)

    # ---- Feature Distribution ----
    st.markdown("---")
    st.subheader("Feature Distribution")
    numeric_cols = df.select_dtypes("number").columns.tolist()

    col_select, chart_type = st.columns([3, 1])
    with col_select:
        selected = st.selectbox("Select a feature", numeric_cols, key="dist_feat")
    with chart_type:
        chart = st.radio("Chart type", ["Bar", "Line", "Area"], horizontal=True, key="dist_chart")

    if selected:
        data = df[selected].dropna()
        if chart == "Bar":
            st.bar_chart(data.value_counts().sort_index(), height=300)
        elif chart == "Line":
            st.line_chart(data.value_counts().sort_index(), height=300)
        else:
            st.area_chart(data.value_counts().sort_index(), height=300)

        # Quick stats for selected feature
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Mean", f"{data.mean():.2f}")
        s2.metric("Median", f"{data.median():.2f}")
        s3.metric("Std Dev", f"{data.std():.2f}")
        s4.metric("Min", f"{data.min():.2f}")
        s5.metric("Max", f"{data.max():.2f}")

    # ---- Correlation ----
    st.markdown("---")
    st.subheader("Correlation with Target")

    if target_col in df.columns and len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()[target_col].drop(target_col, errors="ignore").sort_values(ascending=False)
        top_n = st.slider("Top N features", 5, 30, 15, key="corr_n")

        pos = corr.head(top_n)
        neg = corr.tail(top_n)

        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("**Most positively correlated**")
            st.bar_chart(pos, height=300)
        with tc2:
            st.markdown("**Most negatively correlated**")
            st.bar_chart(neg, height=300)

    # ---- Filter & Download ----
    st.markdown("---")
    with st.expander("Filter & Download"):
        filter_col = st.selectbox("Filter by column", df.columns, key="filter_col")
        unique_vals = df[filter_col].dropna().unique()

        if len(unique_vals) <= 30:
            selected_vals = st.multiselect("Select values", sorted(unique_vals), default=list(unique_vals)[:5])
            filtered = df[df[filter_col].isin(selected_vals)]
        else:
            min_v, max_v = float(df[filter_col].min()), float(df[filter_col].max())
            range_vals = st.slider("Range", min_v, max_v, (min_v, max_v), key="filter_range")
            filtered = df[(df[filter_col] >= range_vals[0]) & (df[filter_col] <= range_vals[1])]

        st.write(f"Filtered: {len(filtered)} rows")
        st.dataframe(filtered.head(20), use_container_width=True)

        st.download_button(
            "Download filtered CSV",
            filtered.to_csv(index=False).encode(),
            "filtered_dataset.csv",
            "text/csv",
        )
