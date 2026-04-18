import streamlit as st
import pandas as pd
import os

BASE = os.path.join(os.path.dirname(__file__), "..", "..")
PLOTS = os.path.join(BASE, "outputs", "plots")
REPORTS = os.path.join(BASE, "outputs", "reports")


def run():
    st.title("Model Analysis")

    # ---- Model Comparison Table ----
    report_path = os.path.join(REPORTS, "model_comparison.csv")
    if os.path.exists(report_path):
        df = pd.read_csv(report_path, index_col=0)
        st.subheader("Performance Comparison")

        # Display as styled metrics
        cols = st.columns(len(df))
        for i, (model_name, row) in enumerate(df.iterrows()):
            with cols[i]:
                st.markdown(f"**{model_name.replace('_', ' ').title()}**")
                st.metric("Accuracy", f"{row['accuracy']:.2%}")
                st.metric("F1 Score", f"{row['f1_score']:.2%}")
                st.metric("Precision", f"{row['precision']:.2%}")
                st.metric("Recall", f"{row['recall']:.2%}")

        st.markdown("---")
        with st.expander("Raw comparison table"):
            st.dataframe(
                df.style.format("{:.4f}").highlight_max(axis=0, color="#2a6e3f"),
                use_container_width=True,
            )
    else:
        st.info("No model comparison report found. Run the evaluation pipeline first.")

    # ---- Plots ----
    st.markdown("---")
    st.subheader("Visualizations")

    tab_roc, tab_pr, tab_feat, tab_shap, tab_cm = st.tabs(
        ["ROC Curve", "Precision-Recall", "Feature Importance", "SHAP Analysis", "Confusion Matrix"]
    )

    with tab_roc:
        path = os.path.join(PLOTS, "roc_curve.png")
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.info("ROC curve plot not found. Run evaluation first.")

    with tab_pr:
        pr_files = [f for f in os.listdir(PLOTS) if f.startswith("pr_curve")] if os.path.isdir(PLOTS) else []
        if pr_files:
            for f in sorted(pr_files):
                model_name = f.replace("pr_curve_", "").replace(".png", "").replace("_", " ").title()
                st.markdown(f"**{model_name}**")
                st.image(os.path.join(PLOTS, f), use_container_width=True)
        else:
            st.info("Precision-recall plots not found. Run evaluation first.")

    with tab_feat:
        found_feat = False
        for fname in ["model_feature_importance.png", "feature_importance.png"]:
            path = os.path.join(PLOTS, fname)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                st.image(path, use_container_width=True)
                found_feat = True
        if not found_feat:
            st.info("Feature importance plots not found. Run the explainability pipeline first.")

    with tab_shap:
        shap_files = [
            ("shap_summary.png", "SHAP Summary Plot"),
            ("shap_bar.png", "SHAP Feature Importance (Bar)"),
        ]
        found = False
        for fname, title in shap_files:
            path = os.path.join(PLOTS, fname)
            if os.path.exists(path):
                st.markdown(f"**{title}**")
                st.image(path, use_container_width=True)
                found = True
        if not found:
            st.info("SHAP plots not found. Run the explainability pipeline first.")

    with tab_cm:
        path = os.path.join(PLOTS, "confusion_matrix.png")
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.info("Confusion matrix plot not found. Run evaluation first.")

    # ---- Report ----
    report_md = os.path.join(REPORTS, "model_report.md")
    if os.path.exists(report_md):
        st.markdown("---")
        with st.expander("Full Model Report"):
            st.markdown(open(report_md).read())
