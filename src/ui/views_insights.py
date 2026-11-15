"""
Sentinel AI - Insights & Session History View
Session analytics, sentiment distributions, interactive charts,
and CSV export functionality.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.utils.session_state import clear_session_history

def render_insights_view():
    """Renders session analytics, distribution charts, and history table."""
    st.markdown("### SESSION ANALYTICS & INFERENCE LOGS")

    history = st.session_state.get("history", [])

    if not history:
        st.markdown(
            """
            <div class="hud-panel" style="text-align:center; padding:2.5rem;">
                <div style="font-family:'JetBrains Mono'; font-size:1.1rem; color:#94A3B8; margin-bottom:0.5rem;">
                    NO INFERENCES RECORDED IN CURRENT SESSION
                </div>
                <div style="font-size:0.85rem; color:#64748B;">
                    Inference events from Text, Webcam, or Demonstration runs will be automatically aggregated here.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    df = pd.DataFrame(history)

    # Top Metric Counters
    col1, col2, col3, col4 = st.columns(4)

    total_records = len(df)
    bimodal_count = len(df[df["modality"] == "BIMODAL"]) if "modality" in df.columns else 0
    agreements = len(df[df["agreement"].str.contains("Agreement|CONGRUENT", case=False, na=False)])
    agreement_rate = (agreements / total_records * 100) if total_records > 0 else 0

    with col1:
        st.metric("Total Inferences", total_records)
    with col2:
        st.metric("Bimodal Events", bimodal_count)
    with col3:
        st.metric("Concordance Count", agreements)
    with col4:
        st.metric("Agreement Rate", f"{agreement_rate:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Distribution Charts
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown(
            """
            <div class="hud-panel">
                <div class="hud-title">TEXT SENTIMENT DISTRIBUTION</div>
            """,
            unsafe_allow_html=True
        )
        text_counts = df["text_sentiment"].value_counts().reset_index()
        text_counts.columns = ["Sentiment", "Count"]

        color_map = {"Positive": "#00E676", "Negative": "#FF3366", "Neutral": "#00F0FF"}
        pie_fig = px.pie(
            text_counts,
            names="Sentiment",
            values="Count",
            color="Sentiment",
            color_discrete_map=color_map,
            hole=0.45
        )
        pie_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=240,
            font=dict(family="JetBrains Mono", color="#F1F5F9")
        )
        st.plotly_chart(pie_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart2:
        st.markdown(
            """
            <div class="hud-panel">
                <div class="hud-title">CROSS-MODAL CONCORDANCE RATIO</div>
            """,
            unsafe_allow_html=True
        )
        agree_counts = df["agreement"].value_counts().reset_index()
        agree_counts.columns = ["Status", "Count"]

        bar_fig = px.bar(
            agree_counts,
            x="Count",
            y="Status",
            orientation="h",
            color="Status",
            color_discrete_sequence=["#00F0FF", "#00E676", "#FFB300", "#FF3366"]
        )
        bar_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=240,
            showlegend=False,
            xaxis=dict(showgrid=False, tickfont=dict(family="JetBrains Mono", color="#94A3B8")),
            yaxis=dict(tickfont=dict(family="JetBrains Mono", color="#94A3B8"))
        )
        st.plotly_chart(bar_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Detailed Tabular History
    st.markdown("#### SESSION RECORD LOG")
    st.dataframe(
        df[["timestamp", "text", "text_sentiment", "text_conf", "face_sentiment", "face_conf", "agreement", "modality"]],
        use_container_width=True,
        hide_index=True
    )

    # Actions: Download CSV & Clear History
    col_dl, col_clr = st.columns([4, 1])
    with col_dl:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="DOWNLOAD SESSION CSV LOG",
            data=csv_data,
            file_name="sentinel_session_history.csv",
            mime="text/csv"
        )
    with col_clr:
        if st.button("Clear Session Log", use_container_width=True):
            clear_session_history()
            st.rerun()
