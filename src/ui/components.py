"""
Sentinel AI - Interactive UI Visualizations & Charts
Plotly charts, confidence gauges, and telemetry status widgets.
"""

import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Any, List

def create_confidence_gauge(confidence: float, title: str = "CONFIDENCE", color: str = "#00F0FF") -> go.Figure:
    """Creates a sleek cyber command-centre semi-circular gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={"suffix": "%", "font": {"size": 26, "color": "#F1F5F9", "family": "JetBrains Mono"}},
        title={"text": title, "font": {"size": 13, "color": "#94A3B8", "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748B", "tickwidth": 1, "nticks": 5},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#0E131F",
            "borderwidth": 1,
            "bordercolor": "#212C45",
            "steps": [
                {"range": [0, 50], "color": "rgba(255, 51, 102, 0.15)"},
                {"range": [50, 75], "color": "rgba(255, 179, 0, 0.15)"},
                {"range": [75, 100], "color": "rgba(0, 240, 255, 0.15)"}
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=30, b=15),
        height=170
    )
    return fig

def create_probability_bar_chart(probabilities: Dict[str, float]) -> go.Figure:
    """Creates horizontal probability breakdown bar chart."""
    classes = list(probabilities.keys())
    values = [probabilities[c] * 100 for c in classes]

    colors = []
    for c in classes:
        if c == "Positive":
            colors.append("#00E676")
        elif c == "Negative":
            colors.append("#FF3366")
        else:
            colors.append("#00F0FF")

    fig = go.Figure(go.Bar(
        x=values,
        y=classes,
        orientation="h",
        marker=dict(color=colors, line=dict(color="#212C45", width=1)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#F1F5F9", family="JetBrains Mono", size=12)
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=40, t=10, b=10),
        height=140,
        xaxis=dict(range=[0, 105], showgrid=False, visible=False),
        yaxis=dict(tickfont=dict(color="#94A3B8", family="JetBrains Mono", size=12), autorange="reversed")
    )
    return fig

def create_bimodal_radar_chart(
    text_probs: Dict[str, float],
    vision_probs: Dict[str, float]
) -> go.Figure:
    """Renders dual-modality radar comparison chart."""
    categories = ["Positive", "Neutral", "Negative"]

    t_vals = [text_probs.get(c, 0.0) * 100 for c in categories]
    v_vals = [vision_probs.get(c, 0.0) * 100 for c in categories]
    
    # Close polygon
    categories_closed = categories + [categories[0]]
    t_vals_closed = t_vals + [t_vals[0]]
    v_vals_closed = v_vals + [v_vals[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=t_vals_closed,
        theta=categories_closed,
        fill="toself",
        name="Text Sentiment",
        line=dict(color="#00F0FF", width=2),
        fillcolor="rgba(0, 240, 255, 0.2)"
    ))

    fig.add_trace(go.Scatterpolar(
        r=v_vals_closed,
        theta=categories_closed,
        fill="toself",
        name="Facial Cues (Vision)",
        line=dict(color="#00E676", width=2),
        fillcolor="rgba(0, 230, 118, 0.2)"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#64748B", tickfont=dict(size=10)),
            angularaxis=dict(color="#F1F5F9", tickfont=dict(family="JetBrains Mono", size=12)),
            bgcolor="#0E131F"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=25, r=25, t=25, b=25),
        height=280,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(family="JetBrains Mono", color="#94A3B8")
        )
    )
    return fig

def render_agreement_badge(is_agreement: bool, label_text: str):
    """Renders HUD agreement / divergence status badge."""
    if is_agreement:
        html = f"""
        <div style="background:rgba(0,230,118,0.12); border:1px solid #00E676; border-radius:6px; padding:0.6rem 1rem; color:#00E676; font-family:'JetBrains Mono'; font-weight:700; font-size:0.95rem; text-align:center;">
            ✓ {label_text}
        </div>
        """
    else:
        html = f"""
        <div style="background:rgba(255,51,102,0.12); border:1px solid #FF3366; border-radius:6px; padding:0.6rem 1rem; color:#FF3366; font-family:'JetBrains Mono'; font-weight:700; font-size:0.95rem; text-align:center;">
            ⚠ {label_text}
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)
