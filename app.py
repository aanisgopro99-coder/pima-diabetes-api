from __future__ import annotations

import html
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "https://pima-diabetes-api-1.onrender.com").rstrip("/")
HEALTH_URL = f"{API_BASE_URL}/health"
PREDICT_URL = f"{API_BASE_URL}/predict"
ROOT = Path(__file__).parent

PALETTE = {
    "forest": "#102219",
    "forest_mid": "#1B3427",
    "sage": "#91A88D",
    "sage_light": "#B9C9B2",
    "walnut": "#B58B63",
    "cream": "#F4F0E6",
    "muted": "#A5B2A5",
    "line": "rgba(218, 228, 214, .13)",
    "positive": "#D58B79",
    "negative": "#9EBB91",
}

FEATURES = [
    ("Pregnancies", "Pregnancies", "count", "Number of times pregnant."),
    ("Glucose", "Glucose", "mg/dL", "Plasma glucose concentration from the glucose tolerance test."),
    ("BloodPressure", "Diastolic blood pressure", "mmHg", "Diastolic blood pressure measurement."),
    ("SkinThickness", "Triceps skin thickness", "mm", "Triceps skinfold thickness."),
    ("Insulin", "Two-hour serum insulin", "µU/mL", "Two-hour serum insulin measurement."),
    ("BMI", "Body mass index", "kg/m²", "Calculated by the API from weight and height."),
    ("DiabetesPedigreeFunction", "Diabetes pedigree function", "score", "Dataset-specific score related to family history."),
    ("Age", "Age", "years", "Age in years."),
]

FEATURE_LABELS = {key: label for key, label, _, _ in FEATURES}
FEATURE_UNITS = {key: unit for key, _, unit, _ in FEATURES}

st.set_page_config(
    page_title="Pima | Health Intelligence",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    css_path = ROOT / "frontend" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def authenticate() -> None:
    expected = os.getenv("APP_PASSWORD", "")
    if not expected:
        st.error("Access protection is not configured. Set APP_PASSWORD before launching the app.")
        st.stop()
    if st.session_state.get("authenticated"):
        return

    st.markdown(
        """
        <div class="login-shell">
          <div class="eyebrow">PRIVATE PROJECT ACCESS</div>
          <div class="login-brand">Pima<span>.</span></div>
          <h1 class="login-heading">Health intelligence,<br><em>made clearer.</em></h1>
          <p class="login-copy">A private machine-learning project workspace for diabetes risk assessment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("login_form"):
        password = st.text_input("Access password", type="password", placeholder="Enter your access password")
        submitted = st.form_submit_button("Enter workspace", width="stretch")
    if submitted:
        if password == expected:
            st.session_state.authenticated = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


def init_state() -> None:
    defaults = {
        "authenticated": False,
        "nav_page": "Dashboard",
        "api_ready": False,
        "last_health": None,
        "last_prediction": None,
        "last_input": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame | None:
    path = ROOT / "Data" / "diabetes_clean.csv"
    if not path.exists():
        path = ROOT / "Data" / "diabetes.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path)
    if "Outcome" in data.columns:
        data["Outcome label"] = data["Outcome"].map({0: "No diabetes recorded", 1: "Diabetes recorded"})
    return data


def bmi_category(value: float) -> str:
    if value < 18.5:
        return "Below standard adult range"
    if value < 25:
        return "Standard adult range"
    if value < 30:
        return "Above standard adult range"
    return "Obesity range"


def compute_bmi(weight: float, height: float) -> float:
    return weight / ((height / 100) ** 2)


def request_health(timeout: int = 90) -> tuple[int, dict, float]:
    started = time.perf_counter()
    response = requests.get(HEALTH_URL, timeout=timeout)
    elapsed = time.perf_counter() - started
    try:
        body = response.json()
    except ValueError:
        body = {"raw": response.text[:1000]}
    return response.status_code, body, elapsed


def make_payload(values: dict) -> dict:
    return {
        "Pregnancies": int(values["pregnancies"]),
        "Glucose": float(values["glucose"]),
        "BloodPressure": float(values["blood_pressure"]),
        "SkinThickness": float(values["skin_thickness"]),
        "Insulin": float(values["insulin"]),
        "Age": int(values["age"]),
        "Waight": float(values["weight"]),
        "Hight": float(values["height"]),
        "DiabetesPedigreeFunction": float(values["pedigree"]),
    }


def style_figure(fig, height: int = 330):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans, sans-serif", "color": PALETTE["cream"], "size": 12},
        margin={"l": 12, "r": 12, "t": 28, "b": 12},
        height=height,
        transition={"duration": 0},
        hoverlabel={"bgcolor": "#1B3427", "bordercolor": "#536B50", "font": {"color": "#F4F0E6"}},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": PALETTE["muted"]}},
    )
    fig.update_xaxes(gridcolor="rgba(218,228,214,.09)", zerolinecolor="rgba(218,228,214,.12)")
    fig.update_xaxes(
        gridcolor="rgba(218,228,214,.09)",
        zerolinecolor="rgba(218,228,214,.12)",
        tickfont={"color": PALETTE["muted"], "size": 10},
        title_font={"color": PALETTE["sage_light"], "size": 11},
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor="rgba(218,228,214,.09)",
        zerolinecolor="rgba(218,228,214,.12)",
        tickfont={"color": PALETTE["muted"], "size": 10},
        title_font={"color": PALETTE["sage_light"], "size": 11},
        automargin=True,
    )
    return fig


@st.fragment(run_every="1s")
def render_live_clock() -> None:
    now = datetime.now().astimezone()
    date_text = html.escape(now.strftime("%A, %d %B %Y"))
    zone_text = html.escape(now.tzname() or "LOCAL")
    st.markdown(
        f'<div class="live-clock-shell"><div class="live-clock">'
        f'<span class="clock-dot"></span><div class="clock-time-group">'
        f'<span class="clock-label">LOCAL TIME</span><strong>{now.strftime("%H:%M:%S")}</strong></div>'
        f'<span class="clock-divider"></span><div class="clock-date-group">'
        f'<span class="clock-date">{date_text}</span><span class="clock-zone">{zone_text}</span>'
        f'</div></div></div>',
        unsafe_allow_html=True,
    )


CURSOR_SCRIPT = """
<script>
(() => {
  if (window.__pimaCursorAuraReady) return;
  window.__pimaCursorAuraReady = true;
  const aura = document.createElement("div");
  aura.id = "pima-cursor-aura";
  aura.setAttribute("aria-hidden", "true");
  document.body.appendChild(aura);
  let frame = 0;
  let pointerX = -500;
  let pointerY = -500;
  window.addEventListener("pointermove", (event) => {
    if (event.pointerType !== "mouse") return;
    pointerX = event.clientX;
    pointerY = event.clientY;
    if (frame) return;
    frame = window.requestAnimationFrame(() => {
      aura.style.transform = "translate3d(" + (pointerX - 180) + "px, " + (pointerY - 180) + "px, 0)";
      aura.classList.add("is-visible");
      frame = 0;
    });
  }, { passive: true });
  window.addEventListener("blur", () => aura.classList.remove("is-visible"));
  document.addEventListener("mouseleave", () => aura.classList.remove("is-visible"));
})();
</script>
"""


def render_cursor_effect() -> None:
    st.html(CURSOR_SCRIPT, unsafe_allow_javascript=True)


def render_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{html.escape(kicker)}</div>'
        f'<h1 class="page-title">{html.escape(title)}</h1>'
        f'<p class="page-subtitle">{html.escape(subtitle)}</p>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str, accent: str = "sage") -> None:
    st.markdown(
        f'<div class="metric-card accent-{accent}"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div><div class="metric-note">{html.escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero">
          <div class="hero-topline"><span class="eyebrow">PIMA INDIANS DIABETES · MACHINE LEARNING</span><span class="live-tag"><i></i> HEALTH INTELLIGENCE</span></div>
          <h1>See the signals.<br><em>Understand the model.</em></h1>
          <p class="hero-copy">A focused workspace connecting clinical measurements, a trained classifier and a live prediction API.</p>
          <svg class="ecg" viewBox="0 0 1200 78" preserveAspectRatio="none" aria-hidden="true">
            <path class="ecg-base" d="M0 39 H160 L180 35 L198 39 H340 L360 43 L378 8 L398 70 L418 37 L438 39 H590 L610 34 L628 39 H780 L800 43 L818 8 L838 70 L858 37 L878 39 H1030 L1050 34 L1068 39 H1200"/>
            <path class="ecg-live" pathLength="1000" d="M0 39 H160 L180 35 L198 39 H340 L360 43 L378 8 L398 70 L418 37 L438 39 H590 L610 34 L628 39 H780 L800 43 L818 8 L838 70 L858 37 L878 39 H1030 L1050 34 L1068 39 H1200"/>
          </svg>
        </section>
        """,
        unsafe_allow_html=True,
    )


def api_status_label() -> tuple[str, str]:
    health = st.session_state.get("last_health")
    if not health:
        return "NOT CHECKED", "idle"
    if health.get("http_status") == 200:
        return "ONLINE", "good"
    return "UNREACHABLE", "bad"


def run_health_check(key: str) -> None:
    if st.button("Wake / check API", key=key, width="stretch"):
        with st.spinner("Contacting the cloud API. Render may need time to wake the service..."):
            try:
                code, body, elapsed = request_health()
                st.session_state.last_health = {
                    "http_status": code,
                    "response": body,
                    "seconds": round(elapsed, 3),
                    "checked_at": datetime.now().isoformat(timespec="seconds"),
                }
                st.session_state.api_ready = code == 200
                if code == 200:
                    st.success(f"API responded in {elapsed:.1f} seconds.")
                else:
                    st.error(f"Health check returned HTTP {code}.")
            except requests.RequestException as exc:
                st.session_state.api_ready = False
                st.session_state.last_health = {"error": str(exc)}
                st.error(f"Could not reach the API: {exc}")


def render_prediction_result(data: pd.DataFrame | None) -> None:
    result = st.session_state.get("last_prediction")
    if not result:
        return

    st.markdown('<div class="section-heading"><span>02</span> Latest prediction</div>', unsafe_allow_html=True)
    if "error" in result:
        st.markdown(
            f'<div class="result-panel result-error"><div class="result-kicker">REQUEST FAILED</div>'
            f'<h2>Could not complete prediction</h2><p>{html.escape(result["error"])}</p></div>',
            unsafe_allow_html=True,
        )
        return

    body = result.get("response") or {}
    if result.get("http_status") != 200 or "prediction" not in body:
        if result.get("http_status") == 422:
            st.error("The API rejected the submitted measurements. Please review these fields:")
            details = body.get("detail", [])
            if isinstance(details, list):
                for item in details:
                    field = ".".join(str(part) for part in item.get("loc", []) if part != "body")
                    message = item.get("msg", "Invalid value")
                    st.write(f"- **{field or 'Request'}:** {message}")
            else:
                st.write(details)
        else:
            st.error(f"API returned HTTP {result.get('http_status', '—')}.")
            st.json(body)
        return

    bmi = body.get("data_ready", {}).get("BMI")
    probability = body.get("model_probability")
    api_threshold = body.get("decision_threshold")
    if probability is None or api_threshold is None:
        st.warning("The connected API must be redeployed with the probability and decision-threshold response fields before the adjustable result can be used.")
        return
    probability = max(0.0, min(float(probability), 1.0))
    api_threshold = max(0.0, min(float(api_threshold), 1.0))
    selected_threshold = st.slider(
        "Decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=api_threshold,
        step=0.01,
        format="%.2f",
        key="decision_threshold_control",
        help="Move the threshold to change how the same model probability is classified. This does not rerun or retrain the model.",
    )
    prediction = int(probability >= selected_threshold)
    percentage = probability * 100
    threshold_percentage = selected_threshold * 100
    positive = prediction == 1
    tone = "result-alert" if positive else "result-calm"
    title = "Has diabetes" if positive else "Does not have diabetes"
    copy = (
        f"The model probability ({percentage:.1f}%) is at or above the selected threshold ({threshold_percentage:.0f}%). This is a model classification, not a medical diagnosis."
        if positive
        else f"The model probability ({percentage:.1f}%) is below the selected threshold ({threshold_percentage:.0f}%). This result does not rule out diabetes."
    )
    st.markdown(
        f'<div class="result-panel {tone}"><div class="result-kicker">MODEL-ESTIMATED PROBABILITY</div>'
        f'<h2>{title}</h2><div class="probability-value">{percentage:.1f}%</div>'
        f'<div class="probability-track"><span style="width:{percentage:.1f}%"></span></div>'
        f'<p>{copy}</p><div class="result-footer">'
        f'<span>Probability {percentage:.1f}%</span>'
        f'<span>Threshold {threshold_percentage:.0f}%</span>'
        f'<span>{f"BMI {float(bmi):.1f}" if bmi is not None else "BMI unavailable"}</span>'
        f'<span>Not a diagnosis</span></div></div>',
        unsafe_allow_html=True,
    )
    st.caption("Important: this percentage is the model's predict_proba score, not a clinically validated personal risk. Its calibration and accuracy for an individual have not been established. Do not use it to diagnose, rule out diabetes or make treatment decisions.")

    if bmi is not None:
        st.markdown(
            f'<div class="bmi-strip"><div><span class="metric-label">BMI CALCULATED BY THE API</span>'
            f'<div class="bmi-number">{float(bmi):.1f} <small>kg/m²</small></div></div>'
            f'<div class="bmi-category">{bmi_category(float(bmi))}</div></div>',
            unsafe_allow_html=True,
        )
    st.caption("BMI is one of the model's inputs. It does not determine the prediction by itself.")

    ready = body.get("data_ready")
    if data is not None and isinstance(ready, dict):
        columns = [item[0] for item in FEATURES]
        percentiles = []
        labels = []
        for col in columns:
            if col in ready and col in data.columns:
                percentiles.append(float((data[col] <= float(ready[col])).mean() * 100))
                labels.append(FEATURE_LABELS.get(col, col))
        if labels:
            fig = go.Figure(
                go.Bar(
                    x=percentiles,
                    y=labels,
                    orientation="h",
                    marker={"color": PALETTE["walnut"], "line": {"width": 0}},
                    text=[f"{v:.0f}th percentile" for v in percentiles],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Dataset percentile: %{x:.1f}%<extra></extra>",
                )
            )
            fig.update_xaxes(range=[0, 112], title="Percentile in the project dataset", ticksuffix="th")
            fig.update_yaxes(autorange="reversed")
            st.markdown('<div class="chart-heading">Input profile against the dataset</div>', unsafe_allow_html=True)
            st.plotly_chart(style_figure(fig, 340), width="stretch", config={"displayModeBar": False, "responsive": True})
            st.caption("Descriptive comparison only. These percentiles are not risk scores, feature contributions or clinical reference ranges.")

    with st.expander("Technical request and response"):
        st.markdown("**POST /predict — request JSON**")
        st.json(result.get("request", {}))
        st.markdown("**API response JSON**")
        st.json(body)


def page_dashboard(data: pd.DataFrame | None) -> None:
    render_hero()
    if data is None:
        st.warning("The project dataset was not found. Dataset charts are unavailable in this deployment.")
        records, positive_rate = 0, 0
    else:
        records = len(data)
        positive_rate = float(data["Outcome"].mean() * 100) if "Outcome" in data else 0

    api_text, api_tone = api_status_label()
    last = st.session_state.get("last_prediction") or {}
    last_time = f'{last.get("duration", 0):.2f} s' if "duration" in last else "—"
    prediction_text = "—"
    if isinstance(last.get("response"), dict) and "prediction" in last["response"]:
        prediction_text = "High model estimate" if int(last["response"]["prediction"]) == 1 else "Low model estimate"

    st.markdown('<div class="section-heading"><span>LIVE OVERVIEW</span> Workspace pulse</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        metric_card("DATASET RECORDS", f"{records:,}" if records else "—", "Clean project dataset", "sage")
    with k2:
        metric_card("MODEL FEATURES", "08", "BMI is calculated from 2 extra inputs", "walnut")
    with k3:
        metric_card("DIABETES RECORDED", f"{positive_rate:.1f}%" if records else "—", "Dataset share · not population prevalence", "rose")
    with k4:
        metric_card("API CONNECTION", api_text, "Last health check in this session", "sage" if api_tone == "good" else "walnut")

    st.markdown('<div class="section-heading"><span>01</span> Dataset landscape</div>', unsafe_allow_html=True)
    if data is not None:
        left, right = st.columns([0.85, 1.5], gap="large")
        with left:
            counts = data["Outcome"].value_counts().sort_index()
            fig = go.Figure(
                go.Pie(
                    labels=["No diabetes recorded" if int(x) == 0 else "Diabetes recorded" for x in counts.index],
                    values=counts.values,
                    hole=0.72,
                    marker={"colors": [PALETTE["negative"], PALETTE["positive"]], "line": {"color": PALETTE["forest"], "width": 4}},
                    textinfo="percent",
                    hovertemplate="<b>%{label}</b><br>%{value} records<br>%{percent} of dataset<extra></extra>",
                )
            )
            fig.update_layout(showlegend=False, annotations=[{"text": f"{records}<br><span style='font-size:11px'>records</span>", "showarrow": False, "font": {"size": 23, "color": PALETTE["cream"]}}])
            st.markdown('<div class="chart-card"><div class="chart-heading">Class balance</div><div class="chart-subtitle">Distribution in the cleaned project dataset</div>', unsafe_allow_html=True)
            st.plotly_chart(style_figure(fig, 310), width="stretch", config={"displayModeBar": False, "responsive": True})
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            color_map = {"No diabetes recorded": PALETTE["negative"], "Diabetes recorded": PALETTE["sage_light"]}
            fig = px.scatter(
                data,
                x="Glucose",
                y="BMI",
                color="Outcome label",
                color_discrete_map=color_map,
                hover_data=["Age", "Pregnancies"],
                opacity=0.68,
                labels={"Outcome label": "Dataset group", "Glucose": "Glucose (mg/dL)", "BMI": "Body mass index (kg/m²)"},
            )
            fig.update_traces(marker={"size": 8, "line": {"width": 0}}, hovertemplate="<b>%{fullData.name}</b><br>Glucose: %{x:.0f} mg/dL<br>Body mass index: %{y:.1f} kg/m²<extra></extra>")
            st.markdown('<div class="chart-card"><div class="chart-heading">Glucose vs. body mass index</div><div class="chart-subtitle">Each point is one dataset record. Hover to inspect its measurements.</div>', unsafe_allow_html=True)
            st.plotly_chart(style_figure(fig, 310), width="stretch", config={"displayModeBar": False, "responsive": True})
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-heading"><span>02</span> Explore a feature</div>', unsafe_allow_html=True)
        feature = st.selectbox(
            "Choose a measurement",
            ["Glucose", "BMI", "Age", "Insulin", "BloodPressure", "Pregnancies", "SkinThickness", "DiabetesPedigreeFunction"],
            key="dashboard_feature",
            format_func=lambda value: FEATURE_LABELS.get(value, value),
            label_visibility="collapsed",
        )
        fig = px.histogram(
            data,
            x=feature,
            color="Outcome label",
            color_discrete_map=color_map,
            nbins=32,
            barmode="overlay",
            opacity=0.72,
            labels={"Outcome label": "Dataset group", feature: f"{FEATURE_LABELS.get(feature, feature)} ({FEATURE_UNITS.get(feature, '')})"},
        )
        feature_label = FEATURE_LABELS.get(feature, feature)
        feature_unit = FEATURE_UNITS.get(feature, "")
        axis_label = f"{feature_label} ({feature_unit})" if feature_unit else feature_label
        fig.update_layout(
            xaxis_title=axis_label,
            yaxis_title="Number of records",
            legend_title_text="Dataset group",
            bargap=0.04,
        )
        fig.update_traces(hovertemplate="Value: %{x}<br>Records: %{y}<extra>%{fullData.name}</extra>")
        st.plotly_chart(style_figure(fig, 300), width="stretch", config={"displayModeBar": False, "responsive": True})

    st.markdown('<div class="section-heading"><span>03</span> Recent activity</div>', unsafe_allow_html=True)
    activity_left, activity_right = st.columns([1.2, 1], gap="large")
    with activity_left:
        if prediction_text == "—":
            st.markdown('<div class="activity-card"><div class="activity-icon"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12h4l3-8 4 16 3-8h4"/></svg></div><div><strong>No prediction in this session yet</strong><p>Open the assessment workspace to submit a test profile.</p></div></div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="activity-card"><div class="activity-icon"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M7 7h10v10"/></svg></div><div><strong>Latest output · {prediction_text}</strong>'
                f'<p>API response time: {last_time}. Session-only activity; no patient history is stored.</p></div></div>',
                unsafe_allow_html=True,
            )
    with activity_right:
        st.markdown('<div class="activity-card"><div class="activity-icon"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8M12 17v4"/></svg></div><div><strong>API monitor</strong><p>Health endpoint: GET /health. Prediction endpoint: POST /predict.</p></div></div>', unsafe_allow_html=True)
        run_health_check("dashboard_health_check")
    if st.button("Open assessment workspace", type="primary", width="stretch"):
        st.session_state.nav_page = "Assessment"
        st.rerun()


def page_assessment(data: pd.DataFrame | None) -> None:
    render_header("PREDICTION WORKSPACE", "Diabetes risk assessment", "Enter the measurements, review the live BMI calculation and submit a request to the deployed model.")
    st.markdown(
        '<div class="notice"><strong>Educational demonstration</strong><span>This interface is not a medical device. Model output is not a diagnosis and must not guide treatment decisions.</span></div>',
        unsafe_allow_html=True,
    )
    status, tone = api_status_label()
    status_col, action_col = st.columns([2.5, 1])
    with status_col:
        st.markdown(f'<div class="status-line"><span class="status-dot {tone}"></span> API status · <strong>{status}</strong></div>', unsafe_allow_html=True)
    with action_col:
        run_health_check("assessment_health_check")

    st.markdown('<div class="section-heading"><span>01</span> Patient measurements</div>', unsafe_allow_html=True)
    st.caption("Nine request fields are collected. Weight and height are sent to the API, which calculates BMI for the eight-feature model input.")
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        pregnancies = st.number_input("Pregnancies · count", min_value=0, max_value=20, value=0, step=1, help="Number of times pregnant.", key="in_pregnancies")
        glucose = st.number_input("Glucose · mg/dL", min_value=0.0, max_value=400.0, value=120.0, step=1.0, help="Plasma glucose concentration.", key="in_glucose")
        blood_pressure = st.number_input("Diastolic blood pressure · mmHg", min_value=1.0, max_value=200.0, value=70.0, step=1.0, help="The API requires a value greater than zero.", key="in_blood_pressure")
    with c2:
        skin_thickness = st.number_input("Skin thickness · mm", min_value=0.0, max_value=100.0, value=20.0, step=1.0, help="Triceps skinfold thickness.", key="in_skin_thickness")
        insulin = st.number_input("Insulin · µU/mL", min_value=0.0, max_value=1000.0, value=80.0, step=1.0, help="Two-hour serum insulin.", key="in_insulin")
        age = st.number_input("Age · years", min_value=1, max_value=119, value=30, step=1, help="Age in years.", key="in_age")
    with c3:
        weight = st.number_input("Weight · kg", min_value=30.1, max_value=249.9, value=70.0, step=0.5, help="API request field: Waight.", key="in_weight")
        height = st.number_input("Height · cm", min_value=60.1, max_value=249.9, value=170.0, step=0.5, help="API request field: Hight.", key="in_height")
        pedigree = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.5, step=0.01, help="Family-history-related dataset score.", key="in_pedigree")

    bmi = compute_bmi(weight, height)
    if bmi < 12 or bmi > 60:
        st.warning("Please double-check the entered height and weight units. The resulting BMI is unusual for this input check; this is not a medical assessment.")
    st.markdown(
        f'<div class="bmi-strip"><div><span class="metric-label">LIVE BMI CALCULATION</span>'
        f'<div class="bmi-number">{bmi:.1f} <small>kg/m²</small></div></div>'
        f'<div class="bmi-category">{bmi_category(bmi)}</div></div>',
        unsafe_allow_html=True,
    )

    values = {
        "pregnancies": pregnancies,
        "glucose": glucose,
        "blood_pressure": blood_pressure,
        "skin_thickness": skin_thickness,
        "insulin": insulin,
        "age": age,
        "weight": weight,
        "height": height,
        "pedigree": pedigree,
    }
    result_slot = st.empty()
    if st.button("Run prediction  →", type="primary", width="stretch"):
        st.session_state.last_prediction = None
        st.session_state.last_input = None
        result_slot.markdown(
            '<div class="result-panel result-calm result-loading">'
            '<div class="result-loader" aria-label="Prediction in progress"></div>'
            '<div class="result-kicker">ASSESSMENT IN PROGRESS</div>'
            '<h2>Analysing the measurements</h2>'
            '<p>Contacting the prediction service. This may take a moment if the cloud service is waking up.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        payload = make_payload(values)
        st.session_state.last_input = values.copy()
        started = time.perf_counter()
        try:
            response = requests.post(PREDICT_URL, json=payload, timeout=90)
            st.session_state.api_ready = response.status_code == 200
            duration = time.perf_counter() - started
            try:
                body = response.json()
            except ValueError:
                body = {"raw": response.text[:2000]}
            st.session_state.last_prediction = {
                "http_status": response.status_code,
                "duration": duration,
                "request": payload,
                "response": body,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        except requests.RequestException as exc:
            st.session_state.last_prediction = {"error": str(exc), "request": payload}
        st.rerun()

    if not st.session_state.api_ready:
        st.info("You can submit directly. The first request may take longer while the cloud API wakes up.")
    render_prediction_result(data)


def page_api_monitor() -> None:
    render_header("SERVICE MONITORING", "API connection", "Inspect the deployed FastAPI service, health-check response and latest prediction request.")
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown(
            f'<div class="endpoint-card"><div class="metric-label">HEALTH ENDPOINT</div>'
            f'<code>{html.escape(HEALTH_URL)}</code><span class="endpoint-method">GET · SERVICE HEALTH</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="endpoint-card"><div class="metric-label">PREDICTION ENDPOINT</div>'
            f'<code>{html.escape(PREDICT_URL)}</code><span class="endpoint-method">POST · JSON INPUT</span></div>',
            unsafe_allow_html=True,
        )
        run_health_check("monitor_health_check")
    with right:
        health = st.session_state.get("last_health") or {}
        code = str(health.get("http_status", "—"))
        seconds = f'{health.get("seconds", 0):.2f} s' if "seconds" in health else "—"
        label, _ = api_status_label()
        st.markdown('<div class="monitor-card">', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            metric_card("SERVICE STATE", label, "Last checked status", "sage" if label == "ONLINE" else "walnut")
        with m2:
            metric_card("HTTP STATUS", code, "Health endpoint response", "sage" if code == "200" else "walnut")
        metric_card("HEALTH CHECK TIME", seconds, "Measured from this browser session", "walnut")
        st.markdown('</div>', unsafe_allow_html=True)
    if health:
        with st.expander("Latest /health response"):
            st.json(health)
    last = st.session_state.get("last_prediction")
    if last:
        st.markdown('<div class="section-heading"><span>01</span> Latest prediction request</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.metric("HTTP response", str(last.get("http_status", "Error")))
        b.metric("Request duration", f'{last.get("duration", 0):.2f} s' if "duration" in last else "—")
        b.metric("Timestamp", last.get("created_at", "—"))
        with st.expander("Request and response payloads"):
            st.markdown("**Request JSON**")
            st.json(last.get("request", {}))
            st.markdown("**Response JSON**")
            st.json(last.get("response", {}))
    st.markdown('<div class="notice"><strong>What the health check does</strong><span>GET /health checks whether the API is reachable. It does not submit patient information or execute a prediction. The first request may take longer if the Render service was sleeping.</span></div>', unsafe_allow_html=True)


def page_model_insights(data: pd.DataFrame | None) -> None:
    render_header("MODEL DOCUMENTATION", "What the model evaluates", "Explore the eight model features and the dataset distributions behind this educational classifier.")
    st.markdown(
        '<div class="notice"><strong>Input pipeline</strong><span>9 API request fields → BMI calculated from weight and height → 8 model features → model probability estimate.</span></div>',
        unsafe_allow_html=True,
    )
    for start in range(0, len(FEATURES), 2):
        left, right = st.columns(2, gap="medium")
        for col, item in zip((left, right), FEATURES[start:start + 2]):
            key, label, unit, description = item
            with col:
                st.markdown(
                    f'<div class="feature-card"><div class="feature-top"><span>{html.escape(key)}</span><small>{html.escape(unit)}</small></div>'
                    f'<h3>{html.escape(label)}</h3><p>{html.escape(description)}</p></div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="section-heading"><span>01</span> Explore feature distributions</div>', unsafe_allow_html=True)
    if data is not None:
        feature = st.selectbox("Feature", [item[0] for item in FEATURES], key="model_feature", format_func=lambda value: FEATURE_LABELS.get(value, value))
        fig = px.box(
            data,
            x="Outcome label",
            y=feature,
            color="Outcome label",
            color_discrete_map={"No diabetes recorded": PALETTE["negative"], "Diabetes recorded": PALETTE["sage_light"]},
            points="outliers",
            labels={"Outcome label": "Dataset group", feature: f"{FEATURE_LABELS.get(feature, feature)} ({FEATURE_UNITS.get(feature, '')})"},
        )
        fig.update_layout(legend_title_text="Dataset group")
        fig.update_traces(hovertemplate="<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>")
        st.plotly_chart(style_figure(fig, 350), width="stretch", config={"displayModeBar": False, "responsive": True})
        st.caption("These plots describe the project dataset. They do not establish causation or explain an individual model prediction.")
    st.markdown('<div class="section-heading"><span>02</span> Decision threshold</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="feature-card"><h3>How the probability estimate is used</h3>'
        '<p>The API now returns the model probability score for the diabetes-labelled class. The saved decision threshold uses that score to select the higher or lower group. '
        'This percentage is not clinically calibrated or validated as an individual medical risk estimate.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-heading"><span>03</span> Request field mapping</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame([
            {"Form input": "Pregnancies", "API key": "Pregnancies", "Purpose": "Model feature"},
            {"Form input": "Glucose", "API key": "Glucose", "Purpose": "Model feature"},
            {"Form input": "Diastolic blood pressure", "API key": "BloodPressure", "Purpose": "Model feature"},
            {"Form input": "Skin thickness", "API key": "SkinThickness", "Purpose": "Model feature"},
            {"Form input": "Insulin", "API key": "Insulin", "Purpose": "Model feature"},
            {"Form input": "Age", "API key": "Age", "Purpose": "Model feature"},
            {"Form input": "Weight", "API key": "Waight", "Purpose": "BMI calculation input"},
            {"Form input": "Height", "API key": "Hight", "Purpose": "BMI calculation input"},
            {"Form input": "Diabetes pedigree function", "API key": "DiabetesPedigreeFunction", "Purpose": "Model feature"},
        ]),
        hide_index=True,
        width="stretch",
    )


def page_project(data: pd.DataFrame | None) -> None:
    render_header("PROJECT PRESENTATION", "From measurements to prediction", "A clear walkthrough of the dataset, model pipeline and deployed application architecture.")
    st.markdown(
        '<div class="overview-hero"><div class="eyebrow">MACHINE LEARNING · FASTAPI · STREAMLIT</div>'
        '<h2>Turning structured measurements into a model output.</h2>'
        '<p>A private web interface connected to a separately deployed prediction API.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-heading"><span>01</span> Project snapshot</div>', unsafe_allow_html=True)
    if data is not None:
        a, b, c = st.columns(3)
        a.metric("Dataset records", f"{len(data):,}")
        b.metric("Model features", "8")
        c.metric("Request fields", "9")
        st.write('The cleaned project dataset contains 768 records: 500 with "No diabetes recorded" and 268 with "Diabetes recorded". These are dataset labels, not clinical assessments.')
    st.markdown('<div class="section-heading"><span>02</span> System architecture</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="architecture"><div class="arch-node"><small>01 · USER</small><strong>Assessment form</strong><span>9 request fields</span></div>'
        '<div class="arch-arrow">↓</div><div class="arch-node"><small>02 · FRONTEND</small><strong>Streamlit interface</strong><span>Validation · BMI preview · API request</span></div>'
        '<div class="arch-arrow">↓</div><div class="arch-node arch-highlight"><small>03 · API</small><strong>FastAPI · POST /predict</strong><span>Input validation · BMI calculation · feature row</span></div>'
        '<div class="arch-arrow">↓</div><div class="arch-node"><small>04 · MODEL</small><strong>Saved classifier</strong><span>8 features · saved decision threshold</span></div>'
        '<div class="arch-arrow">↓</div><div class="arch-node arch-output"><small>05 · RESPONSE</small><strong>Probability estimate</strong><span>Readable model estimate returned to the frontend</span></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-heading"><span>03</span> Technology stack</div>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.columns(4)
    for col, name, detail in [
        (t1, "Python", "Application language"),
        (t2, "Scikit-learn", "Saved ML pipeline"),
        (t3, "FastAPI", "Prediction service"),
        (t4, "Streamlit", "Interactive frontend"),
    ]:
        with col:
            st.markdown(f'<div class="tech-card"><h3>{name}</h3><p>{detail}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading"><span>04</span> Responsible use & limitations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="notice"><strong>Educational project only</strong><span>The dataset is limited and may not represent every population. Predictions depend on the training data, preprocessing and saved decision threshold. '
        'A class result must not be used to diagnose, treat or rule out diabetes. Consult a qualified healthcare professional for medical decisions.</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("The frontend password restricts access to this interface. It does not, by itself, secure the separately deployed public API.")


def main() -> None:
    init_state()
    load_css()
    authenticate()
    render_cursor_effect()

    with st.sidebar:
        st.markdown('<div class="brand-mark">P<span>.</span></div><div class="brand-name">PIMA <small>HEALTH INTELLIGENCE</small></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-caption">WORKSPACE</div>', unsafe_allow_html=True)
        st.radio(
            "Navigate",
            ["Dashboard", "Assessment", "API monitor", "Model insights", "Project"],
            key="nav_page",
            label_visibility="collapsed",
        )
        st.markdown("---")
        state, tone = api_status_label()
        st.markdown(f'<div class="sidebar-status"><span class="status-dot {tone}"></span><span>API · {state}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-url">{html.escape(API_BASE_URL)}</div>', unsafe_allow_html=True)
        if st.button("Sign out", width="stretch"):
            st.session_state.authenticated = False
            st.rerun()

    render_live_clock()
    data = load_dataset()
    page = st.session_state.nav_page
    if page == "Dashboard":
        page_dashboard(data)
    elif page == "Assessment":
        page_assessment(data)
    elif page == "API monitor":
        page_api_monitor()
    elif page == "Model insights":
        page_model_insights(data)
    else:
        page_project(data)

    st.markdown('<div class="app-footer">PIMA HEALTH INTELLIGENCE <span>·</span> EDUCATIONAL MACHINE-LEARNING PROJECT <span>·</span> NOT A MEDICAL DEVICE</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
