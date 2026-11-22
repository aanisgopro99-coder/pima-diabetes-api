import json
import os
import time
from datetime import datetime

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "https://pima-diabetes-api-1.onrender.com").rstrip("/")
HEALTH_URL = f"{API_BASE_URL}/health"
PREDICT_URL = f"{API_BASE_URL}/predict"

st.set_page_config(
    page_title="Pima | Diabetes Risk Assessment",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "frontend", "style.css")
    try:
        with open(css_path, encoding="utf-8") as file:
            st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)
    except OSError:
        pass

load_css()

def authenticate():
    expected = os.getenv("APP_PASSWORD", "")
    if not expected:
        st.error("Access protection is not configured. Set the APP_PASSWORD environment variable before launching the app.")
        st.stop()
    if st.session_state.get("authenticated"):
        return
    st.markdown('<div class="login-wrap"><div class="eyebrow">PRIVATE PROJECT ACCESS</div><h1 class="login-title">Pima<span>.</span></h1><p class="muted">A machine-learning demonstration for diabetes risk assessment.</p></div>', unsafe_allow_html=True)
    with st.form("login_form"):
        password = st.text_input("Access password", type="password")
        submitted = st.form_submit_button("Enter workspace", use_container_width=True)
    if submitted:
        if password == expected:
            st.session_state.authenticated = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()

authenticate()

FEATURES = [
    ("Pregnancies", "Pregnancies", "count", "Number of times pregnant."),
    ("Glucose", "Glucose", "mg/dL", "Plasma glucose concentration from the glucose tolerance test."),
    ("BloodPressure", "Diastolic blood pressure", "mmHg", "Diastolic blood pressure measurement."),
    ("SkinThickness", "Skin thickness", "mm", "Triceps skinfold thickness."),
    ("Insulin", "Insulin", "µU/mL", "Two-hour serum insulin measurement."),
    ("BMI", "Body mass index", "kg/m²", "Calculated from the entered weight and height."),
    ("DiabetesPedigreeFunction", "Diabetes pedigree function", "score", "A dataset-specific score related to family history."),
    ("Age", "Age", "years", "Age in years."),
]

def request_health(timeout=90):
    started = time.perf_counter()
    response = requests.get(HEALTH_URL, timeout=timeout)
    elapsed = time.perf_counter() - started
    try:
        body = response.json()
    except ValueError:
        body = {"raw": response.text[:1000]}
    return response.status_code, body, elapsed

def bmi_category(value):
    if value < 18.5:
        return "Below the standard adult range"
    if value < 25:
        return "Standard adult range"
    if value < 30:
        return "Above the standard adult range"
    return "Obesity range"
    
def page_header(kicker, title, subtitle):
    st.markdown(f'<div class="eyebrow">{kicker}</div><h1 class="page-title">{title}</h1><p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="brand-mark">P<span>.</span></div><div class="brand-name">PIMA <small>ML PROJECT</small></div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("WORKSPACE", ["Assessment", "API status", "Model & features", "Project overview"], label_visibility="visible")
    st.markdown("---")
    st.markdown('<div class="sidebar-caption">DEPLOYED SERVICE</div><div class="sidebar-url">pima-diabetes-api-1.onrender.com</div>', unsafe_allow_html=True)
    if st.button("Sign out", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

if page == "Assessment":
    page_header("PREDICTION WORKSPACE", "Diabetes risk assessment", "Enter the measurements below to submit a prediction request to the deployed machine-learning API.")
    st.markdown('<div class="notice"><strong>Demonstration only</strong><span>This tool is an educational machine-learning project, not a medical device or a diagnosis.</span></div>', unsafe_allow_html=True)

    if "api_ready" not in st.session_state:
        st.session_state.api_ready = False
    if "last_health" not in st.session_state:
        st.session_state.last_health = None
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    status_col, action_col = st.columns([3, 1])
    with status_col:
        if st.session_state.api_ready:
            st.markdown('<div class="status-line"><span class="status-dot good"></span> API marked as reachable in this session</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-line"><span class="status-dot idle"></span> API connection has not been checked yet</div>', unsafe_allow_html=True)
    with action_col:
        if st.button("Wake / check API", use_container_width=True):
            with st.spinner("Contacting the cloud service. A sleeping instance may take a little while to start…"):
                try:
                    code, body, elapsed = request_health()
                    st.session_state.last_health = {"http_status": code, "response": body, "seconds": round(elapsed, 2), "checked_at": datetime.now().isoformat(timespec="seconds")}
                    st.session_state.api_ready = code == 200
                    if code == 200:
                        st.success(f"API responded successfully in {elapsed:.1f} seconds.")
                    else:
                        st.error(f"Health check returned HTTP {code}.")
                except requests.RequestException as exc:
                    st.session_state.api_ready = False
                    st.session_state.last_health = {"error": str(exc)}
                    st.error(f"Could not reach the API: {exc}")

    st.markdown('<div class="section-heading"><span>01</span> Patient measurements</div>', unsafe_allow_html=True)
    st.caption("All fields are required. Use the measurement values from your project dataset or a clearly labelled demonstration example.")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1, help="Number of pregnancies.")
            glucose = st.number_input("Glucose · mg/dL", min_value=0.0, max_value=400.0, value=120.0, step=1.0, help="Plasma glucose concentration.")
            blood_pressure = st.number_input("Diastolic blood pressure · mmHg", min_value=0.0, max_value=200.0, value=70.0, step=1.0, help="Diastolic blood pressure. The API requires a value greater than zero.")
        with c2:
            skin = st.number_input("Skin thickness · mm", min_value=0.0, max_value=100.0, value=20.0, step=1.0, help="Triceps skinfold thickness.")
            insulin = st.number_input("Insulin · µU/mL", min_value=0.0, max_value=1000.0, value=80.0, step=1.0, help="Two-hour serum insulin.")
            age = st.number_input("Age · years", min_value=1, max_value=119, value=30, step=1, help="Age in years.")
        with c3:
            weight = st.number_input("Weight · kg", min_value=30.1, max_value=249.9, value=70.0, step=0.5, help="Used by the API to calculate BMI. API field: Waight.")
            height = st.number_input("Height · cm", min_value=60.1, max_value=249.9, value=170.0, step=0.5, help="Used by the API to calculate BMI. API field: Hight.")
            pedigree = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.5, step=0.01, help="Family-history-related score in the dataset.")
        bmi = weight / ((height / 100) ** 2)
        st.markdown(f'<div class="bmi-strip"><div><span class="metric-label">LIVE BMI ESTIMATE</span><div class="bmi-number">{bmi:.1f} <small>kg/m²</small></div></div><div class="bmi-category">{bmi_category(bmi)}</div></div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("Run prediction", use_container_width=True, type="primary")

    if submitted:
        if blood_pressure <= 0:
            st.error("Blood pressure must be greater than zero.")
        elif not st.session_state.api_ready:
            st.warning("Check or wake the API before submitting a prediction.")
        else:
            payload = {
                "Pregnancies": int(pregnancies),
                "Glucose": float(glucose),
                "BloodPressure": float(blood_pressure),
                "SkinThickness": float(skin),
                "Insulin": float(insulin),
                "Age": int(age),
                "Waight": float(weight),
                "Hight": float(height),
                "DiabetesPedigreeFunction": float(pedigree),
            }
            started = time.perf_counter()
            with st.spinner("Sending measurements to the model…"):
                try:
                    response = requests.post(PREDICT_URL, json=payload, timeout=90)
                    duration = time.perf_counter() - started
                    try:
                        body = response.json()
                    except ValueError:
                        body = {"raw": response.text[:2000]}
                    st.session_state.last_prediction = {
                        "http_status": response.status_code,
                        "duration": round(duration, 3),
                        "request": payload,
                        "response": body,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    }
                except requests.RequestException as exc:
                    st.session_state.last_prediction = {"error": str(exc), "request": payload}

    result = st.session_state.last_prediction
    if result:
        st.markdown('<div class="section-heading"><span>02</span> Prediction result</div>', unsafe_allow_html=True)
        if "error" in result:
            st.error(f"Request failed: {result['error']}")
        elif result["http_status"] == 200 and "prediction" in result["response"]:
            prediction = result["response"]["prediction"]
            if prediction == 1:
                st.markdown('<div class="result-panel result-alert"><div class="result-kicker">MODEL OUTPUT · CLASS 1</div><h2>Positive prediction</h2><p>The model classified this input as the positive class according to its configured decision threshold.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-panel result-calm"><div class="result-kicker">MODEL OUTPUT · CLASS 0</div><h2>Negative prediction</h2><p>The model classified this input as the negative class according to its configured decision threshold.</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-note">BMI calculated by the API: <strong>{result["response"].get("data_ready", {}).get("BMI", bmi):.1f} kg/m²</strong> · {bmi_category(result["response"].get("data_ready", {}).get("BMI", bmi))}. BMI is a separate measurement and does not determine the model prediction by itself.</div>', unsafe_allow_html=True)
            st.caption(f"HTTP {result['http_status']} · API response time {result['duration']:.2f} s")
        else:
            st.error(f"The API returned HTTP {result.get('http_status', '—')}.")
            st.json(result.get("response", {}))
        with st.expander("Technical request and response details"):
            st.markdown("**Request JSON**")
            st.json(result.get("request", {}))
            st.markdown("**API response JSON**")
            st.json(result.get("response", {}))
    if st.session_state.last_health:
        with st.expander("Latest health-check details"):
            st.json(st.session_state.last_health)

elif page == "API status":
    page_header("SERVICE MONITORING", "API connection", "Check whether the deployed FastAPI service is awake and inspect the health endpoint response.")
    st.markdown(f'<div class="endpoint-card"><div class="metric-label">HEALTH ENDPOINT</div><code>{HEALTH_URL}</code><div class="endpoint-method">GET</div></div>', unsafe_allow_html=True)
    if st.button("Wake / check cloud API", type="primary"):
        with st.spinner("Sending GET /health. Render may need time to wake a sleeping instance…"):
            try:
                code, body, elapsed = request_health()
                st.session_state.last_health = {"http_status": code, "response": body, "seconds": round(elapsed, 2), "checked_at": datetime.now().isoformat(timespec="seconds")}
                st.session_state.api_ready = code == 200
            except requests.RequestException as exc:
                st.session_state.api_ready = False
                st.session_state.last_health = {"error": str(exc)}
    health = st.session_state.get("last_health")
    if health:
        if health.get("http_status") == 200:
            st.success("Service is reachable.")
        elif "error" in health:
            st.error(health["error"])
        else:
            st.warning(f"Health check returned HTTP {health.get('http_status')}.")
        st.json(health)
    else:
        st.info("No health check has been run in this session.")
    st.markdown("**Prediction endpoint**")
    st.code(f"POST {PREDICT_URL}", language="text")
    st.caption("The health endpoint checks service availability only. It does not submit patient data or run a prediction.")

elif page == "Model & features":
    page_header("MODEL DOCUMENTATION", "What the model evaluates", "The API accepts measurements, calculates BMI, then prepares the eight-feature row used by the classifier.")
    st.markdown('<div class="notice"><strong>Input pipeline</strong><span>9 request fields → BMI calculation from weight and height → 8 model features → binary prediction.</span></div>', unsafe_allow_html=True)
    for i in range(0, len(FEATURES), 2):
        left, right = st.columns(2)
        for col, item in zip((left, right), FEATURES[i:i+2]):
            key, label, unit, description = item
            with col:
                st.markdown(f'<div class="feature-card"><div class="feature-top"><span>{key}</span><small>{unit}</small></div><h3>{label}</h3><p>{description}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading"><span>API INPUTS</span> Request field mapping</div>', unsafe_allow_html=True)
    st.dataframe([
        {"Form field": "Pregnancies", "API key": "Pregnancies", "Role": "Model feature"},
        {"Form field": "Glucose", "API key": "Glucose", "Role": "Model feature"},
        {"Form field": "Diastolic blood pressure", "API key": "BloodPressure", "Role": "Model feature"},
        {"Form field": "Skin thickness", "API key": "SkinThickness", "Role": "Model feature"},
        {"Form field": "Insulin", "API key": "Insulin", "Role": "Model feature"},
        {"Form field": "Age", "API key": "Age", "Role": "Model feature"},
        {"Form field": "Weight", "API key": "Waight", "Role": "BMI calculation input"},
        {"Form field": "Height", "API key": "Hight", "Role": "BMI calculation input"},
        {"Form field": "Diabetes pedigree function", "API key": "DiabetesPedigreeFunction", "Role": "Model feature"},
    ], hide_index=True, use_container_width=True)
    st.markdown('<div class="notice"><strong>Interpretation</strong><span>The API returns class 0 or class 1. This frontend does not label that output as a probability or confidence score because the current API response does not provide one.</span></div>', unsafe_allow_html=True)

else:
    page_header("PROJECT PRESENTATION", "From measurements to prediction", "A compact overview of the Pima diabetes classification project and its deployed architecture.")
    st.markdown('<div class="overview-hero"><div class="eyebrow">MACHINE LEARNING · FASTAPI · STREAMLIT</div><h2>Turning structured health measurements into a model output.</h2><p>A demonstration interface connected to a separately deployed prediction API.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading"><span>01</span> Project objective</div>', unsafe_allow_html=True)
    st.write("Build and deploy a binary classification workflow using the Pima Indians Diabetes dataset. The user submits structured measurements, the API calculates BMI, prepares the model input and returns a class prediction.")
    st.markdown('<div class="section-heading"><span>02</span> System architecture</div>', unsafe_allow_html=True)
    st.code("User\n  ↓\nStreamlit frontend (this app)\n  ├── GET /health  → check / wake API\n  └── POST /predict → JSON measurements\n                         ↓\n                    FastAPI backend\n                         ↓\n             BMI calculation + feature row\n                         ↓\n                 Saved ML model\n                         ↓\n               Class 0 / Class 1", language="text")
    st.markdown('<div class="section-heading"><span>03</span> Technology stack</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    t1.markdown('<div class="tech-card"><h3>Python</h3><p>Core application language</p></div>', unsafe_allow_html=True)
    t2.markdown('<div class="tech-card"><h3>FastAPI</h3><p>Prediction API and validation</p></div>', unsafe_allow_html=True)
    t3.markdown('<div class="tech-card"><h3>Streamlit</h3><p>Interactive web interface</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading"><span>04</span> Responsible use & limitations</div>', unsafe_allow_html=True)
    st.write("This project is an educational demonstration, not a clinical decision-support system. Model outputs depend on the training data, feature quality, preprocessing and selected decision threshold. A prediction must not be used to diagnose, treat or rule out diabetes.")
    st.caption("The API service is independently reachable at its public endpoint. A password on this frontend restricts access to the interface but does not, by itself, protect the backend API.")
