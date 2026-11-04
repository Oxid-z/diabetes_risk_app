"""AI Diabetes Risk Assessment - Streamlit GUI v2 (RCAI classroom project).

Run with:  py -m streamlit run DIABETIC.py
Needs diabetes.csv in the same folder (the dataset from the notebook).
Models are trained when the app starts, so no pickle file is needed.
"""
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="AI Diabetes Risk Assessment", page_icon="🩺", layout="wide")

# ------------------------------------------------------------------ constants
FEATURES = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Pregnancies", "Age"]
LABELS = {
    "Glucose": "Glucose", "BloodPressure": "Blood pressure", "SkinThickness": "Skin thickness",
    "Insulin": "Insulin", "BMI": "BMI", "DiabetesPedigreeFunction": "Family history score",
    "Pregnancies": "Pregnancies", "Age": "Age",
}
# In this dataset a 0 in these columns means "not measured", not a real value.
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=150, min_samples_leaf=3, random_state=0),
    "Gradient Boosting": lambda: GradientBoostingClassifier(
        n_estimators=120, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=15),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=0),
}
FEATURE_SETS = {
    "Original app (Glucose, Insulin, BMI, Age)": ["Glucose", "Insulin", "BMI", "Age"],
    "+ Blood pressure": ["Glucose", "Insulin", "BMI", "Age", "BloodPressure"],
    "All 8 measurements": FEATURES,
}
RED, AMBER, GREEN, BLUE = "#ff5d73", "#ffb84d", "#2ee6a6", "#4da3ff"

CSS = """
<style>
.block-container{padding-top:2rem;max-width:1250px;}
.hero{background:linear-gradient(120deg,#0ea5a4 0%,#2563eb 55%,#7c3aed 100%);border-radius:20px;
      padding:26px 34px;color:#fff;margin-bottom:18px;box-shadow:0 12px 40px rgba(37,99,235,.28);}
.hero h1{margin:0;padding:0;font-size:2.1rem;color:#fff;}
.hero p{margin:6px 0 14px;opacity:.93;font-size:1.02rem;}
.chip{display:inline-block;background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.3);
      border-radius:999px;padding:4px 13px;margin:0 8px 4px 0;font-size:.82rem;}
.card{background:#131d30;border:1px solid rgba(255,255,255,.08);border-radius:18px;
      padding:22px;color:#e8eefc;text-align:center;}
.card.empty{padding:60px 22px;opacity:.85;}
.gauge{position:relative;width:260px;height:130px;margin:0 auto;border-radius:260px 260px 0 0;overflow:hidden;}
.gauge .tick{position:absolute;left:50%;bottom:0;width:2px;height:130px;margin-left:-1px;
             transform-origin:50% 100%;background:rgba(255,255,255,.75);}
.gauge .hole{position:absolute;left:50%;bottom:0;transform:translateX(-50%);width:196px;height:98px;
             border-radius:196px 196px 0 0;background:#131d30;display:flex;align-items:flex-end;
             justify-content:center;padding-bottom:4px;}
.gauge .pct{font-size:2.5rem;font-weight:700;line-height:1;}
.badge{display:inline-block;margin-top:14px;padding:6px 18px;border-radius:999px;font-weight:600;}
.note{margin-top:12px;font-size:.92rem;opacity:.9;}
.small{font-size:.8rem;opacity:.7;margin-top:10px;}
div[data-testid="stMetric"]{background:#131d30;border:1px solid rgba(255,255,255,.08);
                            border-radius:14px;padding:12px 16px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------- data + model
def find_csv():
    for folder in (Path(__file__).resolve().parent, Path.cwd()):
        path = folder / "diabetes.csv"
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    df[ZERO_MEANS_MISSING] = df[ZERO_MEANS_MISSING].replace(0, np.nan)
    return df[FEATURES + ["Outcome"]]


def make_pipeline(name):
    # Imputing and scaling live inside the pipeline, so they are learned from
    # training data only (no leakage into the test data).
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", MODELS[name]()),
    ])


@st.cache_resource(show_spinner="Training the model...")
def train_model(path, name):
    data = load_data(path)
    return make_pipeline(name).fit(data[FEATURES], data["Outcome"])


@st.cache_data(show_spinner="Evaluating models (first run only)...")
def cv_probabilities(path, name, cols):
    """Out-of-fold probabilities: every patient is scored by a model that never saw them."""
    data = load_data(path)
    X, y = data[list(cols)], data["Outcome"]
    repeats = 3
    probs = np.zeros(len(y))
    for seed in range(repeats):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        probs += cross_val_predict(make_pipeline(name), X, y, cv=cv, method="predict_proba")[:, 1] / repeats
    return y.to_numpy(), probs


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "Accuracy": accuracy_score(y, pred),
        "Sensitivity": recall_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, p),
    }


# ------------------------------------------------------------------------ setup
csv_path = find_csv()
if csv_path is None:
    st.error("diabetes.csv was not found. Put it in the same folder as DIABETIC.py and rerun.")
    st.stop()

csv_path = str(csv_path)
data = load_data(csv_path)
ALL_FEATURES = tuple(FEATURES)

auc_by_model = {m: roc_auc_score(*cv_probabilities(csv_path, m, ALL_FEATURES)) for m in MODELS}
best_model_name = max(auc_by_model, key=auc_by_model.get)

st.session_state.setdefault("history", [])
st.session_state.setdefault("patient", None)

# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Prediction model", list(MODELS), index=list(MODELS).index(best_model_name),
        format_func=lambda m: f"{m}  ★ best" if m == best_model_name else m,
        help="The star marks the model with the highest cross-validated ROC-AUC.",
    )
    threshold = st.slider(
        "Alert threshold", 0.20, 0.60, 0.40, 0.05,
        help="Risk at or above this value is flagged as elevated. Lower = catches more "
             "diabetic patients but gives more false alarms.",
    )
    st.caption(
        "For screening, missing a diabetic patient is worse than a false alarm, so the "
        "default (0.40) is lower than the usual 0.50."
    )
    if st.button("Clear patient history", width="stretch"):
        st.session_state.history = []
        st.session_state.patient = None
        st.rerun()
    st.divider()
    st.caption("Educational project. Not a medical device and not a diagnosis.")

model = train_model(csv_path, model_name)
y_true, y_prob = cv_probabilities(csv_path, model_name, ALL_FEATURES)
perf = metrics_at(y_true, y_prob, threshold)

# ------------------------------------------------------------------------ hero
st.markdown(
    '<div class="hero"><h1>🩺 AI Diabetes Risk Assessment</h1>'
    "<p>Enter a patient's measurements to estimate their likelihood of diabetes. "
    "Trained on historical patient records.</p>"
    f'<span class="chip">Model: {model_name}</span>'
    f'<span class="chip">ROC-AUC {perf["ROC-AUC"]:.2f}</span>'
    f'<span class="chip">Sensitivity {perf["Sensitivity"]:.0%}</span>'
    f'<span class="chip">{len(data)} patient records</span></div>',
    unsafe_allow_html=True,
)

tab_assess, tab_lab, tab_about = st.tabs(["🧑‍⚕️ Assessment", "📊 Model lab", "ℹ️ About"])


# ------------------------------------------------------------------ helpers
def classify(risk):
    if risk >= threshold:
        return "Elevated risk", RED, "Clinical testing and follow-up are recommended."
    if risk >= threshold - 0.15:
        return "Borderline", AMBER, "Consider monitoring and a repeat check."
    return "Lower risk", GREEN, "Based on these measurements. Keep up routine screening."


def gauge_card(risk):
    label, color, msg = classify(risk)
    deg = risk * 180
    return (
        '<div class="card">'
        f'<div class="gauge" style="background:conic-gradient(from 270deg at 50% 100%,{color} 0deg {deg:.1f}deg,'
        f'rgba(255,255,255,.09) {deg:.1f}deg 180deg,transparent 180deg 360deg);">'
        f'<div class="tick" style="transform:rotate({threshold * 180 - 90:.1f}deg)"></div>'
        f'<div class="hole"><span class="pct" style="color:{color}">{risk:.0%}</span></div></div>'
        f'<div class="badge" style="background:{color}22;color:{color};border:1px solid {color}66">{label}</div>'
        f'<div class="note">{msg}</div>'
        f'<div class="small">White tick = alert threshold ({threshold:.0%})</div></div>'
    )


def drivers(row):
    """Model-agnostic explanation: how much does each input move the risk versus a typical value?"""
    base = float(model.predict_proba(row)[0, 1])
    effects = {}
    for f in FEATURES:
        if pd.notna(row[f].iloc[0]):
            alt_row = row.copy()
            alt_row[f] = np.nan  # the pipeline fills this with the training median
            effects[LABELS[f]] = base - float(model.predict_proba(alt_row)[0, 1])
    return pd.DataFrame({"Input": list(effects), "Impact": list(effects.values())})


def percentile(feature, value):
    s = data[feature].dropna()
    return float((s <= value).mean() * 100)


def render_result(patient):
    row = pd.DataFrame([patient], columns=FEATURES, dtype=float)
    risk = float(model.predict_proba(row)[0, 1])
    st.markdown(gauge_card(risk), unsafe_allow_html=True)

    outside = [
        f"{LABELS[f]} ({row[f].iloc[0]:g}) is outside the range seen in training "
        f"({data[f].min():g} to {data[f].max():g})"
        for f in FEATURES
        if pd.notna(row[f].iloc[0]) and not data[f].min() <= row[f].iloc[0] <= data[f].max()
    ]
    if outside:
        st.warning("The estimate is less reliable because " + "; ".join(outside) + ".")
    missing = [LABELS[f] for f in FEATURES if pd.isna(row[f].iloc[0])]
    if missing:
        st.info("Not provided, filled with a typical value: " + ", ".join(missing) + ".")

    st.markdown("**What drove this result**")
    d = drivers(row)
    chart = (
        alt.Chart(d).mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("Impact:Q", title="Change in estimated risk", axis=alt.Axis(format="+.0%")),
            y=alt.Y("Input:N", sort="-x", title=None),
            color=alt.condition(alt.datum.Impact > 0, alt.value(RED), alt.value(GREEN)),
            tooltip=["Input", alt.Tooltip("Impact:Q", format="+.1%")],
        )
        .properties(height=max(120, 30 * len(d)))
    )
    st.altair_chart(chart, width="stretch")
    st.caption("Each bar compares this patient with a typical (median) value for that input. "
               "Red raises the estimated risk, green lowers it.")

    pct_cols = st.columns(3)
    for col, f in zip(pct_cols, ["Glucose", "BMI", "Age"]):
        col.metric(f"{LABELS[f]} percentile", f"{percentile(f, row[f].iloc[0]):.0f}th")
    st.caption("Percentile = how this patient compares with the 768 patients in the dataset.")

    label = classify(risk)[0]
    report = (
        "AI DIABETES RISK ASSESSMENT\n"
        f"Date: {datetime.now():%Y-%m-%d %H:%M}\nModel: {model_name}\n"
        f"Alert threshold: {threshold:.0%}\n\n"
        + "\n".join(f"{LABELS[f]}: {'not provided' if pd.isna(row[f].iloc[0]) else f'{row[f].iloc[0]:g}'}"
                    for f in FEATURES)
        + f"\n\nEstimated risk: {risk:.0%}\nResult: {label}\n\n"
        "Educational project. Not a medical device and not a diagnosis.\n"
    )
    st.download_button("⬇️ Download report", report, file_name="diabetes_assessment.txt", width="stretch")
    return risk, label


def optional_number(container, label, key, lo, hi, default, step, help_text):
    value = container.number_input(label, lo, hi, default, step, help=help_text, key=key)
    unknown = container.checkbox("Not available", key=key + "_na")
    return np.nan if unknown else value


# ------------------------------------------------------------------ assessment
with tab_assess:
    form_col, result_col = st.columns([1.05, 1], gap="large")

    with form_col:
        st.subheader("Patient details")
        with st.expander("Don't know the BMI? Calculate it here"):
            h_col, w_col = st.columns(2)
            height_cm = h_col.number_input("Height (cm)", 100.0, 230.0, 170.0, 1.0)
            weight_kg = w_col.number_input("Weight (kg)", 30.0, 250.0, 70.0, 0.5)
            st.info(f"BMI = **{weight_kg / (height_cm / 100) ** 2:.1f}**  (type it into the BMI field below)")

        with st.form("patient_form"):
            c1, c2 = st.columns(2)
            glucose = c1.number_input(
                "Glucose (mg/dL)", 40.0, 400.0, 120.0, 1.0,
                help="Plasma glucose 2 hours into an oral glucose tolerance test. "
                     "A fasting glucose reading is not the same measurement.")
            bmi = c1.number_input("BMI (kg/m²)", 10.0, 80.0, 30.0, 0.1, help="Body mass index.")
            age = c2.number_input("Age (years)", 18, 100, 35, 1)
            bp = optional_number(c2, "Blood pressure (mm Hg)", "bp", 30.0, 140.0, 72.0, 1.0,
                                 "Diastolic blood pressure.")
            insulin = optional_number(c1, "Insulin (µU/mL)", "ins", 0.0, 900.0, 100.0, 1.0,
                                      "2-hour serum insulin from the same test.")
            with st.expander("More measurements (optional)"):
                e1, e2 = st.columns(2)
                skin = optional_number(e1, "Skin thickness (mm)", "skin", 5.0, 100.0, 29.0, 1.0,
                                       "Triceps skin fold thickness.")
                pedigree = e2.number_input("Family history score", 0.05, 2.5, 0.37, 0.01,
                                           help="Diabetes pedigree function: higher = stronger family history.")
                pregnancies = e1.number_input("Pregnancies", 0, 20, 0, 1)
            submitted = st.form_submit_button("Assess risk", type="primary", width="stretch")

        if submitted:
            st.session_state.patient = {
                "Glucose": glucose, "BloodPressure": bp, "SkinThickness": skin, "Insulin": insulin,
                "BMI": bmi, "DiabetesPedigreeFunction": pedigree, "Pregnancies": float(pregnancies),
                "Age": float(age),
            }
            row_now = pd.DataFrame([st.session_state.patient], columns=FEATURES, dtype=float)
            r_now = float(model.predict_proba(row_now)[0, 1])
            st.session_state.history.append({
                "Time": datetime.now().strftime("%H:%M:%S"), "Glucose": glucose, "BMI": bmi, "Age": age,
                "Model": model_name, "Risk": r_now, "Result": classify(r_now)[0],
            })

    with result_col:
        st.subheader("Result")
        if st.session_state.patient is None:
            st.markdown(
                '<div class="card empty"><div style="font-size:2.6rem">🩺</div>'
                "<b>No assessment yet</b><div class=\"note\">Fill in the patient details and press "
                "<i>Assess risk</i>. Changing the model or threshold updates the result instantly.</div></div>",
                unsafe_allow_html=True,
            )
        else:
            render_result(st.session_state.patient)

    if st.session_state.history:
        st.subheader("Assessment history")
        hist = pd.DataFrame(st.session_state.history).iloc[::-1]
        st.dataframe(
            hist, hide_index=True, width="stretch",
            column_config={"Risk": st.column_config.ProgressColumn("Risk", format="percent", min_value=0, max_value=1)},
        )
        st.download_button("⬇️ Download history (CSV)", hist.to_csv(index=False),
                           file_name="assessment_history.csv")

# ------------------------------------------------------------------- model lab
with tab_lab:
    st.subheader(f"How well does {model_name} work?")
    st.caption(
        f"5-fold cross-validation repeated 3 times on {len(data)} patients, at an alert threshold of "
        f"{threshold:.2f}. Every patient is scored by a model that never saw them."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{perf['Accuracy']:.1%}")
    m2.metric("Sensitivity", f"{perf['Sensitivity']:.1%}", help="Share of diabetic patients that were flagged.")
    m3.metric("Precision", f"{perf['Precision']:.1%}", help="Share of flagged patients that were diabetic.")
    m4.metric("ROC-AUC", f"{perf['ROC-AUC']:.2f}", help="Ranking quality, 0.5 = guessing, 1.0 = perfect.")

    roc_col, thr_col = st.columns(2, gap="large")
    with roc_col:
        st.markdown("**ROC curve**")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_df = pd.DataFrame({"False alarm rate": fpr, "Sensitivity": tpr})
        line = alt.Chart(roc_df).mark_area(line={"color": BLUE}, color=alt.Gradient(
            gradient="linear", stops=[alt.GradientStop(color="#4da3ff55", offset=0),
                                      alt.GradientStop(color="#4da3ff05", offset=1)],
            x1=1, x2=1, y1=1, y2=0)).encode(x="False alarm rate:Q", y="Sensitivity:Q")
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
            strokeDash=[5, 5], color="#8a94a8").encode(x="x:Q", y="y:Q")
        st.altair_chart((line + diag).properties(height=300), width="stretch")
        st.caption("The dashed line is random guessing. The further the curve bows to the top-left, the better.")

    with thr_col:
        st.markdown("**Threshold trade-off**")
        rows = []
        for t in np.arange(0.10, 0.91, 0.05):
            p = (y_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, p, labels=[0, 1]).ravel()
            rows += [
                {"Threshold": t, "Metric": "Sensitivity", "Value": tp / max(tp + fn, 1)},
                {"Threshold": t, "Metric": "Specificity", "Value": tn / max(tn + fp, 1)},
                {"Threshold": t, "Metric": "Precision", "Value": tp / max(tp + fp, 1)},
            ]
        thr_df = pd.DataFrame(rows)
        lines = alt.Chart(thr_df).mark_line(strokeWidth=2.5).encode(
            x=alt.X("Threshold:Q", axis=alt.Axis(format=".0%")),
            y=alt.Y("Value:Q", axis=alt.Axis(format=".0%"), title=None),
            color=alt.Color("Metric:N", scale=alt.Scale(range=[RED, BLUE, GREEN]), legend=alt.Legend(orient="bottom")),
        )
        rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(color="white", strokeDash=[4, 4]).encode(x="t:Q")
        st.altair_chart((lines + rule).properties(height=300), width="stretch")
        st.caption("Moving the threshold up trades sensitivity for fewer false alarms. The white line is your setting.")

    cm_col, cmp_col = st.columns([1, 1.3], gap="large")
    with cm_col:
        st.markdown("**Confusion matrix**")
        tn, fp, fn, tp = confusion_matrix(y_true, (y_prob >= threshold).astype(int), labels=[0, 1]).ravel()
        cm_df = pd.DataFrame({
            "Actual": ["Not diabetic", "Not diabetic", "Diabetic", "Diabetic"],
            "Predicted": ["Not diabetic", "Diabetic", "Not diabetic", "Diabetic"],
            "Patients": [tn, fp, fn, tp],
        })
        base = alt.Chart(cm_df).encode(
            x=alt.X("Predicted:N", sort=["Not diabetic", "Diabetic"], axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Actual:N", sort=["Not diabetic", "Diabetic"]))
        heat = base.mark_rect(cornerRadius=6).encode(
            color=alt.Color("Patients:Q", scale=alt.Scale(range=["#17304f", "#0ea5a4"]), legend=None))
        text = base.mark_text(fontSize=20, fontWeight="bold", color="white").encode(text="Patients:Q")
        st.altair_chart((heat + text).properties(height=260), width="stretch")
        st.caption(f"{fn} diabetic patients were missed, and {fp} were flagged without diabetes.")

    with cmp_col:
        st.markdown("**Model comparison** (same threshold, all 8 inputs)")
        comp = pd.DataFrame({m: metrics_at(*cv_probabilities(csv_path, m, ALL_FEATURES), threshold)
                             for m in MODELS}).T
        st.dataframe(
            comp.style.format({"Accuracy": "{:.1%}", "Sensitivity": "{:.1%}", "Precision": "{:.1%}", "ROC-AUC": "{:.3f}"})
            .highlight_max(color="#0ea5a455"),
            width="stretch",
        )
        st.markdown("**Does adding more inputs help?** (Logistic Regression, ROC-AUC)")
        fs = pd.DataFrame({
            "Inputs": list(FEATURE_SETS),
            "ROC-AUC": [roc_auc_score(*cv_probabilities(csv_path, "Logistic Regression", tuple(cols)))
                        for cols in FEATURE_SETS.values()],
        })
        st.dataframe(fs.style.format({"ROC-AUC": "{:.3f}"}).highlight_max(subset="ROC-AUC", color="#0ea5a455"),
                     hide_index=True, width="stretch")

# ----------------------------------------------------------------------- about
with tab_about:
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("The model")
        st.markdown(
            f"""
- **Data:** {len(data)} patient records (Pima Indians diabetes dataset).
- **Inputs:** glucose, blood pressure, skin thickness, insulin, BMI, family history score,
  pregnancies and age. Any input can be left out and is filled with the training median.
- **Cleaning:** a 0 in glucose, blood pressure, skin thickness, insulin or BMI means "not measured",
  so it is treated as missing. Filling is learned from training data only.
- **Models compared:** logistic regression, random forest, gradient boosting, KNN and decision tree.
  The app starts on the one with the best cross-validated ROC-AUC (currently **{best_model_name}**).
- **Evaluation:** repeated stratified cross-validation, so accuracy is not inflated by testing on training data.
- **Explanations:** each result shows how far every input moved the estimate from a typical value.
            """
        )
    with right:
        st.subheader("Limitations and ethics")
        st.markdown(
            """
- Accuracy on this dataset tops out at roughly 75-80%, so some patients will be misclassified.
- The data covers one population, with no subgroup analysis, so results may not transfer to others.
- It estimates risk from a few measurements. It cannot replace a clinical test or a doctor.
- Use with expert oversight, and protect any patient data entered.
- Assessment history lives only in this browser session and is not saved to disk.
            """
        )
