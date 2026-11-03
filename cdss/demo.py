"""Scripted demo patients (kept free of heavy imports so the landing page loads fast)."""

DEMO_PATIENTS = {
    "Clear present": dict(age=63, sex="male", cp="asymptomatic", trestbps=150, chol=280, fbs=True, restecg="lv hypertrophy",
                          thalch=105, exang=True, oldpeak=3.2, slope="flat", ca=2, thal="reversable defect"),
    "Clear absent": dict(age=38, sex="female", cp="non-anginal", trestbps=115, chol=190, fbs=False, restecg="normal",
                         thalch=175, exang=False, oldpeak=0.0, slope="upsloping", ca=0, thal="normal"),
    "Borderline / missing data": dict(age=52, sex="male", cp="non-anginal", trestbps=140, chol=250, fbs=False,
                                      restecg="normal", thalch=130, exang=True, oldpeak=1.5, slope=None, ca=None, thal=None),
    "Rule vs ML disagreement": dict(age=57, sex="male", cp="atypical angina", trestbps=135, chol=240, fbs=False,
                                    restecg="st-t abnormality", thalch=135, exang=False, oldpeak=1.2, slope=None, ca=None, thal=None),
}
