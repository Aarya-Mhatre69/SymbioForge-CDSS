"""Plain-language meaning of every input, plus a simple status for the patient's own value.
Cut-offs are common, simplified guideline thresholds used for orientation only, not diagnosis."""
from __future__ import annotations

FIELDS: dict[str, dict] = {
    "age": dict(label="Age", unit="years", short="Your age",
                what="Your age in years.",
                matters="Heart-disease risk rises steadily with age.",
                normal="No normal range; risk factors count more after about 55.",
                how="Known"),
    "sex": dict(label="Sex", unit="", short="Sex assigned at birth",
                what="Biological sex as recorded in the training data.",
                matters="On average, men develop heart disease earlier; the data also holds far fewer women.",
                normal="-", how="Known"),
    "cp": dict(label="Chest-pain pattern", unit="", short="What your chest discomfort feels like",
               what="Doctors sort chest pain into four patterns. 'Typical' angina has three features: a pressure or tightness behind the breastbone, "
                    "brought on by exertion or stress, and eased within minutes by rest. Two features is 'atypical'; one or none is 'non-anginal'.",
               matters="It is the single strongest signal in the data. Surprisingly, people with no chest pain at all ('asymptomatic') "
                       "were more often found to have heart disease on testing, so no pain is not proof of a healthy heart.",
               normal="-", how="Your description, sorted by the three features above"),
    "trestbps": dict(label="Resting blood pressure", unit="mmHg", short="Blood pressure at rest (the top number)",
                     what="The pressure in your arteries while resting, the top (systolic) number of a reading like 130/85.",
                     matters="Long-term high pressure strains the heart and damages arteries.",
                     normal="Below 120 is ideal; 120-139 is elevated; 140 or more is high.",
                     how="Cuff on the upper arm at a check-up"),
    "chol": dict(label="Cholesterol", unit="mg/dl", short="Total cholesterol in blood",
                 what="The total amount of cholesterol in your blood.",
                 matters="Excess cholesterol can build up as plaque inside the arteries that feed the heart.",
                 normal="Below 200 is desirable; 200-239 is borderline; 240 or more is high.",
                 how="Blood test (lipid panel)"),
    "fbs": dict(label="Fasting blood sugar", unit="", short="Blood sugar after fasting above 120 mg/dl?",
                what="Whether your blood sugar, measured after at least 8 hours without food, is above 120 mg/dl.",
                matters="High blood sugar (diabetes or pre-diabetes) damages blood vessels over time.",
                normal="Up to about 100 mg/dl is normal; above 120 is flagged here.",
                how="Blood test after fasting"),
    "restecg": dict(label="Resting ECG", unit="", short="Heart tracing while at rest",
                    what="An electrocardiogram (ECG) records the heart's electrical activity. It can be normal, show a thickened left "
                         "ventricle ('LV hypertrophy'), or show an ST-T wave abnormality (a sign the heart muscle may be strained).",
                    matters="Abnormal tracings can reflect strain or past damage.",
                    normal="Normal", how="ECG at a clinic, lying down"),
    "thalch": dict(label="Peak heart rate", unit="bpm", short="Highest heart rate reached during exercise",
                   what="The highest heart rate you reached during an exercise (treadmill) stress test.",
                   matters="A heart that cannot speed up as expected can signal reduced fitness or heart problems. "
                           "It is judged against your age: the expected maximum is roughly 220 minus your age.",
                   normal="Reaching at least 85% of (220 - age) is typical.",
                   how="Exercise stress test"),
    "exang": dict(label="Chest pain during exercise", unit="", short="Did exercise bring on chest pain?",
                  what="Whether chest pain appeared while exercising during the stress test.",
                  matters="Pain that appears when the heart works harder suggests it is not getting enough blood.",
                  normal="No", how="Reported during a stress test"),
    "oldpeak": dict(label="ST depression", unit="mm", short="How far the ECG dips during exercise",
                    what="On an exercise ECG, the ST segment can dip below its resting line. The size of the dip, in millimetres, is the 'ST depression'.",
                    matters="A bigger dip suggests the heart muscle is short of oxygen under strain.",
                    normal="About 0-1 is typical; above 2 is notable.",
                    how="Exercise stress test ECG compared with the resting ECG"),
    "slope": dict(label="ST slope", unit="", short="Shape of the ECG line at peak exercise",
                  what="The direction the ST segment takes at peak exercise: rising (upsloping), flat, or falling (downsloping).",
                  matters="Flat or falling slopes are more often linked to reduced blood flow.",
                  normal="Upsloping", how="Exercise stress test ECG (optional test)"),
    "ca": dict(label="Major vessels with narrowing", unit="0-3", short="Number of main heart arteries showing narrowing",
               what="How many of the 3 major heart arteries show narrowing on a special X-ray with contrast dye (fluoroscopy / angiography).",
               matters="More narrowed vessels means a higher chance that disease is present and significant.",
               normal="0", how="Fluoroscopy or angiography (an optional, more invasive test)"),
    "thal": dict(label="Thallium stress result", unit="", short="Blood-flow scan result",
                 what="A nuclear scan (thallium test) shows how well blood reaches the heart muscle. 'Normal', 'fixed defect' (an area that never "
                      "gets blood, such as old damage) or 'reversible defect' (an area short of blood only under exertion).",
                 matters="A reversible defect is one of the strongest signs of narrowed arteries.",
                 normal="Normal", how="Nuclear stress scan (an optional test)"),
}

# plain names for tests the planner may recommend
TEST_PLAIN = {
    "slope": "Stress-test ECG slope (the shape of the ECG line at peak exercise)",
    "thal": "Thallium stress scan (a blood-flow scan of the heart)",
    "ca": "Fluoroscopy / angiography vessel count (an X-ray with contrast dye; the most invasive of the three)",
}

ALIASES = {
    "bp": "trestbps", "blood pressure": "trestbps", "pressure": "trestbps", "cholesterol": "chol", "sugar": "fbs",
    "glucose": "fbs", "ecg": "restecg", "ekg": "restecg", "heart rate": "thalch", "pulse": "thalch", "st depression": "oldpeak",
    "oldpeak": "oldpeak", "st slope": "slope", "slope": "slope", "vessel": "ca", "fluoroscopy": "ca", "thal": "thal",
    "thallium": "thal", "chest pain": "cp", "angina": "cp", "exercise": "exang", "age": "age", "sex": "sex",
}


def lookup(text: str) -> str | None:
    t = text.lower()
    for k in sorted(ALIASES, key=len, reverse=True):
        if k in t:
            return ALIASES[k]
    for key, f in FIELDS.items():
        if f["label"].lower() in t:
            return key
    return None


def status(field: str, p: dict) -> tuple[str, str]:
    """-> (level, note); level in ok / watch / flag / info / missing."""
    v = p.get(field)
    if v is None:
        return "missing", "Not provided"
    if field == "age":
        return ("watch", "Above 55, age counts as a risk factor") if v > 55 else ("ok", "Below the usual age cut-off")
    if field == "sex":
        return "info", "Included because rates differ by sex in the data"
    if field == "cp":
        return {"typical angina": ("flag", "Pressure on exertion, relieved by rest: the classic pattern"),
                "atypical angina": ("watch", "Some, but not all, features of classic angina"),
                "non-anginal": ("ok", "Does not look like classic heart-related pain"),
                "asymptomatic": ("watch", "No chest pain: not reassuring on its own, as disease is often silent")}[v]
    if field == "trestbps":
        return (("flag", "High (140 or more)") if v >= 140 else ("watch", "Elevated (120-139)") if v >= 120
                else ("watch", "Low (under 90)") if v < 90 else ("ok", "In the ideal range"))
    if field == "chol":
        return ("flag", "High (240 or more)") if v >= 240 else ("watch", "Borderline (200-239)") if v >= 200 else ("ok", "Desirable (under 200)")
    if field == "fbs":
        return ("flag", "Above 120 mg/dl") if v else ("ok", "At or below 120 mg/dl")
    if field == "restecg":
        return ("ok", "Normal") if v == "normal" else ("watch", "Abnormal tracing: " + v)
    if field == "thalch":
        exp = 220 - p["age"]
        pct = v / exp
        return (("ok", f"{pct:.0%} of the age-expected maximum ({exp} bpm)") if pct >= 0.85 else
                ("watch", f"{pct:.0%} of the age-expected maximum ({exp} bpm)") if pct >= 0.70 else
                ("flag", f"Only {pct:.0%} of the age-expected maximum ({exp} bpm)"))
    if field == "exang":
        return ("flag", "Exercise brought on chest pain") if v else ("ok", "No pain on exercise")
    if field == "oldpeak":
        return ("flag", "Notable dip (above 2)") if v > 2 else ("watch", "Mild dip (1-2)") if v > 1 else ("ok", "Little or no dip")
    if field == "slope":
        return {"upsloping": ("ok", "Rising: typical"), "flat": ("watch", "Flat"), "downsloping": ("flag", "Falling")}[v]
    if field == "ca":
        return ("ok", "No narrowed major vessels") if v == 0 else ("flag", f"{v} major vessel(s) narrowed")
    if field == "thal":
        return {"normal": ("ok", "Normal blood flow"), "fixed defect": ("watch", "An area that never gets blood"),
                "reversable defect": ("flag", "An area short of blood under exertion")}[v]
    return "info", ""


def value_text(field: str, v) -> str:
    if v is None:
        return "not provided"
    if isinstance(v, bool):
        return "yes" if v else "no"
    unit = FIELDS[field]["unit"]
    return f"{v} {unit}".strip() if field in ("age", "trestbps", "chol", "thalch", "oldpeak") else str(v)
