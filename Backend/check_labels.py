# check_labels.py
import glob, joblib
from classifier import classify_complaint

r = classify_complaint("not enough benches in cw202")
print([(label, " " in label) for label, _ in r.reasoning["ml_runner_up"]])

def find_classes(obj):
    if hasattr(obj, "classes_"):
        return list(obj.classes_)
    if isinstance(obj, dict):
        for v in obj.values():
            found = find_classes(v)
            if found:
                return found
    for _, step in getattr(obj, "steps", []):
        found = find_classes(step)
        if found:
            return found

for path in glob.glob("ml/model/*"):
    try:
        print(path, "->", find_classes(joblib.load(path)))
    except Exception as e:
        print(path, "-> skipped:", type(e).__name__) 