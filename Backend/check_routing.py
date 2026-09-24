# check_routing.py
from classifier import classify_complaint

for t in [
    "wifi is not working in the library",
    "internet is very slow in cw202",
    "lab computers are not turning on",
    "projector in cw203 is not working",
    "cannot log in to the student portal",
    "no benches in class",
    "not enough benches in cw202",
]:
    print(t)
    print("   ", classify_complaint(t))