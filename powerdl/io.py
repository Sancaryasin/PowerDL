from __future__ import annotations
import csv

def save_samples_csv(path, samples):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "p_w", "gpu_util", "mem_util"])
        for s in samples:
            w.writerow([s.t, s.p_w, s.gpu_util, s.mem_util])

def save_marks_csv(path, marks):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "t", "meta"])
        for m in marks:
            w.writerow([m.name, m.t, dict(m.meta)])

def save_rows_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
