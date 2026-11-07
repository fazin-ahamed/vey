"""Latency by lane and resident memory of a running ``vey-serve``.

    vey-serve --port 8000 &
    python scripts/systemone_bench.py http://127.0.0.1:8000 --requests 60 --threads 4

Reports, from the client, p50 and p95 round-trip milliseconds for requests that
resolve on each lane, the one-time cost of the first NLI call (the model loads on
first use), and the server's own resident memory before and after, read from
``/v1/info``. Report the host with every number: latency here is hardware
specific, as CONTRIBUTING.md requires.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import statistics
import threading
import time
import urllib.request

STATE = ("Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan. "
         "The invoice number is 4411 and the card ends in 0042.")
TEAM = {"type": "choice", "instructions": "Which team should handle this?", "criteria": {
    "billing": "Charges, refunds and invoices", "technical": "Bugs, outages and errors",
    "shipping": "Delivery and parcels", "other": "Everything else"}}
YESNO = {"type": "noul", "instructions": "Is the customer asking for a refund?"}
URGENCY = {"type": "score", "instructions": "How urgent is this ticket?",
           "criteria": ["Can wait", "Needs attention this week", "Needs attention today"]}
WORKLOADS = {
    "exact (state field)": {"state": {"status": "overdue", "note": "pay"},
                            "questions": {"status": {"type": "choice", "criteria": {"paid": "", "overdue": "", "draft": ""}}}},
    "choice, 4 options": {"state": STATE, "questions": {"team": TEAM}},
    "score, 3 levels": {"state": STATE, "questions": {"urgency": URGENCY}},
    "noul": {"state": STATE, "questions": {"refund": YESNO}},
    "3 questions": {"state": STATE, "questions": {"refund": YESNO, "team": TEAM, "urgency": URGENCY}},
}


def post(url: str, body: dict, key: str | None) -> float:
    payload = dict(body, model="vey-latest")
    req = urllib.request.Request(url + "/v1/systemone", data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
    ms = (time.perf_counter() - t) * 1000
    assert set(out["answers"]) == set(body["questions"])
    for name, q in body["questions"].items():
        a = out["answers"][name]
        assert a["type"] == q["type"]
        if q["type"] == "noul":
            assert 0 <= a["noul"] <= 1
        else:
            p = a["probabilities"]
            keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
            assert list(p) == keys and abs(sum(p.values()) - 1) < 1e-4
            assert all(0 <= v <= 1 for v in p.values()) and abs(a["confidence"] - max(p.values())) < 1e-5
            if q["type"] == "choice":
                assert p[a["choice"]] == max(p.values())
            else:
                assert abs(a["score"] - sum(int(k) * v for k, v in p.items())) < 1e-4
    return ms


def image_request(side: int) -> dict:
    """The same synthetic receipt at two sizes, with two questions sharing its vision pass."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.new("RGB", (side, side), "white")
    draw, font = ImageDraw.Draw(im), ImageFont.load_default(size=max(10, side // 16))
    y = side // 12
    for line in ["ACME STORE", "Widget x2   $500.00", "Gadget x1   $750.00", "", "TOTAL USD $1,250.00"]:
        draw.text((side // 10, y), line, fill="black", font=font)
        y += side // 8
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return {"state": "photo", "assets": {"photo": {"type": "image", "format": "png",
            "data_base64": base64.b64encode(buf.getvalue()).decode()}}, "questions": {
                "legible": {"type": "noul", "instructions": "Is the receipt total legible?"},
                "cur": {"type": "choice", "instructions": "Which currency?", "criteria": {"USD": "dollars", "EUR": "euros", "GBP": "pounds"}}}}


def info(url: str, key: str | None) -> dict:
    req = urllib.request.Request(url + "/v1/info")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--requests", type=int, default=60)
    ap.add_argument("--threads", type=int, default=4, help="concurrent clients for the throughput run")
    ap.add_argument("--key", default=os.environ.get("VEY_API_KEY"))
    ap.add_argument("--images", action="store_true", help="include 512px and 2000px receipts (needs Pillow and a --d1 server)")
    ap.add_argument("--skip-throughput", action="store_true", help="only measure serial workloads")
    a = ap.parse_args()
    before = info(a.url, a.key)
    print(f"host: {os.cpu_count()} CPUs | resident before: {before['resident_mb']} MiB | embedder {before['lanes']['matcher']['embedder']}")
    t_first = post(a.url, WORKLOADS["noul"], a.key)
    print(f"first yes/no request (may load a model): {t_first:.0f} ms")
    workloads = dict(WORKLOADS)
    if a.images:
        workloads.update({f"image {s}px, 2 questions": image_request(s) for s in (512, 2000)})
    for name, body in workloads.items():
        for _ in range(3):
            post(a.url, body, a.key)
        ts = [post(a.url, body, a.key) for _ in range(a.requests)]
        print(f"{name:24s} p50 {statistics.median(ts):7.1f} ms   p95 {pct(ts, .95):7.1f} ms   resident {info(a.url, a.key)['resident_mb']} MiB", flush=True)
    if a.skip_throughput:
        return
    body = WORKLOADS["3 questions"]
    done, lock, errs = [], threading.Lock(), []

    def worker(n: int) -> None:
        for _ in range(n):
            try:
                post(a.url, body, a.key)
                with lock:
                    done.append(1)
            except Exception as e:  # noqa: BLE001
                errs.append(str(e))

    per = max(1, a.requests // a.threads)
    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker, args=(per,)) for _ in range(a.threads)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    dt = time.perf_counter() - t0
    print(f"{a.threads} concurrent clients, 3 questions each: {len(done) / dt:.1f} requests/s, {len(errs)} errors")
    after = info(a.url, a.key)
    print(f"resident after: {after['resident_mb']} MiB | counts {after['counts']}")


if __name__ == "__main__":
    main()
