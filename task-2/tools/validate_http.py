"""Check API behavior and real HTTP latency against a running local server."""
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url",default="http://127.0.0.1:8000")
    p.add_argument("--report",default="reports/http.json")
    p.add_argument("--positions",default="tests/positions.json")
    p.add_argument("--repeat",type=int,default=3)
    args=p.parse_args()
    def call(path,data=None):
        raw=json.dumps(data).encode() if data is not None else None
        req=urllib.request.Request(args.url+path,data=raw,headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=1.0) as r:return r.status,json.load(r)
    assert call("/")[1]["apiversion"]=="1"
    states=json.loads(Path(args.positions).read_text())
    assert call("/start",states[0])[0]==200
    assert call("/end",states[0])[0]==200
    try:call("/move",{})
    except urllib.error.HTTPError as e:assert e.code==400
    else:raise AssertionError("Malformed request accepted")
    elapsed=[]
    for _ in range(args.repeat):
        for s in states:
            started=time.perf_counter();status,data=call("/move",s)
            elapsed.append((time.perf_counter()-started)*1000)
            assert status==200 and data["move"] in ("up","down","left","right")
    report={"url":args.url,"moves":len(elapsed),"p50_ms":round(statistics.median(elapsed),3),
            "p95_ms":round(sorted(elapsed)[int((len(elapsed)-1)*.95)],3),"max_ms":round(max(elapsed),3),
            "over_500ms":sum(t>=500 for t in elapsed),"samples_ms":[round(t,3) for t in elapsed]}
    report_path=Path(args.report)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    assert not report["over_500ms"], "Exceeded course timeout"


if __name__=="__main__":main()
