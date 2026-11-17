"""Turn a local JSON or official JSONL match into an offline replay player."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("--output",default="replays/latest.html")
    args=p.parse_args()
    text=Path(args.input).read_text()
    if text.lstrip().startswith("["):
        data=json.loads(text)
    else:
        data=[{"state":f,"stats":{},"moves":{}} for f in map(json.loads,text.splitlines()) if "board" in f]
    template=Path(__file__).with_name("replay.html").read_text()
    encoded=json.dumps(data,separators=(",",":"),ensure_ascii=False).replace("<","\\u003c")
    output=Path(args.output);output.parent.mkdir(exist_ok=True)
    output.write_text(template.replace("/*REPLAY_DATA*/[]",encoded))
    print(output.resolve())


if __name__=="__main__":main()
