import sys, io, json, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, r"D:\Claude\voice-to-rx-repo")
from voicerx.gate import judge_medication, VERIFIED, PROBABLE, REJECTED

names=[]
for fp in glob.glob(r"D:\Claude\voicerx-pipeline\v3\*_rx.json"):
    for rx in json.load(open(fp,encoding="utf-8"))["extractions"]:
        for m in rx.get("medications",[]):
            n=(m.get("drug") or "").strip()
            if n and n not in names: names.append(n)

print("EVERY medication the SLM proposed on real audio\n")
c={VERIFIED:0,PROBABLE:0,REJECTED:0}
for n in names:
    v=judge_medication(n); c[v.tier]+=1
    tag={"verified":"VERIFIED","probable":"PROBABLE","rejected":"reject  "}[v.tier]
    extra=f" -> {v.canonical}" if v.canonical else ""
    sim=f" [{v.similarity}]" if v.tier==PROBABLE else ""
    print(f"  {tag}  {n[:40]:42}{extra}{sim}")
print(f"\n  verified={c[VERIFIED]}  probable={c[PROBABLE]}  rejected={c[REJECTED]}")
