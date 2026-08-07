import sys, io, json, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, r"D:\Claude\voice-to-rx-repo")
from voicerx.glossary import scan_labs, scan_drugs, scan_terms, collisions
print("collisions:", collisions() or "CLEAN")
texts=[]
for fp in glob.glob(r"D:\Claude\voicerx-pipeline\v3\*_rx.json"):
    for s in json.load(open(fp,encoding="utf-8"))["segments"]: texts.append(("asr",s["text"]))
ws=json.load(open(r"D:\Claude\voicerx-pipeline\all_10_files_correction_worksheet.json",encoding="utf-8"))
for s in ws:
    if s["correct_text"].strip(): texts.append(("human",s["correct_text"].strip()))
print("\nMULTI-HIT LAB SCAN")
n=0
for src,t in texts:
    l=scan_labs(t)
    if l:
        n+=1
        print(f"  {l}")
        print(f"      [{src}] {t[:72]}")
print(f"\n{n} segments ordering tests")
