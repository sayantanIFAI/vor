"""CLI entry point. Run over every WAV in /root/voicerx and write one JSON
report per file plus a combined summary.

Usage: python3 run_pipeline.py [audio1.wav audio2.wav ...]
       (defaults to every *.wav in the current directory if none given)
"""
from __future__ import annotations

import dataclasses
import glob
import json
import sys
import time

from voicerx.pipeline import VoiceToRxPipeline
from voicerx.schema import ExtractedRx, Medication


def rx_to_dict(rx: ExtractedRx) -> dict:
    return rx.model_dump()


def main():
    audio_files = sys.argv[1:] or sorted(glob.glob("*.wav"))
    if not audio_files:
        print("No audio files given and none found in cwd.")
        sys.exit(1)

    print(f"Processing {len(audio_files)} file(s): {audio_files}", flush=True)
    print("Loading models (ASR + VAD)...", flush=True)
    t0 = time.time()
    pipeline = VoiceToRxPipeline()
    print(f"models loaded in {time.time()-t0:.1f}s", flush=True)

    summary = {
        "files_processed": 0,
        "total_segments": 0,
        "total_extractions": 0,
        "segments_needing_review": 0,
        "extract_errors": 0,
        "decoder_rnnt_used": 0,
        "decoder_ctc_fallback_used": 0,
        "low_decoder_agreement_flags": 0,
    }
    all_results = []

    for audio_path in audio_files:
        print(f"\n{'='*70}\n{audio_path}\n{'='*70}", flush=True)
        result = pipeline.process_file(audio_path)

        print(f"  {len(result.segments)} VAD segments, "
              f"asr={result.timing['asr_s']}s extract={result.timing['extract_s']}s", flush=True)

        for seg, rx in zip(result.segments, result.extractions):
            flag = "REVIEW" if rx.needs_human_review else "ok"
            print(f"  [{seg.start_s:6.1f}-{seg.end_s:6.1f}s] ({flag}) "
                  f"[{seg.decoder_used} agree={seg.decoder_agreement:.2f}] {seg.text[:60]}", flush=True)
            if rx.needs_human_review:
                for reason in rx.review_reasons:
                    print(f"      -> {reason}", flush=True)

        summary["files_processed"] += 1
        summary["total_segments"] += len(result.segments)
        summary["total_extractions"] += len(result.extractions)
        summary["segments_needing_review"] += sum(
            1 for rx in result.extractions if rx.needs_human_review
        )
        summary["extract_errors"] += len(result.timing["extract_errors"])
        summary["decoder_rnnt_used"] += sum(
            1 for s in result.segments if s.decoder_used == "rnnt"
        )
        summary["decoder_ctc_fallback_used"] += sum(
            1 for s in result.segments if s.decoder_used == "ctc_fallback"
        )
        summary["low_decoder_agreement_flags"] += sum(
            1 for rx in result.extractions
            if any("low CTC/RNNT agreement" in r for r in rx.review_reasons)
        )

        out_path = audio_path.rsplit(".", 1)[0] + "_rx.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "audio_file": audio_path,
                "timing": result.timing,
                "segments": [dataclasses.asdict(s) for s in result.segments],
                "extractions": [rx_to_dict(rx) for rx in result.extractions],
            }, f, ensure_ascii=False, indent=2)
        print(f"  -> {out_path}", flush=True)

        all_results.append({"audio_file": audio_path, "out": out_path})

    with open("pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "files": all_results}, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
