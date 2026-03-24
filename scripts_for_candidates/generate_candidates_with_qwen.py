#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generiert Kandidaten mit lokalem Qwen2.5 via Ollama
"""

import json
import os
import glob
from pathlib import Path
from llm_manager import LLMManager

# === KONFIGURATION ===
BASE_DIR = r"C:\Users\Norita\DMML Datamining Projekt Julia\data\BIRD\train\train\Address Table Candidates\with_query_ids"
OUTPUT_FILE = r"C:\Users\Norita\DMML Datamining Projekt Julia\data\BIRD\train\train\candidates_500_qwen.json"

# Anzahl der Fragen und Kandidaten pro Frage
NUM_QUESTIONS = 50
CANDIDATES_PER_QUESTION = 10

def find_input_file():
    """Findet die Eingabedatei"""
    json_files = glob.glob(os.path.join(BASE_DIR, "*.json"))
    
    if not json_files:
        return None
    
    print("\nGefundene JSON-Dateien:")
    for i, file in enumerate(json_files, 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    return json_files[0]

def load_input_data(filepath, num_questions=50):
    """Lädt die Eingabedaten"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data[:num_questions]
    else:
        return [data]

def main():
    print("=" * 70)
    print("Kandidaten-Generator mit lokalem Qwen2.5 (Ollama)")
    print("=" * 70)
    
    # LLMManager initialisieren
    print("\n🔧 Initialisiere LLMManager...")
    try:
        llm = LLMManager(model_name="qwen2.5", temperature=0.8)
    except Exception as e:
        print(f"❌ Fehler: {e}")
        print("\nBitte führen Sie zuerst aus:")
        print("1. ollama serve (in einem separaten Terminal)")
        print("2. ollama pull qwen2.5")
        return
    
    # Eingabedatei finden
    print(f"\n📁 Suche nach Eingabedatei...")
    input_file = find_input_file()
    
    if not input_file:
        print(f"\n❌ Keine JSON-Dateien gefunden in: {BASE_DIR}")
        return
    
    print(f"\n✅ Verwende: {os.path.basename(input_file)}")
    
    # Daten laden
    print(f"\n📚 Lade Daten...")
    questions = load_input_data(input_file, NUM_QUESTIONS)
    print(f"✅ {len(questions)} Fragen geladen")
    
    # Kandidaten generieren
    print(f"\n🚀 Starte Generierung von {len(questions) * CANDIDATES_PER_QUESTION} Kandidaten...")
    
    all_candidates = llm.batch_generate(
        questions, 
        candidates_per_question=CANDIDATES_PER_QUESTION,
        delay=2.0
    )
    
    # Ergebnisse speichern
    print(f"\n💾 Speichere {len(all_candidates)} Kandidaten...")
    
    output_data = {
        "metadata": {
            "model": "qwen2.5",
            "input_file": os.path.basename(input_file),
            "total_questions": len(questions),
            "total_candidates": len(all_candidates),
            "candidates_per_question": CANDIDATES_PER_QUESTION
        },
        "candidates": all_candidates
    }
    
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ FERTIG!")
    print("=" * 70)
    print(f"Verarbeitete Fragen: {len(questions)}")
    print(f"Generierte Kandidaten: {len(all_candidates)}")
    print(f"\n📁 Ausgabedatei: {OUTPUT_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    main()
