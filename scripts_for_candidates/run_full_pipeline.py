#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VOLLSTÄNDIGE PIPELINE - Alle drei Schritte in einem Skript:

1. Extrahiert SQL-Abfragen aus train_gold.sql und führt sie auf address.sqlite aus
2. Fügt query_ids zu den Fragen hinzu
3. Generiert Kandidaten mit lokalem Qwen2.5 via Ollama
4. Fügt candidate_ids im Format "X.YY" hinzu (z.B. 1.01, 1.02)
"""

import sqlite3
import json
import os
import re
import glob
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from llm_manager import LLMManager

# ============================================================
# KONFIGURATION
# ============================================================

# Pfade für Schritt 1: SQL Extraktion
DB_PATH = r"C:\Users\Norita\DMML Datamining Projekt Julia\data\BIRD\train\train\train_databases\train_databases\address\address.sqlite"
SQL_FILE_PATH = r"C:\Users\Norita\DMML Datamining Projekt Julia\data\BIRD\train\train\train_gold.sql"
GOLD_QUERIES_DIR = r"C:\Users\Norita\DMML Datamining Projekt Julia\data\BIRD\train\train\gold_queries_address"

# Zeilenbereich für address Queries in train_gold.sql
START_LINE = 5083
END_LINE = 5323

# Pfade für Schritt 2 & 3: Questions und Candidates
BASE_DIR = r"C:\Users\Norita\DMML Datamining Projekt Julia\data\BIRD\train\train"
QUESTIONS_DIR = os.path.join(BASE_DIR, "Address Table Candidates")
OUTPUT_FILE = os.path.join(BASE_DIR, "address.json")  # Geändert von candidates_500_qwen.json

# Anzahl der Fragen und Kandidaten pro Frage
NUM_QUESTIONS = 50
CANDIDATES_PER_QUESTION = 5  # Wir haben damals 10 genommen, hat aber viel zu lang gedauert

# ============================================================
# SCHRITT 1: SQL Queries extrahieren und ausführen
# ============================================================

def extract_queries_from_file(file_path, start_line, end_line):
    """Extrahiert SQL-Abfragen aus train_gold.sql."""
    queries = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        selected_lines = lines[start_line-1:end_line]
        current_query = []
        line_counter = start_line

        for line in selected_lines:
            line = line.strip()

            if not line:
                line_counter += 1
                continue

            if line.endswith('address'):
                clean_line = line[:-7].strip()
                if clean_line:
                    queries.append({
                        'line_number': line_counter,
                        'original': line,
                        'clean_query': clean_line,
                        'full_text': line
                    })
            elif 'address' in line and not line.startswith('--'):
                parts = line.rsplit('address', 1)
                if len(parts) > 1:
                    clean_line = parts[0].strip()
                    if clean_line:
                        queries.append({
                            'line_number': line_counter,
                            'original': line,
                            'clean_query': clean_line,
                            'full_text': line
                        })

            line_counter += 1

        return queries

    except Exception as e:
        print(f"Fehler beim Lesen der Datei: {e}")
        return []

def execute_query(db_path, query, query_id):
    """Führt eine einzelne SQL-Abfrage aus."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(query)
        rows = cursor.fetchall()
        result_list = [dict(row) for row in rows]

        conn.close()

        return {
            'success': True,
            'query_id': query_id,
            'row_count': len(result_list),
            'data': result_list,
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'query_id': query_id,
            'row_count': 0,
            'data': [],
            'error': str(e)
        }

def save_results_to_file(results, output_dir):
    """Speichert die SQL-Ergebnisse in einer JSON-Datei."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"all_queries_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    all_results = {
        'summary': {
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success'])
        },
        'queries': []
    }

    for i, result in enumerate(results, 1):
        query_entry = {
            'query_id': i,
            'line_number': result.get('line_number', 'unknown'),
            'original_query': result.get('original', ''),
            'clean_query': result.get('clean_query', ''),
            'execution_result': {
                'success': result.get('success', False),
                'row_count': result.get('row_count', 0),
                'error': result.get('error', None),
                'data': result.get('data', [])
            }
        }
        all_results['queries'].append(query_entry)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    return filepath, all_results['summary']

def run_sql_extraction():
    """Führt Schritt 1 aus: SQL Extraktion und Ausführung."""
    print("\n" + "=" * 70)
    print("SCHRITT 1: SQL Extraktion und Ausführung")
    print("=" * 70)

    # Prüfen ob Dateien existieren
    if not os.path.exists(DB_PATH):
        print(f"❌ Datenbank nicht gefunden: {DB_PATH}")
        return None

    if not os.path.exists(SQL_FILE_PATH):
        print(f"❌ SQL-Datei nicht gefunden: {SQL_FILE_PATH}")
        return None

    # Queries extrahieren
    print(f"\n📄 Extrahiere Queries von Zeile {START_LINE} bis {END_LINE}...")
    queries = extract_queries_from_file(SQL_FILE_PATH, START_LINE, END_LINE)

    if not queries:
        print("❌ Keine Queries gefunden.")
        return None

    print(f"✅ {len(queries)} Queries gefunden.")

    # Queries ausführen
    print(f"\n🚀 Führe {len(queries)} Queries aus...")
    results = []

    for i, query_info in enumerate(queries, 1):
        print(f"  Query {i}/{len(queries)} (Zeile {query_info['line_number']})...", end=" ")

        result = execute_query(DB_PATH, query_info['clean_query'], i)
        result['line_number'] = query_info['line_number']
        result['original'] = query_info['original']
        result['clean_query'] = query_info['clean_query']

        results.append(result)

        if result['success']:
            print(f"✅ {result['row_count']} Zeilen")
        else:
            print(f"❌ {result['error'][:50]}")

    # Ergebnisse speichern
    print(f"\n💾 Speichere Ergebnisse...")
    output_file, summary = save_results_to_file(results, GOLD_QUERIES_DIR)

    print(f"\n✅ Schritt 1 abgeschlossen!")
    print(f"   Erfolgreich: {summary['successful']}/{summary['failed']+summary['successful']}")
    print(f"   Ergebnisdatei: {output_file}")

    return output_file

# ============================================================
# SCHRITT 2: Query IDs zu Fragen hinzufügen
# ============================================================

def normalize_sql(sql):
    """Normalisiert SQL für Vergleich."""
    if not sql:
        return ""
    sql = str(sql)
    sql = re.sub(r"['\"]?\s*address\s*['\"]?$", "", sql, flags=re.IGNORECASE)
    return ' '.join(sql.lower().split()).replace('"', '').replace("'", "")

def load_results_file(filepath):
    """Lädt die Ergebnisdatei aus Schritt 1."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    queries = []
    for q in data.get('queries', []):
        queries.append({
            'query_id': q.get('query_id'),
            'clean_query': q.get('clean_query', '')
        })

    print(f"  Gefunden: {len(queries)} Queries")
    return queries

def find_input_questions():
    """Findet die Eingabedatei mit den Fragen (überspringt address_from_OG_with_ids.json)."""
    json_files = glob.glob(os.path.join(QUESTIONS_DIR, "*.json"))
    json_files = [f for f in json_files if "address_from_OG_with_ids.json" not in f]

    if not json_files:
        return None

    print("\nGefundene JSON-Dateien:")
    for i, file in enumerate(json_files, 1):
        print(f"  {i}. {os.path.basename(file)}")

    return json_files[0]

def load_questions(filepath, num_questions=50):
    """Lädt die Fragen aus der JSON-Datei."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data[:num_questions]
    else:
        return [data]

def add_query_ids_to_questions(questions, results_queries):
    """Fügt query_ids zu den Fragen hinzu."""
    matched = 0
    for q in questions:
        if 'SQL' not in q:
            continue

        q_sql = normalize_sql(q['SQL'])
        best_match = None
        best_sim = 0

        for rq in results_queries:
            r_sql = normalize_sql(rq['clean_query'])
            sim = SequenceMatcher(None, q_sql, r_sql).ratio()
            if sim > best_sim:
                best_sim = sim
                best_match = rq

        if best_match and best_sim >= 0.8:
            q['query_id'] = best_match['query_id']
            matched += 1

    return questions, matched

def run_query_id_assignment(results_file):
    """Führt Schritt 2 aus: Query IDs zu Fragen hinzufügen."""
    print("\n" + "=" * 70)
    print("SCHRITT 2: Query IDs zu Fragen hinzufügen")
    print("=" * 70)

    # Ergebnisse aus Schritt 1 laden
    print(f"\n📂 Lade Ergebnisdatei: {os.path.basename(results_file)}")
    results_queries = load_results_file(results_file)

    if not results_queries:
        print("❌ Keine Queries gefunden.")
        return None, None

    # Fragen-Datei finden
    questions_file = find_input_questions()

    if not questions_file:
        print(f"❌ Keine Fragen-JSON-Dateien gefunden in: {QUESTIONS_DIR}")
        return None, None

    print(f"\n✅ Verwende Fragen-Datei: {os.path.basename(questions_file)}")

    # Fragen laden
    print(f"\n📚 Lade Fragen...")
    questions = load_questions(questions_file, NUM_QUESTIONS)
    print(f"✅ {len(questions)} Fragen geladen")

    # Query IDs hinzufügen
    print(f"\n🔍 Füge Query IDs hinzu...")
    questions_with_ids, matched = add_query_ids_to_questions(questions, results_queries)
    print(f"✅ {matched}/{len(questions)} Fragen zugeordnet ({matched/len(questions)*100:.1f}%)")

    # Prüfen ob alle Fragen eine query_id haben
    missing_ids = [i for i, q in enumerate(questions_with_ids) if 'query_id' not in q]
    if missing_ids:
        print(f"\n⚠️ Warnung: {len(missing_ids)} Fragen haben keine query_id:")
        for i in missing_ids[:5]:
            print(f"   - Frage {i+1}: {questions_with_ids[i].get('question', 'Keine Frage')[:50]}...")

    return questions_with_ids, questions_file

# ============================================================
# SCHRITT 3: Kandidaten generieren mit Qwen2.5
# ============================================================

def add_candidate_ids(candidates, query_id):
    """Fügt candidate_ids im Format 'X.YY' hinzu."""
    for idx, candidate in enumerate(candidates, 1):
        if idx < 10:
            candidate_id = f"{query_id}.0{idx}"
        else:
            candidate_id = f"{query_id}.{idx}"
        candidate['candidate_id'] = candidate_id
    return candidates

def run_candidate_generation(questions_with_ids, questions_file):
    """Führt Schritt 3 aus: Kandidaten generieren."""
    print("\n" + "=" * 70)
    print("SCHRITT 3: Kandidaten mit Qwen2.5 generieren")
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
        return False

    # Kandidaten generieren
    total_candidates = len(questions_with_ids) * CANDIDATES_PER_QUESTION
    print(f"\n🚀 Starte Generierung von {total_candidates} Kandidaten...")

    all_candidates = llm.batch_generate(
        questions_with_ids,
        candidates_per_question=CANDIDATES_PER_QUESTION,
        delay=2.0
    )

    # Candidate IDs im Format "X.YY" hinzufügen
    print(f"\n🏷️ Füge Candidate IDs hinzu (Format: X.YY)...")
    current_idx = 0
    for q_idx, question in enumerate(questions_with_ids, 1):
        query_id = question.get('query_id', q_idx)
        question_candidates = all_candidates[current_idx:current_idx + CANDIDATES_PER_QUESTION]
        add_candidate_ids(question_candidates, query_id)
        current_idx += CANDIDATES_PER_QUESTION

    # Ergebnisse speichern
    print(f"\n💾 Speichere {len(all_candidates)} Kandidaten...")

    output_data = {
        "candidates": all_candidates
    }

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return True

# ============================================================
# MAIN: Alle drei Schritte ausführen
# ============================================================

def main():
    print("=" * 70)
    print("VOLLSTÄNDIGE PIPELINE - Alle drei Schritte")
    print("=" * 70)
    print(f"\n📁 Arbeitsverzeichnis: {BASE_DIR}")
    print(f"📊 Verarbeite {NUM_QUESTIONS} Fragen mit {CANDIDATES_PER_QUESTION} Kandidaten pro Frage")
    print(f"📁 Ausgabedatei: {OUTPUT_FILE}")

    # Schritt 1: SQL Extraktion und Ausführung
    results_file = run_sql_extraction()
    if not results_file:
        print("\n❌ Pipeline abgebrochen bei Schritt 1.")
        return

    # Schritt 2: Query IDs zu Fragen hinzufügen
    questions_with_ids, questions_file = run_query_id_assignment(results_file)
    if not questions_with_ids:
        print("\n❌ Pipeline abgebrochen bei Schritt 2.")
        return

    # Schritt 3: Kandidaten generieren
    success = run_candidate_generation(questions_with_ids, questions_file)

    # Abschluss
    print("\n" + "=" * 70)
    if success:
        print("✅ PIPELINE VOLLSTÄNDIG ABGESCHLOSSEN!")
        print("=" * 70)
        print(f"\n📊 Zusammenfassung:")
        print(f"   Schritt 1: SQL Ergebnisse gespeichert in {GOLD_QUERIES_DIR}")
        print(f"   Schritt 2: {len(questions_with_ids)} Fragen mit Query IDs versehen")
        print(f"   Schritt 3: {len(questions_with_ids) * CANDIDATES_PER_QUESTION} Kandidaten generiert")
        print(f"\n📁 Endgültige Ausgabedatei: {OUTPUT_FILE}")
    else:
        print("❌ PIPELINE ABGEBROCHEN - Schritt 3 fehlgeschlagen")
    print("=" * 70)

if __name__ == "__main__":
    main()

