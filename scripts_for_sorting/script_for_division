import json
import sqlite3
import os
import argparse
from pathlib import Path

def load_gold_standard(gold_file_path, target_db=None):
    """
    Lädt die Gold-Standard SQL-Datei.
    Format pro Zeile: SQL-Query \t Datenbankname
    """
    gold_queries = []
    with open(gold_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Annahme: Tabulator-getrennt (letztes Feld ist der Datenbankname)
                parts = line.rsplit('\t', 1)
                if len(parts) == 2:
                    sql_query, db_name = parts
                    # Optional: Nur Queries für bestimmte Datenbank laden
                    if target_db is None or db_name == target_db:
                        gold_queries.append({
                            'sql': sql_query,
                            'db': db_name
                        })
                else:
                    # Fallback: Durch Leerzeichen trennen (wie ursprünglich angenommen)
                    parts = line.rsplit(' ', 1)
                    if len(parts) == 2:
                        sql_query, db_name = parts
                        if target_db is None or db_name == target_db:
                            gold_queries.append({
                                'sql': sql_query,
                                'db': db_name
                            })
    return gold_queries

def execute_query(db_path, sql_query):
    """
    Führt eine SQL-Abfrage auf einer SQLite-Datenbank aus und gibt das Ergebnis zurück.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        print(f"Fehler bei der Ausführung der Query: {e}")
        print(f"Query: {sql_query}")
        return None

def compare_results(result1, result2):
    """
    Vergleicht zwei Abfrageergebnisse.
    """
    if result1 is None or result2 is None:
        return False
    
    # Konvertiere zu vergleichbarem Format
    set1 = set(tuple(row) for row in result1)
    set2 = set(tuple(row) for row in result2)
    
    return set1 == set2

def evaluate_candidates(json_file_path, db_dir_path, gold_file_path, output_file_path):
    """
    Hauptfunktion zur Bewertung der Kandidaten.
    """
    # Lade JSON mit Kandidaten
    with open(json_file_path, 'r', encoding='utf-8') as f:
        candidates = json.load(f)
    print(f"Kandidaten: {len(candidates)} Einträge geladen")
    
    if not candidates:
        print("Keine Kandidaten gefunden!")
        return
    
    # Extrahiere Datenbankname aus erstem Eintrag
    current_db_id = candidates[0].get('db_id')
    if not current_db_id:
        print("Keine db_id in den Kandidaten gefunden!")
        return
    
    print(f"Datenbank: {current_db_id}")
    
    # Lade Gold-Standard (nur für diese Datenbank)
    gold_queries = load_gold_standard(gold_file_path, current_db_id)
    print(f"Gold-Standard: {len(gold_queries)} Queries für {current_db_id} geladen")
    
    # Pfad zur Datenbank
    db_path = os.path.join(db_dir_path, f"{current_db_id}.sqlite")
    if not os.path.exists(db_path):
        print(f"WARNUNG: Datenbank nicht gefunden: {db_path}")
        print("Fahre fort, aber Ausführung wird fehlschlagen...")
    
    # Gruppiere Kandidaten nach Fragen
    question_groups = []
    current_group = []
    current_question = None
    
    for candidate in candidates:
        question = candidate.get('question')
        if question != current_question:
            if current_group:
                question_groups.append(current_group)
            current_group = [candidate]
            current_question = question
        else:
            current_group.append(candidate)
    
    if current_group:
        question_groups.append(current_group)
    
    print(f"Gefundene Fragegruppen: {len(question_groups)}")
    
    # Stelle sicher, dass wir genug Gold-Queries haben
    if len(question_groups) != len(gold_queries):
        print(f"WARNUNG: Anzahl der Fragen ({len(question_groups)}) stimmt nicht mit Gold-Standard ({len(gold_queries)}) überein!")
        print("Verwende die kleinere Anzahl für den Vergleich.")
        min_len = min(len(question_groups), len(gold_queries))
        question_groups = question_groups[:min_len]
        gold_queries = gold_queries[:min_len]
    
    results = []
    
    # Bewerte jede Gruppe
    for idx, group in enumerate(question_groups):
        gold_query = gold_queries[idx]
        
        print(f"\nVerarbeite Frage {idx + 1}: {group[0]['question'][:50]}...")
        
        # Führe Gold-Query aus
        gold_result = execute_query(db_path, gold_query['sql'])
        
        if gold_result is None:
            print(f"  Gold-Query fehlgeschlagen. Überspringe Gruppe.")
            for candidate in group:
                results.append({
                    'candidate_id': candidate['candidate_id'],
                    'db_id': candidate['db_id'],
                    'question': candidate['question'],
                    'correct': False,
                    'error': 'Gold query failed'
                })
            continue
        
        print(f"  Gold-Result: {gold_result[:3] if len(gold_result) > 3 else gold_result}...")
        
        # Bewerte jeden Kandidaten in der Gruppe
        for candidate in group:
            candidate_result = execute_query(db_path, candidate['SQL'])
            is_correct = compare_results(gold_result, candidate_result)
            
            results.append({
                'candidate_id': candidate['candidate_id'],
                'db_id': candidate['db_id'],
                'question': candidate['question'][:100],  # Kürzen für bessere Lesbarkeit
                'correct': is_correct
            })
            
            print(f"    Kandidat {candidate['candidate_id']}: {'✓ Korrekt' if is_correct else '✗ Falsch'}")
    
    # Speichere Ergebnisse
    output_data = []
    for result in results:
        output_data.append({
            'candidate_id': result['candidate_id'],
            'db_id': result['db_id'],
            'correct': result['correct']
        })
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Statistik
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    print(f"\n{'='*50}")
    print(f"Ergebnisse gespeichert in: {output_file_path}")
    print(f"Total: {total}, Korrekt: {correct}, Falsch: {total-correct}")
    print(f"Genauigkeit: {correct/total*100:.2f}%")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Bewertung von SQL-Kandidaten gegen Gold-Standard')
    parser.add_argument('json_file', help='Pfad zur JSON-Datei mit Kandidaten')
    parser.add_argument('db_dir', help='Pfad zum Verzeichnis mit den SQLite-Datenbanken')
    parser.add_argument('gold_file', help='Pfad zur Gold-Standard SQL-Datei')
    parser.add_argument('-o', '--output', default='evaluation_results.json', 
                       help='Ausgabedatei (Standard: evaluation_results.json)')
    
    args = parser.parse_args()
    
    # Prüfe, ob Dateien existieren
    if not os.path.exists(args.json_file):
        print(f"Fehler: JSON-Datei nicht gefunden: {args.json_file}")
        return
    
    if not os.path.exists(args.gold_file):
        print(f"Fehler: Gold-Standard-Datei nicht gefunden: {args.gold_file}")
        return
    
    if not os.path.exists(args.db_dir):
        print(f"Fehler: Datenbankverzeichnis nicht gefunden: {args.db_dir}")
        return
    
    # Führe Bewertung durch
    evaluate_candidates(args.json_file, args.db_dir, args.gold_file, args.output)

if __name__ == "__main__":
    main()
