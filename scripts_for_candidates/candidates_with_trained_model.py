#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANGEPASSTE PIPELINE fuer trainiertes Modell auf Linux-Server (CPU only)
- Verwendet trainiertes Modell aus ./trained_model
- CPU-only Betrieb (GPU nicht nutzbar)
- 20 Fragen mit je 10 Kandidaten
- Behaelt originales Ausgabeformat bei
"""

import sqlite3
import json
import os
import re
import glob
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional
import time

# ============================================================
# NEUER: TrainedLLMManager fuer lokales Modell (CPU only)
# ============================================================

class TrainedLLMManager:
    """
    Manager fuer trainiertes lokales LLM via Transformers
    CPU-only Betrieb (fuer Server ohne nutzbare GPU)
    """
    
    def __init__(self, model_path: str = "./trained_model", temperature: float = 0.8, max_length: int = 2048):
        """
        Initialisiert das trainierte Modell fuer CPU.
        """
        self.model_path = model_path
        self.temperature = temperature
        self.max_length = max_length
        self.system_prompt = self._get_system_prompt()
        
        # CPU only
        self.device = "cpu"
        print(f"Betrieb auf: CPU")
        
        # Modell laden
        self.model, self.tokenizer = self._load_model()
        
    def _load_model(self):
        """
        Laedt das trainierte Modell und den Tokenizer fuer CPU.
        """
        print(f"\nLade trainiertes Modell von: {self.model_path}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Lade Tokenizer
            print("  Lade Tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Setze Padding-Token falls noetig
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Lade Modell fuer CPU (float32 fuer Stabilitaet)
            print("  Lade Modell auf CPU (dies kann etwas dauern)...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            model.eval()  # Inference mode
            model = model.to(self.device)
            
            print("Modell erfolgreich geladen!")
            
            # Zeige Modellgroesse
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  Modellgroesse: {num_params / 1e9:.2f}B Parameter")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"Fehler beim Laden des Modells: {e}")
            raise
    
    def _get_system_prompt(self) -> str:
        """System-Prompt fuer SQL-Generierung (unveraendert)"""
        return """You are an expert SQL query generator for the 'address' database schema.

The database has the following tables:

Table: zip_data (zip_code, households, male_population, female_population, avg_house_value)
Table: country (zip_code, county, city)
Table: congress (cognress_rep_id, party, state, district)
Table: zip_congress (zip_code, district)

Generate valid SQLite queries. Return ONLY valid JSON with a 'candidates' array.
Each candidate must have 'evidence' (Chain-of-Thought) and 'SQL' fields."""
    
    def generate_sql_variants(self, 
                              question: str, 
                              original_sql: str, 
                              original_evidence: str,
                              num_variants: int = 10) -> List[Dict[str, str]]:
        """
        Generiert verschiedene SQL-Varianten fuer eine Frage.
        """
        prompt = self._build_prompt(question, original_sql, original_evidence, num_variants)
        
        try:
            return self._call_model(prompt, num_variants)
        except Exception as e:
            print(f"Generierungsfehler: {e}")
            return self._generate_fallback_variants(question, original_sql, original_evidence, num_variants)
    
    def _call_model(self, prompt: str, num_variants: int) -> List[Dict[str, str]]:
        """
        Ruft das trainierte Modell auf (CPU).
        """
        import torch
        
        # Formatierung mit Chat-Template (falls vorhanden)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Versuche Chat-Template zu verwenden
        try:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            # Fallback: Einfaches Format
            formatted_prompt = f"System: {self.system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - 500  # Platz fuer Antwort
        ).to(self.device)
        
        # Generate
        print(f"  Generiere Antwort (CPU)...", end=" ", flush=True)
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        elapsed = time.time() - start_time
        print(f"fertig in {elapsed:.1f}s")
        
        # Decode
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # JSON extrahieren
        try:
            # Suche nach JSON in der Antwort
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                result = json.loads(json_str)
                candidates = result.get('candidates', [])
                print(f"  {len(candidates)} Kandidaten erhalten")
                return candidates[:num_variants]
        except Exception as e:
            print(f"  JSON-Parsing Fehler: {e}")
            print(f"  Antwort: {response[:200]}...")
        
        return []
    
    def _generate_fallback_variants(self, question, original_sql, original_evidence, num_variants):
        """Fallback-Varianten (unveraendert)"""
        variants = []
        techniques = [
            {
                "name": "Direct JOIN",
                "evidence": f"Chain-of-Thought: Using INNER JOIN to combine tables and filter by county.",
                "sql": original_sql
            },
            {
                "name": "Subquery",
                "evidence": f"Chain-of-Thought: First find zip codes in the county, then aggregate.",
                "sql": original_sql.replace("INNER JOIN", "WHERE zip_code IN (SELECT zip_code FROM")
            },
            {
                "name": "CTE",
                "evidence": f"Chain-of-Thought: Break down into Common Table Expression for clarity.",
                "sql": f"WITH target_zips AS (SELECT zip_code FROM country WHERE county = 'ARECIBO') SELECT SUM(households) FROM zip_data WHERE zip_code IN (SELECT zip_code FROM target_zips)"
            }
        ]
        
        for i in range(min(num_variants, len(techniques))):
            tech = techniques[i]
            variants.append({
                "evidence": tech["evidence"],
                "SQL": tech["sql"]
            })
        
        while len(variants) < num_variants:
            variants.extend(variants[:num_variants - len(variants)])
        
        return variants[:num_variants]
    
    def _build_prompt(self, question, original_sql, original_evidence, num_variants):
        """Baut den Prompt fuer das LLM"""
        return f"""Generate {num_variants} different SQL queries for this question:

Question: "{question}"

Original SQL example: {original_sql}

Return a JSON object with a 'candidates' array containing {num_variants} objects.
Each object must have:
- "evidence": detailed chain of thought explanation
- "SQL": the SQL query

Example format:
{{
  "candidates": [
    {{
      "evidence": "First, we need to find...",
      "SQL": "SELECT ..."
    }}
  ]
}}

Generate {num_variants} different approaches now:"""
    
    def batch_generate(self, questions_data, candidates_per_question=10, delay=1.0):
        """Batch-Generierung fuer mehrere Fragen"""
        all_candidates = []
        
        for idx, item in enumerate(questions_data, 1):
            query_id = item.get('query_id', idx)
            print(f"\n  [{idx}/{len(questions_data)}] Frage {query_id}: {item.get('question', '')[:80]}...")
            
            variants = self.generate_sql_variants(
                question=item.get('question', ''),
                original_sql=item.get('SQL', ''),
                original_evidence=item.get('evidence', ''),
                num_variants=candidates_per_question
            )
            
            for i, variant in enumerate(variants[:candidates_per_question]):
                candidate = {
                    "candidate_id": f"{query_id}.{i+1:02d}",  # Format: 1.01, 1.02
                    "db_id": "address",
                    "question": item.get('question', ''),
                    "query_id": query_id,
                    "evidence": variant.get('evidence', item.get('evidence', '')),
                    "SQL": variant.get('SQL', item.get('SQL', ''))
                }
                all_candidates.append(candidate)
            
            print(f"    {len(variants)} Kandidaten generiert")
            
            # Kurze Pause zwischen Anfragen (vermeidet Ueberlastung)
            if delay > 0 and idx < len(questions_data):
                time.sleep(delay)
        
        return all_candidates


# ============================================================
# LINUX-SERVER PFADE (ANZUPASSEN)
# ============================================================

# PFADE FUER LINUX-SERVER - BITTE ANPASSEN
BASE_DIR = "/home/akuzg/dmml/axolotl/second_round"  # Aktuelles Verzeichnis
DB_PATH = os.path.join(BASE_DIR, "data/BIRD/train/train_databases/train_databases/address/address.sqlite")
SQL_FILE_PATH = os.path.join(BASE_DIR, "data/BIRD/train/train/train_gold.sql")
GOLD_QUERIES_DIR = os.path.join(BASE_DIR, "data/BIRD/train/train/gold_queries_address")
QUESTIONS_DIR = os.path.join(BASE_DIR, "data/BIRD/train/train/Address Table Candidates")
OUTPUT_FILE = os.path.join(BASE_DIR, "address_candidates_trained_model.json")

# Modell-Pfad (relativ zu diesem Skript)
MODEL_PATH = "./trained_model"

# Anzahl der Fragen und Kandidaten (angepasst: 20 Fragen, 10 Kandidaten)
NUM_QUESTIONS = 20
CANDIDATES_PER_QUESTION = 10

# Zeilenbereich fuer address Queries in train_gold.sql (unveraendert)
START_LINE = 5083
END_LINE = 5323

# ============================================================
# REST DER PIPELINE (unveraendert)
# ============================================================

def extract_queries_from_file(file_path, start_line, end_line):
    """Extrahiert SQL-Abfragen aus train_gold.sql."""
    queries = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        selected_lines = lines[start_line-1:end_line]
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
    """Fuehrt eine einzelne SQL-Abfrage aus."""
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
    """Fuehrt Schritt 1 aus: SQL Extraktion und Ausfuehrung."""
    print("\n" + "=" * 70)
    print("SCHRITT 1: SQL Extraktion und Ausfuehrung")
    print("=" * 70)
    if not os.path.exists(DB_PATH):
        print(f"Datenbank nicht gefunden: {DB_PATH}")
        return None
    if not os.path.exists(SQL_FILE_PATH):
        print(f"SQL-Datei nicht gefunden: {SQL_FILE_PATH}")
        return None
    print(f"\nExtrahiere Queries von Zeile {START_LINE} bis {END_LINE}...")
    queries = extract_queries_from_file(SQL_FILE_PATH, START_LINE, END_LINE)
    if not queries:
        print("Keine Queries gefunden.")
        return None
    print(f"{len(queries)} Queries gefunden.")
    print(f"\nFuehre {len(queries)} Queries aus...")
    results = []
    for i, query_info in enumerate(queries, 1):
        print(f"  Query {i}/{len(queries)} (Zeile {query_info['line_number']})...", end=" ")
        result = execute_query(DB_PATH, query_info['clean_query'], i)
        result['line_number'] = query_info['line_number']
        result['original'] = query_info['original']
        result['clean_query'] = query_info['clean_query']
        results.append(result)
        if result['success']:
            print(f"{result['row_count']} Zeilen")
        else:
            print(f"{result['error'][:50]}")
    print(f"\nSpeichere Ergebnisse...")
    output_file, summary = save_results_to_file(results, GOLD_QUERIES_DIR)
    print(f"\nSchritt 1 abgeschlossen!")
    print(f"   Erfolgreich: {summary['successful']}/{summary['failed']+summary['successful']}")
    print(f"   Ergebnisdatei: {output_file}")
    return output_file

def normalize_sql(sql):
    """Normalisiert SQL fuer Vergleich."""
    if not sql:
        return ""
    sql = str(sql)
    sql = re.sub(r"['\"]?\s*address\s*['\"]?$", "", sql, flags=re.IGNORECASE)
    return ' '.join(sql.lower().split()).replace('"', '').replace("'", "")

def load_results_file(filepath):
    """Laedt die Ergebnisdatei aus Schritt 1."""
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
    """Findet die Eingabedatei mit den Fragen."""
    json_files = glob.glob(os.path.join(QUESTIONS_DIR, "*.json"))
    json_files = [f for f in json_files if "address_from_OG_with_ids.json" not in f]
    if not json_files:
        return None
    print("\nGefundene JSON-Dateien:")
    for i, file in enumerate(json_files, 1):
        print(f"  {i}. {os.path.basename(file)}")
    return json_files[0]

def load_questions(filepath, num_questions=20):
    """Laedt die Fragen aus der JSON-Datei."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data[:num_questions]
    else:
        return [data]

def add_query_ids_to_questions(questions, results_queries):
    """Fuegt query_ids zu den Fragen hinzu."""
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
    """Fuehrt Schritt 2 aus: Query IDs zu Fragen hinzufuegen."""
    print("\n" + "=" * 70)
    print("SCHRITT 2: Query IDs zu Fragen hinzufuegen")
    print("=" * 70)
    print(f"\nLade Ergebnisdatei: {os.path.basename(results_file)}")
    results_queries = load_results_file(results_file)
    if not results_queries:
        print("Keine Queries gefunden.")
        return None, None
    questions_file = find_input_questions()
    if not questions_file:
        print(f"Keine Fragen-JSON-Dateien gefunden in: {QUESTIONS_DIR}")
        return None, None
    print(f"\nVerwende Fragen-Datei: {os.path.basename(questions_file)}")
    print(f"\nLade Fragen...")
    questions = load_questions(questions_file, NUM_QUESTIONS)
    print(f"{len(questions)} Fragen geladen")
    print(f"\nFuege Query IDs hinzu...")
    questions_with_ids, matched = add_query_ids_to_questions(questions, results_queries)
    print(f"{matched}/{len(questions)} Fragen zugeordnet ({matched/len(questions)*100:.1f}%)")
    missing_ids = [i for i, q in enumerate(questions_with_ids) if 'query_id' not in q]
    if missing_ids:
        print(f"\nWarnung: {len(missing_ids)} Fragen haben keine query_id:")
        for i in missing_ids[:5]:
            print(f"   - Frage {i+1}: {questions_with_ids[i].get('question', 'Keine Frage')[:50]}...")
    return questions_with_ids, questions_file

def run_candidate_generation(questions_with_ids, questions_file):
    """Fuehrt Schritt 3 aus: Kandidaten mit trainiertem Modell generieren."""
    print("\n" + "=" * 70)
    print("SCHRITT 3: Kandidaten mit trainiertem Modell generieren")
    print("=" * 70)
    
    print("\nInitialisiere TrainedLLMManager...")
    try:
        llm = TrainedLLMManager(
            model_path=MODEL_PATH,
            temperature=0.8,
            max_length=2048
        )
    except Exception as e:
        print(f"Fehler: {e}")
        print("\nMoegliche Loesungen:")
        print("1. Pruefen ob torch installiert: pip install torch transformers")
        print("2. Pruefen ob Modell existiert: ls -la ./trained_model")
        return False
    
    total_candidates = len(questions_with_ids) * CANDIDATES_PER_QUESTION
    print(f"\nStarte Generierung von {total_candidates} Kandidaten...")
    print(f"   (CPU Betrieb - dies wird ca. 30-60 Minuten dauern)")
    
    all_candidates = llm.batch_generate(
        questions_with_ids,
        candidates_per_question=CANDIDATES_PER_QUESTION,
        delay=0.5
    )
    
    # Ergebnisse speichern
    print(f"\nSpeichere {len(all_candidates)} Kandidaten...")
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "candidates": all_candidates
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Kandidaten gespeichert in: {OUTPUT_FILE}")
    return True

def main():
    print("=" * 70)
    print("PIPELINE mit trainiertem Modell auf Linux-Server (CPU)")
    print("=" * 70)
    print(f"\nArbeitsverzeichnis: {BASE_DIR}")
    print(f"Modell-Pfad: {MODEL_PATH}")
    print(f"Verarbeite {NUM_QUESTIONS} Fragen mit {CANDIDATES_PER_QUESTION} Kandidaten pro Frage")
    print(f"Ausgabedatei: {OUTPUT_FILE}")
    
    # Schritt 1: SQL Extraktion und Ausfuehrung
    results_file = run_sql_extraction()
    if not results_file:
        print("\nPipeline abgebrochen bei Schritt 1.")
        return
    
    # Schritt 2: Query IDs zu Fragen hinzufuegen
    questions_with_ids, questions_file = run_query_id_assignment(results_file)
    if not questions_with_ids:
        print("\nPipeline abgebrochen bei Schritt 2.")
        return
    
    # Schritt 3: Kandidaten generieren
    success = run_candidate_generation(questions_with_ids, questions_file)
    
    # Abschluss
    print("\n" + "=" * 70)
    if success:
        print("PIPELINE VOLLSTAENDIG ABGESCHLOSSEN")
        print("=" * 70)
        print(f"\nZusammenfassung:")
        print(f"   Schritt 1: SQL Ergebnisse gespeichert")
        print(f"   Schritt 2: {len(questions_with_ids)} Fragen mit Query IDs versehen")
        print(f"   Schritt 3: {len(questions_with_ids) * CANDIDATES_PER_QUESTION} Kandidaten generiert")
        print(f"\nEndgueltige Ausgabedatei: {OUTPUT_FILE}")
    else:
        print("PIPELINE ABGEBROCHEN - Schritt 3 fehlgeschlagen")
    print("=" * 70)

if __name__ == "__main__":
    main()
