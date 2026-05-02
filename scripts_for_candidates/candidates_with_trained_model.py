#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PIPELINE fuer trainiertes Modell mit BIRD-Datenbanken
- Datenbank-Name als Kommandozeilen-Argument
- Verwendet train_tables.sql fuer Schema
- Verwendet train.json fuer Fragen
- Generiert 10 Kandidaten pro Frage (max. 20 Fragen pro DB)
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# KONFIGURATION
# ============================================================

BASE_DIR = "/home/akuzg/dmml/axolotl"
MODEL_PATH = "./trained_model"
TEMPERATURE = 0.8
MAX_LENGTH = 2048
CANDIDATES_PER_QUESTION = 10
MAX_QUESTIONS = 20

# ============================================================
# SCHEMA EXTRAKTION AUS TRAIN_TABLES.SQL
# ============================================================

def load_schema_from_bird(db_id: str, schema_file: str) -> str:
    """
    Laedt das Schema fuer eine Datenbank aus der BIRD train_tables.sql Datei
    """
    with open(schema_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Finde die richtige Datenbank
    db_schema_data = None
    for item in data:
        if item.get('db_id') == db_id:
            db_schema_data = item
            break
    
    # Baue lesbares Schema
    schema_parts = []
    
    # Tabellen
    for idx, table_name in enumerate(db_schema_data['table_names']):
        schema_parts.append(f"Table: {table_name}")
        
        # Spalten fuer diese Tabelle
        columns = []
        for col_idx, (table_idx, col_name) in enumerate(db_schema_data['column_names']):
            if table_idx == idx:
                col_type = db_schema_data['column_types'][col_idx]
                columns.append(f"{col_name} ({col_type})")
        
        schema_parts.append(f"Columns: {', '.join(columns)}")
        schema_parts.append("")
    
    # Fremdschluessel
    if db_schema_data.get('foreign_keys'):
        schema_parts.append("Relationships:")
        for fk in db_schema_data['foreign_keys']:
            col_name = db_schema_data['column_names'][fk[0]][1]
            ref_col_name = db_schema_data['column_names'][fk[1]][1]
            
            col_table_idx = db_schema_data['column_names'][fk[0]][0]
            ref_table_idx = db_schema_data['column_names'][fk[1]][0]
            
            col_table = db_schema_data['table_names'][col_table_idx] if col_table_idx >= 0 else "unknown"
            ref_table = db_schema_data['table_names'][ref_table_idx] if ref_table_idx >= 0 else "unknown"
            
            schema_parts.append(f"- {col_table}.{col_name} references {ref_table}.{ref_col_name}")
    
    return '\n'.join(schema_parts)

# ============================================================
# FRAGEN AUS TRAIN.JSON LADEN
# ============================================================

def load_questions_from_bird(db_id: str, questions_file: str, max_questions: int = 20) -> List[Dict]:
    """
    Laedt die Fragen fuer eine Datenbank aus der BIRD train.json Datei
    Behaelt die originale Reihenfolge bei
    """
    with open(questions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filtere Fragen fuer die gewuenschte Datenbank
    db_questions = [item for item in data if item.get('db_id') == db_id]
    
    # Begrenze auf max_questions
    limited_questions = db_questions[:max_questions]
    
    print(f"Gefunden: {len(db_questions)} Fragen fuer {db_id}")
    print(f"Verwende: {len(limited_questions)} Fragen (max. {max_questions})")
    
    return limited_questions

# ============================================================
# MODELL MANAGER MIT CHAIN-OF-THOUGHT PROMPT
# ============================================================

class CoTLLMManager:
    """Manager mit Chain-of-Thought Prompting fuer Text2SQL"""
    
    def __init__(self, model_path: str, db_schema: str, temperature: float = 0.8):
        print(f"\nLade Modell von: {model_path}")
        
        self.db_schema = db_schema
        self.temperature = temperature
        self.device = "cpu"
        
        # Modell laden
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        self.model = self.model.to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Modell geladen. Groesse: {num_params / 1e9:.2f}B Parameter")
    
    def generate_sql_variant(self, question: str, variant_num: int) -> Dict[str, str]:
        """
        Generiert eine SQL-Variante mit Chain-of-Thought Prompting.
        Jede Variante verwendet leicht variierende Temperatur fuer Diversitaet.
        """
        # Temperatur leicht variieren fuer verschiedene Varianten
        temp = min(0.9, self.temperature + (variant_num * 0.02))
        
        prompt = self._build_cot_prompt(question)
        
        # Generieren
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                                max_length=MAX_LENGTH - 500).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_LENGTH,
                temperature=temp,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                        skip_special_tokens=True)
        
        # Extrahiere SQL aus der Antwort
        sql = self._extract_sql(response)
        evidence = self._extract_evidence(response)
        
        return {
            "evidence": evidence,
            "SQL": sql
        }
    
    def _build_cot_prompt(self, question: str) -> str:
        """Baut den Chain-of-Thought Prompt mit Divide and Conquer Strategie"""
        
        prompt = f"""System: As a Text2SQL assistant, your main task is to formulate an SQL query in response to a given natural language inquiry. This process involves a chain-of-thought (CoT) approach, which includes a 'divide and conquer' strategy.

In the 'divide' phase of this CoT process, we break down the presented question into smaller, more manageable sub-problems using pseudo-SQL queries. During the 'conquer' phase, we aggregate the solutions of these sub-problems to form the final response.

Lastly, we refine the constructed query in the optimization step, eliminating any unnecessary clauses and conditions to ensure efficiency.

User: Below I will provide a DB schema and a question that can be answered by querying the provided DB. You will then write out your thought process in detail followed by a single SQL query enclosed in ```sql ... ``` that answers the question.

Database Info: Database Schema: {self.db_schema}

Question: {question}

Main Question: {question}
Analysis: Let me break down this question into smaller sub-problems.

"""
        return prompt
    
    def _extract_sql(self, response: str) -> str:
        """Extrahiert SQL aus der Modellantwort"""
        import re
        
        # Suche nach SQL in ```sql ... ``` Bloecken
        sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        
        # Fallback: Suche nach SELECT statement
        select_match = re.search(r'(SELECT.*?)(?=\n\n|\Z)', response, re.DOTALL | re.IGNORECASE)
        if select_match:
            return select_match.group(1).strip()
        
        return "SELECT * FROM zip_data LIMIT 1"  # Fallback
    
    def _extract_evidence(self, response: str) -> str:
        """Extrahiert die Chain-of-Thought Erklaerung"""
        import re
        sql_match = re.search(r'```sql', response)
        if sql_match:
            return response[:sql_match.start()].strip()
        
        # Fallback: erste 500 Zeichen
        return response[:500].strip()
    
    def batch_generate(self, questions: List[Dict]) -> List[Dict]:
        """Generiert fuer alle Fragen Kandidaten"""
        all_candidates = []
        
        for idx, q in enumerate(questions, 1):
            question_text = q.get('question', '')
            query_id = idx  # Verwende Index als query_id fuer die Reihenfolge
            
            print(f"\n[{idx}/{len(questions)}] Frage {query_id}: {question_text[:80]}...")
            
            for v in range(CANDIDATES_PER_QUESTION):
                print(f"  Generiere Kandidat {v+1}/{CANDIDATES_PER_QUESTION}...", end=" ", flush=True)
                start = time.time()
                
                variant = self.generate_sql_variant(question_text, v)
                
                elapsed = time.time() - start
                print(f"fertig in {elapsed:.1f}s")
                
                candidate = {
                    "candidate_id": f"{query_id}.{v+1:02d}",
                    "db_id": q.get('db_id'),  # Originale db_id aus der Frage
                    "question": question_text,
                    "query_id": query_id,
                    "evidence": variant.get("evidence", ""),
                    "SQL": variant.get("SQL", "")
                }
                all_candidates.append(candidate)
        
        return all_candidates

# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()
    
    # Datenbank-ID aus Kommandozeilen-Argument holen
    if len(sys.argv) < 2:
        print("Usage: python3 generate_candidates.py <db_name>")
        print("Example: python3 generate_candidates.py citeseer")
        print("Available databases: address, citeseer, <and others from BIRD>")
        sys.exit(1)
    
    db_id = sys.argv[1]
    
    print("=" * 70)
    print(f"PIPELINE: Text2SQL mit Chain-of-Thought Prompting")
    print(f"Datenbank: {db_id}")
    print("=" * 70)
    
    # Pfade setzen
    SCHEMA_FILE = os.path.join(BASE_DIR, "data/bird/train/train/train_tables.json")
    QUESTIONS_FILE = os.path.join(BASE_DIR, "data/bird/train/train/train.json")
    OUTPUT_FILE = os.path.join(BASE_DIR, f"{db_id}_candidates_cot.json")
    
    # 1. Schema laden
    print(f"\nLade Schema fuer {db_id}...")
    db_schema = load_schema_from_bird(db_id, SCHEMA_FILE)
    print("Schema geladen")
    
    # 2. Fragen laden
    print(f"\nLade Fragen fuer {db_id}...")
    questions = load_questions_from_bird(db_id, QUESTIONS_FILE, MAX_QUESTIONS)
    
    if not questions:
        print(f"Keine Fragen fuer {db_id} gefunden.")
        sys.exit(1)
    
    print(f"{len(questions)} Fragen geladen (Reihenfolge bleibt erhalten)")
    
    # 3. Modell initialisieren
    print(f"\nInitialisiere Modell...")
    llm = CoTLLMManager(
        model_path=MODEL_PATH,
        db_schema=db_schema,
        temperature=TEMPERATURE
    )
    
    # 4. Kandidaten generieren
    total_candidates = len(questions) * CANDIDATES_PER_QUESTION
    print(f"\nGeneriere {total_candidates} Kandidaten ({len(questions)} Fragen x {CANDIDATES_PER_QUESTION})")
    print("WARNUNG: CPU Betrieb - dies wird ca. 30-60 Minuten dauern")
    
    all_candidates = llm.batch_generate(questions)
    
    # 5. Ergebnisse speichern
    print(f"\nSpeichere {len(all_candidates)} Kandidaten...")
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "metadata": {
            "db_id": db_id,
            "num_questions": len(questions),
            "candidates_per_question": CANDIDATES_PER_QUESTION,
            "total_candidates": len(all_candidates),
            "timestamp": datetime.now().isoformat(),
            "model_path": MODEL_PATH,
            "prompt_type": "chain-of-thought_divide_and_conquer"
        },
        "candidates": all_candidates
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    end_time = time.time()    #Endzeit
    required_time = end_time - start_time
    
    print("\n" + "=" * 70)
    print("PIPELINE ABGESCHLOSSEN")
    print("=" * 70)
    print(f"\nDatenbank: {db_id}")
    print(f"Verarbeitete Fragen: {len(questions)}")
    print(f"Generierte Kandidaten: {len(all_candidates)}")
    print(f"\nAusgabedatei: {OUTPUT_FILE}")
    print("\nFormat: Jeder Kandidat enthaelt:")
    print("  - candidate_id: Frage.Kandidat (z.B. 1.01)")
    print("  - db_id: Originaler Datenbank-Name")
    print("  - query_id: Index der Frage (Reihenfolge)")
    print("  - evidence: Chain-of-Thought Erklaerung")
    print("  - SQL: Die generierte SQL Query")

    
    print("\n" + "="*50)
    print("Benötigte Zeit für Kandidatengenerierung")
    print("="*50)
    print(f"Total execution time: {required_time:.2f} seconds")
    print(f"Total execution time: {required_time/60:.2f} minutes")
    print(f"Total execution time: {required_time/3600:.2f} hours")

if __name__ == "__main__":
    main()
