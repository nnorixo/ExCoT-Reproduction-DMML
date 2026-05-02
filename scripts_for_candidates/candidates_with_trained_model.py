#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EINFACHE PIPELINE fuer trainiertes Modell auf Linux-Server (CPU only)
- Nur Kandidatengenerierung aus Text-Fragen
- Verwendet Chain-of-Thought Prompting mit Divide and Conquer
- Keine Verwendung von train_gold.sql
"""

import json
import os
import glob
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# KONFIGURATION
# ============================================================

BASE_DIR = "/home/akuzg/dmml/axolotl/second_round"
QUESTIONS_DIR = os.path.join(BASE_DIR, "data/BIRD/train/train/Address Table Candidates")
OUTPUT_FILE = os.path.join(BASE_DIR, "address_candidates_cot.json")
MODEL_PATH = "./trained_model"

NUM_QUESTIONS = 20
CANDIDATES_PER_QUESTION = 10
TEMPERATURE = 0.8
MAX_LENGTH = 2048

# ============================================================
# MODELL MANAGER MIT CHAIN-OF-THOUGHT PROMPT
# ============================================================

class CoTLLMManager:
    """Manager mit Chain-of-Thought Prompting fuer Text2SQL"""
    
    # Database Schema fuer address Datenbank
    DB_SCHEMA = """
Table: zip_data
Columns: zip_code, households, male_population, female_population, avg_house_value

Table: country  
Columns: zip_code, county, city

Table: congress
Columns: cognress_rep_id, party, state, district

Table: zip_congress
Columns: zip_code, district

Relationships:
- country.zip_code references zip_data.zip_code
- zip_congress.zip_code references zip_data.zip_code
- congress.district relates to zip_congress.district
"""
    
    def __init__(self, model_path: str = "./trained_model", temperature: float = 0.8):
        print(f"\nLade Modell von: {model_path}")
        
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
        
        print(f"Modell geladen. Groesse: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B Parameter")
    
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

Database Info: Database Schema: {self.DB_SCHEMA}

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
        # Nimm den gesamten Text vor dem SQL
        import re
        sql_match = re.search(r'```sql', response)
        if sql_match:
            return response[:sql_match.start()].strip()
        
        # Fallback: erste 500 Zeichen
        return response[:500].strip()
    
    def batch_generate(self, questions: List[Dict], candidates_per_question: int = 10) -> List[Dict]:
        """Generiert fuer alle Fragen Kandidaten"""
        all_candidates = []
        
        for idx, q in enumerate(questions, 1):
            question_text = q.get('question', '')
            query_id = q.get('query_id', idx)
            
            print(f"\n[{idx}/{len(questions)}] Frage {query_id}: {question_text[:80]}...")
            
            for v in range(candidates_per_question):
                print(f"  Generiere Kandidat {v+1}/{candidates_per_question}...", end=" ", flush=True)
                start = time.time()
                
                variant = self.generate_sql_variant(question_text, v)
                
                elapsed = time.time() - start
                print(f"fertig in {elapsed:.1f}s")
                
                candidate = {
                    "candidate_id": f"{query_id}.{v+1:02d}",
                    "db_id": "address",
                    "question": question_text,
                    "query_id": query_id,
                    "evidence": variant.get("evidence", ""),
                    "SQL": variant.get("SQL", "")
                }
                all_candidates.append(candidate)
            
            # Pause zwischen Fragen
            if idx < len(questions):
                time.sleep(0.5)
        
        return all_candidates

# ============================================================
# DATEN LADEN
# ============================================================

def load_questions(num_questions: int = 20) -> List[Dict]:
    """Laedt die Fragen aus der JSON-Datei"""
    json_files = glob.glob(os.path.join(QUESTIONS_DIR, "*.json"))
    json_files = [f for f in json_files if "address_from_OG_with_ids.json" not in f]
    
    if not json_files:
        raise FileNotFoundError(f"Keine JSON-Dateien in {QUESTIONS_DIR} gefunden")
    
    questions_file = json_files[0]
    print(f"Lade Fragen von: {os.path.basename(questions_file)}")
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data[:num_questions]
    else:
        return [data]

# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()
    
    print("=" * 70)
    print("PIPELINE: Text2SQL mit Chain-of-Thought Prompting")
    print("=" * 70)
    
    # Fragen laden
    print(f"\nLade {NUM_QUESTIONS} Fragen...")
    questions = load_questions(NUM_QUESTIONS)
    print(f"{len(questions)} Fragen geladen")
    
    # Modell initialisieren
    print(f"\nInitialisiere Modell...")
    llm = CoTLLMManager(model_path=MODEL_PATH, temperature=TEMPERATURE)
    
    # Kandidaten generieren
    total = len(questions) * CANDIDATES_PER_QUESTION
    print(f"\nGeneriere {total} Kandidaten ({len(questions)} Fragen x {CANDIDATES_PER_QUESTION})")
    print("WARNUNG: CPU Betrieb - dies wird ca. 30-60 Minuten dauern")
    
    all_candidates = llm.batch_generate(questions, CANDIDATES_PER_QUESTION)
    
    # Ergebnisse speichern
    print(f"\nSpeichere {len(all_candidates)} Kandidaten...")
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "metadata": {
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

    end_time = time.time()
    required_time = end_time - start_time
    
    print("\n" + "=" * 70)
    print("PIPELINE ABGESCHLOSSEN")
    print("=" * 70)
    print(f"\nAusgabedatei: {OUTPUT_FILE}")
    print(f"Generierte Kandidaten: {len(all_candidates)}")
    print("\nFormat: Jeder Kandidat enthaelt:")
    print("  - candidate_id: Frage.Kandidat (z.B. 1.01)")
    print("  - evidence: Chain-of-Thought Erklaerung")
    print("  - SQL: Die generierte SQL Query")
    
    print("\n" + "="*50)
    print("Gesamtzeit für Kandidatengenerierung")
    print("="*50)
    print(f"Total execution time: {required_time:.2f} seconds")
    print(f"Total execution time: {required_time/60:.2f} minutes")
    print(f"Total execution time: {required_time/3600:.2f} hours")

if __name__ == "__main__":
    main()
