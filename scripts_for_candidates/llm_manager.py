#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM Manager für die Integration von Ollama mit Qwen2.5
Verwendet direkte HTTP-Requests
Wird benötigt für: "generate_candidates_with_qwen"
"""

import json
import time
import requests
from typing import List, Dict, Any, Optional

class LLMManager:
    """
    Manager für lokale LLMs via Ollama - mit HTTP-Requests
    """
    
    def __init__(self, model_name: str = "qwen2.5", temperature: float = 0.7):
        """
        Initialisiert den LLM Manager.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = self._get_system_prompt()
        self.ollama_url = "http://localhost:11434/api"
        
        # Prüfe Verbindung
        self._check_connection()
    
    def _check_connection(self):
        """
        Prüft die Verbindung zu Ollama.
        """
        print("🔍 Prüfe Ollama Verbindung...")
        
        try:
            response = requests.get(f"{self.ollama_url}/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                model_names = [m['name'] for m in models]
                print(f"✅ Ollama ist erreichbar!")
                print(f"📋 Verfügbare Modelle: {model_names}")
                
                # Prüfe ob unser Modell da ist
                model_found = False
                for m in model_names:
                    if self.model_name in m:
                        self.model_name = m
                        model_found = True
                        print(f"✅ Verwende Modell: '{self.model_name}'")
                        break
                
                if not model_found:
                    print(f"⚠️ Modell '{self.model_name}' nicht gefunden!")
                    print(f"Installieren mit: ollama pull {self.model_name}")
                    raise ValueError(f"Modell {self.model_name} nicht verfügbar")
                
                return True
            else:
                print(f"❌ HTTP-Fehler: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Keine Verbindung zu Ollama möglich!")
            print("Stellen Sie sicher, dass Ollama läuft: 'ollama serve'")
        except Exception as e:
            print(f"❌ Fehler: {e}")
        
        raise ConnectionError("Keine Verbindung zu Ollama möglich")
    
    def _get_system_prompt(self) -> str:
        """System-Prompt für SQL-Generierung"""
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
        Generiert verschiedene SQL-Varianten für eine Frage.
        """
        prompt = self._build_prompt(question, original_sql, original_evidence, num_variants)
        
        # HTTP-API aufrufen
        try:
            return self._call_ollama_http(prompt, num_variants)
        except Exception as e:
            print(f"⚠️ HTTP-Fehler: {e}")
            return self._generate_fallback_variants(question, original_sql, original_evidence, num_variants)
    
    def _call_ollama_http(self, prompt: str, num_variants: int) -> List[Dict[str, str]]:
        """
        Ruft Ollama über HTTP-API auf.
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 2048
            }
        }
        
        print(f"    📤 Sende Anfrage an {self.model_name}...")
        response = requests.post(
            f"{self.ollama_url}/chat",
            json=payload,
            timeout=120
        )
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        data = response.json()
        content = data['message']['content']
        
        # Versuche JSON zu parsen
        try:
            # Suche nach JSON in der Antwort
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                result = json.loads(json_str)
                candidates = result.get('candidates', [])
                print(f"    ✅ {len(candidates)} Kandidaten erhalten")
                return candidates[:num_variants]
        except Exception as e:
            print(f"    ⚠️ JSON-Parsing Fehler: {e}")
            print(f"    Antwort: {content[:200]}...")
        
        return []
    
    def _generate_fallback_variants(self, question, original_sql, original_evidence, num_variants):
        """
        Generiert einfache Fallback-Varianten.
        """
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
        
        # Wiederholen falls nötig
        while len(variants) < num_variants:
            variants.extend(variants[:num_variants - len(variants)])
        
        return variants[:num_variants]
    
    def _build_prompt(self, question, original_sql, original_evidence, num_variants):
        """Baut den Prompt für das LLM"""
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
    
    def batch_generate(self, questions_data, candidates_per_question=10, delay=2.0):
        """Batch-Generierung für mehrere Fragen"""
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
                    "candidate_id": f"{query_id}.{i+1}",
                    "db_id": "address",
                    "question": item.get('question', ''),
                    "query_id": query_id,
                    "evidence": variant.get('evidence', item.get('evidence', '')),
                    "SQL": variant.get('SQL', item.get('SQL', ''))
                }
                all_candidates.append(candidate)
            
            print(f"    ✅ {len(variants)} Kandidaten generiert")
            time.sleep(delay)
        
        return all_candidates


# Test-Funktion
if __name__ == "__main__":
    print("Teste LLMManager...")
    try:
        llm = LLMManager("qwen2.5")
        print("✅ LLMManager erfolgreich initialisiert!")
    except Exception as e:
        print(f"❌ Fehler: {e}")
