import json
import os
from pathlib import Path
from typing import Dict, List, Optional

def load_json(file_path: str) -> dict:
    """Lädt eine JSON-Datei."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: List[dict], file_path: str) -> None:
    """Speichert Daten als JSON-Datei."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def create_dpo_pairs(
        candidates_file: str,
        preferences_file: str,
        output_file: str,
        db_name: str
) -> None:
    """
    Erstellt DPO-Paare aus Kandidaten und Präferenzdaten.

    Format: Für jede Frage wird die beste (correct=true) als chosen
    und eine zufällige schlechte (correct=false) als rejected genommen.
    """
    # Daten laden
    candidates_data = load_json(candidates_file)
    preferences_data = load_json(preferences_file)

    # Index der Präferenzen nach candidate_id
    pref_index = {item["candidate_id"]: item for item in preferences_data}

    # Gruppiere Kandidaten nach Frage (question als Key)
    questions_map = {}
    for candidate in candidates_data: #.get("candidates", []):
        question = candidate.get("question", "")
        if question not in questions_map:
            questions_map[question] = {"chosen": [], "rejected": []}

        # Prüfe, ob dieser Kandidat in den Präferenzen ist
        candidate_id = candidate.get("candidate_id")
        if candidate_id in pref_index:
            pref = pref_index[candidate_id]
            if pref.get("correct", False):
                questions_map[question]["chosen"].append(candidate)
            else:
                questions_map[question]["rejected"].append(candidate)

    # Erstelle DPO-Dataset
    dpo_dataset = []
    for question, candidates in questions_map.items():
        chosen_list = candidates["chosen"]
        rejected_list = candidates["rejected"]

        # Für jede chosen, eine rejected auswählen
        for chosen in chosen_list:
            if rejected_list:
                # Nimm die erste rejected (oder randomisieren)
                rejected = rejected_list[0]

                # Erstelle Eintrag im gewünschten Format
                dpo_entry = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "As a Text2SQL assistant, your main task is to formulate an SQL query in response to a given natural language inquiry. This process involves a chain-of-thought (CoT) approach, which includes a 'divide and conquer' strategy. In the 'divide' phase of this CoT process, we break down the presented question into smaller, more manageable sub-problems using pseudo-SQL queries. During the 'conquer' phase, we aggregate the solutions of these sub-problems to form the final response. Lastly, we refine the constructed query in the optimization step, eliminating any unnecessary clauses and conditions to ensure efficiency."
                        },
                        {
                            "role": "user",
                            "content": f"Question: {question}\n\nDatabase: {db_name}"
                        }
                    ],
                    "chosen": {
                        "role": "assistant",
                        "content": chosen.get("SQL", "")
                    },
                    "rejected": {
                        "role": "assistant",
                        "content": rejected.get("SQL", "")
                    }
                }
                dpo_dataset.append(dpo_entry)

    # Speichere Ergebnis
    save_json(dpo_dataset, output_file)
    print(f"Erstellt: {len(dpo_dataset)} DPO-Paare für {db_name}")
    print(f"Gespeichert in: {output_file}")

def process_all_datasets(
        data_dir: str,
        output_dir: str,
        file_pairs: List[tuple]
) -> None:
    """
    Verarbeitet alle Datasets.

    Args:
        data_dir: Verzeichnis mit den Quelldateien
        output_dir: Verzeichnis für die Ausgabe
        file_pairs: Liste von Tupeln (candidates_file, preferences_file, db_name)
    """
    os.makedirs(output_dir, exist_ok=True)

    for candidates_file, preferences_file, db_name in file_pairs:
        candidates_path = os.path.join(data_dir, candidates_file)
        pref_path = os.path.join(data_dir, preferences_file)
        output_path = os.path.join(output_dir, f"{db_name}_dpo.json")

        if os.path.exists(candidates_path) and os.path.exists(pref_path):
            create_dpo_pairs(candidates_path, pref_path, output_path, db_name)
        else:
            print(f"Warnung: Dateien für {db_name} nicht gefunden")

# Beispiel-Nutzung
if __name__ == "__main__":
    # Konfiguration
    DATA_DIR = "./data"  # Ihr Verzeichnis mit den JSON-Dateien
    OUTPUT_DIR = "./dpo_datasets"

    # Definieren Sie alle Ihre Dateipaare
    file_pairs = [
        ("./../candidates/address.json", "./../candidates_sorted/address_sorted.json", "address"),
        ("./../candidates/citeseer.json", "./../candidates_sorted/citeseer_sorted.json", "citeseer"),
        ("./../candidates/craftbeer.json", "./../candidates_sorted/craftbeer_sorted.json", "craftbeer"),
        ("./../candidates/disney.json", "./../candidates_sorted/disney_sorted.json", "disney"),
        ("./../candidates/restaurant.json", "./../candidates_sorted/restaurant_sorted.json", "restaurant"),
        # Fügen Sie hier weitere Paare hinzu
        # ("customers.json", "customers_sorted.json", "customers"),
    ]

    # Führen Sie die Konvertierung durch
    process_all_datasets(DATA_DIR, OUTPUT_DIR, file_pairs)

    # Optional: Alle Datasets zu einem großen zusammenführen
    all_dpo_data = []
    for _, _, db_name in file_pairs:
        dataset_file = os.path.join(OUTPUT_DIR, f"{db_name}_dpo.json")
        if os.path.exists(dataset_file):
            with open(dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_dpo_data.extend(data)

    # Gesamtdataset speichern
    combined_path = os.path.join(OUTPUT_DIR, "all_dpo_combined.json")
    save_json(all_dpo_data, combined_path)
    print(f"\nKombiniertes Dataset: {len(all_dpo_data)} Einträge")