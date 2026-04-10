import json
import os
import glob
from datasets import Dataset

def load_json_file(file_path):
    """Lädt eine JSON-Datei."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_training_messages(candidate):
    """
    Erstellt das Nachrichtenformat für SFT.
    Format:
    [
        {"role": "user", "content": "Frage + Chain-of-Thought"},
        {"role": "assistant", "content": "SQL Query"}
    ]
    """
    # Kombiniere Frage mit der Chain-of-Thought (evidence)
    user_content = f"Question: {candidate['question']}\n\nChain-of-Thought: {candidate['evidence']}"

    # Assistant bekommt das korrekte SQL
    assistant_content = candidate['SQL']

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]

def main():
    # Pfade zu den Ordnern
    candidates_dir = "candidates"
    sorting_dir = "Sorting"
    output_file = "sft_candidates.json"

    # Lade alle Bewertungen aus dem Sorting-Ordner
    print(f"Loading evaluations from {sorting_dir}/...")
    all_evaluations = {}

    # Finde alle JSON-Dateien im Sorting-Ordner
    eval_files = glob.glob(os.path.join(sorting_dir, "*.json"))

    for eval_file in eval_files:
        print(f"  Loading {os.path.basename(eval_file)}...")
        evaluations = load_json_file(eval_file)

        # Füge jede Bewertung zum Dictionary hinzu
        for eval_item in evaluations:
            key = (eval_item['candidate_id'], eval_item['db_id'])
            all_evaluations[key] = eval_item

    print(f"Total evaluations loaded: {len(all_evaluations)}")

    # Lade alle Kandidaten aus dem candidates-Ordner
    print(f"\nLoading candidates from {candidates_dir}/...")
    all_candidates = []

    # Finde alle JSON-Dateien im candidates-Ordner
    candidate_files = glob.glob(os.path.join(candidates_dir, "*.json"))

    for candidate_file in candidate_files:
        print(f"  Loading {os.path.basename(candidate_file)}...")
        candidates = load_json_file(candidate_file)
        all_candidates.extend(candidates)

    print(f"Total candidates loaded: {len(all_candidates)}")

    # Sammle alle korrekten Kandidaten
    correct_messages = []
    correct_count = 0
    skipped_count = 0
    missing_evaluation_count = 0

    for candidate in all_candidates:
        key = (candidate['candidate_id'], candidate['db_id'])

        if key in all_evaluations:
            if all_evaluations[key].get('correct', False):
                messages = create_training_messages(candidate)
                correct_messages.append({"messages": messages})
                correct_count += 1
            else:
                skipped_count += 1
        else:
            missing_evaluation_count += 1

    print(f"\n--- Summary ---")
    print(f"Correct candidates: {correct_count}")
    print(f"Incorrect candidates: {skipped_count}")
    print(f"Candidates without evaluation: {missing_evaluation_count}")

    # Erstelle das Dataset im Axolotl-kompatiblen Format
    if correct_messages:
        dataset = Dataset.from_list(correct_messages)

        # Speichere als JSON
        dataset.to_json(output_file)

        print(f"\n✅ Dataset saved to {output_file}")
        print(f"Total training examples: {len(dataset)}")

        # Zeige ein Beispiel
        if len(dataset) > 0:
            print("\n📝 Example training entry:")
            print(json.dumps(dataset[0], indent=2, ensure_ascii=False))
    else:
        print("\n❌ No correct candidates found. Dataset not created.")

if __name__ == "__main__":
    main()