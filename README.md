# ExCoT-Reproduction-DMML
Reproduzierung des ExCoT: Optimizing Reasoning for Text-to-SQL with Execution Feedback von Snowflake Inc.

Ordner candidates: Kandidaten (address) der verschiedenen Datenbanken von BIRD. Kandidaten: Zu 50 Fragen aus den Gold-Datensätzen wurden je 10 Querys generiert. Die Kandidaten aus adress wurden mit dem Script (ordner: scripts_for_candidates) mit qwen2.5:7b erstellt. 
Modelle: Über Ollama herunterladen
Weitere erstellte Kandidaten aus anderen BIRD-Datenbanken.

Ordner candidates_sorted: Mit script_for_division.py (Ordner script_for_sorting) werden die Kandidaten unterteilt (true/false). 
Ordner script_for_SFT: Für das Supervised-Fine-Tuning werden mit create_sft_dataset.py nur die korrekten Kandidate in einer der Datei sft_candidates.json (Ordner first_round_sft_datasets) gespeichert.
Ordner script_for_DPO: Für die Direct Preference Optimization werden je eine korrekte und eine falsche SQL-Anfrage zu einer Text-Frage hinzugefügt und in all_dpo_combined.json (Ordner first_round_dpo_datasets) gespeichert. 

Ordner config_files: Zuerst wird mit training_sft.py das Base-Model nur mit den richtigen Antworten trainiert, anschließend wird es mit training_dpo.py mit den Paaren trainiert.

Ordner merge_LoRA_und_baseModel: Da das Model mit LoRA trainiert wurde, wird es anschließend  mit dem Base-Model gemerged, so dass man am Ende ein vollständiges, trainiertes Model erhält. 
Ordner trained_model: Hier liegt das Model, das ind er ersten Runde trainiert wurde.



Die Hauptskript-Dateien müssen im gleichen Verzeichnis liegen: llm_manager.py heir wird die Klasse LLMManager importiert
ollama pull qwen2.5 (falls noch nicht vorhanden), für unsere Rechner wären auch noch
Kleinere und schnelleres Modelle denkbar gewesen zu verwenden, da Qwen2.5 zu langsam war oder die
Batch-Größe stark reduzieren
Das BIRD-Dataset muss heruntergeladen werden: link hinzufügen!

Aus dem Paper: Ausf¨uhrungsbasiertes Feedback Um L¨osungen als richtig oder falsch zu kennzeichnen,
betten wir jedes relevante Datenbankschema in eine lokale SQLite-Instanz ein, müssen wir das erwähnen?



