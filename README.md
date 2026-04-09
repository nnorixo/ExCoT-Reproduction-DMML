# ExCoT-Reproduction-DMML
Reproduzierung des ExCoT: Optimizing Reasoning for Text-to-SQL with Execution Feedback von Snowflake Inc.

Ordner candidates: Kandidaten (address) der verschiedenen Datenbanken von BIRD. Kandidaten: Zu 50 Fragen aus den Gold-Datensätzen wurden je 10 Querys generiert. Die Kandidaten aus adress wurden mit dem Script (ordner: scripts_for_candidates) mit qwen2.5:7b erstellt. 
Modelle: Über Ollama herunterladen


Die Hauptskript-Dateien müssen im gleichen Verzeichnis liegen: llm_manager.py heir wird die Klasse LLMManager importiert
ollama pull qwen2.5 (falls noch nicht vorhanden), für unsere Rechner wären auch noch
Kleinere und schnelleres Modelle denkbar gewesen zu verwenden, da Qwen2.5 zu langsam war oder die
Batch-Größe stark reduzieren
Das BIRD-Dataset muss heruntergeladen werden: link hinzufügen!

Aus dem Paper: Ausf¨uhrungsbasiertes Feedback Um L¨osungen als richtig oder falsch zu kennzeichnen,
betten wir jedes relevante Datenbankschema in eine lokale SQLite-Instanz ein, müssen wir das erwähnen?



