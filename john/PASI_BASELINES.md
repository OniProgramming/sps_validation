# Ioan 1: ablațiile și benchmarkurile, pas cu pas

Sunt aceleași ablații (A1–A6) și aceleași benchmarkuri (COMETKiwi, xCOMET, GEMBA-MQM cu GPT) ca la
Geneza și Efeseni. Folosesc același cod (`sps_validation/ablation.py` și `baselines.py`),
neschimbat, aplicat pe rezultatele din Ioan 1.

| Ce | Unde rulează | Cost |
|---|---|---|
| Ablații A1–A6 | PyCharm | gratuit (refolosește verdictele existente) |
| GEMBA-MQM cu GPT | PyCharm | ≈ $1,8 (erori plantate) + ≈ $0,3 (editări neutre) |
| COMETKiwi și xCOMET | Kaggle (GPU gratuit) | gratuit |

## 0. Codul nou

1. Descarcă din nou zip-ul branch-ului:
   https://github.com/OniProgramming/sps_validation/archive/refs/heads/claude/gallant-einstein-07m6zf.zip
2. Din zip copiază **doar folderul `john`** peste cel din proiectul Ioan, cu Overwrite. Nu
   atinge `data` și `build`, pentru că acolo sunt rezultatele tale.
3. În `john` trebuie să apară două fișiere noi: `ablation.py` și `baselines.py`.

## 1. Ablațiile (gratuit, câteva secunde)

```
python -m john.ablation
```
Rezultat: `build\john\report\ablation\ablation.md` (plus CSV-uri).

## 2. Pregătești fișierele pentru benchmarkuri (gratuit)

```
python -m john.baselines export
```
Trebuie să scrie `exported 168 planted-error pairs, 84 meaning-preserving controls (21 per
translation) and 226 main segments (646 texts)`. Se creează `build\john\baselines\items.jsonl`.

## 3. GEMBA-MQM cu GPT (≈ $2,1 în total)

În terminal setezi cheia (doar acolo, nu în chat):
```
$env:OPENAI_API_KEY="cheia-ta-openai"
```
Apoi rulezi două comenzi. Fiecare îți arată costul și pornește doar după ce scrii `yes`:
```
python -m john.baselines gemba gpt --set pairs
python -m john.baselines gemba gpt --set neutral
```
Dacă la final scrie că au rămas texte fără răspuns, rulezi din nou aceeași comandă: le reia
doar pe acelea.

## 4. COMETKiwi și xCOMET pe Kaggle (gratuit)

Folosești același notebook Kaggle ca la Geneza și Efeseni.

1. Pe Kaggle, deschide notebook-ul de data trecută.
2. **Scoate inputul vechi:** în panoul din dreapta, la Input, ștergi (×) datasetul cu
   `items.jsonl` de la Geneza și Efeseni. Altfel notebook-ul l-ar lua pe cel vechi.
3. **+ Add Input → Upload:** încarci `build\john\baselines\items.jsonl`, cu numele
   `sps-john-items`.
4. Verifici setările: Accelerator **GPU T4 x2** sau P100, Internet **on**, secretul `HF_TOKEN` atașat.
5. Rulezi celulele pe rând, de sus în jos. La celula 3 trebuie să scrie **646 texts**.
6. La celula cu xCOMET pune din start varianta care a mers data trecută:
   ```
   !PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /kaggle/working/cenv/bin/python -m sps_validation.baselines comet xcomet --set all --gpus 1 --batch 2 --half
   ```
7. Ultima celulă creează `comet_scores.zip`. Îl descarci din panoul Output.
8. Pe calculator, dezarhivezi zip-ul în `build\john\baselines\scores\`. Acolo trebuie să ajungă
   `cometkiwi.jsonl` și `xcomet.jsonl`, lângă `gemba-gpt.jsonl`.

`items.jsonl` conține textul SPS și e procesat pe serverele Kaggle, ca data trecută.

## 5. Raportul (gratuit)

```
python -m john.baselines report
```
Rezultat: `build\john\report\baselines\baselines.md` (plus CSV-uri).

## 6. Ce îmi trimiți

- `build\john\report\ablation\ablation.md`
- `build\john\report\baselines\baselines.md`

Nu trimite nimănui `items.jsonl` și fișierele din `scores\`, pentru că conțin textul SPS.
