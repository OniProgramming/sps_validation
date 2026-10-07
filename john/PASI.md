# SATE pe Ioan 1 — pașii de la zero

Proiectul pentru Ioan 1 se pune într-un **folder nou**, separat de cel pentru Geneza și Efeseni.
Proiectul vechi rămâne neatins. Zip-ul descărcat conține tot ce trebuie:
- codul SATE (`sps_validation/`), neschimbat;
- folderul `john/`;
- setările (`config/`).

Timp: aproximativ 30 de minute de pregătire. Judecătorii lucrează apoi de la câteva minute la
câteva ore.
Cost: aproximativ $11 în total. Îl vezi exact înainte să confirmi.

---

## A. O singură dată: programele

1. **Python 3.11.** Îl ai deja, din proiectul vechi.
2. **Git for Windows.** Dacă nu l-ai instalat, descarcă-l de la https://git-scm.com/download/win.
   Instalează-l lăsând toate opțiunile implicite (Next, Next…).
3. **PyCharm.** Îl ai deja.

## B. Descarcă proiectul

1. Deschide în browser
   https://github.com/OniProgramming/sps_validation/archive/refs/heads/claude/gallant-einstein-07m6zf.zip.
   Descărcarea pornește singură.
2. Dezarhivează zip-ul: click dreapta → **Extract All…**. Alege un loc nou, de exemplu
   `C:\Users\User\Documents\SATE_Ioan`.
3. În folderul dezarhivat trebuie să vezi, printre altele:
   - `john`
   - `sps_validation`
   - `config`
   - `requirements.txt`

   Acesta este **folderul proiectului**.

## C. Deschide proiectul în PyCharm și creează mediul Python

1. PyCharm → **File → Open…** → alege folderul proiectului (cel cu `john` înăuntru) → **OK**.
   Dacă te întreabă, alege **Trust Project**.
2. Click jos-dreapta, pe zona care arată interpretorul (de exemplu „No interpreter” sau „Python 3.11”).
   Alege **Add New Interpreter → Add Local Interpreter…**.
3. Alege **Virtualenv Environment → New**:
   - Base interpreter: **Python 3.11**;
   - Location: lasă cea propusă (`...\venv` sau `...\.venv`).

   Apasă **OK**.
4. Deschide terminalul: **View → Tool Windows → Terminal** (sau Alt+F12).
   La începutul rândului trebuie să scrie `(venv)` sau `(.venv)`. Dacă nu scrie, închide terminalul
   și deschide-l din nou.
5. În terminal, instalează pachetele:
   ```
   pip install -r requirements.txt
   ```

## D. Pune fișierele de intrare

1. În folderul proiectului creează folderul `data`. În el creează folderul `input`, iar în acesta
   folderul `john`. Calea completă: `data\input\john\`.
2. Pune în `data\input\john\`:
   - docx-ul tău cu Ioan 1, redenumit exact **`SPS_John1.docx`**;
   - Biblia BSB: deschide https://bereanbible.com/bsb.txt, apasă Ctrl+S și salvează ca
     **`bsb.txt`** în același folder.

Aceste fișiere nu ajung niciodată pe GitHub: folderul `data\input` este exclus.

## E. Verifică totul (gratuit)

În terminal:
```
python -m john.check
```
Toate rândurile trebuie să înceapă cu `OK`, cu excepția celor două chei API. Pe acestea le
setezi la pasul G. Dacă alt rând arată `MISSING`, sub el scrie ce trebuie făcut.

## F. Pregătirea (gratuită)

```
python -m john.prepare
```
Descarcă textele grecești și WEB/OEB, apoi construiește propozițiile, itemii, alinierea și
cererile. Durează câteva minute la prima rulare. La final trebuie să vezi:
- `BSB: 51 verses — official download …`
- `SPS: 51 verses — …SPS_John1.docx … transliterations: italics`
- `JHN: 57 sentence units …` și `JHN: 1693 information items …`
- patru rânduri `WEB/BSB/OEB/SPS {…'accuracy': …}`
- `226 judge requests per judge`
- `planted errors: 168 pairs …; retest: 24 requests`

Fă o captură de ecran și trimite-mi-o înainte să mergi mai departe.

## G. Cheile API

1. Cheile le găsești la:
   - Claude: https://console.anthropic.com/settings/keys (creditul: https://console.anthropic.com/settings/billing)
   - GPT: https://platform.openai.com/api-keys (creditul: https://platform.openai.com/settings/organization/billing/overview)

   Pe fiecare cont trebuie să ai cel puțin $6.
2. Scrie-le **doar în terminal**, nu în chat și nu în fișiere. Pune cheia între ghilimele:
   ```
   $env:ANTHROPIC_API_KEY="cheia-ta-claude"
   $env:OPENAI_API_KEY="cheia-ta-openai"
   ```
3. Verifică din nou: `python -m john.check`. Acum toate rândurile trebuie să fie `OK`.

Cheile rămân setate doar cât timp terminalul e deschis. Dacă îl închizi, le scrii din nou.

## H. Rularea judecătorilor

```
python -m john.run --skip-prepare
```
1. Programul îți arată costul estimat.
2. Scrie `yes` și apasă Enter.
3. Programul trimite cererile la ambii judecători (Claude Haiku 4.5 și GPT-5 mini) și așteaptă.
   Lasă terminalul deschis și calculatorul pornit.
4. La final scrie `written …\build\john\report`.

**Dacă se întrerupe** (închizi terminalul sau cade internetul):
1. Deschide terminalul.
2. Setează din nou cheile (pasul G.2).
3. Rulează aceeași comandă: `python -m john.run --skip-prepare`. Scrie din nou `yes`.

Programul continuă cu cererile deja trimise. Nu plătești de două ori.

**Nu rula `john.prepare` din nou după ce ai trimis cererile.**

## I. Rezultatele

În folderul `build\john\report\` găsești:
- `report.md`: tabelele 1–4 și detaliile (judecători, retest, erori plantate);
- `table1.csv` … `table4.csv`: tabelele, care se deschid în Excel;
- `summary.json`: toate cifrele;
- `sentences.csv`: scorul fiecărei propoziții. Conține textul SPS, deci nu îl trimite nimănui.

Trimite-mi **`report.md`** și **`summary.json`** și le interpretăm împreună.

Nu trimite nimănui folderul `build\john\judge\`, pentru că acolo e textul SPS.

## Probleme frecvente

| Mesaj | Ce faci |
|---|---|
| `No module named john` | Terminalul nu e în folderul proiectului. Închide-l și deschide-l din PyCharm, cu proiectul deschis. |
| `No module named anthropic` (sau `openai`, `numpy`) | Venv-ul nu e activ (lipsește `(venv)` la începutul rândului). Pasul C.4, apoi C.5. |
| `'git' is not recognized` | Instalează Git (pasul A.2), apoi închide și redeschide PyCharm. |
| `verse numbers not consecutive` | Lipsește un număr de verset în docx; mesajul arată care. |
| `authentication` / `401` | Cheia e greșită sau nu e setată în acest terminal. Pasul G. |
| `credit` / `billing` / `insufficient_quota` | Adaugă credit în contul respectiv. Apoi rulezi din nou comanda de la H. |
| `belongs to a different experiment` | Ai rulat din nou `john.prepare` după trimitere. Scrie-mi înainte să ștergi ceva. |
