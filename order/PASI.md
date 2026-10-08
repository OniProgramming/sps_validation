# Ordinea cuvintelor (ORD) — pașii

## Ce măsoară

SATE avea 11 clase de itemi și **niciuna pentru ordinea cuvintelor**. De aceea nu vedea diferența
dintre „*Theos* was the *Logos*” (ordinea greacă: θεὸς ἦν ὁ λόγος) și „the Word was God”.

`order/` adaugă clasa a 12-a, **ORD**. Codul existent rămâne neschimbat.

**Cum se creează un item ORD (automat, din arborii sintactici MACULA):** în fiecare propoziție
cu verb, fiecare parte de propoziție așezată **înaintea verbului** devine un item. Contează
complementul direct, complementul indirect, predicatul (numele predicativ), complementul
circumstanțial și grupul prepozițional. În greacă și ebraică, o parte așezată înaintea verbului
primește prin asta **importanță**: e cadrul propoziției, tema ei sau accentul ei.

Exemple din Ioan 1:1:
- „**Ἐν ἀρχῇ** ἦν ὁ λόγος”: „la început” stă înaintea verbului;
- „καὶ **θεὸς** ἦν ὁ λόγος”: „Dumnezeu” stă înaintea verbului și înaintea subiectului.

**Ce nu devine item:**
- subiectul, pentru că engleza îl pune oricum înaintea verbului;
- vocativele și interjecțiile („Doamne”, „iată”);
- cuvintele care stau mereu primele: relativele, întrebările (ποῦ, πῶς), negațiile, conjuncțiile,
  καί cu sensul „și / chiar”;
- o propoziție întreagă (participială, infinitivală, subordonată), pentru că acolo e locul ei
  obișnuit;
- infinitivul absolut ebraic, care e deja item ASP;
- propozițiile fără verb.

**Judecătorii** (Claude și GPT, aceiași ca în studiu) răspund la două întrebări pentru fiecare
item:
- **Sens:** cititorul obișnuit simte aceeași importanță? Contează orice mijloc: ordinea, „it was
  X that…”, „as for X”, un cuvânt de accent. Dacă engleza îl face pe cititor să ia predicatul
  drept subiect, verdictul e **distorted**.
- **Sursă:** engleza păstrează partea respectivă înaintea verbului?

**Câți itemi sunt:**

| | Itemi ORD | Propoziții | Cereri per judecător |
|---|---|---|---|
| Geneza (eșantionul) + Efeseni | 95 | 66 | 263 + 27 de retestare |
| Ioan 1 | 38 | 24 | 96 + 8 de retestare |

**Cost estimat:** sub 1 dolar pentru Geneza + Efeseni și sub 0,40 dolari pentru Ioan. Programul
îți arată costul exact înainte să confirmi.

---

## A. Ia folderul `order`

1. Deschide în browser
   https://github.com/OniProgramming/sps_validation/archive/refs/heads/claude/gallant-einstein-07m6zf.zip.
2. Dezarhivează zip-ul într-un loc temporar (click dreapta → **Extract All…**).
3. În el găsești folderul **`order`**. Copiază-l (click dreapta → Copy) în **ambele** proiecte,
   lângă folderul `sps_validation`:
   - proiectul vechi (Geneza + Efeseni);
   - proiectul Ioan.

   Nu copia nimic altceva și nu înlocui nimic.

## B. Geneza + Efeseni (în proiectul vechi)

1. Deschide proiectul vechi în PyCharm, apoi terminalul (Alt+F12). Trebuie să scrie `(venv)` sau
   `(.venv)` la începutul rândului.
2. Pune cheile, ca de obicei, doar în terminal:
   ```
   $env:ANTHROPIC_API_KEY="cheia ta Anthropic"
   $env:OPENAI_API_KEY="cheia ta OpenAI"
   ```
3. Pornește:
   ```
   python -m order.run
   ```
   Programul scrie câți itemi a găsit și cât costă, apoi întreabă `Type yes to start:`.
   Scrie `yes` și apasă Enter.
4. Așteaptă. De obicei durează de la câteva minute la o oră. Dacă se întrerupe, scrie aceeași
   comandă din nou: programul continuă de unde a rămas.
5. La sfârșit apare `written ...\build\order\report\report.md`.

## C. Ioan 1 (în proiectul Ioan)

Ca la B, în proiectul Ioan, cu comanda:
```
python -m order.run --john
```
Raportul apare în `build\john\order\report\report.md`.

## D. Ce îmi trimiți

- din proiectul vechi: `build\order\report\report.md`, `summary.json` și `items.csv`;
- din proiectul Ioan: `build\john\order\report\report.md`, `summary.json` și `items.csv`.

`items.csv` conține textul SPS, deci nu-l trimite profesoarei. Are fiecare verdict ORD al
ambilor judecători, cu motivele lor.

## Ce conține raportul

1. **ORD singur:** scorul F pentru sens și pentru sursă, la fiecare traducere, cu interval de
   încredere și teste statistice.
2. **Fiecare judecător separat.**
3. **Fiabilitatea:**
   - cât de des cei doi judecători dau același verdict;
   - cât de des același judecător dă același răspuns când e întrebat din nou;
   - o verificare automată: dacă un judecător spune că ordinea s-a păstrat, cuvintele chiar apar
     înaintea verbului în engleză?
4. **Scorurile principale cu ORD adăugat**, puse lângă cele din studiu. Coloanele „main” trebuie
   să fie exact cifrele din raportul principal; dacă diferă, spune-mi.

**Limite:**
- subiectul așezat înaintea verbului nu e itemizat;
- propozițiile fără verb nu sunt itemizate;
- pentru ORD nu există încă test cu erori plantate.
