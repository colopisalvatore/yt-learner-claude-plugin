# yt_learner

CLI Python che guarda un video tutorial di YouTube al posto tuo e produce **due
tipi di memoria** per un sistema agentico:

| Output | Cosa e | Dove finisce |
|---|---|---|
| **SKILL** (memoria procedurale) | istruzioni eseguibili passo-passo, comandi, snippet, error handling, criteri di validazione | `skills/<skill-name>/SKILL.md` |
| **KNOWLEDGE** (memoria dichiarativa/episodica) | sintesi concettuale, teoria, timeline con timestamp cliccabili, cross-reference | `brain/knowledge/<slug-video>.md` |

I due file si linkano a vicenda: la skill sa da quale scheda proviene, la scheda
sa quale skill ha generato.

---

## Plugin per Claude Code

Il repo è anche un plugin: installa la skill `yt-learner`, che sa installare il
tool, lanciarlo e rivedere l'output contro la trascrizione.

```text
/plugin install yt-learner --marketplace colopisalvatore/yt-learner-claude-plugin
```

Su versioni più vecchie di Claude Code:

```text
/plugin marketplace add colopisalvatore/yt-learner-claude-plugin
/plugin install yt-learner@yt-learner
```

Poi basta incollare un link: «elabora questo video https://youtu.be/...».
**Senza chiave API** la skill usa `--offline --dry-run --transcript-out`: il tool
scarica solo la trascrizione e skill e scheda le scrive Claude.

---

## Installazione

```bash
git clone https://github.com/colopisalvatore/yt-learner-claude-plugin && cd yt-learner-claude-plugin

python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

In alternativa, come pacchetto installabile (aggiunge il comando `yt-learner`):

```bash
pip install -e ".[gemini]"      # oppure ".[openai]", ".[anthropic]", ".[all]"
```

Requisiti: **Python >= 3.10**. `ffmpeg` serve per `--slides` e per
`--allow-audio-fallback` (`winget install Gyan.FFmpeg` / `apt install ffmpeg`).

### Chiavi API

Copia `.env.example` in `.env` e riempi **solo** la chiave del provider che usi:

```dotenv
GEMINI_API_KEY=...            # default (gemini-2.5-flash)
OPENAI_API_KEY=...            # --model gpt-4o
ANTHROPIC_API_KEY=...         # --model claude-sonnet-5
YT_LEARNER_MODEL=gemini-2.5-flash
```

Il provider viene dedotto dal nome del modello (`gemini*` -> Gemini, `gpt-*`/`o*`
-> OpenAI, `claude*` -> Anthropic); puoi forzarlo con `--provider`.

---

## Uso

```bash
# caso base: usa ./skills e ./brain/knowledge
yt-learner "https://www.youtube.com/watch?v=XXXXXXXXXXX"

# equivalente senza installare il pacchetto
python -m yt_learner "https://youtu.be/XXXXXXXXXXX"
```

Esempi utili:

```bash
# modello e lingua dei sottotitoli diversi
python -m yt_learner "https://youtu.be/XXXXXXXXXXX" --model gpt-4o --lang en,it

# scrivere in una cartella qualsiasi e aggiornare un indice MEMORY.md
python -m yt_learner "https://youtu.be/XXXXXXXXXXX" \
    --brain-dir "C:/Users/me/note/fonti" \
    --skills-dir "C:/Users/me/.claude/skills" \
    --index "C:/Users/me/note/MEMORY.md"

# opzionale: scrivere in un vault Manent (vedi sotto)
MANENT_VAULT=~/vault python -m yt_learner "https://youtu.be/XXXXXXXXXXX" --brain manent

# provare senza scrivere niente su disco
python -m yt_learner "https://youtu.be/XXXXXXXXXXX" --dry-run

# nessuna chiamata LLM (bozze euristiche dalla trascrizione)
python -m yt_learner "https://youtu.be/XXXXXXXXXXX" --offline

# solo la skill, con nome forzato, senza sovrascrivere l'esistente
python -m yt_learner "https://youtu.be/XXXXXXXXXXX" \
    --only skill --skill-name deploy-fastapi-su-vps --no-overwrite

# video senza sottotitoli: scarica l'audio e lo trascrive con l'LLM
python -m yt_learner "https://youtu.be/XXXXXXXXXXX" --allow-audio-fallback
```

### Flag

| Flag | Default | Descrizione |
|---|---|---|
| `--brain` | `default` | destinazione: nome di un preset (`default`, `manent`, o tuoi) oppure un percorso |
| `--list-brains` | — | elenca le destinazioni configurate ed esce |
| `--brain-dir` | dal preset | cartella esplicita delle schede (override del preset) |
| `--skills-dir` | `./skills` | cartella delle skill |
| `--skill-name` | dal video | forza il nome (kebab-case) della skill |
| `--only` | `both` | genera solo `skill` o solo `knowledge` |
| `--index` | dal preset | file indice (es. `MEMORY.md`) da aggiornare con un puntatore |
| `--no-index` | off | non aggiornare l'indice anche se il preset ne definisce uno |
| `--no-lint` | off | non lanciare il lint del vault previsto dal preset |
| `--no-overwrite` | off | non sovrascrive: aggiunge un suffisso numerico |
| `--dump-json` | — | salva estrazione + spec LLM in JSON (debug/riuso) |
| `--transcript-out` | — | salva la trascrizione con timestamp |
| `--dry-run` | off | esegue tutto senza scrivere file |
| `--model` | `$YT_LEARNER_MODEL` o `gemini-2.5-flash` | modello LLM |
| `--provider` | dedotto | `gemini` \| `openai` \| `anthropic` |
| `--api-key` | da env/`.env` | chiave esplicita |
| `--temperature` | `0.2` | temperatura del modello |
| `--output-language` | `italiano` | lingua dei file generati |
| `--offline` | off | nessuna chiamata LLM |
| `--lang` | `it,en` | lingue dei sottotitoli, in ordine di preferenza |
| `--allow-audio-fallback` | off | scarica l'audio e lo trascrive se mancano i sottotitoli |
| `--slides` | off | scarica il video ed estrae le slide reali con ffmpeg |
| `--slides-mode` | `sample` | `sample` (ogni cambio slide), `timeline` (una per voce), `scene` (stacchi ffmpeg) |
| `--slides-interval` | `5` | secondi tra due campioni in modalita `sample` |
| `--slides-change-bits` | `45` | bit su 256 che devono cambiare perche sia una slide nuova |
| `--slides-dir` | `<brain-dir>/assets/<slug>` | dove salvare le immagini |
| `--slides-width` / `--slides-max-height` | `1280` / `720` | risoluzione immagini / video scaricato |
| `--max-slides` | `60` | tetto di slide estratte |
| `--keep-video` | — | conserva il video scaricato nella cartella indicata |
| `-v/--verbose`, `-q/--quiet` | — | verbosita del log |

Exit code: `0` ok, `1` errore di runtime (estrazione o LLM), `2` uso errato
(es. API key mancante).

---

## Destinazioni del brain (`--brain`)

Un **preset** descrive un vault: cartella, come si chiamano i file, che `type`
mettere nel frontmatter, quale indice aggiornare e che lint lanciare dopo.

| Preset | Dove scrive | Layout | `type` | Extra |
|---|---|---|---|---|
| `default` | `./brain/knowledge` | `<slug>.md` | `reference` | — |
| `manent` (opzionale) | `$MANENT_VAULT/library` (senza variabile: `./library`) | `<YYYY-MM-DD>-<slug>.md` | `raw-source` | aggiorna `MEMORY.md`, lancia `manent lint` se la CLI c'è |

`manent` segue il layout di un vault [Manent](https://github.com/colopisalvatore/manent)
(memoria file-first per agenti, servita via MCP): le fonti datate vanno in
`library/` e non in `knowledge/`, che in un vault Manent è un mirror in sola
lettura escluso dall'indicizzazione. Se non usi Manent ignoralo: `default` o un
percorso bastano.

Variabili d'ambiente: `MANENT_VAULT` (radice del vault), `MANENT_CLI` (percorso di
`packages/cli/dist/index.js` di un checkout di Manent; in alternativa il comando
`manent` nel PATH; senza CLI il lint viene semplicemente saltato).

Puoi aggiungerne altri in `yt-learner.json` (nella cwd) o `~/.yt-learner.json`:

```json
{
  "brains": {
    "lavoro": {
      "root": "D:/vault",
      "subdir": "fonti",
      "layout": "dated",
      "note_type": "raw-source",
      "index": "MEMORY.md",
      "lint_cmd": "node D:/tools/manent/cli.js lint {root}"
    }
  }
}
```

`--brain <percorso>` accetta anche una cartella qualsiasi, e `--brain-dir`
resta l'override esplicito che batte tutto.

---

## Slide (`--slides`)

Con `--slides` il video viene scaricato (solo traccia video, max 720p di
default) e le slide vengono estratte con ffmpeg in
`<brain-dir>/assets/<slug>/`, poi linkate nella scheda: quelle che cadono
dentro un segmento della timeline finiscono sotto la voce corrispondente, le
altre in una galleria in fondo. Il video temporaneo viene cancellato a fine
lavoro (`--keep-video` per tenerlo).

| Modalita | Come sceglie i frame | Quando usarla |
|---|---|---|
| `sample` (default) | campiona ogni `--slides-interval` secondi e tiene il frame quando l'immagine **cambia** (dHash a 256 bit, soglia `--slides-change-bits`) | deck di slide, anche quando il template resta identico e cambia solo il testo |
| `timeline` | un frame per voce della timeline, scegliendo il piu ricco di dettaglio in una finestra dopo il timestamp | vuoi esattamente una immagine per segmento della scheda |
| `scene` | stacchi rilevati da ffmpeg (`select='gt(scene,N)'`) | video con tagli netti (montaggi, demo a schermo) |

Perche `sample` e il default: il rilevamento scena di ffmpeg non scatta quando
due slide condividono lo stesso sfondo. Misurando le distanze di hash su un
tutorial con webcam in sovrimpressione, il movimento del volto sta sotto i ~33
bit su 256 mentre un cambio slide vero supera i 58 — da qui la soglia a 45.
I frame quasi vuoti (stacchi, schermate bianche) e i duplicati vengono scartati.

---

## Cosa viene generato

```
skills/<skill-name>/
  SKILL.md            # frontmatter (name, description, version, tags, triggers) +
                      # prerequisiti, workflow passo-passo, snippet, troubleshooting,
                      # criteri di validazione, riferimenti
  scripts/…           # solo se il video mostra script corposi
  examples/…          # solo se il video mostra template/esempi

brain/knowledge/<slug-video>.md
                      # frontmatter (title, source_url, channel, duration,
                      # learned_date, related_skill) + executive summary,
                      # concetti chiave, architettura, timeline con timestamp
                      # cliccabili, takeaway, trappole, cross-reference
```

Ogni voce della timeline e un link diretto al minuto del video
(`https://www.youtube.com/watch?v=ID&t=135s`).

**Rieseguire sullo stesso video aggiorna, non duplica.** Il modello propone un
nome di skill diverso a ogni giro, quindi i file esistenti vengono ritrovati dal
`source_url` nel loro frontmatter e riusati: stessa cartella di skill, stesso
file di scheda (anche se il layout e datato e il giorno e cambiato). Lo slug
della scheda viene dal titolo del video, non dalla proposta del modello. Con
`--skill-name` decidi tu il nome e vince su tutto.

---

## Architettura

```
yt_learner/
  main.py            # CLI (argparse), orchestrazione della pipeline
  yt_extractor.py    # metadati + trascrizione, con fallback a cascata
  llm_processor.py   # provider LLM, prompt, map-reduce, normalizzazione JSON
  skill_builder.py   # spec JSON -> skills/<name>/SKILL.md (+ asset)
  brain_writer.py    # spec JSON -> <brain>/<slug>.md (+ slide, + indice)
  slide_extractor.py # download video, frame ffmpeg, dedup via dHash
  config.py          # preset delle destinazioni del brain (built-in + JSON)
  utils.py           # slug, timestamp, frontmatter YAML, IO
tests/test_pipeline.py
```

**Fallback della trascrizione** (in ordine):

1. `youtube-transcript-api` — sottotitoli manuali, poi automatici (supporta sia
   l'API 0.6.x sia la 1.x);
2. `yt-dlp` — caption track scaricata e parsata (`json3`, `srv3`, `vtt`, `srt`);
3. audio + LLM multimodale — solo con `--allow-audio-fallback`
   (Gemini via upload file, OpenAI via `whisper-1`).

I metadati usano `yt-dlp` e, se fallisce, l'endpoint pubblico oEmbed.

**Trascrizioni lunghe**: oltre ~120k caratteri la trascrizione viene condensata
con un passaggio map-reduce (blocchi da 40k caratteri, timestamp preservati)
prima di generare skill e knowledge.

**Robustezza dell'output LLM**: la risposta viene ripulita da fence markdown e
testo attorno, riparsata con estrazione dell'oggetto JSON bilanciato, e in caso
di formato invalido il modello viene richiamato con l'errore in prompt (2 retry).
Le spec vengono poi normalizzate: campi mancanti riempiti, tipi sbagliati
convertiti, timestamp oltre la durata del video scartati, asset con path
traversal o estensioni non ammesse rifiutati.

---

## Errori comuni

| Sintomo | Causa | Soluzione |
|---|---|---|
| `API key mancante per il provider …` | nessuna chiave in ambiente | riempi `.env` o usa `--api-key`, oppure `--offline` |
| `Nessuna trascrizione disponibile` | video senza sottotitoli in quelle lingue | prova `--lang en` o `--allow-audio-fallback` |
| `Video non accessibile … privato, rimosso o con restrizioni` | video privato/age-gated | serve un video pubblico |
| `Provider non deducibile dal modello …` | nome modello non standard | passa `--provider` |
| `Il modello non ha prodotto JSON valido` | modello piccolo o troppo caldo | abbassa `--temperature` o cambia `--model` |
| `Pacchetto mancante: pip install …` | SDK del provider non installato | installa l'extra corrispondente |
| `ffmpeg non trovato nel PATH` | serve per le slide | installalo, oppure lancia senza `--slides` |
| slide troppe / troppo poche | soglia di cambio non tarata sul video | alza o abbassa `--slides-change-bits` (default 45 su 256) |
| `Preset di brain sconosciuto` | nome non definito | `--list-brains`, o passa direttamente un percorso |

---

## Test

Suite completamente offline (nessuna rete, nessuna API key):

```bash
python -m unittest discover -s tests -v
```

Copre: parsing URL/VTT/json3, scelta della caption track, tolleranza del parser
JSON, retry sul formato, map-reduce, normalizzazione di spec malformate,
rendering dei due markdown, protezione dai path traversal negli asset,
aggiornamento idempotente dell'indice e i percorsi principali della CLI.
