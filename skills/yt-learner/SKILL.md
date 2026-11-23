---
name: yt-learner
description: Trasforma un video tutorial di YouTube in una SKILL eseguibile (passi, comandi, troubleshooting) e in una scheda di knowledge con timeline e timestamp cliccabili. Usala quando l'utente incolla un link YouTube e dice "elabora questo video", "impara da questo tutorial", "fammi una skill da questo video", "salva questo video nel brain".
---

# yt-learner

Dato un URL YouTube produce due file collegati fra loro:
- `skills/<nome>/SKILL.md`: memoria procedurale (prerequisiti, workflow numerato con timestamp, snippet, troubleshooting, criteri di validazione);
- una scheda Markdown di knowledge: executive summary, concetti, architettura, timeline con deep link `&t=<sec>s`, takeaway, trappole, domande aperte.

## 1. Installa il tool (una volta sola)

Controlla con `yt-learner --help`. Se manca:

```bash
pip install "yt-learner[gemini] @ git+https://github.com/colopisalvatore/yt-learner-claude-plugin"
```

Extra alternativi: `[openai]`, `[anthropic]`, `[all]`. Serve Python >= 3.10; `ffmpeg` solo per le slide.

## 2. Scegli la modalità

Guarda se c'è una chiave: variabili `GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, oppure un `.env` nella cartella corrente. Non stampare mai il valore.

**A. Con chiave: lo fa il tool.**

```bash
yt-learner "<url>" --transcript-out out/<video_id>-transcript.txt
# opzioni utili:
#   --skills-dir <dir> --brain-dir <dir>   dove scrivere skill e scheda
#   --brain <preset|percorso>              destinazione preconfigurata (yt-learner --list-brains)
#   --slides                               estrae le slide vere con ffmpeg
#   --model gpt-4o | claude-...            cambia provider (dedotto dal nome)
#   --lang en,it                           lingue dei sottotitoli
```

Salva sempre la trascrizione con `--transcript-out`: serve per la revisione al passo 3.

**B. Senza chiave, oppure il provider risponde 402/429: scrivi tu.**

```bash
yt-learner "<url>" --offline --dry-run --transcript-out out/<video_id>-transcript.txt
```

Il tool scarica solo metadati (titolo, canale, durata, stampati a video) e la trascrizione con i timestamp, senza scrivere altro. Leggi la trascrizione intera e scrivi tu i due file con le sezioni elencate sopra:
- frontmatter della skill: `name`, `description`, `version`, `tags`, `triggers`, `source_video`, `source_url`, `source_channel`, `generated_at`, `related_knowledge`;
- passi con il timestamp `[mm:ss]` da cui vengono e un **Risultato atteso**;
- comandi che il video non mostra per intero marcati `(ricostruito)`;
- nella scheda la timeline con link `https://www.youtube.com/watch?v=<id>&t=<secondi>s`.

Correggi gli errori tipici dei sottotitoli automatici (nomi di prodotto e di comandi storpiati) e non inventare link o codici che il video "mette in descrizione": nella trascrizione non ci sono.

## 3. Rivedi l'output contro la trascrizione (sempre, anche in modalità A)

Il modello scrive con sicurezza cose che il video non dice. Prima di consegnare cerca nella trascrizione, con `grep`, ogni:
- URL e dominio;
- codice sconto o affiliato;
- nome di modello (il modello tende a mettere quelli del proprio addestramento);
- nome di prodotto, comando, chiave API o credenziale nei prerequisiti.

Quello che non trovi nella trascrizione: correggilo, oppure segnalo come "non verificato". Per i nomi di prodotto dubbi fai una ricerca web. Correggi con modifiche puntuali.

⚠️ Rilanciare il tool sullo stesso video **aggiorna** skill e scheda e cancella le correzioni fatte a mano: se servono di nuovo le slide (es. download andato in timeout), rilancia prima di correggere.

## 4. Consegna

Riporta all'utente: titolo, canale, durata; percorsi dei due file; cosa hai corretto nella revisione; cosa è rimasto non verificato (link in descrizione, slide non estratte).
