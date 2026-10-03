# Dataset sources

Where each dataset was obtained from, and the exact command or page used. Companion
to `licensing.md`, which holds the license terms. Retrieval date is recorded here so
a re-download can be compared against it.

---

## iisc-mile-ta — IISc-MILE Tamil ASR Corpus

**Authoritative source (reference):**
- OpenSLR SLR127 — https://www.openslr.org/127/
- Archive: `mile_tamil_asr_corpus.tar.gz` [13 G]
- Mirror used for reference: `https://openslr.trmal.net/resources/127/mile_tamil_asr_corpus.tar.gz`
  (also `https://openslr.elda.org/resources/127/`, `https://openslr.magicdatatech.com/resources/127/`)

**Obtained via (on 2026-10-03):** the Kaggle reupload
`vickythefire2000/iisc-mile-tamil-asr-corpus`, with:

```python
import kagglehub
path = kagglehub.dataset_download("vickythefire2000/iisc-mile-tamil-asr-corpus")
```

kagglehub cache path:
`C:\Users\Rajan\.cache\kagglehub\datasets\vickythefire2000\iisc-mile-tamil-asr-corpus\versions\1`

Then moved (not copied) into the repo at:
`data/raw/iisc_mile_ta/mile_tamil_asr_corpus/`

**Why the mirror and not OpenSLR directly:** the OpenSLR archive is 13 G; the Kaggle
mirror resolved to the same corpus (`mile_tamil_asr_corpus/` with `train/` and
`test/`, each holding `audio_files/` + `trans_files/`) and was faster to fetch. The
governing license is still SLR127's CC BY 2.0 — see `licensing.md`.

**Payload size:** 16.125 GB on disk after extraction (89,401 WAV + 89,401 TXT).

**Re-download note:** the Kaggle URL is a mirror and may be removed or changed. If it
disappears, fall back to the OpenSLR mirror above; the corpus and license are the
same.

**Not yet obtained:** AI4Bharat IndicVoices (Tamil) and Kathbath (Tamil). Planned
sources: Hugging Face `ai4bharat/IndicVoices` / AI4Bharat, or AIKosh
(https://aikosh.indiaai.gov.in). Record license and retrieval here when acquired.
