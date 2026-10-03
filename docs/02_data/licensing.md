# Dataset licensing register

Every dataset used by this project is recorded here **before it is used**. A
dataset with no clear license is not used. This register holds what the license is,
where the data came from, when it was retrieved, and what it may be used for.

Numbers here were read from the source pages and from the files actually on disk on
2026-10-03.

---

## iisc-mile-ta — IISc-MILE Tamil ASR Corpus

| Field | Value |
| --- | --- |
| Dataset | IISc-MILE Tamil ASR Corpus (OpenSLR **SLR127**) |
| Language | Tamil (`ta`) |
| License | **CC BY 2.0** — Attribution 2.0 Generic |
| License URL | https://creativecommons.org/licenses/by/2.0/ |
| Authoritative source | https://www.openslr.org/127/ |
| Archive (authoritative) | `mile_tamil_asr_corpus.tar.gz` [13 G] |
| Mirrors | https://openslr.trmal.net/resources/127/, https://openslr.elda.org/resources/127/, https://openslr.magicdatatech.com/resources/127/ |
| Retrieved via | Kaggle reupload `vickythefire2000/iisc-mile-tamil-asr-corpus` using `kagglehub` 1.0.2 |
| Retrieval date | 2026-10-03 |
| Version | none stated in the OpenSLR archive; treated as SLR127 as published |
| Publisher | MILE Lab (Medical Intelligence and Language Engineering), Indian Institute of Science, Bangalore |
| Speakers | 531 |
| Duration | ~150 hours, read speech, noise-free studio recording (high-quality USB mics) |
| Audio format | 16 kHz, 16-bit, mono, PCM WAV |
| Transcription | one UTF-8 `.txt` per `.wav`, same basename, Tamil script |
| Shipped split | `train/` (77,314 utterances) and `test/` (12,087 utterances); no `dev` |
| Permitted use | Commercial and non-commercial use permitted **with attribution** |
| Attribution | Credit MILE Lab, Indian Institute of Science, Bangalore. Cite A. Madhavaraj, Bharathi Pilar, A. G. Ramakrishnan, *Subword Dictionary Learning and Segmentation Techniques for ASR in Tamil and Kannada* (arXiv:2207.13331) and *Knowledge-driven Subword Grammar Modeling for ASR in Tamil and Kannada* (arXiv:2207.13333) |
| Local path | `data/raw/iisc_mile_ta/mile_tamil_asr_corpus/` (outside git) |
| On-disk verified | 89,401 `.wav` + 89,401 `.txt` (77,314 train / 12,087 test); 16.125 GB uncompressed; sample file `ISTL_0000202_0000009.wav` read as 16000 Hz, mono, PCM_16, 8.0 s |

**Provenance caveat.** The archive was obtained from a third-party Kaggle mirror, not
from OpenSLR directly. Its structure, file counts and a sample WAV/transcript match
SLR127, but per-file checksums were **not** compared against the OpenSLR archive, so
byte-identical integrity is unverified. The license that governs this data is
SLR127's **CC BY 2.0**; the Kaggle mirror does not relicense it.

**Split caveat.** The shipped `train`/`test` split is not known to be
speaker-disjoint. Per this project's rule (split by speaker, never by recording),
speaker IDs must be derived from the filenames and a disjoint train/dev/test split
built before any training, or the test numbers will be worthless.

**Not yet obtained (not usable yet):** AI4Bharat IndicVoices (Tamil) and Kathbath
(Tamil) for spontaneous/conversational diversity — licenses and retrieval to be
recorded here when acquired.
