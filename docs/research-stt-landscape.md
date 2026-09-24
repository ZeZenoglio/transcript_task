# Research: local vs. cloud speech-to-text — quality vs. speed vs. cost vs. privacy

Phase 1 of [PLAN.md](../PLAN.md). Desk research only, no code changed. Numbers
below are as reported by vendors/aggregators in September 2026; treat published
WER figures the way the rest of this repo treats FLEURS — a relative signal
across providers, not an absolute promise for any specific recording.

## Where this repo already stands

| | |
|---|---|
| ASR | `mlx-whisper` large-v3-turbo, local, Metal-accelerated |
| Measured | ~15x realtime on an M4 base (13:35 audio → 77 s) |
| Measured PT quality | correct on noisy conversational audio where a same-size competitor (Parakeet v3) drifted into English — see [README.md](../README.md) |
| Reference WER | FLEURS pt: large-v3 3.65%, Parakeet TDT 0.6B v3 4.76% (published) |
| Cost | $0 marginal, after a one-time ~1.6 GB download |
| Data egress | none — audio never leaves the machine |

## Cloud/API comparison

| Provider / model | Reported WER (EN, various benchmarks) | Batch price | Portuguese | Diarization | Notes |
|---|---|---|---|---|---|
| **OpenAI `gpt-transcribe`** (successor to `whisper-1`) | competitive with Whisper large-v3 | $0.006/min ≈ $0.36/hr | yes, same multilingual training as Whisper | no | `whisper-1`/`gpt-4o-transcribe` on a deprecation path to Feb 2027; cheapest big-name API |
| **Deepgram Nova-3** | ~5.3% (vendor's own suite); independent academic test found it behind AssemblyAI/Speechmatics on read speech | $0.0043/min batch, $0.0077/min streaming | supported | yes | lowest streaming latency (~450 ms median) — built for live captioning/voice agents, not this repo's batch use case |
| **AssemblyAI Universal-3 Pro** | ~5.6% mean WER | ~$0.37/hr (~$0.006/min) | one of 6 languages in the Pro streaming tier | yes | keyterm prompting up to 1,500 words — useful for domain vocabulary |
| **Azure Speech (Standard)** | competitive, not independently top-ranked | ~$1/hr | supported | yes | priced highest of the majors; buys Azure's compliance/enterprise tooling |
| **Google Chirp 2 (STT v2)** | competitive | ~$0.96/hr ($16/1000 min) | supported | yes | similar price band to Azure |
| **ElevenLabs Scribe v2** | ~3.6% (Realtime, on the AA-WER leaderboard) | $0.22–0.40/hr batch, $0.39–0.48/hr realtime | 99 languages | yes, plus audio-event tagging (laughter, music) | currently the best-reported WER of this group; richest metadata (word/char timestamps, event tags) |

Rough consensus from three independent 2026 comparison write-ups: **the top five
cloud providers now sit within 1–2 WER points of each other** on LibriSpeech/FLEURS-style
benchmarks. The competitive surface has moved to streaming latency, diarization
quality, domain vocabulary support, and cost — not raw accuracy. That matters here:
it means the "cloud is meaningfully more accurate" assumption is no longer
generally true, at least on English/clean-speech benchmarks. Whether it holds
for noisy conversational Portuguese specifically was not independently verifiable
without buying API access and this repo's own actual data — a candidate follow-up
if this project ever grows a cloud fallback.

## Cost at this project's actual scale

The 13:35 of test audio this repo has actually processed:

| Provider | Cost for 13:35 of audio |
|---|---|
| Local (`mlx-whisper`) | $0 (one-time model download only) |
| ElevenLabs Scribe (cheapest tier used) | ~$0.05 |
| OpenAI `gpt-transcribe` | ~$0.08 |
| Azure Speech | ~$0.22 |

At this volume cost is a non-issue either way. It becomes one only at scale — a
few hundred hours a month starts to look like real money against a machine you
already own, and it's the kind of number worth re-running if usage grows.

## Privacy and data residency

This is the axis that actually matters for this project. Two facts from current
guidance are worth stating plainly:

- **Audio containing speech is personal data under GDPR** (voice is treated as a
  potentially identifying characteristic), so sending it to any third-party API
  requires a lawful basis and a signed Data Processing Addendum with that vendor.
- **Data residency and data sovereignty are different things.** Storing the
  original file in an EU bucket does not make the pipeline compliant if a
  US-based API processes it during inference — the processing location is what
  counts, not just storage.

None of the providers above are disqualified by this — all offer DPAs and most
offer an EU processing region — but every one of them requires a contract,
a vendor risk review, and an explicit decision about what leaves the building.
**Running locally removes that entire category of work by construction**: there
is no vendor, no DPA, no residency question, no transfer mechanism to justify.
For a personal utility with no compliance team behind it, that is worth more
than a marginal WER point.

## Recommendation

**Stay local as the default.** The measured quality on this project's actual
audio already competes with, or in Parakeet's case beats, a same-class local
alternative, cloud APIs are converging on similar accuracy to each other, and
the privacy story is categorically simpler with nothing leaving the machine.

Reach for a cloud API instead when:
- **Real-time / streaming is required** (live captioning, a voice agent) — this
  repo's batch pipeline doesn't do this, and Deepgram/AssemblyAI are purpose-built
  for it with sub-500ms latency, which local Whisper is not tuned for.
- **Volume exceeds what one machine can process** in the required turnaround time.
- **Rich diarization/event-tagging is a hard requirement** — ElevenLabs Scribe's
  speaker diarization and audio-event tags are more mature than anything planned
  here (Phase 3's `speakers_detected` is a best-effort LLM guess, not real diarization).
- **A domain vocabulary needs biasing** — AssemblyAI's keyterm prompting handles
  jargon/proper-noun lists better than a general-purpose local model.

None of these apply to this project today. **No model swap is recommended from
this research** — it's logged here for the record, per the Phase 1 exit criterion,
not acted on.

---

Sources: [OpenAI Whisper API pricing 2026](https://diyai.io/ai-tools/speech-to-text/openai-whisper-api-pricing-2026/), [OpenAI transcription pricing (Sep 2026)](https://costgoat.com/pricing/openai-transcription), [Deepgram vs Speechmatics vs AssemblyAI 2026](https://deepgram.com/learn/deepgram-vs-speechmatics-vs-assemblyai), [STT APIs 2026 benchmarks/pricing guide](https://futureagi.com/blog/speech-to-text-apis-in-2026-benchmarks-pricing-developer-s-decision-guide/), [Deepgram Nova-3 pricing explained](https://convertaudiototext.com/blog/deepgram-nova-3-explained), [Introducing Scribe v2](https://elevenlabs.io/blog/introducing-scribe-v2), [ElevenLabs Scribe Portuguese](https://elevenlabs.io/speech-to-text/portuguese), [GDPR-compliant STT with EU data residency](https://dev.to/jamesanderson121/choosing-a-gdpr-compliant-speech-to-text-api-with-eu-data-residency-2nh7), [Data residency for voice/transcription data](https://www.gladia.io/blog/data-residency-for-voice-and-transcription-data-eu-us-and-ai-compliance), [Data residency for speech-to-text: where your audio goes](https://privocio.com/blog/data-residency-speech-to-text-where-audio-goes).
