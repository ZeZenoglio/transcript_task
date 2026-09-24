# Fixture attribution

The four audio clips in `audio/` are sourced from
[fsicoli/common_voice_17_0](https://huggingface.co/datasets/fsicoli/common_voice_17_0)
(`pt` config, `test` split, revision `8262c16bf297c87a9cd88c51997c4758ed7a8ba2`),
a plain-file republication of Mozilla's official
[Common Voice](https://commonvoice.mozilla.org/) corpus, licensed
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain
dedication).

**Why this republication and not the official `mozilla-foundation` org's
Hugging Face repo:** the official repo ships a Python dataset-loading script;
the `datasets` library version this project uses (5.x) has removed support
for running those entirely, so it can't be loaded at all through the
standard `datasets.load_dataset` API any more. This mirror publishes the
same underlying Common Voice release as plain per-language audio archives
and transcript files instead, which this project's own
`scripts/fetch_common_voice.py` downloads directly (bypassing `datasets`
altogether) and re-licenses under the same CC0 terms Common Voice itself
uses.

**Changes made:** each clip was transcoded from Common Voice's native mp3
into a different format (`.mp3` unchanged, `.wav`, `.m4a`, `.opus`) with
ffmpeg, purely to exercise this project's format-normalization code across
codecs. No audio content, pitch, or speed was altered. See
`scripts/build_fixtures.py`.

These four clips were selected for duration diversity (shortest, longest,
and two evenly spaced points between them across the full ~9,467-clip test
split) -- see `manifest.jsonl` for the ground-truth transcript and duration
of each. Per-clip `client_id` (a hashed, pseudonymous contributor id used
upstream to track which clips share a speaker) is deliberately **not**
carried into this repository's manifest -- nothing here needs it.

**Why a second dataset alongside FLEURS at all:** FLEURS (Phase 4) is clean,
studio-quality read speech. Common Voice contributors record themselves on
whatever device they have, in whatever room they're in, so these clips carry
real background noise, mic-quality variance, and accent diversity FLEURS'
recordings don't -- a genuinely noisier acoustic tier for the evaluation
harness (Phase 6), even though the *content* is still someone reading a
prompted sentence rather than free conversation.

No content in this directory depicts a real, identifiable individual in a
private context: Common Voice contributors explicitly donate their reading
of a public-domain or crowd-submitted sentence to this public corpus,
knowing it will be redistributed for exactly this kind of reuse.
