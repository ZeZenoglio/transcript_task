# Fixture attribution

The four audio clips in `audio/` are sourced from
[Google FLEURS](https://huggingface.co/datasets/google/fleurs)
(`pt_br` config, `test` split, revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**Changes made:** each clip was transcoded from FLEURS' original 16kHz mono
float32 WAV into a different format (`.wav` unchanged, `.mp3`, `.m4a`,
`.opus`) with ffmpeg, purely to exercise this project's format-normalization
code across codecs. No audio content, pitch, or speed was altered. See
`scripts/build_fixtures.py`.

These four clips were selected for duration diversity (shortest, longest, and
two evenly spaced points between them across the full 919-clip test split) --
see `manifest.jsonl` for the ground-truth transcript and duration of each.

No content in this directory depicts a real, identifiable individual in a
private context: FLEURS is a public read-speech corpus of volunteers reading
public-domain and Wikipedia-sourced sentences.
