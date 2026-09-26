# Benchmark: main-baseline (fleurs/quick, n=30, seed=42)

- ASR model: `mlx-community/whisper-large-v3-turbo`
- LLM model: `qwen3.5:9b`
- Refine prompt: `refine-pt-v2` · Summarize prompt: `summarize-pt-v2`
- Generated: 2026-09-25T08:23:41+00:00
- Errors: 0/30

## Aggregate metrics

| metric | mean | median | p95 |
|---|---|---|---|
| wer_raw | 0.0214 | 0.0000 | 0.1014 |
| wer_refined | 0.0289 | 0.0000 | 0.1286 |
| cer_raw | 0.0070 | 0.0000 | 0.0240 |
| cer_refined | 0.0103 | 0.0000 | 0.0529 |
| semdist_raw | 0.0126 | 0.0027 | 0.0454 |
| semdist_refined | 0.0111 | 0.0030 | 0.0434 |
| content_recall | 0.9868 | 1.0000 | 1.0000 |
| length_ratio | 0.9980 | 1.0000 | 1.0072 |
| asr_seconds | 7.1989 | 4.8322 | 16.6665 |
| refine_seconds | 12.5118 | 11.6642 | 19.6878 |
| summarize_seconds | 17.7215 | 17.1325 | 22.4478 |
| realtime_factor | 2.3585 | 2.4245 | 3.8661 |
| description_transcript_semdist | 0.1730 | 0.1826 | 0.2728 |
| refine_tokens_per_second | 3.0488 | 2.9318 | 4.9211 |
| summarize_tokens_per_second | 10.3424 | 10.3511 | 12.1580 |

## Summary-stage metrics

| metric | value |
|---|---|
| schema_valid_rate | 1.0000 |
| retry_rate | 0.0000 |
| fallback_rate | 0.0000 |
| slug_uniqueness | 1.0000 |
| title_len_ok_rate | 1.0000 |
| description_len_ok_rate | 1.0000 |
| topic_count_ok_rate | 1.0000 |

## Worst clips by WER (refined)

| clip | duration | wer_refined | wer_raw | error |
|---|---|---|---|---|
| fleurs_row00237 | 11.3s | 0.1905 | 0.0952 |  |
| fleurs_row00879 | 11.0s | 0.1429 | 0.0476 |  |
| fleurs_row00822 | 7.3s | 0.1111 | 0.0556 |  |
| fleurs_row00655 | 18.8s | 0.0930 | 0.0698 |  |
| fleurs_row00077 | 18.5s | 0.0833 | 0.0000 |  |
| fleurs_row00689 | 20.2s | 0.0645 | 0.1290 |  |
| fleurs_row00200 | 9.8s | 0.0476 | 0.0476 |  |
| fleurs_row00737 | 10.9s | 0.0455 | 0.0000 |  |
| fleurs_row00719 | 13.0s | 0.0400 | 0.0400 |  |
| fleurs_row00852 | 15.7s | 0.0312 | 0.0312 |  |
