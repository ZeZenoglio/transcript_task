"""LLM-written interpretation of a benchmark's already-computed metrics.

Per PLAN.md decision #6: no LLM-as-judge scoring, anywhere. This module
does not score anything -- it reads the metrics table `BenchmarkResult`
already computed deterministically and writes a short prose interpretation
of it (what moved, what likely caused it, what to look at), logged as an
MLflow artifact alongside the raw numbers. Every number in the output
traces back to a deterministic metric; the LLM only writes about numbers
that already exist; it never sees the audio or the transcripts, and never
produces one of its own.
"""

from __future__ import annotations

from .results import BenchmarkResult

_SYSTEM_PT = (
    "Escreves relatórios técnicos curtos e objetivos sobre métricas de "
    "avaliação de um pipeline de transcrição de áudio. Baseias-te "
    "exclusivamente na tabela de métricas fornecida -- nunca inventas "
    "números, nunca atribuis uma pontuação própria, e assinalas quando os "
    "dados são insuficientes para uma conclusão firme."
)

_INSTRUCTIONS_PT = """\
Eis a tabela de métricas de uma execução de avaliação (benchmark) do \
pipeline de transcrição:

{table}

Escreve uma interpretação curta em markdown (4-8 frases) desta execução: \
o que se destaca, o que pode explicar os números (ex: erosão de WER pelo \
estágio de refinamento, taxa alta de fallback do resumo), e o que vale a \
pena investigar a seguir. Não inventes números que não estão na tabela. \
Não atribuas uma pontuação geral própria."""


def build_interpretation_prompt(result: BenchmarkResult) -> str:
    return _INSTRUCTIONS_PT.format(table=result.markdown_table())


def write_interpretation(result: BenchmarkResult, model) -> str:
    """Ask a local LLM to interpret the already-computed metrics table.

    Never raises: an LLM outage shouldn't take down a benchmark run whose
    deterministic metrics are already computed and saved. On failure,
    returns a short, honest placeholder instead of a fabricated report.
    """
    try:
        return model.chat(
            system=_SYSTEM_PT,
            user=build_interpretation_prompt(result),
            options={"temperature": 0.2},
        )
    except Exception as exc:  # noqa: BLE001 - interpretation is best-effort
        return (
            "_Interpretação automática indisponível "
            f"(falha ao contactar o modelo local: {exc})._\n"
        )
