"""Prompt templates for the LLM stages.

Each template carries an id so a transcript's provenance record, and later
an eval run, can pin exactly which wording produced a given result -- useful
the moment more than one variant of a prompt exists (which Phase 3's
summarisation stage will add).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    system: str
    user_template: str

    def render(self, **kwargs: str) -> str:
        return self.user_template.format(**kwargs)


REFINE_PT_V1 = PromptTemplate(
    id="refine-pt-v1",
    system=(
        "És um revisor profissional de transcrições em português. "
        "Devolves sempre apenas o texto revisto, sem comentários nem explicações."
    ),
    user_template="""Abaixo está a transcrição automática de um ficheiro de áudio em português.
A transcrição foi gerada por um modelo de reconhecimento de fala e pode conter erros.

A tua tarefa é preparar este texto para revisão humana. Regras obrigatórias:

1. CORRIGE erros evidentes de reconhecimento de fala, ortografia, acentuação e concordância.
2. PONTUA e divide em frases e parágrafos de forma natural e legível.
3. REMOVE apenas ruído de oralidade sem conteúdo (hesitações como "hum", "ãh", gaguejos e
   repetições acidentais da mesma palavra). Mantém repetições que sejam intencionais ou enfáticas.
4. NÃO inventes informação. NÃO resumas. NÃO omitas ideias. NÃO traduzas.
5. PRESERVA o registo do falante (informal, coloquial, ou mesmo grosseiro). Não suavizes o tom.
6. MANTÉM o português original do falante (europeu ou brasileiro). Não converjas para outra variante.
7. Quando um trecho for inaudível ou ambíguo e não conseguires deduzir com confiança,
   mantém a tua melhor hipótese seguida de [?].
8. Se identificares mudança de interlocutor, inicia um novo parágrafo.

Responde APENAS com o texto revisto.

TRANSCRIÇÃO ORIGINAL:
---
{transcript}
---""",
)

# v2 (Phase 6 prompt-tuning follow-up): v1 measurably raised WER on both the
# FLEURS and Common Voice benchmarks (Phase 6) -- refine made 7/30 real
# clips worse and only 1/30 better. Manually inspecting raw-vs-refined pairs
# (not just the aggregate metric) found a consistent pattern: v1 volunteers
# small, unforced rewrites of text that was already correct -- "uma" ->
# "numa" with no rule justifying it, "das ondas na praia voltando ao mar" ->
# "que volta ao mar" (a stylistic rewrite of an already-grammatical
# sentence), and one actively wrong "agreement fix" ("mais baratas" ->
# "mais baratos") that guessed the wrong noun for an ambiguous adjective and
# flipped a correct sentence into an incorrect one. None of these are
# hallucination or the runaway-generation failure mode (Phase 6's other
# follow-up) -- they're a system prompt ("professional reviewer") and rule
# set that never told the model NOT to polish text that didn't need it.
# v2 adds an explicit minimal-edit principle ahead of the other rules, and
# narrows the concordância rule to stop guessing on ambiguous cases instead
# of taking a side.
REFINE_PT_V2 = PromptTemplate(
    id="refine-pt-v2",
    system=(
        "És um revisor de transcrições em português cujo objetivo é fazer o MENOR número "
        "de alterações possível -- corriges apenas o que estiver de facto errado, nunca "
        "reescreves por estilo. Devolves sempre apenas o texto revisto, sem comentários "
        "nem explicações."
    ),
    user_template="""Abaixo está a transcrição automática de um ficheiro de áudio em português.
A transcrição foi gerada por um modelo de reconhecimento de fala e pode conter erros.

A tua tarefa é preparar este texto para revisão humana. Regra de ouro, mais importante
que todas as outras: se uma palavra, expressão ou frase já estiver correta e compreensível,
mantém-na exatamente como está, mesmo que imagines uma forma alternativa, mais elegante ou
mais formal de a escrever. Não estás a melhorar o texto -- estás apenas a corrigir erros
genuínos. Na dúvida, NÃO alteres.

Regras obrigatórias:

1. CORRIGE apenas erros evidentes e inequívocos de reconhecimento de fala, ortografia e
   acentuação. Corrige concordância apenas quando o erro for óbvio e a frase não permitir
   outra leitura; se houver ambiguidade sobre a que palavra um termo se refere (ex.: a que
   substantivo um adjetivo concorda), mantém a frase tal como foi transcrita em vez de
   arriscar uma correção errada.
2. PONTUA e divide em frases e parágrafos de forma natural e legível -- isto não conta como
   reescrever, mas não aproveites para alterar palavras ao mesmo tempo.
3. REMOVE apenas ruído de oralidade sem conteúdo (hesitações como "hum", "ãh", gaguejos e
   repetições acidentais da mesma palavra). Mantém repetições que sejam intencionais ou enfáticas.
4. NÃO substituas palavras por sinónimos, NÃO reordenes frases e NÃO reformules construções
   que já estejam corretas, mesmo que o resultado te pareça mais fluido.
5. NÃO inventes informação. NÃO resumas. NÃO omitas ideias. NÃO traduzas.
6. PRESERVA o registo do falante (informal, coloquial, ou mesmo grosseiro). Não suavizes o tom.
7. MANTÉM o português original do falante (europeu ou brasileiro). Não converjas para outra variante.
8. Quando um trecho for inaudível ou ambíguo e não conseguires deduzir com confiança,
   mantém a tua melhor hipótese seguida de [?].
9. Se identificares mudança de interlocutor, inicia um novo parágrafo.

Responde APENAS com o texto revisto.

TRANSCRIÇÃO ORIGINAL:
---
{transcript}
---""",
)

REFINE_PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    "refine-pt-v1": REFINE_PT_V1,
    "refine-pt-v2": REFINE_PT_V2,
}


def get_refine_template(prompt_id: str) -> PromptTemplate:
    try:
        return REFINE_PROMPT_TEMPLATES[prompt_id]
    except KeyError:
        raise ValueError(
            f"No refine prompt with id {prompt_id!r}; "
            f"available: {sorted(REFINE_PROMPT_TEMPLATES)}"
        ) from None


# The template the pipeline uses by default. `Settings.refine_prompt_id`
# (env-overridable, see settings.py) is what actually selects the prompt at
# runtime via get_refine_template() -- this constant is kept only as the
# literal default value for that setting, not read directly by pipeline.py
# or the eval harness any more.
REFINE_PROMPT_TEMPLATE = REFINE_PT_V1


# ---------------------------------------------------------------------------
# Summarization (Phase 3): title + description + metadata, as JSON.
#
# These render with {raw_transcript} and {refined_transcript}. The schema
# itself (field names, types, allowed values) is enforced by Ollama's
# structured-output decoding -- see summarize.py -- so these prompts describe
# what each field *means*, not its shape. language_variant always asks about
# the recording's own language, regardless of which template (pt/en) is used
# to write the summary itself.
# ---------------------------------------------------------------------------

_SUMMARIZE_FIELDS_PT = """Produz um resumo estruturado com os seguintes campos:
- title: um título curto (até 80 caracteres), descritivo, sem ponto final, seguro para nome de ficheiro.
- description: 3 a 6 frases a explicar do que trata a gravação.
- topics: lista de 3 a 8 palavras-chave ou temas.
- speakers_detected: número estimado de interlocutores distintos (0 se não for possível estimar).
- language_variant: "pt-PT", "pt-BR" ou "unknown" -- a variante do português falada na gravação.
- sensitivity: "low", "medium" ou "high" -- assinala "medium"/"high" se a gravação contiver
  informação pessoal, financeira, médica, legal ou de outro modo sensível.
- confidence: a tua confiança própria neste resumo ("low", "medium" ou "high")."""

# v2 (Phase 3b): added the PII-avoidance rule below. This filename/metadata
# built from title/description/topics can travel more casually than the full
# transcript (file properties, folder listings), so this is the primary --
# not the only -- defense; anonymize.py's NER/regex pass is the deterministic
# backstop for when a model ignores this instruction under real content
# pressure. The instruction is scoped to these three fields only: it does
# not, and must not, affect the transcript itself.
_PII_RULE_PT = """IMPORTANTE -- privacidade: title, description e topics podem ser usados para
nomear o ficheiro e aparecem nas propriedades do documento, que podem circular
independentemente do documento completo. NÃO incluas nomes completos de pessoas privadas,
números de telefone, moradas, ou números de identificação/contribuinte nestes três campos.
Refere-te às pessoas pela função ou relação (ex.: "um interlocutor", "a cliente", "um familiar"),
não pelo nome. Esta regra aplica-se apenas a title/description/topics -- não te preocupes com
isto para os restantes campos, e nunca alteres a transcrição em si."""

SUMMARIZE_PT_V2 = PromptTemplate(
    id="summarize-pt-v2",
    system=(
        "És um assistente que resume transcrições de áudio. Respondes sempre "
        "apenas com um objeto JSON válido, sem comentários, sem markdown, "
        "sem texto antes ou depois."
    ),
    user_template=f"""Analisa a transcrição de um ficheiro de áudio, apresentada em duas versões:
a transcrição automática bruta (com possíveis erros de reconhecimento de fala) e uma
versão revista por um segundo modelo (mais legível, mas ainda passível de erro).

{_SUMMARIZE_FIELDS_PT}

{_PII_RULE_PT}

Usa a transcrição revista como base do conteúdo, mas usa a transcrição bruta como pista
adicional sobre número de falantes, registo, e trechos onde a revisão possa ter alterado
o significado.

Responde APENAS com um objeto JSON válido.

TRANSCRIÇÃO BRUTA:
---
{{raw_transcript}}
---

TRANSCRIÇÃO REVISTA:
---
{{refined_transcript}}
---""",
)

_SUMMARIZE_FIELDS_EN = """Produce a structured summary with these fields:
- title: a short descriptive title (max 80 characters), no trailing period, filename-safe.
- description: 3 to 6 sentences describing what the recording is about.
- topics: a list of 3 to 8 keywords or themes.
- speakers_detected: estimated number of distinct speakers (0 if you cannot tell).
- language_variant: "pt-PT", "pt-BR", or "unknown" -- the Portuguese variant spoken in
  the recording, regardless of the language you write this summary in.
- sensitivity: "low", "medium", or "high" -- flag "medium"/"high" if the recording
  contains personal, financial, medical, legal, or otherwise private information.
- confidence: your own confidence in this summary ("low", "medium", or "high")."""

_PII_RULE_EN = """IMPORTANT -- privacy: title, description and topics may be used to name the
file and appear in the document's properties, which can circulate independently of the full
document. Do NOT include private individuals' full names, phone numbers, addresses, or
ID/tax numbers in these three fields. Refer to people by role or relationship
(e.g. "a caller", "the client", "a family member"), not by name. This rule applies only to
title/description/topics -- don't worry about it for the other fields, and never alter the
transcript itself."""

SUMMARIZE_EN_V2 = PromptTemplate(
    id="summarize-en-v2",
    system=(
        "You are an assistant that summarizes audio transcripts. You always "
        "respond with a single valid JSON object only, no commentary, no "
        "markdown, no text before or after."
    ),
    user_template=f"""Analyse the transcript of an audio recording, given in two versions: the raw
automatic transcript (may contain speech-recognition errors) and a cleaned-up version
produced by a second model (more readable, but still possibly imperfect).

{_SUMMARIZE_FIELDS_EN}

{_PII_RULE_EN}

Use the cleaned-up transcript as the basis for content, but use the raw transcript as
an extra clue about speaker count, register, and any place cleanup may have changed
the meaning.

Respond with ONLY a valid JSON object.

RAW TRANSCRIPT:
---
{{raw_transcript}}
---

CLEANED-UP TRANSCRIPT:
---
{{refined_transcript}}
---""",
)

SUMMARIZE_PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    "pt": SUMMARIZE_PT_V2,
    "en": SUMMARIZE_EN_V2,
}


def get_summarize_template(language: str) -> PromptTemplate:
    try:
        return SUMMARIZE_PROMPT_TEMPLATES[language]
    except KeyError:
        raise ValueError(
            f"No summarize prompt for language {language!r}; "
            f"available: {sorted(SUMMARIZE_PROMPT_TEMPLATES)}"
        ) from None
