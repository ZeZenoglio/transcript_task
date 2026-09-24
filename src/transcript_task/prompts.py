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

# The template the pipeline uses today. Swapping this constant -- or, once
# more variants exist, having Settings point at a template id instead --
# changes the refine prompt everywhere without touching refine.py.
REFINE_PROMPT_TEMPLATE = REFINE_PT_V1
