"""Word document generation for one transcribed recording."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .settings import Settings

# Labels for the summary section (title/abstract/topics/warnings), which
# follows settings.summary_language since its content is LLM-authored prose
# in that language. The provenance block below it (file/model/timing facts)
# stays Portuguese-labeled regardless -- it's fixed technical metadata, not
# narrative content, and was already shipped before Phase 3 added summaries.
_LABELS = {
    "pt": {
        "topics": "Temas: ",
        "sensitivity_high": "⚠ Este documento pode conter informação sensível "
                             "(pessoal, financeira, médica ou legal). Reveja antes de partilhar.",
        "sensitivity_medium": "⚠ Este documento pode conter informação privada. "
                               "Reveja antes de partilhar.",
        "low_confidence": "Resumo automático de baixa confiança — reveja o título e a descrição.",
    },
    "en": {
        "topics": "Topics: ",
        "sensitivity_high": "⚠ This document may contain sensitive information "
                             "(personal, financial, medical, or legal). Review before sharing.",
        "sensitivity_medium": "⚠ This document may contain private information. "
                               "Review before sharing.",
        "low_confidence": "Low-confidence automatic summary — please review the title and description.",
    },
}


def human_duration(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:d}:{s:02d}"


# OOXML core properties (title/subject/keywords/...) are metadata fields, not
# document content -- python-docx enforces the format's 255-unicode-character
# cap and raises ValueError past it. A 3-6 sentence description routinely
# exceeds that, so anything going into a core property needs truncating; the
# full, untruncated description still appears in the document body below.
_CORE_PROP_LIMIT = 255


def _truncated(text: str, limit: int = _CORE_PROP_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def write_docx(key: str, item: dict, out_path: Path, settings: Settings) -> None:
    """Write one reviewed transcript as a .docx.

    `item` is the corresponding entry from the transcripts.json state dict --
    see pipeline.py's stage_docx for the fields it's expected to carry.
    """
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, RGBColor

    body = item.get("refined_transcript") or item.get("raw_transcript")
    if not body:
        raise ValueError(f"{key}: no transcript to write")

    summary = item.get("summary")
    labels = _LABELS.get(settings.summary_language, _LABELS["pt"])

    doc = Document()
    doc.core_properties.title = _truncated(summary["title"] if summary else f"Transcrição — {key}")
    doc.core_properties.comments = _truncated(f"Ficheiro de origem: {key}")
    if summary:
        doc.core_properties.subject = _truncated(summary["description"])
        if summary.get("topics"):
            doc.core_properties.keywords = _truncated(", ".join(summary["topics"]))

    doc.add_heading("Transcrição de Áudio", level=0)

    if summary:
        doc.add_heading(summary["title"], level=1)

        if summary["sensitivity"] in ("medium", "high"):
            banner = doc.add_paragraph()
            banner_run = banner.add_run(labels[f"sensitivity_{summary['sensitivity']}"])
            banner_run.bold = True
            banner_run.font.color.rgb = (
                RGBColor(0xB0, 0x00, 0x00) if summary["sensitivity"] == "high"
                else RGBColor(0xB0, 0x70, 0x00)
            )

        doc.add_paragraph(summary["description"])

        if summary.get("topics"):
            topics_p = doc.add_paragraph()
            topics_p.add_run(labels["topics"]).italic = True
            topics_p.add_run(", ".join(summary["topics"])).italic = True

        if summary["confidence"] == "low":
            conf_note = doc.add_paragraph()
            conf_run = conf_note.add_run(labels["low_confidence"])
            conf_run.italic = True
            conf_run.font.size = Pt(9)
            conf_run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

        doc.add_paragraph()

    # Provenance block: this is what ties the document back to its recording.
    meta = doc.add_paragraph()
    meta.add_run("Ficheiro de origem: ").bold = True
    meta.add_run(key)
    meta.add_run("\nDuração: ").bold = True
    meta.add_run(human_duration(item["duration_seconds"])
                 if item.get("duration_seconds") else "desconhecida")
    meta.add_run("\nFormato original: ").bold = True
    meta.add_run(f"{item.get('source_format', '?')} · "
                 f"{item.get('source_sample_rate', '?')} Hz · "
                 f"{item.get('source_channels', '?')} canal(is)")
    meta.add_run("\nModelo de transcrição: ").bold = True
    meta.add_run(settings.asr_model)
    meta.add_run("\nModelo de revisão: ").bold = True
    meta.add_run(settings.llm_model if item.get("refined_transcript")
                 else "— (texto bruto, sem revisão)")
    if summary:
        meta.add_run("\nFalantes estimados: ").bold = True
        meta.add_run(str(summary["speakers_detected"]) if summary["speakers_detected"] else "desconhecido")
        meta.add_run("\nVariante detectada: ").bold = True
        meta.add_run(summary["language_variant"])
    meta.add_run("\nGerado em: ").bold = True
    meta.add_run(datetime.now().strftime("%Y-%m-%d %H:%M"))
    for run in meta.runs:
        run.font.size = Pt(9)

    note = doc.add_paragraph()
    note_run = note.add_run(
        "Documento gerado automaticamente. Trechos marcados com [?] são "
        "incertos e requerem confirmação humana."
    )
    note_run.italic = True
    note_run.font.size = Pt(9)
    note_run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

    doc.add_paragraph()
    doc.add_heading("Transcrição revista" if item.get("refined_transcript")
                    else "Transcrição (bruta)", level=1)

    for block in [b.strip() for b in body.split("\n") if b.strip()]:
        para = doc.add_paragraph(block)
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        para.paragraph_format.space_after = Pt(8)

    # Keep the unedited ASR output alongside it so a reviewer can check any
    # correction the model made against what was actually heard.
    if item.get("refined_transcript") and item.get("raw_transcript"):
        doc.add_page_break()
        doc.add_heading("Anexo — transcrição automática original", level=1)
        anexo = doc.add_paragraph(item["raw_transcript"])
        anexo.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for run in anexo.runs:
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    doc.save(out_path)
