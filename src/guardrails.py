"""
guardrails.py — Filtros de prompt obligatorios.

Este módulo implementa los guardrails de seguridad no negociables:
1. Bloqueo de contenido que involucre menores de edad en contexto sexual.
2. Bloqueo de generación de imágenes íntimas no consentidas de personas reales.
Los patrones son heuristicas locales: pueden producir falsos positivos y
falsos negativos; no son un clasificador semantico de seguridad.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ==============================================================
# Patrones de detección de menores de edad
# ==============================================================

# Edades explícitas que indican menores (0-17). Se busca el número
# seguido de descriptores de edad o contexto de "años".
_MINOR_AGE_PATTERN = re.compile(
    r"\b(\b(?:1[0-7]|[0-9])\b)\s*"
    r"(?:years?\s*old|yr|años?|añitos?|meses?|months?)\b",
    re.IGNORECASE,
)

# Términos que describen cuerpos/rostros de niños en contexto sexual
_MINOR_BODY_DESCRIPTORS = re.compile(
    r"\b(?:"
    r"little\s*(?:girl|boy|child|kid|teen|loli|shota)"
    r"|prepubescent"
    r"|pubescent"
    r"|jailbait"
    r"|under\s*age"
    r"|underage"
    r"|minor"
    r"|loli(?!con)"  # "loli" sin el sufijo "con" es flagged
    r"|shota"
    r"|child(?:ish)?"
    r"|toddler"
    r"|infant"
    r"|baby"
    r"|infantil"
    r"|prepuber"
    r"|nena"
    r"|nen(?:a|o|ita|ito)"
    r"|chiquit[ao]"
    r")\b",
    re.IGNORECASE,
)

# Contexto sexual en el mismo prompt
SEXUAL_CONTEXT = re.compile(
    r"\b(?:"
    r"nude|desnud[ao]|nsfw|sexual|erot[iao]|porn|xxx"
    r"|sex|fucking|sucking|blowjob|handjob"
    r"|topless|toples|bottomless"
    r"|onlyfans|lewd|pervert"
    r"|nsfw"
    r")\b",
    re.IGNORECASE,
)


# ==============================================================
# Patrones de detección de personas reales
# ==============================================================

# Nombres propios comunes + contexto sexual → bloqueo.
# Lista extensible. Unprompt realista debe ser bloqueado.
_REAL_PERSON_INDICATORS = re.compile(
    r"\b(?:"
    r"(?:photo|picture|image|portrait|selfie|snapshot|pic)\s+of"
    r"|looking\s+like"
    r"|resembling"
    r"|impersonat"
    r"|deepfake"
    r"|real\s+person"
    r"|celebrity"
    r"|actress|actor|singer|model"
    r"|instagram|tiktok|twitter|x\.com"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GuardrailResult:
    """Resultado de la evaluación de un prompt."""
    allowed: bool
    reason: Optional[str] = None
    details: Optional[str] = None


class PromptGuardrail:
    """
    Filtros de prompt obligatorios.

    Diseño:
    - Se rechazan coincidencias con los patrones configurados.
    - Se evalúan patrones combinados (menor + contexto sexual).
    - No se almacenan prompts rechazados en logs por privacidad.
    """

    def __init__(self, extra_blocked_patterns: Optional[list[str]] = None) -> None:
        self._extra_patterns: list[re.Pattern[str]] = []
        if extra_blocked_patterns:
            for pat in extra_blocked_patterns:
                self._extra_patterns.append(re.compile(pat, re.IGNORECASE))

    def evaluate(self, prompt: str) -> GuardrailResult:
        """
        Evalúa un prompt contra todos los guardrails.

        Args:
            prompt: El prompt de entrada del usuario.

        Returns:
            GuardrailResult con allowed=True si es seguro, o
            allowed=False con la razón del bloqueo.
        """
        if not prompt or not prompt.strip():
            return GuardrailResult(allowed=False, reason="El prompt no puede estar vacio.")

        prompt_lower = prompt.lower()

        # --- Guardrail 1: Menores de edad en contexto sexual ---
        has_minor = bool(_MINOR_AGE_PATTERN.search(prompt_lower))
        has_minor = has_minor or bool(_MINOR_BODY_DESCRIPTORS.search(prompt_lower))
        has_sexual = bool(SEXUAL_CONTEXT.search(prompt_lower))

        if has_minor and has_sexual:
            return GuardrailResult(
                allowed=False,
                reason="CONTENIDO BLOQUEADO: El prompt contiene referencias "
                       "a menores de edad en contexto sexual.",
                details="Este proyecto prohíbe estrictamente la generación de "
                        "contenido que involucre menores de edad. Esta restricción "
                        "es no negociable y no puede desactivarse.",
            )

        # --- Guardrail 2: Menores de edad (sin contexto sexual explícito,
        #     pero con descriptores de cuerpo que son ambiguos) ---
        if has_minor:
            return GuardrailResult(
                allowed=False,
                reason="CONTENIDO BLOQUEADO: El prompt contiene términos "
                       "asociados a menores de edad.",
                details="Por seguridad, se bloquean prompts que contengan "
                        "descriptores de edad o cuerpo infantil, ya que "
                        "podrían generar contenido inapropiado.",
            )

        # --- Guardrail 3: Persona real en contexto sexual ---
        has_real_person = bool(_REAL_PERSON_INDICATORS.search(prompt_lower))
        if has_real_person and has_sexual:
            return GuardrailResult(
                allowed=False,
                reason="CONTENIDO BLOQUEADO: El prompt intenta generar "
                       "imágenes íntimas de una persona real.",
                details="Este proyecto restringe la generacion de imagenes "
                        "intimas no consentidas de personas reales.",
            )

        # --- Guardrail 4: Patrones adicionales del usuario ---
        for pattern in self._extra_patterns:
            if pattern.search(prompt_lower):
                return GuardrailResult(
                    allowed=False,
                    reason="CONTENIDO BLOQUEADO: El prompt coincide con un "
                           "patrón de contenido restringido configurado.",
                )

        return GuardrailResult(allowed=True)
