from __future__ import annotations

import itertools
import re
from dataclasses import replace

from .types import VinValidationResult

VIN_LENGTH = 17
FORBIDDEN_CHARS = frozenset("IOQ")
VIN_ALLOWED_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")
VIN_LOOSE_RE = re.compile(r"[A-Z0-9]{17}")
CHASSIS_CHARSET_RE = re.compile(r"^[A-Z0-9]+$")

COMMON_SUBSTITUTIONS: dict[str, tuple[str, ...]] = {
    "O": ("0",),
    "0": ("O",),
    "I": ("1",),
    "1": ("I",),
    "S": ("5",),
    "5": ("S",),
    "B": ("8",),
    "8": ("B",),
    "Z": ("2",),
    "2": ("Z",),
}

MODEL_YEAR_CODES: dict[str, int] = {
    "A": 2010,
    "B": 2011,
    "C": 2012,
    "D": 2013,
    "E": 2014,
    "F": 2015,
    "G": 2016,
    "H": 2017,
    "J": 2018,
    "K": 2019,
    "L": 2020,
    "M": 2021,
    "N": 2022,
    "P": 2023,
    "R": 2024,
    "S": 2025,
    "T": 2026,
    "V": 2027,
    "W": 2028,
    "X": 2029,
    "Y": 2030,
    "1": 2031,
    "2": 2032,
    "3": 2033,
    "4": 2034,
    "5": 2035,
    "6": 2036,
    "7": 2037,
    "8": 2038,
    "9": 2039,
}

# Starter WMI map. Extend from company data as labels are cleaned.
DEFAULT_WMI_MANUFACTURERS: dict[str, str] = {
    "JTD": "Toyota",
    "JTJ": "Lexus",
    "JTM": "Toyota",
    "JTN": "Toyota",
    "KMH": "Hyundai",
    "KMF": "Hyundai",
    "KNA": "Kia",
    "KND": "Kia",
    "KL1": "Chevrolet",
    "MNT": "Nissan",
    "MR0": "Toyota",
    "NMT": "Toyota",
    "SAL": "Land Rover",
    "SCA": "Rolls-Royce",
    "TRU": "Audi",
    "VF1": "Renault",
    "WBA": "BMW",
    "WBS": "BMW",
    "WDD": "Mercedes-Benz",
    "WDB": "Mercedes-Benz",
    "WDC": "Mercedes-Benz",
    "WVW": "Volkswagen",
    "ZFF": "Ferrari",
    "1C4": "Jeep",
    "1FA": "Ford",
    "1FM": "Ford",
    "1G1": "Chevrolet",
    "1G6": "Cadillac",
    "1HG": "Honda",
    "1N4": "Nissan",
    "2T1": "Toyota",
    "3FA": "Ford",
    "3N1": "Nissan",
    "4T1": "Toyota",
    "5N1": "Nissan",
    "5TD": "Toyota",
}


def clean_chassis_no(value: str) -> str:
    """Clean a labelled chassis number into uppercase A-Z/0-9 text."""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def is_valid_chassis_format(chassis_no: str) -> bool:
    """Return True when a chassis number has the basic UAE VIN format."""
    cleaned = clean_chassis_no(chassis_no)
    return (
        len(cleaned) == VIN_LENGTH
        and CHASSIS_CHARSET_RE.fullmatch(cleaned) is not None
        and not any(char in cleaned for char in FORBIDDEN_CHARS)
    )


def validate_chassis_no(chassis_no: str) -> dict:
    """Return simple validation flags for a chassis number.

    This intentionally does not use VIN checksum validation.
    """
    cleaned = clean_chassis_no(chassis_no)
    length_valid = len(cleaned) == VIN_LENGTH
    charset_valid = CHASSIS_CHARSET_RE.fullmatch(cleaned) is not None
    forbidden_chars_present = any(char in cleaned for char in FORBIDDEN_CHARS)

    return {
        "cleaned": cleaned,
        "length_valid": length_valid,
        "charset_valid": charset_valid,
        "forbidden_chars_present": forbidden_chars_present,
        "format_valid": length_valid and charset_valid and not forbidden_chars_present,
    }


def normalize_vin_text(value: str | None) -> str:
    if not value:
        return ""
    return clean_chassis_no(value)


def extract_loose_vins(text: str | None) -> list[str]:
    if not text:
        return []

    normalized_text = text.upper()
    candidates: list[str] = []

    tokens = re.findall(r"[A-Z0-9]+", normalized_text)
    for start in range(len(tokens)):
        combined = ""
        for token in tokens[start:]:
            combined += normalize_vin_text(token)
            if len(combined) == VIN_LENGTH:
                candidates.append(combined)
                break
            if len(combined) > VIN_LENGTH:
                break

    return list(dict.fromkeys(candidates))


def model_year_from_vin(vin: str) -> int | None:
    if len(vin) != VIN_LENGTH:
        return None
    return MODEL_YEAR_CODES.get(vin[9])


def validate_vin(
    vin: str,
    wmi_manufacturers: dict[str, str] | None = None,
    expected_make: str | None = None,
    expected_year: str | int | None = None,
) -> VinValidationResult:
    normalized = normalize_vin_text(vin)
    reasons: list[str] = []
    score = 1.0

    if len(normalized) != VIN_LENGTH:
        reasons.append("vin_length_not_17")
        score -= 0.55

    if not normalized.isalnum():
        reasons.append("vin_contains_non_alphanumeric")
        score -= 0.35

    forbidden = sorted(set(normalized) & FORBIDDEN_CHARS)
    if forbidden:
        reasons.append(f"vin_contains_forbidden_chars:{''.join(forbidden)}")
        score -= 0.45

    if len(normalized) == VIN_LENGTH and not VIN_ALLOWED_RE.fullmatch(normalized):
        reasons.append("vin_not_allowed_pattern")
        score -= 0.35

    wmi_map = wmi_manufacturers or DEFAULT_WMI_MANUFACTURERS
    manufacturer = wmi_map.get(normalized[:3])
    if len(normalized) >= 3 and not manufacturer:
        reasons.append("wmi_unknown")
        score -= 0.12

    if expected_make and manufacturer:
        if _clean_name(expected_make) != _clean_name(manufacturer):
            reasons.append("manufacturer_metadata_mismatch")
            score -= 0.22
        else:
            score += 0.08

    model_year = model_year_from_vin(normalized)
    if expected_year and model_year:
        try:
            if int(expected_year) != model_year:
                reasons.append("year_metadata_mismatch")
                score -= 0.18
            else:
                score += 0.06
        except (TypeError, ValueError):
            reasons.append("year_metadata_unparseable")
            score -= 0.05

    is_valid = (
        len(normalized) == VIN_LENGTH
        and VIN_ALLOWED_RE.fullmatch(normalized) is not None
        and not any(reason.endswith("_mismatch") for reason in reasons)
    )
    return VinValidationResult(
        vin=normalized,
        is_valid=is_valid,
        score=max(0.0, min(1.0, score)),
        reasons=tuple(reasons),
        manufacturer=manufacturer,
        model_year=model_year,
    )


def generate_corrections(
    vin: str,
    max_changes: int = 2,
    max_candidates: int = 250,
) -> list[str]:
    normalized = normalize_vin_text(vin)
    if len(normalized) != VIN_LENGTH:
        return []

    variants: list[list[str]] = []
    mutable_indexes: list[int] = []
    for index, char in enumerate(normalized):
        options = [char, *COMMON_SUBSTITUTIONS.get(char, ())]
        variants.append(list(dict.fromkeys(options)))
        if len(options) > 1:
            mutable_indexes.append(index)

    results: list[str] = []
    seen = {normalized}
    for changes in range(1, max_changes + 1):
        for indexes in itertools.combinations(mutable_indexes, changes):
            replacement_sets = [variants[index][1:] for index in indexes]
            for replacements in itertools.product(*replacement_sets):
                chars = list(normalized)
                for index, replacement in zip(indexes, replacements):
                    chars[index] = replacement
                candidate = "".join(chars)
                if candidate not in seen:
                    seen.add(candidate)
                    results.append(candidate)
                    if len(results) >= max_candidates:
                        return results
    return results


def best_validated_variant(
    vin: str,
    wmi_manufacturers: dict[str, str] | None = None,
    expected_make: str | None = None,
    expected_year: str | int | None = None,
) -> VinValidationResult:
    base = validate_vin(vin, wmi_manufacturers, expected_make, expected_year)
    best = base
    for corrected in generate_corrections(vin):
        validation = validate_vin(corrected, wmi_manufacturers, expected_make, expected_year)
        if validation.score > best.score:
            best = replace(validation, reasons=(*validation.reasons, "ocr_correction_applied"))
    return best


def _clean_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())
