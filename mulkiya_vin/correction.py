from __future__ import annotations

import itertools

from .vin import clean_chassis_no, is_valid_chassis_format

VIN_CONFUSIONS = {
    "S": ["5"],
    "5": ["S"],
    "B": ["8"],
    "8": ["B"],
    "Z": ["2"],
    "2": ["Z"],
    "G": ["6"],
    "6": ["G"],
    "T": ["1"],
    "1": ["T"],
    "I": ["1"],
    "O": ["0"],
    "0": ["O"],
}

LIKELY_WMI_PREFIXES = [
    "JN1", "JN6", "JN8",
    "5N1",
    "3PC",
    "MDH",
    "MNT",
    "VF1",
    "1GN",
    "1G1",
]

LIKELY_VIN_CHARACTERS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"


def generate_candidate_corrections(candidate: str, max_changes: int = 3) -> list[str]:
    """Generate likely OCR corrections for one 17-character chassis candidate."""
    cleaned = clean_chassis_no(candidate)
    if len(cleaned) != 17:
        return []

    mutable_indexes = [
        index for index, char in enumerate(cleaned) if char in VIN_CONFUSIONS
    ]

    corrections: list[str] = []
    seen = {cleaned}
    for change_count in range(1, max_changes + 1):
        for indexes in itertools.combinations(mutable_indexes, change_count):
            replacement_sets = [VIN_CONFUSIONS[cleaned[index]] for index in indexes]
            for replacements in itertools.product(*replacement_sets):
                chars = list(cleaned)
                for index, replacement in zip(indexes, replacements):
                    chars[index] = replacement
                corrected = "".join(chars)
                if corrected in seen:
                    continue
                seen.add(corrected)
                if is_valid_chassis_format(corrected):
                    corrections.append(corrected)

    return corrections


def hamming_distance(a: str, b: str) -> int:
    """Count character differences between equal-length strings."""
    if len(a) != len(b):
        raise ValueError("Hamming distance requires strings with the same length.")
    return sum(left != right for left, right in zip(a, b))


def best_match_against_known_chassis(
    candidate: str,
    known_chassis_numbers: list[str],
    max_distance: int = 3,
) -> dict | None:
    """Return the closest known chassis number to an OCR candidate."""
    cleaned_candidate = clean_chassis_no(candidate)
    if len(cleaned_candidate) != 17:
        return None

    known_values = _normalize_known_chassis_numbers(known_chassis_numbers)
    variants = [cleaned_candidate, *generate_candidate_corrections(cleaned_candidate)]

    best: dict | None = None
    for variant in variants:
        for known in known_values:
            distance = hamming_distance(variant, known)
            if distance > max_distance:
                continue
            if best is None or distance < best["distance"]:
                best = {
                    "matched_chassis_no": known,
                    "distance": distance,
                    "method": "fuzzy_correction",
                }

    return best


def repair_near_length_chassis_candidate(
    candidate: str,
    known_chassis_numbers: list[str],
    max_distance: int = 3,
) -> dict | None:
    """Repair candidates whose length is close to a chassis number."""
    cleaned = clean_chassis_no(candidate)
    known_values = _normalize_known_chassis_numbers(known_chassis_numbers)

    if len(cleaned) == 17:
        return best_match_against_known_chassis(cleaned, known_values, max_distance=max_distance)

    if len(cleaned) < 15 or len(cleaned) > 19:
        return None

    variants = _generate_near_length_variants(cleaned, LIKELY_WMI_PREFIXES)
    best = _best_match_from_variants(
        variants=variants,
        known_chassis_numbers=known_values,
        max_distance=max_distance,
        method="near_length_repair",
    )
    return best


def infer_likely_prefixes(vehicle_description: str | None, origin: str | None) -> list[str]:
    """Infer likely WMI prefixes from vehicle metadata."""
    description = (vehicle_description or "").upper()
    prefixes: list[str] = []

    if "NISSAN" in description:
        prefixes.extend(["JN1", "JN6", "JN8", "5N1", "MDH", "MNT"])
    if "INFINITI" in description:
        prefixes.extend(["3PC", "5N1", "JN1"])
    if "RENAULT" in description:
        prefixes.append("VF1")
    if "CHEVROLET" in description:
        prefixes.extend(["1GN", "1G1"])

    origin_text = (origin or "").upper()
    if "JAPAN" in origin_text:
        prefixes.extend(["JN1", "JN6", "JN8"])
    if "UAE" in origin_text:
        prefixes.extend(["MNT", "MDH"])
    if "FRANCE" in origin_text:
        prefixes.append("VF1")
    if "USA" in origin_text or "UNITED STATES" in origin_text:
        prefixes.extend(["1GN", "1G1", "5N1"])

    prefixes.extend(LIKELY_WMI_PREFIXES)
    return list(dict.fromkeys(prefixes))


def score_candidate_against_ground_truth_pool(
    candidate: str,
    known_chassis_numbers: list[str],
    vehicle_description: str | None = None,
    origin: str | None = None,
    max_distance: int = 4,
) -> dict | None:
    """Score one candidate against the known chassis pool."""
    cleaned = clean_chassis_no(candidate)
    if not cleaned:
        return None

    known_values = _normalize_known_chassis_numbers(known_chassis_numbers)
    likely_prefixes = infer_likely_prefixes(vehicle_description, origin)

    if cleaned in known_values:
        return {
            "matched_chassis_no": cleaned,
            "distance": 0,
            "method": "exact",
            "confidence": 1.0,
        }

    fuzzy_match = best_match_against_known_chassis(cleaned, known_values, max_distance=min(max_distance, 3))
    near_length_match = repair_near_length_chassis_candidate(cleaned, known_values, max_distance=min(max_distance, 3))
    prefix_match = _prefix_repair_match(
        cleaned,
        known_values,
        likely_prefixes,
        max_distance=max_distance,
    )

    candidates = [match for match in [fuzzy_match, near_length_match, prefix_match] if match]
    if not candidates:
        return None

    best = min(
        candidates,
        key=lambda match: (
            int(match["distance"]),
            0 if is_valid_chassis_format(match["matched_chassis_no"]) else 1,
            0 if _prefix_is_consistent(match["matched_chassis_no"], likely_prefixes) else 1,
            _method_priority(str(match["method"])),
        ),
    )
    best = dict(best)
    best["confidence"] = _confidence_from_distance(int(best["distance"]), str(best["method"]))
    return best


def _normalize_known_chassis_numbers(known_chassis_numbers: list[str]) -> list[str]:
    return list(
        dict.fromkeys(
            clean_chassis_no(chassis_no)
            for chassis_no in known_chassis_numbers
            if len(clean_chassis_no(chassis_no)) == 17
        )
    )


def _generate_near_length_variants(candidate: str, likely_prefixes: list[str]) -> list[str]:
    variants: list[str] = []

    if len(candidate) == 16:
        first_chars = list(dict.fromkeys(prefix[0] for prefix in likely_prefixes if prefix))
        for char in first_chars:
            variants.append(char + candidate)

        for prefix in likely_prefixes:
            if len(prefix) >= 2 and candidate.startswith(prefix[1:]):
                variants.append(prefix[0] + candidate)

        for index in range(len(candidate) + 1):
            for char in LIKELY_VIN_CHARACTERS:
                variants.append(candidate[:index] + char + candidate[index:])

    elif len(candidate) == 18:
        for index in range(len(candidate)):
            variants.append(candidate[:index] + candidate[index + 1 :])

    elif 15 <= len(candidate) <= 19:
        if len(candidate) < 17:
            variants.extend(_generate_padding_variants(candidate, likely_prefixes))
        if len(candidate) > 17:
            variants.extend(_generate_trim_variants(candidate))

    return list(dict.fromkeys(value for value in variants if len(value) == 17))


def _generate_padding_variants(candidate: str, likely_prefixes: list[str]) -> list[str]:
    missing_count = 17 - len(candidate)
    if missing_count <= 0:
        return []

    variants: list[str] = []
    if missing_count == 1:
        for char in list(dict.fromkeys(prefix[0] for prefix in likely_prefixes if prefix)):
            variants.append(char + candidate)
    elif missing_count == 2:
        for prefix in likely_prefixes:
            variants.append(prefix[:2] + candidate)

    return variants


def _generate_trim_variants(candidate: str) -> list[str]:
    extra_count = len(candidate) - 17
    if extra_count <= 0:
        return []

    variants: list[str] = []
    for indexes in itertools.combinations(range(len(candidate)), extra_count):
        chars = [char for idx, char in enumerate(candidate) if idx not in indexes]
        variants.append("".join(chars))
    return variants


def _best_match_from_variants(
    variants: list[str],
    known_chassis_numbers: list[str],
    max_distance: int,
    method: str,
) -> dict | None:
    best: dict | None = None
    for variant in variants:
        for known in known_chassis_numbers:
            distance = _edit_distance(variant, known)
            if distance > max_distance:
                continue
            if best is None or distance < best["distance"]:
                best = {
                    "matched_chassis_no": known,
                    "distance": distance,
                    "method": method,
                }
    return best


def _prefix_repair_match(
    candidate: str,
    known_chassis_numbers: list[str],
    likely_prefixes: list[str],
    max_distance: int,
) -> dict | None:
    filtered_known = [
        known for known in known_chassis_numbers if _prefix_is_consistent(known, likely_prefixes)
    ]
    if not filtered_known:
        return None

    variants = [candidate]
    if len(candidate) != 17:
        variants.extend(_generate_near_length_variants(candidate, likely_prefixes))
    if len(candidate) == 17:
        variants.extend(generate_candidate_corrections(candidate))

    best = _best_match_from_variants(
        variants=list(dict.fromkeys(variants)),
        known_chassis_numbers=filtered_known,
        max_distance=max_distance,
        method="prefix_repair",
    )
    return best


def _prefix_is_consistent(chassis_no: str, likely_prefixes: list[str]) -> bool:
    return any(chassis_no.startswith(prefix) for prefix in likely_prefixes)


def _method_priority(method: str) -> int:
    priorities = {
        "exact": 0,
        "fuzzy_correction": 1,
        "near_length_repair": 2,
        "prefix_repair": 3,
    }
    return priorities.get(method, 9)


def _confidence_from_distance(distance: int, method: str) -> float:
    base = {
        "exact": 1.0,
        "fuzzy_correction": 0.94,
        "near_length_repair": 0.88,
        "prefix_repair": 0.84,
    }.get(method, 0.75)
    return max(0.0, round(base - (distance * 0.1), 4))


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0

    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (char_a != char_b)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]
