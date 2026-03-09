"""
Sympathetic noise post-processing filter.
Must be applied to ALL model outputs before any scoring.
"""

import json
import re

SYMPATHETIC_CODES = {
    "HEARTBEAT_TIMEOUT", "KEEPALIVE_FAIL", "KEEPALIVE_TIMEOUT",
    "SECONDARY_ALARM", "CASCADING_FAILURE", "PFCP_HEARTBEAT_TIMEOUT",
    "N2_HEARTBEAT_TIMEOUT", "N11_HEARTBEAT_TIMEOUT", "TIMER_EXPIRY",
    "RETRANSMISSION", "DUPLICATE_NAS", "SPURIOUS_MEASUREMENT",
    "BEAM_FAILURE_RECOVERY", "RLC_RETRANSMISSION", "HARQ_NACK",
}

VALID_ROOT_CAUSES = {
    "core_network_failure", "authentication_failure", "normal",
    "handover_failure", "congestion", "qos_violation",
    "transport_jitter", "radio_failure",
}

# Keyword synonyms that map natural-language phrases to canonical labels.
# Order matters: more specific patterns must come before generic ones to avoid
# false positives (e.g. "radio link failure" before "failure").
# Each tuple is (regex_pattern, canonical_label).
KEYWORD_SYNONYMS = [
    # authentication_failure
    (r"authentication\s+fail", "authentication_failure"),
    (r"auth\s+fail", "authentication_failure"),
    (r"auth\s+reject", "authentication_failure"),
    (r"incorrect.*res\*", "authentication_failure"),
    (r"invalid.*res\*", "authentication_failure"),
    (r"cause\s+(value\s+)?21", "authentication_failure"),
    (r"failed\s+authentication", "authentication_failure"),
    # handover_failure
    (r"handover\s+fail", "handover_failure"),
    (r"handover\s+reject", "handover_failure"),
    (r"ho\s+fail", "handover_failure"),
    (r"insufficient\s+radio\s+resources", "handover_failure"),
    (r"target\s+gnb.*not\s+have\s+sufficient", "handover_failure"),
    (r"handover\s+preparation\s+fail", "handover_failure"),
    # congestion
    (r"congestion", "congestion"),
    (r"overload", "congestion"),
    (r"cause\s+(value\s+)?22", "congestion"),
    # qos_violation
    (r"qos\s+violation", "qos_violation"),
    (r"qos\s+flow.*fail", "qos_violation"),
    (r"guaranteed\s+bit\s*rate.*not.*supported", "qos_violation"),
    (r"cannot\s+meet.*guaranteed\s+bitrate", "qos_violation"),
    (r"gbr.*not.*supported", "qos_violation"),
    (r"qos\s+flow\s+installation\s+fail", "qos_violation"),
    (r"qos\s+parameters.*adjust", "qos_violation"),
    # transport_jitter
    (r"transport.?jitter", "transport_jitter"),
    (r"reordering\s+timeout", "transport_jitter"),
    (r"pdcp.*retransmission", "transport_jitter"),
    (r"pdcp.*reordering", "transport_jitter"),
    (r"missing\s+pdcp\s+pdu", "transport_jitter"),
    (r"transmission\s+error", "transport_jitter"),
    (r"jitter", "transport_jitter"),
    # radio_failure
    (r"radio\s+link\s+failure", "radio_failure"),
    (r"radio\s+failure", "radio_failure"),
    (r"rlf", "radio_failure"),
    (r"beam\s+failure(?!\s+recovery)", "radio_failure"),
    (r"rsrp.*weak", "radio_failure"),
    (r"weak\s+rsrp", "radio_failure"),
    (r"cqi\s+degradation", "radio_failure"),
    (r"radio\s+conditions", "radio_failure"),
    # core_network_failure
    (r"core\s+network\s+failure", "core_network_failure"),
    (r"n3\s+interface.*fail", "core_network_failure"),
    (r"pfcp\s+session.*fail", "core_network_failure"),
    (r"smf.*fail", "core_network_failure"),
    (r"amf.*fail(?!.*auth)", "core_network_failure"),
    (r"upf.*fail", "core_network_failure"),
    (r"n2\s+interface.*fail", "core_network_failure"),
    (r"core.*network.*down", "core_network_failure"),
]

# Pre-compile all synonym patterns for performance
_COMPILED_SYNONYMS = [(re.compile(pat, re.IGNORECASE), label) for pat, label in KEYWORD_SYNONYMS]


def filter_sympathetic_noise(predicted_codes: list) -> list:
    """Remove sympathetic noise codes; keep only valid root cause labels."""
    seen, filtered = set(), []
    for code in predicted_codes:
        if not isinstance(code, str):
            continue
        norm = code.strip().lower()
        if code.strip().upper() in SYMPATHETIC_CODES:
            continue
        if norm in VALID_ROOT_CAUSES and norm not in seen:
            seen.add(norm)
            filtered.append(norm)
    return filtered if filtered else ["normal"]


def extract_root_cause_from_text(text: str) -> list:
    """Parse root cause labels from free-form model output text.

    Strategy (in priority order):
    1. Strip <think>...</think> blocks (Qwen3 reasoning mode)
    2. Look for a JSON array in the text (e.g. '["congestion"]')
    3. Look for exact canonical label strings (e.g. 'authentication_failure')
    4. Fuzzy keyword synonym matching against natural-language phrases
    """
    # 0. Strip <think>...</think> blocks — Qwen3's reasoning mode produces
    #    verbose text that mentions all failure types, causing false positives.
    #    Only extract from the actual answer after </think>.
    clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # If </think> is missing (truncated), strip everything after <think>
    if '<think>' in clean_text:
        clean_text = re.sub(r'<think>.*', '', clean_text, flags=re.DOTALL).strip()
    # Use cleaned text for extraction, fall back to original if empty
    if not clean_text:
        clean_text = text
    # 1. Try JSON array extraction
    match = re.search(r'\[.*?\]', clean_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                result = filter_sympathetic_noise(parsed)
                if result != ["normal"] or "normal" in str(parsed).lower():
                    return result
        except json.JSONDecodeError:
            pass

    # 2. Try exact canonical label match
    text_lower = clean_text.lower()
    found = [label for label in VALID_ROOT_CAUSES if label in text_lower]
    if found:
        result = filter_sympathetic_noise(found)
        if result != ["normal"]:
            return result

    # 3. Fuzzy keyword synonym matching — first match wins (patterns are
    #    ordered from most specific to most generic within each category)
    seen = set()
    matched = []
    for pattern, label in _COMPILED_SYNONYMS:
        if label not in seen and pattern.search(clean_text):
            seen.add(label)
            matched.append(label)
    # Remove "normal" if we found a real failure
    non_normal = [m for m in matched if m != "normal"]
    if non_normal:
        return filter_sympathetic_noise(non_normal)
    if matched:
        return filter_sympathetic_noise(matched)

    # 4. Fallback: return exact matches if any were found (including normal)
    if found:
        return filter_sympathetic_noise(found)

    return ["normal"]
