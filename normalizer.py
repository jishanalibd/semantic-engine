"""BDWebs Bangla / Banglish / English normalization and safety guards."""

import re
from typing import Dict, List

SYNONYMS: Dict[str, str] = {
    "hostng": "hosting", "hostin": "hosting", "hosing": "hosting", "hostig": "hosting", "hoosting": "hosting",
    "doman": "domain", "domian": "domain", "domein": "domain", "domane": "domain", "domai": "domain", "dmn": "domain",
    "rechrg": "recharge", "recharg": "recharge", "richarge": "recharge", "richarj": "recharge", "recharj": "recharge",
    "bikas": "bkash", "bikash": "bkash", "vkash": "bkash", "nogod": "nagad", "ngd": "nagad", "roket": "rocket",
    "pament": "payment", "paymnt": "payment", "peyment": "payment",
    "nambr": "number", "numbr": "number", "numbar": "number", "nombor": "number", "nmbr": "number", "namber": "number", "nambur": "number",
    "pakage": "package", "pakej": "package", "pakge": "package", "packge": "package", "pckg": "package",
    "sarver": "server", "servar": "server", "servr": "server", "serbar": "server",
    "reselar": "reseller", "reseler": "reseller", "riselar": "reseller", "riseller": "reseller",
    "suport": "support", "saport": "support", "sapot": "support", "supprt": "support",
    "tikit": "ticket", "tiket": "ticket", "tikt": "ticket", "invoic": "invoice", "invois": "invoice", "invos": "invoice",
    "pasword": "password", "paswd": "password", "passwrd": "password", "cpenel": "cpanel", "cpnel": "cpanel",
    "wbsite": "website", "websit": "website", "webside": "website", "oebsite": "website",
    "mbl": "mobile", "moble": "mobile", "mobaile": "mobile", "mobail": "mobile",
    "smss": "sms", "bluk": "bulk", "balk": "bulk", "msking": "masking", "maskng": "masking",
    "pbxx": "pbx", "ppx": "pbx", "ippx": "pbx", "ippbx": "pbx",
    "transection": "transaction", "transction": "transaction", "tranjection": "transaction", "tranjaction": "transaction",
    "ordar": "order", "ordr": "order", "panding": "pending", "pendng": "pending",
    "dam": "price", "mullo": "price", "khoroch": "price", "cost": "price", "rate": "price",
    "দাম": "price", "মূল্য": "price", "খরচ": "price", "কত": "price", "প্রাইস": "price", "প্রাইজ": "price",
    "ডোমেইন": "domain", "ডোমেন": "domain", "হোস্টিং": "hosting", "সার্ভার": "server", "ভিপিএস": "vps",
    "নাম্বার": "number", "নম্বর": "number", "এসএমএস": "sms", "মেসেজ": "sms",
    "massge": "sms", "massege": "sms", "massage": "sms", "mesage": "sms",
    "বিকাশ": "bkash", "নগদ": "nagad", "পেমেন্ট": "payment", "অফার": "offer",
    "রিচার্জ": "recharge", "রিচার্জের": "recharge", "রিসার্চ": "recharge",
    "ডকুমেন্ট": "document", "ডকুমেন্টস": "document", "কাগজপত্র": "document", "কাগজ": "document",
    "kagoj": "document", "kagojpotro": "document", "nid": "document", "এনআইডি": "document",
    "সেটআপ": "setup", "সেটাপ": "setup", "রিনিউ": "renew", "ট্রান্সফার": "transfer",
    "রিফান্ড": "refund", "ক্যানসেল": "cancel", "পাসওয়ার্ড": "password", "পাসওয়ার্ড": "password",
    "ব্যবহার": "usage", "ব্যাবহার": "usage", "chalano": "usage", "chalate": "usage", "byabohar": "usage", "bebohar": "usage",
    "মোবাইল": "mobile", "মোবাইলে": "mobile", "ফোনে": "mobile", "কম্পিউটার": "pc", "পিসি": "pc",
    "সময়": "time", "সময়": "time", "somoy": "time", "kotokkhon": "time",
    "সমস্যা": "issue", "প্রবলেম": "issue", "ইন্টিগ্রেশন": "api", "এপিআই": "api",
    "ভেরিফাই": "verify", "পিন": "pin", "রিসেলার": "reseller", "আইপি": "ip", "সিপ্যানেল": "cpanel", "এসএসএল": "ssl",
}

NON_IP_PHONE_TOKENS = {
    "vps", "rdp", "kvm", "hosting", "reseller", "wordpress", "cpanel", "whm",
    "domain", "tld", "sms", "masking", "nonmasking", "whmcs", "dedicated", "colocation", "ssl", "email",
}


def normalize_text(text: str) -> str:
    """Normalize Bangla Unicode forms, whitespace, and lowercase."""
    s = str(text or "")
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    s = s.replace("\u09AF\u09BC", "\u09DF").replace("\u09A1\u09BC", "\u09DC").replace("\u09A2\u09BC", "\u09DD")
    s = s.lower().strip()
    return re.sub(r"\s+", " ", s)


def normalize_token(token: str) -> str:
    """Collapse 3+ repeated Latin chars and map Banglish/typo synonyms."""
    collapsed = re.sub(r"([a-z])\1{2,}", r"\1", token.strip("."))
    return SYNONYMS.get(collapsed, collapsed)


def extract_canonical_tokens(text: str) -> List[str]:
    """Extract ordered unique normalized tokens for embedding enrichment & guards."""
    clean = re.sub(r"[^\w\s\u0980-\u09FF.]", " ", normalize_text(text))
    seen = set()
    out: List[str] = []
    for raw in clean.split():
        mapped = normalize_token(raw)
        if len(mapped) <= 1 or mapped in seen:
            continue
        seen.add(mapped)
        out.append(mapped)
    return out


def enrich_query_for_embedding(query: str, active_topic: str = "", use_e5_prefix: bool = True) -> str:
    """Build query text preserving native text + Banglish->English hints (with optional E5 prefix)."""
    norm = normalize_text(query)
    canon = extract_canonical_tokens(norm)
    parts: List[str] = []
    if active_topic:
        parts.append(f"[{active_topic}]")
    parts.append(norm)
    if canon:
        parts.append("keywords: " + " ".join(canon))
    body = " | ".join(parts)
    return f"query: {body}" if use_e5_prefix else body


def passes_rule_guards(rule: dict, query: str) -> bool:
    """Enforce deterministic BDWebs safety guards before returning semantic hits."""
    if not rule or rule.get("active") is False:
        return False
    src = str(rule.get("source") or "")
    rk = str(rule.get("rule_key") or "")
    if src.startswith("copilot_"):
        return False
    norm_q = normalize_text(query)
    canon = set(extract_canonical_tokens(norm_q))
    has_non_ip_entity = bool(canon & NON_IP_PHONE_TOKENS) or bool(re.search(r"\.[a-z]{2,10}\b", norm_q))
    if rk == "cw_49" and has_non_ip_entity:
        return False
    if src == "whmcs_tld" and rk != "whmcs_tld_summary":
        ext = rk[10:].replace("_", ".") if rk.startswith("whmcs_tld_") else ""
        if ext and f".{ext}" not in norm_q and ext not in canon:
            return False
    if src == "whmcs_plan" and not re.search(r"\d", norm_q):
        title_norm = normalize_text(str(rule.get("title") or ""))
        if not title_norm or title_norm not in norm_q:
            return False
    return True
