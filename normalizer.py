"""BDWebs Bangla / Banglish / English normalization, spam/gibberish detection, and safety guards."""

import re
from typing import Dict, List, Set, Tuple

SYNONYMS: Dict[str, str] = {
    "hostng": "hosting", "hostin": "hosting", "hosing": "hosting", "hostig": "hosting", "hoosting": "hosting",
    "doman": "domain", "domian": "domain", "domein": "domain", "domane": "domain", "domai": "domain", "dmn": "domain",
    "rechrg": "recharge", "recharg": "recharge", "richarge": "recharge", "richarj": "recharge", "recharj": "recharge",
    "topup": "recharge", "fund": "recharge", "balance": "recharge", "ব্যালেন্স": "recharge", "ফান্ড": "recharge",
    "bikas": "bkash", "bikash": "bkash", "vkash": "bkash", "nogod": "nagad", "ngd": "nagad", "roket": "rocket",
    "pament": "payment", "paymnt": "payment", "peyment": "payment", "pay": "payment", "পে": "payment",
    "bill": "bill", "বিল": "bill",
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
    "dam": "price", "mullo": "price", "khoroch": "price", "cost": "price", "rate": "price", "taka": "price", "tk": "price",
    "দাম": "price", "মূল্য": "price", "খরচ": "price", "কত": "price", "প্রাইস": "price", "প্রাইজ": "price", "টাকা": "price",
    "kivabe": "how", "kibhabe": "how", "kemne": "how", "কিভাবে": "how", "কেমনে": "how",
    "banana": "design", "banabo": "design", "banate": "design", "banai": "design", "তৈরি": "design", "বানাতে": "design",
    "bidesh": "abroad", "bideshe": "abroad", "foreign": "abroad", "বিদেশ": "abroad", "বিদেশে": "abroad",
    "slow": "slow", "ধীর": "slow", "স্লো": "slow",
    "company": "business", "কোম্পানি": "business", "ব্যবসা": "business",
    "email": "email", "ইমেইল": "email", "মেইল": "email",
    "call": "call", "কল": "call", "receive": "receive", "রিসিভ": "receive", "office": "office", "অফিস": "office",
    "kinte": "buy", "kinbo": "buy", "nibo": "buy", "nite": "buy", "কিনতে": "buy", "নিতে": "buy",
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
    "হোয়াটসঅ্যাপ": "whatsapp", "হোয়াটসঅ্যাপ": "whatsapp",
}

PHRASE_SYNONYMS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bek\s*sathe\b|এক\s*সাথে"), "concurrent multiple"),
    (re.compile(r"\bauto\s*reply\b|অটো\s*রিপ্লাই"), "auto reply chatbot"),
    (re.compile(r"\bbusiness\s*email\b"), "business email"),
    (re.compile(r"\bbill\s*pay\b"), "bill payment"),
    (re.compile(r"\bwebsite\s*(?:banana|banabo|banate|banai|তৈরি|বানাতে)"), "website design development"),
    (re.compile(r"\bcall\s*receive\b"), "pbx incoming call"),
]

NON_IP_PHONE_TOKENS: Set[str] = {
    "vps", "rdp", "kvm", "hosting", "reseller", "wordpress", "cpanel", "whm",
    "domain", "tld", "sms", "masking", "nonmasking", "whmcs", "dedicated", "colocation", "ssl", "email",
}

BARE_WORD_TLDS: Set[str] = {
    "work", "top", "shop", "store", "online", "site", "cloud", "app", "dev", "pro", "live", "info",
    "net", "org", "co", "me", "io", "tv", "in", "ai", "xyz", "blog", "tech", "email", "host", "news",
    "club", "space", "fun", "life", "world", "today", "vip", "one", "link", "click", "help", "page",
    "name", "digital", "agency", "media", "company", "center", "support", "services", "solutions",
    "systems", "network", "studio", "design", "group", "team", "zone", "plus", "care", "chat", "call",
}

DOMAIN_CONTEXT_TOKENS: Set[str] = {
    "domain", "tld", "extension", "register", "registration", "transfer", "renew", "whois", "nameserver", "dns",
}

SEMANTIC_EXCLUDED_RULE_KEYS: Set[str] = {
    "cw_9", "cw_15", "cw_30", "cw_13", "cw_11", "cw_12", "cw_27", "cw_28", "cw_33", "cw_41",
    "cw_canned_1", "cw_canned_5", "cw_canned_6", "cw_canned_8", "cw_canned_54", "cw_canned_78",
    "cw_canned_88", "cw_canned_96", "cw_canned_105", "learned_155",
}

KNOWN_PRODUCT_TOKENS: Set[str] = NON_IP_PHONE_TOKENS | {
    "ip", "pbx", "number", "phone", "call", "voice", "recharge", "payment", "bkash", "nagad", "rocket",
    "bill", "ticket", "invoice", "password", "pin", "whatsapp", "chatbot", "api", "webhook", "webrtc",
    "wss", "softphone", "autossl", "ssl", "website", "design", "development", "abroad", "slow",
    "business", "office", "renew", "transfer", "refund", "cancel", "setup", "document", "offer", "rate",
}

KNOWN_SIGNAL_TOKENS: Set[str] = set(SYNONYMS.keys()) | set(SYNONYMS.values()) | KNOWN_PRODUCT_TOKENS | {
    "how", "price", "buy", "receive", "concurrent", "multiple", "auto", "reply", "incoming", "order",
    "transaction", "pending", "support", "verify", "mobile", "pc", "time", "issue", "usage", "package",
    "server", "host", "web", "agent", "livechat", "chat", "bot",
}

STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from", "how", "i", "if", "in",
    "is", "it", "me", "my", "no", "of", "on", "or", "our", "so", "that", "the", "there", "this", "to",
    "us", "was", "we", "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
    "ki", "koto", "kivabe", "korbo", "korte", "kore", "kora", "chai", "ase", "ache", "er", "ta", "te",
    "jonno", "diye", "ek", "ekta", "amar", "apnar", "lagbe", "hobe", "jabe", "theke", "ekhon", "hoy",
    "hoye", "gese", "naki", "sudhu", "ba", "na", "ar", "ami", "apni", "sir", "vai", "bhai", "vaiya",
    "কি", "কী", "এর", "জন্য", "থেকে", "করা", "যাবে", "আছে", "হবে", "লাগবে", "দিয়ে", "দিয়ে", "একটা", "আমার", "আপনার",
}

_STUB_RESPONSE_RE = re.compile(r"\b(?:most welcome|forwarding it|i am forwarding)\b", re.I)


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


def extract_domain_tokens(text: str) -> Set[str]:
    """Extract high-signal domain/product tokens for fast lexical overlap reranking."""
    norm = normalize_text(text)
    tokens: Set[str] = {tok for tok in extract_canonical_tokens(norm) if tok not in STOPWORDS and len(tok) > 1}
    for pattern, expansion in PHRASE_SYNONYMS:
        if pattern.search(norm):
            for part in expansion.split():
                if part not in STOPWORDS and len(part) > 1:
                    tokens.add(part)
    return tokens


def is_semantic_eligible_rule(rule: dict) -> bool:
    """Return True if rule belongs in the semantic vector index (skips greetings/stubs/copilot)."""
    if not isinstance(rule, dict) or rule.get("active") is False:
        return False
    src = str(rule.get("source") or "")
    rk = str(rule.get("rule_key") or "")
    if src.startswith("copilot_") or rk in SEMANTIC_EXCLUDED_RULE_KEYS:
        return False
    resp = str(rule.get("response") or "").strip()
    if not resp:
        return False
    resp_low = normalize_text(resp)
    if _STUB_RESPONSE_RE.search(resp_low):
        return False
    if len(resp_low) < 35 and not re.search(r"https?://|\d", resp_low):
        if not (set(extract_canonical_tokens(resp_low)) & KNOWN_PRODUCT_TOKENS):
            return False
    return True


def is_gibberish_or_noise(query: str) -> bool:
    """Detect keyboard mashes / consonant-only noise without rejecting Bangla or Banglish queries."""
    text = str(query or "").strip()
    if not text:
        return True
    if re.search(r"[\u0980-\u09FF]", text):
        return False
    low = text.lower()
    raw_words = re.findall(r"[a-z]+", low)
    canon = extract_canonical_tokens(low)
    if (set(canon) & KNOWN_SIGNAL_TOKENS) or (set(raw_words) & KNOWN_SIGNAL_TOKENS):
        return False
    if re.search(r"\b(?:cw_|whmcs_|wp_)|https?://|\d{2,}", low):
        return False
    if re.search(r"(?:asdf|qwer|zxcv|hjkl|fghj|uiop|bnm)", low):
        return True
    long_words = [w for w in raw_words if len(w) >= 4]
    if long_words:
        vowelless = sum(1 for w in long_words if not re.search(r"[aeiouy]", w))
        if (vowelless / len(long_words)) >= 0.6:
            return True
    if len(raw_words) >= 2:
        letters = "".join(raw_words)
        if len(letters) >= 6:
            vowels = len(re.findall(r"[aeiouy]", letters))
            if (vowels / len(letters)) < 0.18:
                return True
    return False


def enrich_query_for_embedding(query: str, active_topic: str = "", use_e5_prefix: bool = True) -> str:
    """Build natural query text with translated Banglish/Bangla hints (no '| keywords:' boilerplate)."""
    norm = normalize_text(query)
    if not norm:
        return "query: " if use_e5_prefix else ""
    raw_words = set(re.sub(r"[^\w\s\u0980-\u09FF.]", " ", norm).split())
    hints: List[str] = []
    seen_hints: Set[str] = set()

    for pattern, expansion in PHRASE_SYNONYMS:
        if pattern.search(norm):
            for word in expansion.split():
                if word not in raw_words and word not in seen_hints:
                    seen_hints.add(word)
                    hints.append(word)

    for tok in extract_canonical_tokens(norm):
        if tok not in raw_words and tok not in seen_hints:
            seen_hints.add(tok)
            hints.append(tok)

    text_part = f"{norm} ({' '.join(hints)})" if hints else norm
    if active_topic:
        topic_label = active_topic.replace("_", " ").strip()
        if topic_label:
            text_part = f"{topic_label}: {text_part}"
    return f"query: {text_part}" if use_e5_prefix else text_part


def passes_rule_guards(rule: dict, query: str) -> bool:
    """Enforce deterministic BDWebs safety guards before returning semantic hits."""
    if not is_semantic_eligible_rule(rule):
        return False
    src = str(rule.get("source") or "")
    rk = str(rule.get("rule_key") or "")
    norm_q = normalize_text(query)
    canon = set(extract_canonical_tokens(norm_q))
    has_non_ip_entity = bool(canon & NON_IP_PHONE_TOKENS) or bool(re.search(r"\.[a-z]{2,10}\b", norm_q))
    if rk == "cw_49" and has_non_ip_entity:
        return False
    if src == "whmcs_tld" and rk != "whmcs_tld_summary":
        ext = rk[10:].replace("_", ".") if rk.startswith("whmcs_tld_") else ""
        if ext:
            has_dot_ext = bool(re.search(rf"(?<![a-z0-9])\.{re.escape(ext)}\b", norm_q))
            if not has_dot_ext:
                ext_parts = set(ext.split("."))
                if ext not in canon and not (ext_parts <= canon):
                    return False
                if ext in BARE_WORD_TLDS and not (canon & DOMAIN_CONTEXT_TOKENS):
                    return False
    if src == "whmcs_plan" and not re.search(r"\d", norm_q):
        title_norm = normalize_text(str(rule.get("title") or ""))
        if not title_norm or title_norm not in norm_q:
            return False
    return True
