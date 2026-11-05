"""Romance Trope Extraction Engine with Rule-Based, TF-IDF, and Evidence Tracking."""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import logger

# 10 Canonical Romance Tropes with Keywords and Semantic Descriptions
TROPE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "Enemies to Lovers": {
        "keywords": [
            r"\benemies[- ]to[- ]lovers\b",
            r"\bnemesis\b",
            r"\bsworn enemies\b",
            r"\bhate each other\b",
            r"\bhate turned to love\b",
            r"\bsparring\b",
            r"\bhostile\b",
            r"\brivals?\b",
            r"\bantagonist\b",
        ],
        "prototype": (
            "Two sworn rivals, nemeses, or workplace opponents who passionately hate each other "
            "engage in witty sparring and hostile conflict before discovering an irresistible mutual attraction "
            "and falling deeply in love."
        ),
    },
    "Friends to Lovers": {
        "keywords": [
            r"\bfriends[- ]to[- ]lovers\b",
            r"\bbest friends?\b",
            r"\bchildhood friends?\b",
            r"\blifelong friendship\b",
            r"\bfalling for (?:his|her|their) best friend\b",
            r"\bplatonic turned\b",
        ],
        "prototype": (
            "Two longtime best friends or childhood confidants share deep emotional trust and shared history. "
            "Over time or through a sudden catalyst, their platonic bond crosses into undeniable romance."
        ),
    },
    "Fake Dating": {
        "keywords": [
            r"\bfake dating\b",
            r"\bfake boyfriend\b",
            r"\bfake girlfriend\b",
            r"\bfake fiancé\b",
            r"\bfake relationship\b",
            r"\bpretend dating\b",
            r"\bcharade\b",
            r"\bpose as a couple\b",
            r"\bsham relationship\b",
        ],
        "prototype": (
            "Two individuals enter into a staged romantic charade or fake relationship to satisfy family, "
            "attend a wedding, or manipulate a public scenario, only for their faux intimacy to ignite genuine passion."
        ),
    },
    "Slow Burn": {
        "keywords": [
            r"\bslow[- ]burn\b",
            r"\bsimmering tension\b",
            r"\bdelicious slow burn\b",
            r"\bagonizing tension\b",
            r"\bunhurried pace\b",
            r"\bgradual build\b",
            r"\btake (?:hundreds of )?pages to warm up\b",
        ],
        "prototype": (
            "A gradual, deeply developed romantic progression where emotional intimacy, magnetic eye contact, "
            "and delicious romantic tension build agonizingly slowly across hundreds of pages before any culmination."
        ),
    },
    "Second Chance Romance": {
        "keywords": [
            r"\bsecond chance\b",
            r"\bfirst love\b",
            r"\breunited\b",
            r"\breunion after years\b",
            r"\bpast relationship\b",
            r"\bdecades? later\b",
            r"\bestranged\b",
            r"\bruns? into (?:her|his) ex\b",
        ],
        "prototype": (
            "Former lovers who separated due to tragedy, misunderstanding, or bad timing are reunited years later, "
            "confronting past grief and rekindling their unfinished love story for a hard-won second chance."
        ),
    },
    "Grumpy x Sunshine": {
        "keywords": [
            r"\bgrumpy[- ]x[- ]sunshine\b",
            r"\bgrumpy (?:and|meets) sunshine\b",
            r"\bbrooding (?:hero|man)\b",
            r"\bcheerfully optimistic\b",
            r"\bsunny (?:heroine|girl)\b",
            r"\bstern (?:hero|professor|boss)\b",
            r"\bopposites attract\b",
        ],
        "prototype": (
            "A stern, brooding, cynical, or taciturn partner is paired with a vibrant, cheerful, energetic, "
            "and optimistic counterpart who steadily melts their hardened exterior."
        ),
    },
    "Workplace Romance": {
        "keywords": [
            r"\bworkplace romance\b",
            r"\boffice romance\b",
            r"\bcolleague\b",
            r"\bco[- ]worker\b",
            r"\bexecutive assistant\b",
            r"\bpromotion\b",
            r"\bcorporate\b",
            r"\blab partner\b",
            r"\bacademic romance\b",
            r"\bworking together\b",
        ],
        "prototype": (
            "Romantic tension developing in a professional corporate, editorial, athletic, or academic environment "
            "between coworkers, bosses, or career competitors navigating company policies and ambition."
        ),
    },
    "Forced Proximity": {
        "keywords": [
            r"\bforced proximity\b",
            r"\bshare an office\b",
            r"\bshare a flat\b",
            r"\bshare a bed\b",
            r"\bonly one bed\b",
            r"\btrapped together\b",
            r"\bliving next door\b",
            r"\bneighboring\b",
            r"\bclose quarters\b",
            r"\bstuck together\b",
        ],
        "prototype": (
            "Circumstances such as shared living spaces, travel mishaps, only one bed in a hotel room, or snowed-in "
            "cabins force two characters into continuous, inescapable physical closeness that breaks down barriers."
        ),
    },
    "Marriage of Convenience": {
        "keywords": [
            r"\bmarriage of convenience\b",
            r"\barranged marriage\b",
            r"\bcontract marriage\b",
            r"\bconvenient wedding\b",
            r"\bpolitical marriage\b",
            r"\bwed in an alliance\b",
            r"\bsham marriage\b",
            r"\bmarry for (?:a )?visa\b",
        ],
        "prototype": (
            "Two individuals legally bind themselves in matrimony for external pragmatic reasons—such as immigration, "
            "royal alliances, financial inheritances, or social status—promising detachment before falling truly in love."
        ),
    },
    "Friends with Benefits": {
        "keywords": [
            r"\bfriends with benefits\b",
            r"\bno strings attached\b",
            r"\bcasual arrangement\b",
            r"\bstrictly physical\b",
            r"\bjust sex\b",
            r"\brules of engagement\b",
            r"\bno feelings\b",
        ],
        "prototype": (
            "Two acquaintances or friends agree to a strictly sexual partnership governed by explicit rules to prevent "
            "romantic entanglement, until inevitable vulnerability causes genuine emotional feelings to bloom."
        ),
    },
}

class RomanceTropeExtractor:
    """Multi-tiered Romance Trope Classification with Evidence Tracking."""

    def __init__(self, tfidf_similarity_threshold: float = 0.28):
        self.similarity_threshold = tfidf_similarity_threshold
        self.trope_names = list(TROPE_DEFINITIONS.keys())

        # Fit TF-IDF on prototype descriptions
        prototypes = [TROPE_DEFINITIONS[t]["prototype"] for t in self.trope_names]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.prototype_matrix = self.vectorizer.fit_transform(prototypes)

    def extract_sentence_context(self, text: str, match_pattern: str) -> str:
        """Find the sentence containing the pattern match to preserve evidence."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for s in sentences:
            if re.search(match_pattern, s, re.IGNORECASE):
                return s.strip()
        return ""

    def extract_tropes_from_text(
        self, description: str, shelves: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract tropes using:
        1. Rule-based keyword and shelf matching (Baseline)
        2. TF-IDF prototype cosine similarity
        3. Hybrid confidence weighting
        """
        if not description and not shelves:
            return []

        text = description or ""

        # Parse shelves if passed as json string or list
        parsed_shelves: List[Any] = []
        if isinstance(shelves, str):
            try:
                import json
                parsed_shelves = json.loads(shelves)
            except Exception:
                parsed_shelves = [shelves]
        elif isinstance(shelves, list):
            parsed_shelves = shelves

        shelf_tokens = []
        for s in parsed_shelves:
            if isinstance(s, dict):
                shelf_tokens.append(str(s.get("name", "")))
            elif isinstance(s, str):
                shelf_tokens.append(s)

        shelf_text = " ".join(shelf_tokens)
        combined_text = f"{text} {shelf_text}".lower()

        extracted: Dict[str, Dict[str, Any]] = {}

        # 1. Rule-Based Matching
        for trope_name, trope_info in TROPE_DEFINITIONS.items():
            matches = []
            matched_sentences = []
            for pattern in trope_info["keywords"]:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    matches.append(pattern)
                    sent = self.extract_sentence_context(text, pattern)
                    if sent and sent not in matched_sentences:
                        matched_sentences.append(sent)

            if matches:
                # Calibrate confidence based on match count
                if len(matches) == 1:
                    confidence = 0.65
                elif len(matches) == 2:
                    confidence = 0.78
                else:
                    confidence = min(0.92, 0.78 + (len(matches) - 2) * 0.05)

                evidence = " | ".join(matched_sentences) if matched_sentences else f"Matched keywords: {', '.join(matches[:3])}"
                extracted[trope_name] = {
                    "trope_name": trope_name,
                    "confidence": round(confidence, 2),
                    "method": "rule_based_keyword",
                    "evidence": evidence[:400],
                }

        # 2. TF-IDF Similarity Candidate Matching with Sentence-Level Traceability
        if text.strip():
            desc_vec = self.vectorizer.transform([text])
            sims = cosine_similarity(desc_vec, self.prototype_matrix)[0]

            # Candidate sentences for evidence extraction
            candidate_sentences = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15
            ]

            for idx, trope_name in enumerate(self.trope_names):
                sim_score = float(sims[idx])
                if sim_score >= self.similarity_threshold:
                    # Find highest scoring sentence against trope prototype
                    best_sent = ""
                    if candidate_sentences:
                        sent_vecs = self.vectorizer.transform(candidate_sentences)
                        sent_sims = cosine_similarity(sent_vecs, self.prototype_matrix[idx]).flatten()
                        best_idx = int(np.argmax(sent_sims))
                        if sent_sims[best_idx] > 0.10:
                            best_sent = candidate_sentences[best_idx]

                    if trope_name in extracted:
                        # Hybrid detection: boost confidence
                        extracted[trope_name]["method"] = "hybrid"
                        extracted[trope_name]["confidence"] = min(
                            0.95, round(extracted[trope_name]["confidence"] * 0.7 + sim_score * 0.3, 2)
                        )
                        extracted[trope_name]["evidence"] += f" [Prototype similarity: {sim_score:.2f}]"
                    else:
                        evidence_str = (
                            f'Contextual alignment ({sim_score:.2f}): "{best_sent[:250]}"'
                            if best_sent
                            else f"Global description prototype cosine similarity: {sim_score:.2f}"
                        )
                        extracted[trope_name] = {
                            "trope_name": trope_name,
                            "confidence": round(sim_score, 2),
                            "method": "tfidf_matcher",
                            "evidence": evidence_str,
                        }

        # Sort by confidence descending
        return sorted(list(extracted.values()), key=lambda x: x["confidence"], reverse=True)
