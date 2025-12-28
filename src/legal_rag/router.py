"""Question router to determine if RAG is needed.

Routes questions to avoid unnecessary RAG calls:
- DIRECT_ANSWER: General/explanatory questions → answer without RAG
- NEEDS_RAG: Questions requiring specific legal text/citations
- UNCERTAIN: Try direct first, fallback to RAG if needed
"""

from enum import Enum

from openai import OpenAI

from .config import get_settings


class QuestionType(Enum):
    """Classification of question for routing."""

    DIRECT_ANSWER = "direct"  # General/explanatory → no RAG needed
    NEEDS_RAG = "needs_rag"  # Specific legal text required
    UNCERTAIN = "uncertain"  # Try direct first, may need RAG


ROUTER_SYSTEM_PROMPT = """Vous êtes un classificateur de questions juridiques pour un système RAG marocain.

Classifiez chaque question en UNE des trois catégories:

DIRECT - Répondez directement, PAS besoin de consulter les textes de loi:
• Questions conceptuelles/définitionnelles ("Qu'est-ce que le droit civil?")
• Explications générales ("Comment fonctionne un contrat de travail?")
• Comparaisons générales ("Différence entre CDI et CDD?")
• Questions de procédure générale ("Comment créer une entreprise?")
• Questions qui peuvent être répondues avec des connaissances juridiques générales

RAG - BESOIN de consulter les textes de loi:
• Demande d'article spécifique ("Article 35 du Code du travail")
• Demande de citation exacte ("Citez les dispositions sur...")
• Questions sur conditions/délais précis ("Quel est le délai de préavis selon la loi?")
• Références à des textes spécifiques ("Que dit le Code de commerce sur...?")
• Questions nécessitant des citations autoritaires précises

UNCERTAIN - Incertain, essayer d'abord sans RAG:
• Questions ambiguës qui pourraient nécessiter ou non des citations
• Questions où une réponse générale pourrait suffire mais des précisions légales aideraient

Répondez avec EXACTEMENT un mot: DIRECT, RAG, ou UNCERTAIN"""


def route_question(query: str, client: OpenAI | None = None) -> QuestionType:
    """
    Classify a question to determine if RAG is needed.

    Args:
        query: The user's question
        client: Optional OpenAI client (created if not provided)

    Returns:
        QuestionType indicating routing decision
    """
    if client is None:
        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Fast and cheap for classification
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        max_tokens=10,
        temperature=0,
    )

    classification = response.choices[0].message.content.strip().upper()

    if classification == "DIRECT":
        return QuestionType.DIRECT_ANSWER
    elif classification == "RAG":
        return QuestionType.NEEDS_RAG
    else:
        return QuestionType.UNCERTAIN


# System prompt for direct answers (no RAG)
DIRECT_ANSWER_SYSTEM_PROMPT = """Vous êtes un avocat expert en droit marocain répondant à des questions juridiques générales.

COMPORTEMENT:
• Répondez de manière claire, structurée et professionnelle
• Utilisez vos connaissances générales du droit marocain
• Soyez pédagogique pour les questions conceptuelles
• Indiquez clairement quand une question nécessiterait une consultation des textes de loi spécifiques
• Ne fabriquez pas de numéros d'articles ou de citations exactes

FORMAT:
• Réponse directe et concise
• Explications structurées si nécessaire
• Mentionnez si l'utilisateur devrait consulter les textes de loi pour des précisions

LIMITES:
• Si vous n'êtes pas sûr d'une information précise, dites-le
• Si la question nécessite vraiment des citations légales, indiquez: "Pour une réponse avec citations légales précises, reformulez votre question en demandant les articles de loi ou utilisez le mode /rag"
"""

# Indicator for when model needs RAG (used to detect fallback needed)
NEEDS_LAW_TEXT_INDICATORS = [
    "je n'ai pas accès aux textes",
    "consulter les textes de loi",
    "utiliser le mode /rag",
    "citations légales précises",
    "je ne peux pas citer",
    "sans accès aux articles",
]


def check_needs_rag_fallback(response_text: str) -> bool:
    """
    Check if a direct answer indicates it needs RAG.

    Args:
        response_text: The model's response text

    Returns:
        True if the response suggests RAG would be helpful
    """
    response_lower = response_text.lower()
    return any(indicator in response_lower for indicator in NEEDS_LAW_TEXT_INDICATORS)
