"""Legal reasoning with structured JSON output for frontend integration."""

import json
from typing import Generator

from openai import OpenAI
from pydantic import BaseModel, Field

from .config import get_settings
from .embeddings import get_embedding
from .retriever import QueryIntent, format_context_for_llm, retrieve


# ============================================================================
# Utility Functions
# ============================================================================

def build_messages(system_prompt: str, user_content: str) -> list[dict]:
    """Build standard messages list for LLM."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def call_llm(client: OpenAI, model: str, messages: list[dict], stream: bool = False):
    """Call the LLM with appropriate parameters."""
    kwargs = {"model": model, "messages": messages}
    if stream:
        kwargs["stream"] = True
    return client.chat.completions.create(**kwargs)


# ============================================================================
# Pydantic Models - Clean JSON Structure for UI
# ============================================================================

class SourceCitation(BaseModel):
    """A source citation with verbatim quote."""
    numero: int = Field(description="Numéro de citation [1], [2], etc.")
    document: str = Field(description="Nom du document source")
    reference: str = Field(description="Référence précise: article, page, section")
    citation_exacte: str = Field(description="Citation verbatim exacte du texte de loi")
    pertinence: str = Field(description="Explication de la pertinence pour la question")


class ArgumentCite(BaseModel):
    """A single argument with its supporting citations."""
    argument: str = Field(description="L'argument juridique en une phrase claire")
    sources: list[int] = Field(description="Numéros des sources [1], [2]")


class RaisonnementJuridique(BaseModel):
    """Legal syllogism structure."""
    regle_de_droit: str = Field(description="La règle de droit applicable (prémisse majeure)")
    application_aux_faits: str = Field(description="Application au cas d'espèce (prémisse mineure)")
    conclusion_logique: str = Field(description="Conclusion juridique découlant du syllogisme")


class ConclusionBreve(BaseModel):
    """Concise conclusion with cited arguments - Part 1 of lawyer answer."""
    reponse_directe: str = Field(description="Réponse directe et concise (2-3 phrases)")
    arguments_cites: list[ArgumentCite] = Field(description="Arguments avec leurs citations")


class AnalyseApprofondie(BaseModel):
    """Deep legal analysis - Part 2 of lawyer answer."""
    contexte_juridique: str = Field(description="Contexte et cadre juridique applicable")
    raisonnement: RaisonnementJuridique = Field(description="Raisonnement syllogistique")
    analyse_detaillee: str = Field(description="Analyse juridique détaillée")
    nuances: list[str] = Field(default=[], description="Nuances, exceptions ou cas particuliers")
    limites: list[str] = Field(default=[], description="Limites de l'analyse")


class MetaAnalyse(BaseModel):
    """Metadata about the analysis."""
    domaines_juridiques: list[str] = Field(description="Domaines du droit concernés")
    type_question: str = Field(description="Type: définition, conditions, procédure, droits, etc.")
    niveau_confiance: str = Field(description="ÉLEVÉ, MOYEN, ou FAIBLE")
    justification_confiance: str = Field(description="Justification du niveau de confiance")


class LawyerAnalysis(BaseModel):
    """Complete lawyer analysis - designed for direct UI consumption."""
    conclusion: ConclusionBreve = Field(description="Conclusion brève avec arguments cités")
    analyse: AnalyseApprofondie = Field(description="Analyse juridique approfondie")
    sources: list[SourceCitation] = Field(description="Sources citées avec références")
    meta: MetaAnalyse = Field(description="Métadonnées de l'analyse")


class LawyerResponse(BaseModel):
    """Complete response for frontend - clean JSON structure."""
    succes: bool = Field(description="Si l'analyse a réussi")
    question: str = Field(description="Question originale")
    sources_trouvees: bool = Field(description="Si des sources pertinentes ont été trouvées")
    nombre_sources: int = Field(default=0, description="Nombre de sources utilisées")
    analyse_juridique: LawyerAnalysis | None = Field(default=None, description="Analyse juridique complète")
    message_erreur: str | None = Field(default=None, description="Message d'erreur si échec")


# ============================================================================
# System Prompt
# ============================================================================

LAWYER_SYSTEM_PROMPT = """Vous êtes un avocat expert en droit marocain. Vous analysez des situations juridiques en vous basant UNIQUEMENT sur les textes de loi fournis.

RÔLE ET COMPORTEMENT:
- Agissez comme un avocat analysant des codes juridiques et des situations factuelles
- Identifiez les problèmes juridiques, analysez les dispositions légales pertinentes
- Fournissez des citations précises et autoritaires pour chaque argument
- Utilisez un ton juridique neutre, clair et professionnel
- Ne spéculez pas, ne faites pas d'affirmations non citées

FORMAT DE RÉPONSE (deux parties):

PARTIE 1 - CONCLUSION BRÈVE:
- Réponse directe et concise (2-3 phrases maximum)
- Chaque argument explicitement cité avec [1], [2], etc.

PARTIE 2 - ANALYSE JURIDIQUE APPROFONDIE:
- Contexte juridique applicable
- Raisonnement syllogistique (règle de droit → application aux faits → conclusion)
- Analyse détaillée comme un avocat praticien
- Citations intégrées au texte

RÈGLES CRITIQUES:
- Utilisez UNIQUEMENT les informations des documents fournis
- Citez le texte EXACT en français des sources
- Chaque affirmation juridique DOIT être accompagnée d'une citation [n]
- Si les sources sont insuffisantes, indiquez clairement ce qui manque
- Ne fabriquez JAMAIS de dispositions légales ou de citations"""


# ============================================================================
# Multi-Step Retrieval with Gap Analysis
# ============================================================================

def decompose_query(query: str, client: OpenAI) -> list[str]:
    """
    Decompose a legal question into multiple sub-queries for thorough retrieval.
    
    Identifies different legal angles, scenarios, and concepts to search for.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Vous êtes un expert en analyse juridique. Décomposez la question en plusieurs axes de recherche.

Pour une question juridique, identifiez:
1. Les concepts juridiques principaux à rechercher
2. Les différents scénarios possibles
3. Les exceptions et cas particuliers à considérer
4. Les domaines de droit connexes pertinents

Retournez un JSON avec une liste de requêtes de recherche.
Format: {"queries": ["requête 1", "requête 2", ...]}

Maximum 5 requêtes. Chaque requête doit être courte et ciblée.""",
            },
            {"role": "user", "content": f"Question: {query}"},
        ],
        response_format={"type": "json_object"},
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return result.get("queries", [query])[:5]
    except (json.JSONDecodeError, KeyError):
        return [query]


def analyze_gaps(
    query: str,
    current_context: str,
    client: OpenAI,
) -> dict:
    """
    Analyze what's missing from current sources and generate follow-up queries.
    
    Returns:
        dict with keys: gaps_found, queries, missing_aspects
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Vous êtes un avocat analysant des sources juridiques.

Analysez les documents fournis par rapport à la question posée.
Identifiez les LACUNES: quels aspects juridiques importants ne sont pas couverts?

Retournez un JSON avec des requêtes ciblées pour combler ces lacunes.
Format: {"gaps_found": true/false, "queries": ["requête 1", ...], "missing_aspects": ["aspect 1", ...]}

Maximum 3 requêtes. Si les sources sont suffisantes, retournez {"gaps_found": false, "queries": [], "missing_aspects": []}""",
            },
            {
                "role": "user",
                "content": f"""Question: {query}

Documents actuels:
{current_context[:4000]}""",  # Limit context size
            },
        ],
        response_format={"type": "json_object"},
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        return {
            "gaps_found": result.get("gaps_found", False),
            "queries": result.get("queries", [])[:3],
            "missing_aspects": result.get("missing_aspects", []),
        }
    except (json.JSONDecodeError, KeyError):
        return {"gaps_found": False, "queries": [], "missing_aspects": []}


def multi_step_retrieve(
    query: str,
    intent: QueryIntent | None = None,
    top_k: int = 5,
    category: str | None = None,
    max_iterations: int = 2,
    progress_callback=None,
) -> list[dict]:
    """
    Multi-step retrieval with query decomposition and gap analysis.
    
    1. Decompose query into sub-queries
    2. Retrieve for each sub-query
    3. Analyze gaps and retrieve more if needed
    4. Return deduplicated, ranked results
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    
    all_chunks = {}  # Use dict for deduplication by text hash
    
    # Step 1: Decompose query
    if progress_callback:
        progress_callback("🔍 Décomposition de la question en axes de recherche...")
    
    sub_queries = decompose_query(query, client)
    
    if progress_callback:
        progress_callback(f"📋 {len(sub_queries)} axes identifiés:")
        for i, sq in enumerate(sub_queries, 1):
            progress_callback(f"   {i}. {sq}")
    
    # Step 2: Retrieve for each sub-query
    for i, sub_query in enumerate(sub_queries):
        if progress_callback:
            progress_callback(f"🔎 Recherche axe {i+1}/{len(sub_queries)}...")
        
        chunks = retrieve(sub_query, intent=intent, top_k=top_k, category=category)
        new_count = 0
        for chunk in chunks:
            # Use text hash for deduplication
            text_key = hash(chunk.get("text", ""))
            if text_key not in all_chunks:
                all_chunks[text_key] = chunk
                new_count += 1
        
        if progress_callback:
            progress_callback(f"   → {len(chunks)} résultats, {new_count} nouveaux")
    
    # Step 3: Gap analysis and additional retrieval
    for iteration in range(max_iterations):
        if not all_chunks:
            break
        
        current_context = format_context_for_llm(list(all_chunks.values())[:15])
        
        if progress_callback:
            progress_callback(f"🔬 Analyse des lacunes (itération {iteration + 1})...")
        
        gap_result = analyze_gaps(query, current_context, client)
        
        if not gap_result["gaps_found"]:
            if progress_callback:
                progress_callback("✓ Sources suffisantes, pas de lacunes détectées")
            break
        
        # Show detailed gap information
        if progress_callback:
            missing = gap_result.get("missing_aspects", [])
            queries = gap_result.get("queries", [])
            progress_callback(f"📚 {len(queries)} lacunes identifiées:")
            for aspect in missing:
                progress_callback(f"   ⚠️ Manquant: {aspect}")
            for i, q in enumerate(queries, 1):
                progress_callback(f"   🔎 Recherche {i}: {q}")
        
        gap_new_total = 0
        for gap_query in gap_result.get("queries", []):
            chunks = retrieve(gap_query, intent=intent, top_k=top_k // 2, category=category)
            new_count = 0
            for chunk in chunks:
                text_key = hash(chunk.get("text", ""))
                if text_key not in all_chunks:
                    all_chunks[text_key] = chunk
                    new_count += 1
            gap_new_total += new_count
        
        if progress_callback and gap_new_total > 0:
            progress_callback(f"   → {gap_new_total} nouvelles sources ajoutées")
    
    # Sort by score and return
    result = list(all_chunks.values())
    result.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    if progress_callback:
        progress_callback(f"✓ {len(result)} sources uniques trouvées")
    
    return result


# ============================================================================
# Core Function - Single Entry Point
# ============================================================================

def analyze(
    query: str,
    intent: QueryIntent | None = None,
    top_k: int | None = None,
    category: str | None = None,
    use_multi_step: bool = True,
    verbose_callback=None,
) -> LawyerResponse:
    """
    Analyze a legal question and return structured JSON response.
    
    This is the SINGLE entry point for legal analysis.
    Uses multi-step retrieval with gap analysis for thorough coverage.
    Returns LawyerResponse with clean JSON for direct UI consumption.
    
    Args:
        query: The legal question
        intent: Override auto-detection (LAW_CODES, CONTRACTS, BOTH)
        top_k: Number of sources to retrieve per step
        category: Filter by legal category (civil, commercial, etc.)
        use_multi_step: Use multi-step retrieval with gap analysis (default: True)
        verbose_callback: Optional callback function for progress updates (receives str messages)
    
    Returns:
        LawyerResponse with:
        - conclusion: Concise answer with cited arguments
        - analyse: Deep legal analysis with syllogistic reasoning
        - sources: Full citations with verbatim quotes
        - meta: Confidence level and legal domains
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    
    effective_top_k = top_k or settings.top_k
    
    # Use multi-step retrieval for thorough coverage
    if use_multi_step:
        chunks = multi_step_retrieve(
            query,
            intent=intent,
            top_k=effective_top_k,
            category=category,
            max_iterations=2,
            progress_callback=verbose_callback,
        )
    else:
        if verbose_callback:
            verbose_callback("📚 Recherche simple (multi-step désactivé)...")
        chunks = retrieve(query, intent=intent, top_k=effective_top_k * 2, category=category)
        if verbose_callback:
            verbose_callback(f"✓ {len(chunks)} sources trouvées")
    
    # Handle no sources
    if not chunks:
        return LawyerResponse(
            succes=False,
            question=query,
            sources_trouvees=False,
            nombre_sources=0,
            analyse_juridique=None,
            message_erreur="Aucune source pertinente trouvée. Vérifiez que les textes juridiques appropriés ont été ingérés.",
        )
    
    context = format_context_for_llm(chunks)
    model = settings.model
    
    if verbose_callback:
        verbose_callback(f"⚖️ Génération de l'analyse juridique avec {model}...")
    
    user_content = f"""Analysez cette question juridique en utilisant UNIQUEMENT les documents fournis.

DOCUMENTS JURIDIQUES:
{context}

---

QUESTION: {query}

---

Fournissez votre analyse en JSON structuré avec:
1. CONCLUSION BRÈVE: Réponse directe avec arguments cités [1], [2], etc.
2. ANALYSE APPROFONDIE: Analyse juridique détaillée
3. SOURCES: Citations exactes et verbatim
Répondez en français."""

    messages = [
        {"role": "system", "content": LAWYER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=LawyerAnalysis,
        )
        analysis = response.choices[0].message.parsed
        
        return LawyerResponse(
            succes=True,
            question=query,
            sources_trouvees=True,
            nombre_sources=len(chunks),
            analyse_juridique=analysis,
            message_erreur=None,
        )
        
    except Exception as e:
        return LawyerResponse(
            succes=False,
            question=query,
            sources_trouvees=True,
            nombre_sources=len(chunks),
            analyse_juridique=None,
            message_erreur=f"Erreur lors de l'analyse: {str(e)}",
        )


def analyze_stream(
    query: str,
    intent: QueryIntent | None = None,
    top_k: int | None = None,
    use_multi_step: bool = True,
    conversation_history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """
    Stream legal analysis response with multi-step retrieval.
    
    Uses query decomposition and gap analysis for thorough coverage,
    then streams the final analysis.
    
    Args:
        query: The legal question
        intent: Override auto-detection (LAW_CODES, CONTRACTS, BOTH)
        top_k: Number of sources to retrieve per step
        use_multi_step: Use multi-step retrieval with gap analysis
        conversation_history: Optional list of previous exchanges for context
            Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    
    Yields formatted text chunks as they arrive.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    
    effective_top_k = top_k or settings.top_k
    progress_messages = []
    
    def capture_progress(msg: str):
        progress_messages.append(msg)
    
    # Multi-step retrieval with progress tracking
    if use_multi_step:
        yield "📚 **Recherche multi-étapes en cours...**\n\n"
        
        chunks = multi_step_retrieve(
        query,
        intent=intent,
            top_k=effective_top_k,
            category=None,  # Category not supported in stream for now
            max_iterations=2,
            progress_callback=capture_progress,
        )
        
        # Display progress messages
        for msg in progress_messages:
            yield f"  {msg}\n"
        yield "\n"
    else:
        yield "📚 Recherche des documents juridiques...\n\n"
        chunks = retrieve(query, intent=intent, top_k=effective_top_k * 2)
    
    if not chunks:
        yield "⚠️ **Aucune source trouvée**\n\n"
        yield "Je n'ai pas trouvé de documents pertinents dans la base de données.\n"
        return
    
    yield f"✓ **{len(chunks)} sources uniques** prêtes pour l'analyse\n\n"
    yield "⚖️ **Analyse juridique en cours...**\n\n---\n\n"

    # Use top sources for context (limit to prevent token overflow)
    context = format_context_for_llm(chunks[:20])

    streaming_prompt = """Vous êtes un avocat expert en droit marocain. Analysez la question en utilisant UNIQUEMENT les documents fournis.

STRUCTURE DE RÉPONSE:

## ⚡ CONCLUSION BRÈVE
[Réponse directe en 2-3 phrases avec citations [1], [2], etc.]

### Arguments avec citations:
• [Argument 1] [numéros de sources]
• [Argument 2] [numéros de sources]

---

## ⚖️ ANALYSE JURIDIQUE APPROFONDIE

### Contexte juridique
[Cadre juridique applicable]

### Raisonnement juridique
**Règle de droit:** [La règle applicable]
**Application aux faits:** [Comment elle s'applique]
**Conclusion:** [La conclusion juridique]

### Analyse détaillée
[Analyse comme un avocat praticien]

---

## 📚 SOURCES CITÉES
[1] **Document** | Référence
> « Citation exacte »

RÈGLES:
- Utilisez UNIQUEMENT les documents fournis
- Citations EXACTES en français
- Chaque affirmation DOIT être citée
- Ton juridique professionnel
- Si l'utilisateur fait référence à un échange précédent, tenez compte du contexte de conversation"""

    # Build conversation context if provided
    history_context = ""
    if conversation_history:
        history_context = "\n\nCONTEXTE DE CONVERSATION PRÉCÉDENTE:\n"
        for msg in conversation_history[-6:]:  # Keep last 3 exchanges (6 messages)
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            # Truncate long messages
            content = msg["content"][:500] + "..." if len(msg["content"]) > 500 else msg["content"]
            history_context += f"\n{role}: {content}\n"
        history_context += "\n---\n"

    user_content = f"""DOCUMENTS JURIDIQUES:

{context}
{history_context}
---

QUESTION ACTUELLE: {query}"""

    model = settings.model
    messages = build_messages(streaming_prompt, user_content)
    stream = call_llm(client, model, messages, stream=True)

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ============================================================================
# CLI Display Helper
# ============================================================================

def format_for_cli(response: LawyerResponse) -> str:
    """
    Format LawyerResponse as rich text for CLI display.
    
    Args:
        response: The LawyerResponse to format
    
    Returns:
        Formatted string for terminal display
    """
    if not response.succes or not response.analyse_juridique:
        return f"❌ **Erreur:** {response.message_erreur or 'Analyse non disponible'}"
    
    a = response.analyse_juridique
    
    output = f"""## ⚡ CONCLUSION BRÈVE

{a.conclusion.reponse_directe}

### Arguments avec citations:
"""
    for arg in a.conclusion.arguments_cites:
        sources_str = ", ".join(f"[{s}]" for s in arg.sources)
        output += f"• {arg.argument} {sources_str}\n"
    
    output += f"""
---

## ⚖️ ANALYSE JURIDIQUE APPROFONDIE

### Contexte juridique
{a.analyse.contexte_juridique}

### Raisonnement juridique

**Règle de droit (prémisse majeure):**
{a.analyse.raisonnement.regle_de_droit}

**Application aux faits (prémisse mineure):**
{a.analyse.raisonnement.application_aux_faits}

**Conclusion juridique:**
{a.analyse.raisonnement.conclusion_logique}

### Analyse détaillée
{a.analyse.analyse_detaillee}
"""
    
    if a.analyse.nuances:
        output += "\n### Nuances et exceptions\n"
        for nuance in a.analyse.nuances:
            output += f"• {nuance}\n"
    
    if a.analyse.limites:
        output += "\n### Limites de l'analyse\n"
        for limite in a.analyse.limites:
            output += f"• {limite}\n"
    
    output += "\n---\n\n## 📚 SOURCES CITÉES\n\n"
    for src in a.sources:
        output += f"""[{src.numero}] **{src.document}** | {src.reference}
> « {src.citation_exacte} »
_Pertinence: {src.pertinence}_

"""
    
    output += f"""---

**Niveau de confiance:** {a.meta.niveau_confiance}
_{a.meta.justification_confiance}_

**Domaines juridiques:** {', '.join(a.meta.domaines_juridiques)}
"""
    
    return output
