"""Deep legal reasoning with multi-step analysis and chain-of-thought."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Generator

from openai import OpenAI

from .config import get_settings
from .retriever import QueryIntent, format_context_for_llm, retrieve


def is_reasoning_model(model: str) -> bool:
    """Check if the model is an OpenAI reasoning model (o1, o3, etc.)."""
    reasoning_prefixes = ("o1", "o3", "o4")
    return any(model.startswith(prefix) for prefix in reasoning_prefixes)


def build_messages_for_model(
    model: str,
    system_prompt: str,
    user_content: str,
) -> list[dict]:
    """
    Build messages list appropriate for the model type.
    
    Reasoning models (o1, o3) use 'developer' role instead of 'system'.
    """
    if is_reasoning_model(model):
        # Reasoning models: use 'developer' role for instructions
        return [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    else:
        # Standard models: use 'system' role
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]


def call_llm(
    client: OpenAI,
    model: str,
    messages: list[dict],
    stream: bool = False,
):
    """
    Call the LLM with appropriate parameters for the model type.
    
    Reasoning models don't support temperature parameter.
    """
    kwargs = {
        "model": model,
        "messages": messages,
    }
    
    if stream:
        kwargs["stream"] = True
    
    # Only add temperature for non-reasoning models
    if not is_reasoning_model(model):
        kwargs["temperature"] = 0.1
    
    return client.chat.completions.create(**kwargs)


class ReasoningDepth(Enum):
    """Level of reasoning depth for analysis."""

    QUICK = "quick"  # Fast, direct answer
    STANDARD = "standard"  # Balanced analysis
    DEEP = "deep"  # Comprehensive legal reasoning


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""

    step_type: str
    content: str
    sources_used: list[int] = field(default_factory=list)


@dataclass
class LegalReasoning:
    """Complete legal reasoning output."""

    query: str
    intent: QueryIntent
    reasoning_steps: list[ReasoningStep]
    final_answer: str
    sources: list[dict]
    confidence: str  # HIGH, MEDIUM, LOW based on source coverage


# Legal reasoning system prompt for deep analysis
REASONING_SYSTEM_PROMPT = """You are an expert legal analyst specializing in Moroccan law. Your role is to provide deep, structured legal reasoning.

You MUST follow this exact reasoning framework:

## PHASE 1: QUESTION ANALYSIS
First, deeply analyze the legal question:
- What is the core legal issue?
- What sub-questions need to be answered?
- What areas of law are implicated (civil, commercial, procedural, etc.)?
- What legal concepts are at play?

## PHASE 2: SOURCE EXAMINATION
For each provided document:
- Extract the exact relevant legal provisions
- Note article numbers, section headers, and page references
- Identify how each source relates to the question
- Flag any conflicting provisions

## PHASE 3: LEGAL REASONING
Apply structured legal analysis:
- Start from general principles, then move to specific rules
- Consider hierarchy of legal norms
- Apply legal syllogism (major premise → minor premise → conclusion)
- Address any exceptions or special cases
- Consider temporal application (when laws came into effect)

## PHASE 4: SYNTHESIS
- Integrate findings from all sources
- Resolve any apparent conflicts
- State conclusions with confidence levels
- Identify any gaps in the available information

## OUTPUT FORMAT

### RAISONNEMENT JURIDIQUE

**Analyse de la Question:**
[Your analysis of the legal question and its components]

**Dispositions Légales Pertinentes:**
[Cite each relevant legal provision with exact quotes and references [1], [2], etc.]

**Raisonnement:**
[Step-by-step legal reasoning applying law to the question]

**Conclusion:**
[Clear, definitive answer with confidence level]

**Limites:**
[Any limitations or caveats in the analysis]

### SOURCES
[Full citations with verbatim quotes]

CRITICAL RULES:
1. ONLY use information from the provided documents
2. ALWAYS cite with bracketed numbers [1], [2]
3. Use EXACT QUOTES in French from the sources
4. If information is insufficient, clearly state what is missing
5. Never fabricate legal provisions or citations
6. Distinguish between what the law says vs interpretation"""


DECOMPOSITION_PROMPT = """You are a legal question analyzer. Given a complex legal question, break it down into specific sub-questions that need to be researched.

For the following question, identify:
1. The main legal issue
2. 2-5 specific sub-questions that need to be answered
3. The areas of law that should be searched

Output as JSON:
{
    "main_issue": "description of core legal issue",
    "sub_questions": ["question 1", "question 2", ...],
    "legal_areas": ["civil", "commercial", etc.],
    "key_terms": ["term1", "term2", ...] 
}

Only output valid JSON, nothing else."""


SYNTHESIS_PROMPT = """You are a legal synthesis expert. Given multiple pieces of legal analysis about related sub-questions, synthesize them into a coherent, comprehensive answer.

Combine the analyses while:
1. Maintaining all citations
2. Resolving any apparent conflicts between sources
3. Building a logical argument structure
4. Providing a clear final conclusion

Format your synthesis with:
- Introduction stating the question
- Organized presentation of findings
- Final conclusion with confidence level"""


GAP_ANALYSIS_PROMPT = """You are a legal research assistant analyzing whether retrieved documents fully answer a legal question.

Given:
1. The original legal question
2. The documents retrieved so far

Determine:
1. Is the question fully answerable with current documents? (yes/no)
2. What specific information is MISSING to fully answer?
3. What additional search queries would help find the missing information?

Output as JSON:
{
    "is_sufficient": true/false,
    "confidence": "high/medium/low",
    "missing_information": ["list of missing pieces"],
    "follow_up_queries": ["query 1", "query 2", ...]
}

Be strict - if any important aspect is not covered, mark as insufficient.
Only output valid JSON."""


HYDE_PROMPT = """You are a legal expert. Given a legal question, write a hypothetical ideal passage that would perfectly answer this question. 

This passage should:
1. Be written as if from an authoritative legal source (law code, doctrine)
2. Include specific article numbers, legal terms, and precise language
3. Be in French (for Moroccan law context)
4. Be detailed but concise (1-2 paragraphs)

Question: {query}

Hypothetical ideal passage:"""


def generate_hyde_embedding(query: str) -> list[float]:
    """
    Generate a Hypothetical Document Embedding (HyDE).
    
    Creates a hypothetical ideal answer and embeds it for better retrieval.
    """
    from .embeddings import get_embedding
    
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    
    # Generate hypothetical document
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": HYDE_PROMPT.format(query=query)},
        ],
        temperature=0.7,  # Some creativity for diverse hypotheticals
        max_tokens=500,
    )
    
    hypothetical_doc = response.choices[0].message.content
    
    # Embed the hypothetical document
    return get_embedding(hypothetical_doc)


def analyze_retrieval_gaps(
    query: str,
    chunks: list[dict],
) -> dict:
    """
    Analyze if retrieved documents fully answer the question.
    
    Returns gap analysis with follow-up queries if needed.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    
    context = format_context_for_llm(chunks)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": GAP_ANALYSIS_PROMPT},
            {
                "role": "user",
                "content": f"""Question: {query}

Retrieved Documents:
{context}

Analyze if these documents are sufficient to fully answer the question.""",
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    
    import json
    return json.loads(response.choices[0].message.content)


def multi_step_retrieve(
    query: str,
    intent: QueryIntent | None = None,
    top_k: int | None = None,
    max_iterations: int = 3,
    use_hyde: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Multi-step iterative retrieval with gap analysis.
    
    Retrieves documents, analyzes gaps, and retrieves more until sufficient.
    
    Args:
        query: The legal question
        intent: Override intent detection
        top_k: Chunks per retrieval step
        max_iterations: Maximum retrieval iterations
        use_hyde: Whether to use HyDE for initial retrieval
        
    Returns:
        Tuple of (final_chunks, retrieval_steps)
    """
    from .embeddings import get_embedding
    from .pinecone_store import (
        NAMESPACE_LAW_CODES,
        NAMESPACE_USER_CONTRACTS,
        query_similar,
        query_multiple_namespaces,
    )
    
    settings = get_settings()
    effective_top_k = top_k or settings.top_k
    
    all_chunks = []
    seen_ids = set()
    retrieval_steps = []
    
    # Step 1: Initial retrieval (optionally with HyDE)
    if use_hyde:
        retrieval_steps.append({
            "step": 1,
            "type": "hyde",
            "description": "Generating hypothetical document for enhanced retrieval",
        })
        query_embedding = generate_hyde_embedding(query)
    else:
        query_embedding = get_embedding(query)
    
    # Determine namespaces based on intent
    if intent is None:
        from .retriever import detect_query_intent
        intent = detect_query_intent(query)
    
    if intent == QueryIntent.LAW_CODES:
        initial_chunks = query_similar(query_embedding, NAMESPACE_LAW_CODES, effective_top_k)
    elif intent == QueryIntent.CONTRACTS:
        initial_chunks = query_similar(query_embedding, NAMESPACE_USER_CONTRACTS, effective_top_k)
    else:
        initial_chunks = query_multiple_namespaces(
            query_embedding,
            [NAMESPACE_LAW_CODES, NAMESPACE_USER_CONTRACTS],
            effective_top_k,
        )
    
    for chunk in initial_chunks:
        chunk_id = chunk.get("id", hash(chunk.get("text", "")[:100]))
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            all_chunks.append(chunk)
    
    retrieval_steps.append({
        "step": len(retrieval_steps) + 1,
        "type": "initial_retrieval",
        "query": query,
        "chunks_found": len(initial_chunks),
    })
    
    # Iterative refinement
    for iteration in range(max_iterations - 1):
        # Analyze gaps
        gap_analysis = analyze_retrieval_gaps(query, all_chunks)
        
        retrieval_steps.append({
            "step": len(retrieval_steps) + 1,
            "type": "gap_analysis",
            "is_sufficient": gap_analysis.get("is_sufficient", False),
            "confidence": gap_analysis.get("confidence", "low"),
            "missing": gap_analysis.get("missing_information", []),
        })
        
        # Stop if sufficient
        if gap_analysis.get("is_sufficient", False):
            break
        
        # Get follow-up queries
        follow_up_queries = gap_analysis.get("follow_up_queries", [])
        if not follow_up_queries:
            break
        
        # Retrieve for each follow-up query
        new_chunks_count = 0
        for fq in follow_up_queries[:2]:  # Limit to 2 follow-ups per iteration
            fq_embedding = get_embedding(fq)
            
            if intent == QueryIntent.LAW_CODES:
                fq_chunks = query_similar(fq_embedding, NAMESPACE_LAW_CODES, 3)
            elif intent == QueryIntent.CONTRACTS:
                fq_chunks = query_similar(fq_embedding, NAMESPACE_USER_CONTRACTS, 3)
            else:
                fq_chunks = query_multiple_namespaces(
                    fq_embedding,
                    [NAMESPACE_LAW_CODES, NAMESPACE_USER_CONTRACTS],
                    3,
                )
            
            for chunk in fq_chunks:
                chunk_id = chunk.get("id", hash(chunk.get("text", "")[:100]))
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    all_chunks.append(chunk)
                    new_chunks_count += 1
        
        retrieval_steps.append({
            "step": len(retrieval_steps) + 1,
            "type": "follow_up_retrieval",
            "queries": follow_up_queries[:2],
            "new_chunks": new_chunks_count,
        })
        
        # Stop if no new chunks found
        if new_chunks_count == 0:
            break
    
    # Limit total chunks to avoid context overflow
    all_chunks = all_chunks[:20]
    
    return all_chunks, retrieval_steps


def decompose_query(query: str) -> dict:
    """
    Decompose a complex legal query into sub-questions.
    
    Returns structured breakdown of the query.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": DECOMPOSITION_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    import json

    return json.loads(response.choices[0].message.content)


def reason_deep(
    query: str,
    intent: QueryIntent | None = None,
    top_k: int | None = None,
) -> LegalReasoning:
    """
    Perform deep legal reasoning on a query.
    
    Uses multi-step analysis with chain-of-thought prompting.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # Use more chunks for deep reasoning
    effective_top_k = (top_k or settings.top_k) * 2

    # Retrieve relevant chunks
    chunks = retrieve(query, intent=intent, top_k=effective_top_k)

    # Format context
    context = format_context_for_llm(chunks)

    # Build reasoning prompt
    model = settings.reasoning_model
    user_content = f"""Documents Juridiques Disponibles:

{context}

---

Question Juridique: {query}

Veuillez fournir une analyse juridique approfondie en suivant le cadre de raisonnement structuré."""

    messages = build_messages_for_model(model, REASONING_SYSTEM_PROMPT, user_content)

    # Call LLM with appropriate parameters for model type
    response = call_llm(client, model, messages)

    answer = response.choices[0].message.content

    # Determine confidence based on source relevance
    confidence = _assess_confidence(chunks)

    return LegalReasoning(
        query=query,
        intent=intent or QueryIntent.BOTH,
        reasoning_steps=[
            ReasoningStep(
                step_type="analysis",
                content="Question decomposed and analyzed",
                sources_used=list(range(1, len(chunks) + 1)),
            ),
            ReasoningStep(
                step_type="reasoning",
                content="Legal provisions examined and applied",
                sources_used=list(range(1, len(chunks) + 1)),
            ),
        ],
        final_answer=answer,
        sources=chunks,
        confidence=confidence,
    )


def reason_multistep(
    query: str,
    intent: QueryIntent | None = None,
    top_k: int | None = None,
    max_iterations: int = 3,
    use_hyde: bool = False,
) -> LegalReasoning:
    """
    Deep reasoning with multi-step iterative retrieval.
    
    Uses HyDE and gap analysis to iteratively retrieve until sufficient.
    
    Args:
        query: The legal question
        intent: Override intent detection
        top_k: Chunks per retrieval step
        max_iterations: Maximum retrieval iterations
        use_hyde: Whether to use Hypothetical Document Embeddings
        
    Returns:
        LegalReasoning with comprehensive analysis
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    
    # Multi-step retrieval
    chunks, retrieval_steps = multi_step_retrieve(
        query,
        intent=intent,
        top_k=top_k,
        max_iterations=max_iterations,
        use_hyde=use_hyde,
    )
    
    # Format context
    context = format_context_for_llm(chunks)
    
    # Build reasoning prompt
    model = settings.reasoning_model
    user_content = f"""Documents Juridiques Disponibles (récupérés en {len(retrieval_steps)} étapes):

{context}

---

Question Juridique: {query}

Veuillez fournir une analyse juridique approfondie en suivant le cadre de raisonnement structuré."""

    messages = build_messages_for_model(model, REASONING_SYSTEM_PROMPT, user_content)
    response = call_llm(client, model, messages)
    
    answer = response.choices[0].message.content
    confidence = _assess_confidence(chunks)
    
    # Convert retrieval steps to ReasoningSteps
    reasoning_steps = []
    for step in retrieval_steps:
        step_type = step.get("type", "unknown")
        if step_type == "hyde":
            content = "Generated hypothetical document for enhanced retrieval"
        elif step_type == "initial_retrieval":
            content = f"Initial retrieval found {step.get('chunks_found', 0)} documents"
        elif step_type == "gap_analysis":
            if step.get("is_sufficient"):
                content = f"Gap analysis: Sufficient (confidence: {step.get('confidence')})"
            else:
                missing = step.get("missing", [])
                content = f"Gap analysis: Missing {len(missing)} pieces of information"
        elif step_type == "follow_up_retrieval":
            content = f"Follow-up retrieval: +{step.get('new_chunks', 0)} new documents"
        else:
            content = str(step)
        
        reasoning_steps.append(ReasoningStep(
            step_type=step_type,
            content=content,
        ))
    
    reasoning_steps.append(ReasoningStep(
        step_type="synthesis",
        content="Final legal analysis synthesized",
        sources_used=list(range(1, len(chunks) + 1)),
    ))
    
    return LegalReasoning(
        query=query,
        intent=intent or QueryIntent.BOTH,
        reasoning_steps=reasoning_steps,
        final_answer=answer,
        sources=chunks,
        confidence=confidence,
    )


def reason_with_decomposition(
    query: str,
    intent: QueryIntent | None = None,
    top_k: int | None = None,
) -> LegalReasoning:
    """
    Advanced reasoning with query decomposition.
    
    Breaks down complex queries and analyzes each part.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # Step 1: Decompose the query
    decomposition = decompose_query(query)

    # Step 2: Retrieve for main query and each sub-question
    all_chunks = []
    chunk_sources = {}

    # Main query retrieval
    main_chunks = retrieve(query, intent=intent, top_k=top_k or settings.top_k)
    all_chunks.extend(main_chunks)

    # Sub-question retrieval (get additional context)
    for sub_q in decomposition.get("sub_questions", [])[:3]:  # Limit to 3 sub-questions
        sub_chunks = retrieve(sub_q, intent=intent, top_k=3)
        for chunk in sub_chunks:
            # Avoid duplicates
            chunk_id = chunk.get("id", chunk.get("text", "")[:100])
            if chunk_id not in chunk_sources:
                chunk_sources[chunk_id] = chunk
                all_chunks.append(chunk)

    # Deduplicate and limit
    unique_chunks = []
    seen_texts = set()
    for chunk in all_chunks:
        text_key = chunk.get("text", "")[:200]
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            unique_chunks.append(chunk)
    
    unique_chunks = unique_chunks[:15]  # Limit total chunks

    # Step 3: Format enhanced context
    context = format_context_for_llm(unique_chunks)

    # Step 4: Build comprehensive reasoning prompt
    sub_questions_text = "\n".join(
        f"  - {sq}" for sq in decomposition.get("sub_questions", [])
    )
    key_terms = ", ".join(decomposition.get("key_terms", []))

    enhanced_prompt = f"""Documents Juridiques Disponibles:

{context}

---

Question Principale: {query}

Analyse Préliminaire:
- Problématique Centrale: {decomposition.get('main_issue', query)}
- Sous-Questions à Examiner:
{sub_questions_text}
- Domaines Juridiques: {', '.join(decomposition.get('legal_areas', ['général']))}
- Termes Clés: {key_terms}

Veuillez fournir une analyse juridique exhaustive en:
1. Répondant à chaque sous-question identifiée
2. Synthétisant les réponses en une conclusion cohérente
3. Citant précisément les sources pour chaque affirmation"""

    model = settings.reasoning_model
    messages = build_messages_for_model(model, REASONING_SYSTEM_PROMPT, enhanced_prompt)

    response = call_llm(client, model, messages)

    answer = response.choices[0].message.content
    confidence = _assess_confidence(unique_chunks)

    return LegalReasoning(
        query=query,
        intent=intent or QueryIntent.BOTH,
        reasoning_steps=[
            ReasoningStep(
                step_type="decomposition",
                content=f"Query decomposed into {len(decomposition.get('sub_questions', []))} sub-questions",
            ),
            ReasoningStep(
                step_type="retrieval",
                content=f"Retrieved {len(unique_chunks)} relevant legal documents",
            ),
            ReasoningStep(
                step_type="analysis",
                content="Each sub-question analyzed with relevant provisions",
                sources_used=list(range(1, len(unique_chunks) + 1)),
            ),
            ReasoningStep(
                step_type="synthesis",
                content="Findings synthesized into comprehensive answer",
            ),
        ],
        final_answer=answer,
        sources=unique_chunks,
        confidence=confidence,
    )


def reason_stream(
    query: str,
    depth: ReasoningDepth = ReasoningDepth.STANDARD,
    intent: QueryIntent | None = None,
    top_k: int | None = None,
) -> Generator[str, None, None]:
    """
    Stream deep reasoning response for interactive use.
    
    Yields chunks of the reasoning as they arrive.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # Adjust retrieval based on depth
    if depth == ReasoningDepth.QUICK:
        effective_top_k = top_k or settings.top_k
        system_prompt = REASONING_SYSTEM_PROMPT
    elif depth == ReasoningDepth.STANDARD:
        effective_top_k = (top_k or settings.top_k) + 3
        system_prompt = REASONING_SYSTEM_PROMPT
    else:  # DEEP
        effective_top_k = (top_k or settings.top_k) * 2
        # For deep, we do decomposition first
        yield "🔍 Analysing query structure...\n\n"
        decomposition = decompose_query(query)
        yield f"📋 Identified {len(decomposition.get('sub_questions', []))} sub-questions\n\n"
        system_prompt = REASONING_SYSTEM_PROMPT

    # Retrieve
    yield "📚 Retrieving relevant legal documents...\n\n"
    chunks = retrieve(query, intent=intent, top_k=effective_top_k)
    yield f"✓ Found {len(chunks)} relevant passages\n\n"
    yield "⚖️ Applying legal reasoning...\n\n---\n\n"

    # Format context
    context = format_context_for_llm(chunks)

    user_content = f"""Documents Juridiques:

{context}

---

Question: {query}

Analyse juridique approfondie:"""

    model = settings.reasoning_model
    messages = build_messages_for_model(model, system_prompt, user_content)

    # Stream the reasoning
    stream = call_llm(client, model, messages, stream=True)

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def _assess_confidence(chunks: list[dict]) -> str:
    """Assess confidence level based on retrieved sources."""
    if not chunks:
        return "LOW"

    # Check average similarity score
    scores = [c.get("score", 0) for c in chunks]
    avg_score = sum(scores) / len(scores) if scores else 0

    # Check source diversity
    sources = set(c.get("source", "") for c in chunks)

    if avg_score > 0.8 and len(sources) >= 2:
        return "HIGH"
    elif avg_score > 0.6 or len(chunks) >= 3:
        return "MEDIUM"
    else:
        return "LOW"


def compare_provisions(
    topic: str,
    sources: list[str] | None = None,
    intent: QueryIntent | None = None,
) -> str:
    """
    Compare legal provisions from different sources on a topic.
    
    Useful for understanding how different codes address the same issue.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    # Retrieve with high top_k to get diverse sources
    chunks = retrieve(topic, intent=intent, top_k=15)

    # Filter by sources if specified
    if sources:
        chunks = [c for c in chunks if any(s.lower() in c.get("source", "").lower() for s in sources)]

    context = format_context_for_llm(chunks)

    comparison_prompt = """You are a legal comparativist. Given multiple legal provisions on a topic, create a structured comparison.

Output format:
## COMPARATIVE ANALYSIS: [Topic]

### Sources Examined
[List each source with brief description]

### Key Provisions
| Aspect | Source 1 | Source 2 | ... |
|--------|----------|----------|-----|
| ... | ... | ... | ... |

### Analysis
[Discuss similarities, differences, and how they interact]

### Conclusion
[Which provision takes precedence and why]

Always cite with [1], [2], etc. and use exact quotes."""

    user_content = f"""Topic: {topic}

Available Documents:
{context}

Please provide a comparative analysis."""

    model = settings.reasoning_model
    messages = build_messages_for_model(model, comparison_prompt, user_content)

    response = call_llm(client, model, messages)

    return response.choices[0].message.content

