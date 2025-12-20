"""Command-line interface for Legal RAG."""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import LEGAL_CATEGORIES, get_settings
from .embeddings import get_embeddings
from .ingest.chunker import chunk_document
from .ingest.pdf_extractor import extract_pdf, extract_pdfs_from_directory
from .pinecone_store import (
    NAMESPACE_LAW_CODES,
    NAMESPACE_USER_CONTRACTS,
    ensure_index_exists,
    get_index_stats,
    upsert_chunks,
)
from .rag import ask, chat_stream
from .reasoning import (
    LegalReasoning,
    ReasoningDepth,
    compare_provisions,
    reason_deep,
    reason_multistep,
    reason_stream,
    reason_with_decomposition,
)
from .retriever import QueryIntent

app = typer.Typer(
    name="mouhamia",
    help="Mouham'IA - Your AI Legal Assistant (محامي + IA)",
    add_completion=False,
)
console = Console()


@app.command()
def ingest_laws(
    path: Path = typer.Argument(..., help="Path to PDF file or directory of PDFs"),
    chunk_size: int = typer.Option(512, help="Chunk size in tokens"),
    chunk_overlap: int = typer.Option(50, help="Chunk overlap in tokens"),
):
    """Ingest law code documents into the law_codes namespace."""
    _ingest(path, NAMESPACE_LAW_CODES, chunk_size, chunk_overlap)


@app.command()
def ingest_contracts(
    path: Path = typer.Argument(..., help="Path to PDF file or directory of PDFs"),
    chunk_size: int = typer.Option(512, help="Chunk size in tokens"),
    chunk_overlap: int = typer.Option(50, help="Chunk overlap in tokens"),
):
    """Ingest contract documents into the user_contracts namespace."""
    _ingest(path, NAMESPACE_USER_CONTRACTS, chunk_size, chunk_overlap)


def _ingest(path: Path, namespace: str, chunk_size: int, chunk_overlap: int):
    """Common ingestion logic."""
    if not path.exists():
        console.print(f"[red]Error: Path does not exist: {path}[/red]")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Ensure index exists
        progress.add_task("Ensuring Pinecone index exists...", total=None)
        ensure_index_exists()

        # Extract PDFs
        progress.add_task("Extracting PDFs...", total=None)
        if path.is_file():
            documents = [extract_pdf(path)]
        else:
            documents = extract_pdfs_from_directory(path)

        if not documents:
            console.print("[yellow]No PDF documents found.[/yellow]")
            raise typer.Exit(0)

        console.print(f"[green]Found {len(documents)} document(s)[/green]")

        # Chunk documents
        all_chunks = []
        for doc in documents:
            progress.add_task(f"Chunking {doc.source}...", total=None)
            chunks = chunk_document(doc, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
            console.print(f"  {doc.source}: {len(chunks)} chunks")

        console.print(f"[green]Total chunks: {len(all_chunks)}[/green]")

        # Generate embeddings
        progress.add_task("Generating embeddings...", total=None)
        texts = [chunk.text for chunk in all_chunks]
        embeddings = get_embeddings(texts)

        # Upsert to Pinecone
        progress.add_task("Uploading to Pinecone...", total=None)
        count = upsert_chunks(all_chunks, embeddings, namespace)

    console.print(
        Panel(
            f"[green]Successfully ingested {count} chunks into '{namespace}' namespace[/green]",
            title="Ingestion Complete",
        )
    )


@app.command(name="ask")
def ask_question(
    question: str = typer.Argument(..., help="Your question"),
    source: str = typer.Option(
        "auto",
        help="Source to search: 'law', 'contracts', 'both', or 'auto'",
    ),
    category: str = typer.Option(
        None,
        "--category",
        "-c",
        help=f"Filter by legal category: {', '.join(LEGAL_CATEGORIES.keys())}",
    ),
    top_k: int = typer.Option(5, help="Number of chunks to retrieve"),
    deep: bool = typer.Option(False, "--deep", "-d", help="Enable deep reasoning mode"),
):
    """Ask a question and get an answer with citations."""
    # Parse intent
    intent = None
    if source == "law":
        intent = QueryIntent.LAW_CODES
    elif source == "contracts":
        intent = QueryIntent.CONTRACTS
    elif source == "both":
        intent = QueryIntent.BOTH
    # else auto-detect

    # Validate category
    if category and category not in LEGAL_CATEGORIES:
        console.print(f"[red]Invalid category: {category}[/red]")
        console.print(f"Valid categories: {', '.join(LEGAL_CATEGORIES.keys())}")
        raise typer.Exit(1)

    if category:
        console.print(f"[dim]Filtering by category: {category} ({LEGAL_CATEGORIES[category]})[/dim]\n")

    if deep:
        # Use deep reasoning mode
        console.print("\n[bold magenta]🧠 Deep Reasoning Mode[/bold magenta]\n")
        try:
            for chunk in reason_stream(question, ReasoningDepth.DEEP, intent, top_k):
                console.print(chunk, end="")
            console.print()
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
    else:
        with console.status("[bold green]Thinking..."):
            response = ask(question, intent=intent, top_k=top_k, category=category)

        console.print()
        console.print(Panel(response.answer, title="Answer", border_style="green"))


@app.command()
def reason(
    question: str = typer.Argument(..., help="Your legal question"),
    source: str = typer.Option(
        "auto",
        help="Source to search: 'law', 'contracts', 'both', or 'auto'",
    ),
    category: str = typer.Option(
        None,
        "--category",
        "-c",
        help=f"Filter by legal category: {', '.join(LEGAL_CATEGORIES.keys())}",
    ),
    depth: str = typer.Option(
        "deep",
        help="Reasoning depth: 'quick', 'standard', or 'deep'",
    ),
    decompose: bool = typer.Option(
        False,
        "--decompose",
        "-x",
        help="Decompose query into sub-questions for comprehensive analysis",
    ),
    multistep: bool = typer.Option(
        False,
        "--multistep",
        "-m",
        help="Use multi-step iterative retrieval with gap analysis",
    ),
    hyde: bool = typer.Option(
        False,
        "--hyde",
        help="Use HyDE (Hypothetical Document Embeddings) - can hallucinate, use with caution",
    ),
    iterations: int = typer.Option(
        3,
        help="Maximum retrieval iterations for multi-step mode",
    ),
    top_k: int = typer.Option(5, help="Number of chunks to retrieve"),
):
    """Deep legal reasoning with structured analysis."""
    # Parse intent
    intent = None
    if source == "law":
        intent = QueryIntent.LAW_CODES
    elif source == "contracts":
        intent = QueryIntent.CONTRACTS
    elif source == "both":
        intent = QueryIntent.BOTH

    # Validate category
    if category and category not in LEGAL_CATEGORIES:
        console.print(f"[red]Invalid category: {category}[/red]")
        console.print(f"Valid categories: {', '.join(LEGAL_CATEGORIES.keys())}")
        raise typer.Exit(1)

    # Parse depth
    reasoning_depth = ReasoningDepth.DEEP
    if depth == "quick":
        reasoning_depth = ReasoningDepth.QUICK
    elif depth == "standard":
        reasoning_depth = ReasoningDepth.STANDARD

    # Determine mode
    if multistep:
        mode_desc = f"Multi-Step Retrieval (max {iterations} iterations)"
        if hyde:
            mode_desc += " + HyDE (⚠️ may hallucinate)"
    elif decompose:
        mode_desc = "Query Decomposition"
    else:
        mode_desc = f"Depth: {reasoning_depth.value.upper()}"

    if category:
        mode_desc += f"\nCategory: {category} ({LEGAL_CATEGORIES[category]})"

    console.print()
    console.print(
        Panel(
            f"[bold magenta]🧠 Legal Reasoning Mode[/bold magenta]\n\n"
            f"Mode: {mode_desc}",
            title="Mouham'IA Reasoning",
            border_style="magenta",
        )
    )
    console.print()

    try:
        if multistep:
            # Multi-step iterative retrieval
            with console.status("[bold magenta]Multi-step retrieval and analysis..."):
                result = reason_multistep(
                    question,
                    intent=intent,
                    top_k=top_k,
                    max_iterations=iterations,
                    use_hyde=hyde,
                )

            # Show reasoning/retrieval steps
            console.print("[bold cyan]Retrieval & Reasoning Steps:[/bold cyan]")
            for i, step in enumerate(result.reasoning_steps, 1):
                icon = {
                    "hyde": "🔮",
                    "initial_retrieval": "📚",
                    "gap_analysis": "🔍",
                    "follow_up_retrieval": "🔄",
                    "synthesis": "⚖️",
                }.get(step.step_type, "•")
                console.print(f"  {icon} {step.content}")
            console.print()

            console.print(
                Panel(
                    result.final_answer,
                    title=f"Legal Analysis (Confidence: {result.confidence})",
                    border_style="green",
                )
            )

        elif decompose:
            with console.status("[bold magenta]Decomposing query and analyzing..."):
                result = reason_with_decomposition(question, intent=intent, top_k=top_k)

            # Show reasoning steps
            console.print("[bold cyan]Reasoning Steps:[/bold cyan]")
            for i, step in enumerate(result.reasoning_steps, 1):
                console.print(f"  {i}. {step.step_type}: {step.content}")
            console.print()

            console.print(
                Panel(
                    result.final_answer,
                    title=f"Legal Analysis (Confidence: {result.confidence})",
                    border_style="green",
                )
            )
        else:
            # Streaming mode
            for chunk in reason_stream(question, reasoning_depth, intent, top_k):
                console.print(chunk, end="")
            console.print()

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def compare(
    topic: str = typer.Argument(..., help="Legal topic to compare"),
    source: str = typer.Option(
        "auto",
        help="Source to search: 'law', 'contracts', 'both', or 'auto'",
    ),
):
    """Compare legal provisions from different sources on a topic."""
    intent = None
    if source == "law":
        intent = QueryIntent.LAW_CODES
    elif source == "contracts":
        intent = QueryIntent.CONTRACTS
    elif source == "both":
        intent = QueryIntent.BOTH

    console.print()
    console.print(
        Panel(
            "[bold cyan]⚖️ Comparative Legal Analysis[/bold cyan]",
            title="Mouham'IA Compare",
            border_style="cyan",
        )
    )
    console.print()

    with console.status("[bold cyan]Comparing provisions..."):
        try:
            result = compare_provisions(topic, intent=intent)
            console.print(result)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def chat(
    reasoning: bool = typer.Option(
        False,
        "--reasoning",
        "-r",
        help="Enable deep reasoning mode for all responses",
    ),
    multistep: bool = typer.Option(
        False,
        "--multistep",
        "-m",
        help="Use multi-step iterative retrieval in reasoning mode",
    ),
):
    """Start an interactive chat session."""
    if multistep:
        mode_text = "[bold magenta]Multi-Step Reasoning Mode[/bold magenta]"
        border = "magenta"
    elif reasoning:
        mode_text = "[bold magenta]Reasoning Mode[/bold magenta]"
        border = "magenta"
    else:
        mode_text = "[bold blue]Standard Mode[/bold blue]"
        border = "blue"

    console.print(
        Panel(
            f"[bold]Mouham'IA Chat[/bold] محامي\n\n"
            f"Mode: {mode_text}\n\n"
            "Ask questions about your legal documents.\n"
            "Commands:\n"
            "  • 'quit' or 'exit' - End session\n"
            "  • 'stats' - Show index statistics\n"
            "  • 'mode' - Cycle modes: Standard → Reasoning → Multi-Step\n"
            "  • '/deep <question>' - One-off deep reasoning\n"
            "  • '/multi <question>' - One-off multi-step reasoning",
            title="Welcome",
            border_style=border,
        )
    )

    # Mode: 0 = standard, 1 = reasoning, 2 = multistep
    if multistep:
        current_mode = 2
    elif reasoning:
        current_mode = 1
    else:
        current_mode = 0

    mode_names = ["Standard", "Reasoning", "Multi-Step Reasoning"]
    mode_colors = ["cyan", "magenta", "magenta"]

    while True:
        try:
            prompt_color = mode_colors[current_mode]
            question = console.input(f"\n[bold {prompt_color}]You:[/bold {prompt_color}] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        if question.lower() == "stats":
            try:
                stats = get_index_stats()
                console.print(Panel(str(stats), title="Index Statistics"))
            except Exception as e:
                console.print(f"[red]Error getting stats: {e}[/red]")
            continue

        if question.lower() == "mode":
            current_mode = (current_mode + 1) % 3
            console.print(f"[yellow]Switched to {mode_names[current_mode]} mode[/yellow]")
            continue

        if question.lower() == "help":
            console.print(
                Panel(
                    "Commands:\n"
                    "  • 'mode' - Cycle: Standard → Reasoning → Multi-Step\n"
                    "  • '/deep <q>' - One-off deep reasoning\n"
                    "  • '/multi <q>' - One-off multi-step reasoning\n"
                    "  • 'stats' - Index statistics\n"
                    "  • 'quit' - Exit",
                    title="Help",
                    border_style="cyan",
                )
            )
            continue

        # Check for command prefixes
        use_mode = current_mode
        if question.lower().startswith("/deep "):
            use_mode = 1
            question = question[6:].strip()
        elif question.lower().startswith("/multi "):
            use_mode = 2
            question = question[7:].strip()

        try:
            if use_mode == 2:
                # Multi-step reasoning
                console.print("\n[bold magenta]🧠 Multi-Step Reasoning...[/bold magenta]\n")
                with console.status("[bold magenta]Iterative retrieval and analysis..."):
                    result = reason_multistep(question)

                # Show retrieval steps
                console.print("[bold cyan]Retrieval Steps:[/bold cyan]")
                for step in result.reasoning_steps:
                    icon = {
                        "hyde": "🔮",
                        "initial_retrieval": "📚",
                        "gap_analysis": "🔍",
                        "follow_up_retrieval": "🔄",
                        "synthesis": "⚖️",
                    }.get(step.step_type, "•")
                    console.print(f"  {icon} {step.content}")
                console.print()

                console.print(
                    Panel(
                        result.final_answer,
                        title=f"Analysis (Confidence: {result.confidence})",
                        border_style="green",
                    )
                )

            elif use_mode == 1:
                # Deep reasoning (streaming)
                console.print("\n[bold magenta]🧠 Reasoning...[/bold magenta]\n")
                for chunk in reason_stream(question, ReasoningDepth.DEEP):
                    console.print(chunk, end="")
                console.print()

            else:
                # Standard mode
                console.print("\n[bold green]Assistant:[/bold green] ", end="")
                for chunk in chat_stream(question):
                    console.print(chunk, end="")
                console.print()

        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


@app.command()
def categories():
    """List all available legal categories."""
    console.print()
    console.print(
        Panel(
            "[bold]Available Legal Categories[/bold]\n\n"
            "Use with --category / -c flag in ask, reason, or compare commands.",
            title="Categories",
            border_style="cyan",
        )
    )
    console.print()

    from rich.table import Table

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Code", style="green")
    table.add_column("Description")

    for code, description in LEGAL_CATEGORIES.items():
        table.add_row(code, description)

    console.print(table)
    console.print()
    console.print("[dim]Example: mouhamia ask 'question' --category numerique[/dim]")


@app.command()
def stats():
    """Show Pinecone index statistics."""
    try:
        index_stats = get_index_stats()
        console.print(Panel(str(index_stats), title="Pinecone Index Statistics"))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def init():
    """Initialize Pinecone index (run once before first use)."""
    with console.status("[bold green]Creating Pinecone index..."):
        try:
            ensure_index_exists()
            console.print("[green]Pinecone index ready![/green]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()

