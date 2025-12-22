"""Command-line interface for Legal RAG."""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import get_settings
from .embeddings import get_embeddings
from .ingest.chunker import chunk_document
from .ingest.pdf_extractor import extract_pdf, extract_pdfs_from_directory
from .pinecone_store import (
    NAMESPACE_LAW_CODES,
    NAMESPACE_USER_CONTRACTS,
    clear_all_namespaces,
    clear_namespace,
    delete_index,
    ensure_index_exists,
    get_all_categories,
    get_index_stats,
    upsert_chunks,
)
from .rag import analyze, analyze_stream, format_for_cli, QueryIntent

app = typer.Typer(
    name="mouhamia",
    help="Mouham'IA - Your AI Legal Assistant (محامي + IA)",
    add_completion=False,
)
console = Console()


# ============================================================================
# Ingestion Commands
# ============================================================================

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
        progress.add_task("Ensuring Pinecone index exists...", total=None)
        ensure_index_exists()

        progress.add_task("Extracting PDFs...", total=None)
        if path.is_file():
            documents = [extract_pdf(path)]
        else:
            documents = extract_pdfs_from_directory(path)

        if not documents:
            console.print("[yellow]No PDF documents found.[/yellow]")
            raise typer.Exit(0)

        console.print(f"[green]Found {len(documents)} document(s)[/green]")

        all_chunks = []
        for doc in documents:
            progress.add_task(f"Chunking {doc.source}...", total=None)
            chunks = chunk_document(doc, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
            console.print(f"  {doc.source}: {len(chunks)} chunks")

        console.print(f"[green]Total chunks: {len(all_chunks)}[/green]")

        progress.add_task("Generating embeddings...", total=None)
        texts = [chunk.text for chunk in all_chunks]
        embeddings = get_embeddings(texts)

        progress.add_task("Uploading to Pinecone...", total=None)
        count = upsert_chunks(all_chunks, embeddings, namespace)

    console.print(
        Panel(
            f"[green]Successfully ingested {count} chunks into '{namespace}' namespace[/green]",
            title="Ingestion Complete",
        )
    )


# ============================================================================
# Query Commands
# ============================================================================

def _parse_intent(source: str) -> QueryIntent | None:
    """Parse source string to QueryIntent."""
    if source == "law":
        return QueryIntent.LAW_CODES
    elif source == "contracts":
        return QueryIntent.CONTRACTS
    elif source == "both":
        return QueryIntent.BOTH
    return None  # auto-detect


def _validate_category(category: str | None) -> bool:
    """Validate category if provided. Returns True if valid or None."""
    if not category:
        return True
    
    try:
        available = get_all_categories()
        if category not in available:
            console.print(f"[red]Invalid category: {category}[/red]")
            if available:
                console.print(f"Available categories: {', '.join(sorted(available.keys()))}")
            else:
                console.print("[yellow]No categories found. Ingest documents first.[/yellow]")
            return False
        return True
    except Exception:
        # If we can't check, allow it
        return True


@app.command(name="ask")
def ask_question(
    question: str = typer.Argument(..., help="Your legal question"),
    source: str = typer.Option("auto", help="Source: 'law', 'contracts', 'both', or 'auto'"),
    category: str = typer.Option(None, "-c", "--category", help="Filter by legal category"),
    top_k: int = typer.Option(5, help="Number of sources to retrieve"),
    stream: bool = typer.Option(False, "-s", "--stream", help="Stream the response"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON for API use"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show multi-step thinking process"),
):
    """Ask a legal question and get a lawyer-style analysis."""
    intent = _parse_intent(source)
    if not _validate_category(category):
        raise typer.Exit(1)

    if category:
        console.print(f"[dim]Category filter: {category}[/dim]\n")

    console.print()
    console.print(
        Panel(
            "[bold magenta]⚖️ Analyse Juridique[/bold magenta]",
            title="Mouham'IA",
            border_style="magenta",
        )
    )
    console.print()

    try:
        if stream:
            # Streaming mode (already shows progress)
            for chunk in analyze_stream(question, intent=intent, top_k=top_k):
                console.print(chunk, end="")
            console.print()
        else:
            # Structured mode
            if verbose:
                # Verbose mode - show thinking process
                console.print("[bold cyan]🧠 Processus de réflexion:[/bold cyan]\n")
                
                def print_progress(msg: str):
                    console.print(f"  [dim]{msg}[/dim]")
                
                response = analyze(
                    question, 
                    intent=intent, 
                    top_k=top_k, 
                    category=category,
                    verbose_callback=print_progress,
                )
                console.print()
            else:
                # Silent mode - just spinner
                with console.status("[bold magenta]Analyse en cours..."):
                    response = analyze(question, intent=intent, top_k=top_k, category=category)

            if json_output:
                # Raw JSON output
                import json
                console.print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
            else:
                # Formatted CLI output
                if response.succes and response.analyse_juridique:
                    formatted = format_for_cli(response)
                    console.print(
                        Panel(
                            formatted,
                            title=f"Analyse (Confiance: {response.analyse_juridique.meta.niveau_confiance})",
                            border_style="green",
                        )
                    )
                else:
                    console.print(f"[red]❌ {response.message_erreur}[/red]")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def chat():
    """Start an interactive chat session."""
    console.print(
        Panel(
            "[bold]Mouham'IA Chat[/bold] محامي\n\n"
            "Posez vos questions juridiques.\n\n"
            "Commandes:\n"
            "  • 'quit' ou 'exit' - Quitter\n"
            "  • 'stats' - Statistiques de l'index\n"
            "  • '/json <question>' - Réponse en JSON",
            title="Bienvenue",
            border_style="magenta",
        )
    )

    while True:
        try:
            question = console.input("\n[bold magenta]Vous:[/bold magenta] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Au revoir![/yellow]")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit"):
            console.print("[yellow]Au revoir![/yellow]")
            break

        if question.lower() == "stats":
            try:
                stats = get_index_stats()
                console.print(Panel(str(stats), title="Index Statistics"))
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            continue

        # Check for JSON prefix
        json_mode = False
        if question.lower().startswith("/json "):
            json_mode = True
            question = question[6:].strip()

        try:
            console.print("\n[bold green]Assistant:[/bold green]\n")
            
            if json_mode:
                with console.status("[bold magenta]Analyse..."):
                    response = analyze(question)
                import json
                console.print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
            else:
                # Stream the response
                for chunk in analyze_stream(question):
                    console.print(chunk, end="")
                console.print()

        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


# ============================================================================
# Utility Commands
# ============================================================================

@app.command()
def categories():
    """List all available legal categories (from ingested documents)."""
    from rich.table import Table

    console.print()
    console.print(
        Panel(
            "[bold]Catégories Juridiques Disponibles[/bold]\n\n"
            "Catégories détectées automatiquement depuis les documents ingérés.\n"
            "Utilisez avec --category / -c",
            title="Categories",
            border_style="cyan",
        )
    )
    console.print()

    try:
        with console.status("[bold cyan]Récupération des catégories..."):
            available = get_all_categories()
        
        if not available:
            console.print("[yellow]Aucune catégorie trouvée. Ingérez des documents d'abord.[/yellow]")
            return
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Catégorie", style="green")
        table.add_column("Documents", justify="right")
        
        # Sort by count descending
        sorted_cats = sorted(available.items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_cats:
            table.add_row(category, str(count))
        
        console.print(table)
        console.print()
        console.print(f"[dim]Total: {len(available)} catégories[/dim]")
        console.print("[dim]Example: mouhamia ask 'question' --category civil[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


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


@app.command()
def reset(
    target: str = typer.Option("all", help="What to reset: 'all', 'laws', 'contracts', or 'index'"),
    force: bool = typer.Option(False, "-f", "--force", help="Skip confirmation"),
):
    """Reset/clear the vector database."""
    if target == "index":
        action = "DELETE the entire Pinecone index"
        warning = "[bold red]This will permanently delete the index![/bold red]"
    elif target == "all":
        action = "clear ALL vectors"
        warning = "[yellow]This will delete all data.[/yellow]"
    elif target == "laws":
        action = "clear all LAW CODES"
        warning = "[yellow]This will delete all law documents.[/yellow]"
    elif target == "contracts":
        action = "clear all CONTRACTS"
        warning = "[yellow]This will delete all contract documents.[/yellow]"
    else:
        console.print(f"[red]Invalid target: {target}[/red]")
        raise typer.Exit(1)

    console.print()
    console.print(Panel(f"[bold]Action:[/bold] {action}\n\n{warning}", title="⚠️ Confirmation", border_style="red"))

    if not force:
        confirm = typer.confirm("\nContinue?")
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    try:
        if target == "index":
            with console.status("[bold red]Deleting index..."):
                deleted = delete_index()
            console.print("[green]✓ Index deleted.[/green]" if deleted else "[yellow]Index did not exist.[/yellow]")

        elif target == "all":
            with console.status("[bold yellow]Clearing..."):
                deleted = clear_all_namespaces()
            console.print("[green]✓ Cleared:[/green]")
            for ns, count in deleted.items():
                console.print(f"  - {ns}: {count} vectors")

        elif target == "laws":
            with console.status("[bold yellow]Clearing laws..."):
                count = clear_namespace(NAMESPACE_LAW_CODES)
            console.print(f"[green]✓ Cleared {count} vectors from law_codes.[/green]")

        elif target == "contracts":
            with console.status("[bold yellow]Clearing contracts..."):
                count = clear_namespace(NAMESPACE_USER_CONTRACTS)
            console.print(f"[green]✓ Cleared {count} vectors from contracts.[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
