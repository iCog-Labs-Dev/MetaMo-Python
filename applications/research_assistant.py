import os
import sys

if __package__ in (None, ""):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from core.engine import MetaMoEngine, AssistantResponse, format_response
from applications.papers import PaperContextService, PaperIngestionService


def _chat_loop(engine: MetaMoEngine, paper_context: str | None):
    """Run the REPL chat loop, optionally augmenting user input with paper context."""
    print("\nSystem Ready. Subsystems: [Curiosity] & [Ethics]. Type 'quit' to exit.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ('quit', 'exit'):
                break

            print("\n[MetaMo Internal Processing...]")

            if paper_context:
                response = engine.process_with_context(user_input, paper_context)
            else:
                response = engine.process(user_input)

            print(format_response(response))

        except Exception as e:
            print(f"\nAn error occurred: {e}")


def interactive_loop():
    """Start the chat interface, auto-loading the latest stored paper as context."""
    print("Initializing MetaMo Multi-Subsystem Chat Interface...")
    engine = MetaMoEngine()
    context_svc = PaperContextService()

    paper_context: str | None = None
    if context_svc.has_papers():
        latest = context_svc.get_latest_paper()
        print(f"  Found stored paper: {latest.title}")
        paper_context = PaperIngestionService().build_context(latest)
        print(f"  Loaded {len(paper_context)} chars of context.")

    _chat_loop(engine, paper_context)


def load_and_chat():
    """Ingest a paper file from the CLI argument and start a context-aware chat."""
    path = sys.argv[1]
    print(f"Loading paper from {path}...")

    ingest_svc = PaperIngestionService()
    paper = ingest_svc.ingest(path)
    print(f"Ingested: {paper.title} ({paper.total_chars} chars, {len(paper.chunks)} chunks)")

    print("Initializing MetaMo Multi-Subsystem Chat Interface...")
    engine = MetaMoEngine()
    paper_context = ingest_svc.build_context(
        PaperContextService().get_latest_paper()
    )

    print("\n" + "=" * 60)
    print(f"Paper '{paper.title}' ingested. You can now ask questions.")
    print("=" * 60)

    _chat_loop(engine, paper_context)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        load_and_chat()
    else:
        interactive_loop()
