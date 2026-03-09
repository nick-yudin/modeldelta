"""CLI entrypoint for modeldelta."""

import json
import sys
import time

import click


def _progress(current, total, elapsed, name):
    click.echo(f"  [{current}/{total}] {elapsed:.0f}s — {name}", err=True)


@click.command()
@click.argument("model_a")
@click.argument("model_b")
@click.option("--output", "-o", default=None, help="Output file path (.json, .html, or omit for text)")
@click.option("--top-k", default=20, help="Number of top singular values to track")
@click.option("--top-n", default=20, help="Number of modules to show in text output")
@click.option("--token", default=None, help="HuggingFace token (or set HF_TOKEN env var)")
@click.option("--quiet", "-q", is_flag=True, default=None, help="Suppress download progress bars (auto-enabled in notebooks)")
def compare(model_a, model_b, output, top_k, top_n, token, quiet):
    """Compare two model checkpoints and report weight deltas.

    MODEL_A and MODEL_B can be HuggingFace model IDs or local paths.
    """
    from modeldelta.utils.model_loader import resolve_model, get_tensor_map
    from modeldelta.core.weight_diff import compare_models

    # Auto-detect non-TTY (notebooks, pipes) — suppress HF progress bars
    if quiet is None:
        quiet = not sys.stderr.isatty()

    click.echo(f"modeldelta: {model_a} → {model_b}", err=True)

    # Resolve and load
    click.echo("Downloading/resolving models...", err=True)
    t0 = time.time()
    path_a = resolve_model(model_a, token=token, quiet=quiet)
    path_b = resolve_model(model_b, token=token, quiet=quiet)
    click.echo(f"Models ready in {time.time() - t0:.0f}s", err=True)

    tensor_map_a = get_tensor_map(path_a)
    tensor_map_b = get_tensor_map(path_b)

    # Compare
    click.echo("Analyzing weight deltas...", err=True)
    results, n_skipped = compare_models(
        tensor_map_a, tensor_map_b,
        top_k=top_k,
        progress_callback=_progress,
    )
    click.echo(f"Done: {len(results)} tensors, {n_skipped} skipped", err=True)

    # Output
    if output is None:
        from modeldelta.report.text_report import generate_text
        click.echo(generate_text(results, model_a, model_b, n_skipped, top_n=top_n))

    elif output.endswith(".json"):
        from modeldelta.report.json_report import generate_json
        with open(output, "w") as f:
            f.write(generate_json(results, model_a, model_b, n_skipped))
        click.echo(f"JSON report saved to {output}", err=True)

    elif output.endswith(".html"):
        from modeldelta.report.html_report import generate_html
        html = generate_html(results, model_a, model_b, top_k=top_k)
        with open(output, "w") as f:
            f.write(html)
        click.echo(f"HTML report saved to {output}", err=True)

    else:
        click.echo(f"Unknown output format: {output}. Use .json or .html", err=True)
        sys.exit(1)


def main():
    compare()


if __name__ == "__main__":
    main()
