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
@click.option("--share", is_flag=True, default=False, help="Share results to modeldelta gallery on HuggingFace")
def compare(model_a, model_b, output, top_k, top_n, token, quiet, share):
    """Compare two model checkpoints and report weight deltas.

    MODEL_A and MODEL_B can be HuggingFace model IDs or local paths.
    """
    from modeldelta.utils.model_loader import resolve_model, get_tensor_map
    from modeldelta.utils.model_meta import validate_pair
    from modeldelta.core.weight_diff import compare_models

    # Auto-detect non-TTY (notebooks, pipes) — suppress HF progress bars
    if quiet is None:
        quiet = not sys.stderr.isatty()

    click.echo(f"modeldelta: {model_a} → {model_b}", err=True)

    # Pre-flight validation (fast, no download)
    click.echo("Validating models...", err=True)
    meta_a, meta_b, warnings = validate_pair(model_a, model_b, token=token)
    if not meta_a.exists:
        click.echo(f"Error: model not found — {model_a}", err=True)
        if meta_a.error:
            click.echo(f"  {meta_a.error}", err=True)
        sys.exit(1)
    if not meta_b.exists:
        click.echo(f"Error: model not found — {model_b}", err=True)
        if meta_b.error:
            click.echo(f"  {meta_b.error}", err=True)
        sys.exit(1)
    for w in warnings:
        click.echo(f"Warning: {w}", err=True)

    click.echo(f"  A: {meta_a.one_liner()}", err=True)
    click.echo(f"  B: {meta_b.one_liner()}", err=True)

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
            f.write(generate_json(results, model_a, model_b, n_skipped,
                                  meta_a=meta_a, meta_b=meta_b))
        click.echo(f"JSON report saved to {output}", err=True)

    elif output.endswith(".html"):
        from modeldelta.report.html_report import generate_html
        html = generate_html(results, model_a, model_b, top_k=top_k,
                             meta_a=meta_a, meta_b=meta_b)
        with open(output, "w") as f:
            f.write(html)
        click.echo(f"HTML report saved to {output}", err=True)

    else:
        click.echo(f"Unknown output format: {output}. Use .json or .html", err=True)
        sys.exit(1)

    # Share to HF gallery
    if share:
        click.echo("Sharing to modeldelta gallery...", err=True)
        try:
            from modeldelta.database.hub import push_results
            pair_id = push_results(
                results, model_a, model_b, n_skipped,
                meta_a=meta_a, meta_b=meta_b,
                token=token, top_k=top_k,
            )
            click.echo(
                f"Shared! View at https://huggingface.co/datasets/NikolayYudin/modeldelta-results",
                err=True,
            )
        except Exception as e:
            click.echo(f"Warning: share failed — {e}", err=True)


def main():
    compare()


if __name__ == "__main__":
    main()
