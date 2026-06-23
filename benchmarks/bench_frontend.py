"""
Benchmark: Feature Projection View (frontend, Playwright).

Measures wall-clock time for:
  - fpv_page_load : navigation to /annotate → PCA plot fully rendered (cold, page reload per rep)
  - fpv_switch_*  : thumbnail click → switched method plot rendered (warm in-memory cache)

Projection computation is server-side and cached in DB; what this measures is:
  API fetch latency + JSON parse + Plotly scattergl WebGL render for 30,687 points.

Requirements (run on the host machine, NOT inside Docker):
    pip install playwright
    playwright install chromium

Usage:
    python -m benchmarks.bench_frontend \
        --url https://yapat.ni.dfki.de \
        --dataset-id 6 \
        --username testuser123 \
        --password testuser67890 \
        --repeats 3
"""

import argparse
import os
import statistics
import time

_BENCH_FIELDS = [
    "operation", "device", "dataset", "N", "repeats",
    "time_mean_s", "time_std_s", "throughput_per_s",
    "peak_mem_mb", "gpu_peak_mem_mb", "timestamp",
]

# AnuraSet full snippet count (fixed N — projection always returns all points)
N_POINTS = 30_687

# Methods to switch to after initial PCA load (PCA is the default)
_SWITCH_METHODS = ["umap", "tsne", "isomap"]
_METHOD_LABELS = {
    "pca":    "PCA",
    "umap":   "UMAP",
    "tsne":   "t‑SNE",   # t‑SNE (non-breaking hyphen, matches React render)
    "isomap": "Isomap",
}


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="https://yapat.ni.dfki.de",
                   help="Frontend base URL")
    p.add_argument("--dataset-id", type=int, default=6)
    p.add_argument("--dataset", default="anuraset",
                   help="Dataset label written to results.csv")
    p.add_argument("--username", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--no-headless", dest="headless", action="store_false", default=True,
                   help="Show browser window (useful for debugging)")
    p.add_argument("--auth-state", default="/tmp/yapat_auth_state.json",
                   help="Path to save/load Playwright auth storage state")
    p.add_argument("--background-wait", type=float, default=20.0,
                   help="Seconds to wait for all 4 methods to background-fetch after initial load")
    p.add_argument("--render-sizes", default="1000,5000,10000,20000,30687",
                   help="Comma-separated N values for render sweep benchmark")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Login helper
# ---------------------------------------------------------------------------

def _login(pw, args):
    """Open a fresh context, log in, persist storage state, close context."""
    ctx = pw.chromium.launch(headless=args.headless).new_context(
        viewport={"width": 1440, "height": 900}
    )
    page = ctx.new_page()
    page.goto(f"{args.url}/login", wait_until="networkidle")
    page.fill("input[name='username']", args.username)
    page.fill("input[name='password']", args.password)
    page.click("button[type='submit']")
    page.wait_for_url("**/dashboard", timeout=20_000)
    ctx.storage_state(path=args.auth_state)
    ctx.close()
    print(f"  Auth state saved to {args.auth_state}")


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _wait_for_plot(page, timeout_ms=90_000):
    """
    Block until the Plotly scattergl canvas is present and painted.

    Flow:
      1. Wait for the 'Loading feature projection…' spinner to disappear —
         this covers both the initial load and the method-loading state.
      2. Wait for the .js-plotly-plot canvas element to exist.
      3. Two rAF ticks so WebGL has time to flush the first draw.
    """
    page.wait_for_selector(
        "text=Loading feature projection",
        state="hidden",
        timeout=timeout_ms,
    )
    page.wait_for_selector(".js-plotly-plot canvas", timeout=timeout_ms)
    # Two animation frames to let WebGL finish the first paint
    page.evaluate(
        "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
    )


def _time_page_load(browser, auth_state, url, args):
    """
    Cold page load: fresh context (no JS cache) → measure navigation → plot ready.
    Returns elapsed seconds.
    """
    ctx = browser.new_context(
        storage_state=auth_state,
        viewport={"width": 1440, "height": 900},
    )
    page = ctx.new_page()
    t0 = time.perf_counter()
    page.goto(url, wait_until="domcontentloaded")
    _wait_for_plot(page)
    elapsed = time.perf_counter() - t0
    ctx.close()
    return elapsed


def _time_method_switch(page, method):
    """
    Warm method switch: click thumbnail → plot updated.
    Assumes all methods are already background-fetched (in JS memory cache).
    Returns elapsed seconds.
    """
    label = _METHOD_LABELS[method]
    t0 = time.perf_counter()
    # The sidebar thumbnails are <button> elements containing the method label text
    page.click(f"button:has-text('{label}')")
    # For cached methods the loading overlay does NOT appear — just wait for rAF
    page.evaluate(
        "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
    )
    return time.perf_counter() - t0


def _time_render_n(page, n: int) -> float:
    """
    Slice the already-loaded Plotly traces to N points and measure re-render time.
    Uses Plotly.react() then gl.readPixels() to force GPU synchronization before
    stopping the clock — ensures we capture actual WebGL draw time, not just JS
    scheduling time.
    Returns elapsed seconds (pure render, no fetch).
    """
    elapsed_ms = page.evaluate(
        """(n) => {
            const div = document.querySelector('.js-plotly-plot');
            if (!div || !div.data) return -1;

            // Slice every trace to n points
            const sliced = div.data.map(trace => {
                const t = Object.assign({}, trace);
                if (Array.isArray(t.x)) t.x = t.x.slice(0, n);
                if (Array.isArray(t.y)) t.y = t.y.slice(0, n);
                if (Array.isArray(t.customdata)) t.customdata = t.customdata.slice(0, n);
                if (Array.isArray(t.text)) t.text = t.text.slice(0, n);
                if (t.marker) {
                    t.marker = Object.assign({}, t.marker);
                    if (Array.isArray(t.marker.color)) t.marker.color = t.marker.color.slice(0, n);
                    if (Array.isArray(t.marker.size))  t.marker.size  = t.marker.size.slice(0, n);
                    if (t.marker.line) {
                        t.marker.line = Object.assign({}, t.marker.line);
                        if (Array.isArray(t.marker.line.color)) t.marker.line.color = t.marker.line.color.slice(0, n);
                        if (Array.isArray(t.marker.line.width)) t.marker.line.width = t.marker.line.width.slice(0, n);
                    }
                }
                return t;
            });

            return new Promise(resolve => {
                const layout = Object.assign({}, div.layout);
                const config = {displayModeBar: false, responsive: true};

                // newPlot() tears down and recreates the WebGL context from scratch
                // with only N points — true cold render, not a diff/mask over 30K.
                const t0 = performance.now();
                Plotly.newPlot(div, sliced, layout, config).then(() => {
                    // gl.readPixels() is a synchronous GPU fence — blocks until
                    // all pending draw calls are flushed to the framebuffer.
                    const canvas = div.querySelector('canvas.gl-canvas');
                    if (canvas) {
                        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
                        if (gl) {
                            const px = new Uint8Array(4);
                            gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
                        }
                    }
                    resolve(performance.now() - t0);
                });
            });
        }""",
        n,
    )
    return elapsed_ms / 1000.0  # convert ms → seconds


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _write_row(operation, times, args):
    _write_row_n(operation, times, N_POINTS, args)


def _write_row_n(operation, times, n, args):
    from benchmarks.stage_timer import write_csv_row

    mean_t = statistics.mean(times)
    std_t = statistics.stdev(times) if len(times) > 1 else 0.0
    throughput = round(n / mean_t, 1) if mean_t > 0 else None

    write_csv_row(
        {
            "operation": operation,
            "device": "cpu",
            "dataset": args.dataset,
            "N": n,
            "repeats": args.repeats,
            "time_mean_s": round(mean_t, 4),
            "time_std_s": round(std_t, 4),
            "throughput_per_s": throughput,
            "peak_mem_mb": 0,
            "gpu_peak_mem_mb": None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        extra_fields=_BENCH_FIELDS,
    )
    print(f"  → {operation} N={n:,}: mean={mean_t:.3f}s  std={std_t:.4f}s  ({throughput:,.0f} pts/s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    from playwright.sync_api import sync_playwright

    annotate_url = f"{args.url}/annotate?mode=al&dataset_id={args.dataset_id}"
    render_sizes = [int(s) for s in args.render_sizes.split(",")]

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=args.headless)

        # --- Step 1: Login once, save session ---
        print("\n[1/4] Logging in...")
        _login(pw, args)

        # --- Step 2: Cold page load (PCA is default) ---
        print(f"\n[2/4] Benchmarking fpv_page_load ({args.repeats} reps, cold context each)...")
        load_times = []
        for rep in range(args.repeats):
            t = _time_page_load(browser, args.auth_state, annotate_url, args)
            load_times.append(t)
            print(f"  rep {rep + 1}/{args.repeats}: {t:.3f}s")
        _write_row("fpv_page_load", load_times, args)

        # --- Step 3: Method switch (warm cache) ---
        print(f"\n[3/4] Benchmarking fpv_switch (warm cache, {args.repeats} reps each)...")

        ctx = browser.new_context(
            storage_state=args.auth_state,
            viewport={"width": 1440, "height": 900},
        )
        page = ctx.new_page()
        page.goto(annotate_url, wait_until="domcontentloaded")
        _wait_for_plot(page)
        print(f"  Initial PCA loaded. Waiting {args.background_wait:.0f}s for background fetches...")
        time.sleep(args.background_wait)

        for method in _SWITCH_METHODS:
            times = []
            for rep in range(args.repeats):
                page.click(f"button:has-text('PCA')")
                page.evaluate(
                    "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
                )
                time.sleep(0.3)
                t = _time_method_switch(page, method)
                times.append(t)
                print(f"  fpv_switch_{method} rep {rep + 1}/{args.repeats}: {t:.3f}s")
            _write_row(f"fpv_switch_{method}", times, args)

        # --- Step 4: Render sweep (slice traces to N, measure WebGL redraw) ---
        print(f"\n[4/4] Benchmarking fpv_render sweep {render_sizes} ({args.repeats} reps each)...")
        for n in render_sizes:
            times = []
            for rep in range(args.repeats):
                t = _time_render_n(page, n)
                times.append(t)
                print(f"  fpv_render N={n:>6,} rep {rep + 1}/{args.repeats}: {t:.3f}s")
            _write_row_n("fpv_render", times, n, args)

        ctx.close()
        browser.close()

    print("\nDone. Results appended to benchmarks/results.csv")


if __name__ == "__main__":
    main()
