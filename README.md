# Archive OCR Pipeline

Downloads newspaper scans from Google Drive, OCRs each page, classifies the
text with a local LLM, uploads obfuscated thumbnails plus per-page master
documents, and registers a publication record for every page.

## Quick start

```bash
pip install -r requirements.txt
export JWT/credentials as before   # service_account.json + folder_ids.txt
python ocr-pipeline.py             # concurrent by default
```

## Concurrency

Processing is dominated by I/O, not CPU: each page performs an Ollama
classification, a Cloudinary thumbnail upload, a master-document upload and a
publication POST, one after another. Those four round-trips are independent of
every other page, so the pipeline runs them on a thread pool.

Two pools are used:

| Pool | What it does | Default size |
|------|--------------|--------------|
| page workers | OCR/classify/upload/publish one page or image | CPU count (max 16) |
| file workers | download a Drive file, then fan its pages out | `3` |

Files are downloaded up front and their pages are queued immediately, so a
long PDF keeps every page worker busy while the next file is still
downloading. Backpressure is applied with `--max-inflight-files` so a huge
Drive tree cannot fill the disk or the queue.

### Options

```
-p, --page-workers N     pages processed concurrently (default: CPU count, max 16)
-f, --file-workers N     files downloaded concurrently (default: 3)
    --cpus N             shorthand: N page workers, 1 tesseract thread each
    --omp-thread-limit N threads per tesseract subprocess (default: 1)
    --max-inflight-files N   cap on files held at once (default: file_workers*2, min 4)
    --limit N            process at most N files (smoke tests)
    --keep-files         do not delete downloads after processing
    --dry-run            list work without downloading or uploading
    --sequential         control run: 1 page + 1 file worker
    --folder-ids PATH    folder list file (default: folder_ids.txt)
```

Equivalent environment variables: `OCR_PAGE_WORKERS`, `OCR_FILE_WORKERS`,
`OMP_THREAD_LIMIT`, `OCR_FITZ_LOCK`, `OCR_HTTP_RETRIES`,
`OCR_HTTP_RETRY_DELAY`, `OCR_DOWNLOAD_RETRIES`, `OLLAMA_HOST`,
`OLLAMA_TIMEOUT`.

```bash
python ocr-pipeline.py -p 16 -f 4
python ocr-pipeline.py --limit 5 --keep-files      # smoke test
python ocr-pipeline.py --sequential                # baseline for timing
```

Tune `-p` until the bottleneck moves: Ollama serialises inference, so a single
GPU box usually saturates somewhere between 4 and 16 page workers. Uploads and
downloads scale further. More threads than the slowest service can absorb just
adds queueing and 429s.

## Apple Silicon (M-series) and tesseract's OpenMP

This is the single most important setting on a Mac, and it is easy to get
wrong.

`tesseract` runs as a **subprocess** for every OCR call, and the Homebrew/
macOS build is compiled with OpenMP: by default each subprocess starts as many
threads as there are cores. Run 8 page workers on an M2 Max and you do **not**
get 8 threads — you get 8 × 12 = 96 threads competing for 12 cores, and
throughput can end up *worse* than the serial version.

The pipeline therefore exports `OMP_THREAD_LIMIT=1` by default (pytesseract
passes `os.environ` to the subprocess, so this reaches tesseract) and logs the
budget it is using:

```
OMP thread budget: 8 page workers x 1 tesseract thread(s) = 8 of 12 CPU(s)
```

The rule is **`page_workers × OMP_THREAD_LIMIT ≈ cores you want to use`**. If
the product exceeds the CPU count the pipeline logs a warning telling you how
to fix it.

An M2 Max has 12 cores (8 performance + 4 efficiency), and `os.cpu_count()`
reports all of them. To give the pipeline 8 CPUs:

```bash
python ocr-pipeline.py --cpus 8
# equivalent to:
OMP_THREAD_LIMIT=1 python ocr-pipeline.py -p 8
```

`--cpus 8` is the setting you asked for: eight single-threaded OCR workers.
Because the 8 performance cores are the ones that matter for OCR, this keeps
all the heavy work on the P-cores and leaves the 4 E-cores free for macOS,
Ollama and the network callbacks.

### Flag precedence

Only one flag should decide the worker count, so the rules are explicit:

```
--sequential   beats  --cpus   beats  --page-workers   beats  default
```

- `--sequential` always forces **1 page + 1 file worker**, even alongside
  `--cpus`. That makes it a trustworthy baseline to time against.
- `--cpus N` and `--page-workers N` set the same knob; `--cpus` wins if you
  pass both.
- `--omp-thread-limit N` sets the other half of the budget (threads per
  tesseract). An `OMP_THREAD_LIMIT` env var beats it, so shell-level tuning
  stays authoritative.
- `-f/--file-workers` is independent: it only controls downloads, which are
  network-bound and use almost no CPU.

Every run logs exactly what it resolved, so there is no guessing:

```
Concurrency: 8 page worker(s) x 1 tesseract thread(s) = 8 of 12 CPU(s) [--cpus]; 3 file worker(s)
```

If the budget oversubscribes your CPU count (or wastes half of it) the
pipeline says so at startup rather than silently running slowly.

### Recipes

```bash
# Daily run on an M2 Max: 8 OCR workers, 3 downloaders
python ocr-pipeline.py --cpus 8

# Same thing, spelled explicitly
OMP_THREAD_LIMIT=1 python ocr-pipeline.py -p 8 -f 3

# Ollama also on this machine and sharing cores: leave it more room
python ocr-pipeline.py --cpus 6

# Fewer, fatter tesseract processes for the same 12-thread budget
python ocr-pipeline.py -p 6 --omp-thread-limit 2

# Baseline to compare against (1 page, 1 file)
python ocr-pipeline.py --sequential

# Smoke-test 5 files without deleting what it downloaded
python ocr-pipeline.py --cpus 8 --limit 5 --keep-files
```

Do **not** stack `--cpus` with `-p`: `--cpus` silently wins, which is exactly
the kind of ambiguity this precedence list exists to remove.

### What to change if it is still slow

Diagnose before adding threads — more workers past the bottleneck only adds
queueing and 429s.

| Symptom | Likely bottleneck | Action |
|---|---|---|
| All cores busy, workers idle at ~100% CPU | OCR (tesseract/raster) | Already saturated; raise `--cpus` only if you have spare cores |
| Workers waiting on Ollama, low CPU | LLM classification | `OLLAMA_NUM_PARALLEL` on the Ollama side; keep 1 model resident |
| `429`/`5xx` in the log | Upload/publication endpoint | Lower `-p`; retries are already built in |
| Slow to start each file, few pages per file | Download/dispatch | Raise `-f` (cheap: network-bound) |
| Memory pressure / swap | Too many in-flight pages | Lower `--max-inflight-files` |

The classifier already memoises identical page text, so repeated or blank
pages skip the LLM round-trip entirely — expect a bigger gain than the thread
count alone on archive scans that repeat content.

Two other practical notes for M-series:

- **Leave headroom for Ollama.** If Ollama runs on the same machine it needs
  CPU/GPU time too. `--cpus 8` of 12 is a sensible split; if you see the LLM
  call become the bottleneck, drop to `--cpus 6`.
- **Raise Ollama's own concurrency.** Ollama serves only a few requests in
  parallel by default, so a high `--page-workers` just queues requests inside
  Ollama. Set it on the Ollama side (environment of the `ollama serve`
  process, not this script):
  ```bash
  OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
  ```
  Keep `OLLAMA_MAX_LOADED_MODELS=1` so the model stays resident; loading it
  once per request would dominate everything else.
- **`--omp-thread-limit 2` with `-p 6`** is a reasonable alternative: fewer,
  fatter tesseract processes (less per-process startup overhead, better cache
  locality) for the same 12-thread budget. Measure both — on Apple Silicon the
  difference is usually small but directionally in favour of one thread per
  worker.

Thread guidance elsewhere still applies: the file pool is network-bound and
needs very little CPU, so leave `-f` at 3-4 rather than scaling it with cores.

## Thread-safety

The refactor isolates every piece of shared state that is not thread-safe:

- **Google Drive** — one `httplib2` transport per thread (`googleapiclient`'s
  HTTP layer is not thread-safe). Credentials are shared and only refreshed
  when they genuinely expire.
- **PyMuPDF** — documents are never shared across threads; each worker opens
  its own handle (`PdfDocumentPool`), and every MuPDF call is wrapped in a
  short library-wide lock. Measured cost: none (<1% of runtime, since network
  I/O stays fully parallel). Set `OCR_FITZ_LOCK=0` to disable.
- **Ollama** — one client per thread.
- **Classification cache** — locked memo table; identical/repeated pages skip
  the LLM round-trip entirely.
- **File lifecycle** — a reference-counted handle per file guarantees the
  temporary download is deleted and all PDF handles are closed exactly once,
  after the *last* page finishes (not when the queue empties).

Failed pages are isolated: one bad page no longer aborts the rest of the file,
and retries with jittered backoff absorb transient 429/5xx from the
publication endpoint.

## Tests

`test_concurrency.py` stubs every outbound call and exercises the real
threading machinery: page fan-out, progress accounting, backpressure, the
document pool, per-thread Ollama clients, cache behaviour, failure isolation
and file cleanup. It also measures a serial-vs-parallel control run.

```bash
python tests/test_concurrency.py
```

## Behaviour changes

Concurrency makes "process page N" independent of page N-1, which required
these adjustments:

- A failing page is logged and skipped; the remaining pages still run (the old
  `process_pdf` aborted the whole file via one outer `try`).
- Edition number and publication date are read from page 1 *before* fan-out,
  rather than being discovered mid-loop.
- `thumbnail_url` is now initialised before use. Previously a failed thumbnail
  raised `NameError` in both `ocr_image` and `process_pdf`.
- Source files are deleted only after the last page is published, and the
  image master upload now keeps ownership of the file handle so cleanup stays
  deterministic.
