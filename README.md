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
| page workers | OCR/classify/upload/publish one page or image | `min(16, cpu*2)` |
| file workers | download a Drive file, then fan its pages out | `3` |

Files are downloaded up front and their pages are queued immediately, so a
long PDF keeps every page worker busy while the next file is still
downloading. Backpressure is applied with `--max-inflight-files` so a huge
Drive tree cannot fill the disk or the queue.

### Options

```
-p, --page-workers N     pages processed concurrently (default: min(16, cpu*2))
-f, --file-workers N     files downloaded concurrently (default: 3)
    --max-inflight-files N   cap on files held at once (default: file_workers*2, min 4)
    --limit N            process at most N files (smoke tests)
    --keep-files         do not delete downloads after processing
    --dry-run            list work without downloading or uploading
    --sequential         control run: 1 page + 1 file worker
    --folder-ids PATH    folder list file (default: folder_ids.txt)
```

Equivalent environment variables: `OCR_PAGE_WORKERS`, `OCR_FILE_WORKERS`,
`OCR_FITZ_LOCK`, `OCR_HTTP_RETRIES`, `OCR_HTTP_RETRY_DELAY`,
`OCR_DOWNLOAD_RETRIES`, `OLLAMA_HOST`, `OLLAMA_TIMEOUT`.

```bash
python ocr-pipeline.py -p 16 -f 4
python ocr-pipeline.py --limit 5 --keep-files      # smoke test
python ocr-pipeline.py --sequential                # baseline for timing
```

Tune `-p` until the bottleneck moves: Ollama serialises inference, so a single
GPU box usually saturates somewhere between 4 and 16 page workers. Uploads and
downloads scale further. More threads than the slowest service can absorb just
adds queueing and 429s.

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
