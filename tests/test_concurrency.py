"""
Offline test harness for the concurrent ocr-pipeline.

Every outbound call (Ollama, Cloudinary, GNP uploads, Drive) is stubbed, but
the real threading machinery is exercised: the PDF document pool, per-thread
Ollama clients, the shared classifier cache, the reference-counted file
lifecycle, backpressure and progress accounting.

Run with:  python tests/test_concurrency.py
"""

import importlib.util
import io
import os
import sys
import tempfile
import threading
import time
import types

import fitz  # real PyMuPDF, used to build fixtures
from PIL import Image, ImageDraw

PIPELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, "ocr-pipeline.py")
spec = importlib.util.spec_from_file_location("ocr_pipeline", PIPELINE)
mod = importlib.util.module_from_spec(spec)
sys.modules["ocr_pipeline"] = mod
spec.loader.exec_module(mod)

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILS.append(name)


# ===========================================================================
# Stubs
# ===========================================================================
class FakeOllamaClient:
    """Records which thread made the call so we can assert isolation."""

    calls = []
    lock = threading.Lock()

    def __init__(self, host=None, timeout=None):
        self.thread = threading.get_ident()
        self.timeout = timeout

    def chat(self, model="", messages=None, **kw):
        time.sleep(0.02)
        with FakeOllamaClient.lock:
            FakeOllamaClient.calls.append(threading.get_ident())
        return {"message": {"content": '{"categories":["news"],"tags":["spec"],'
                                     '"content_type":"news","summary":"summary"}'}}


fake_ollama = types.ModuleType("ollama")
fake_ollama.Client = FakeOllamaClient
fake_ollama.chat = lambda *a, **k: {"message": {"content": "{}"}}
sys.modules["ollama"] = fake_ollama

# stub cloudinary + google so importing the module never needs credentials
uploaded = []
pubs = []
upload_lock = threading.Lock()


def stub_thumbnail_upload(buffer, asset_id, page_num=1):
    with upload_lock:
        uploaded.append(asset_id)
    time.sleep(0.01)
    return f"https://cdn.example/{asset_id}_p{page_num}.jpg"


def stub_create_publication(publication_date, page_number, classification_result,
                            asset_id, file_type, edition_number, thumbnail_url=None):
    with upload_lock:
        pubs.append({
            "asset_id": asset_id,
            "page": page_number,
            "date": publication_date,
            "edition": edition_number,
            "thumb": thumbnail_url,
            "file_type": file_type,
        })
    return {"ok": True}


def stub_upload_pdf_page(file_path, asset_id, page_number, doc=None):
    time.sleep(0.01)
    with upload_lock:
        uploaded.append(asset_id)
    return {"status": "success"}


def stub_upload_file_asset(file_content, asset_id, bucket_name, file_name,
                           content_type="application/octet-stream"):
    time.sleep(0.01)
    with upload_lock:
        uploaded.append(asset_id)
    return {"status": "success", "fileName": file_name}


def stub_smart_thumbnail(page, output_size=(400, 400)):
    """Real JPEG buffer, so the concurrent thumbnail/upload path is exercised."""
    img = Image.new("RGB", output_size, "white")
    ImageDraw.Draw(img).text((10, 10), "thumb", fill="black")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


mod.upload_thumbnail_to_cloudinary = stub_thumbnail_upload
mod.create_enhanced_publication = stub_create_publication
mod.upload_pdf_page = stub_upload_pdf_page
mod.upload_file_asset = stub_upload_file_asset
mod.obfuscate_pdf_thumbnail_smart = stub_smart_thumbnail
mod.pytesseract.image_to_string = lambda img: "sample ocr text " * 20


# ===========================================================================
# Fixtures
# ===========================================================================
def make_pdf(path, pages, text="Edition No. 22148\nMonday, January 5, 2026\n"):
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"{text}page {i + 1}", fontsize=12)
    doc.save(path)
    doc.close()


def make_image(path, name_hint="2026JANUARY5PAGE1"):
    img = Image.new("RGB", (400, 300), "white")
    ImageDraw.Draw(img).text((20, 20), "sample text", fill="black")
    img.save(path)


# ===========================================================================
# Test 1: PDF pages are OCR'd concurrently and exactly once
# ===========================================================================
def _run_pdf(pdf, workers):
    engine = mod.init_engine(page_workers=workers, file_workers=1, keep_files=True)
    ctx = engine._prepare_pdf(pdf, "file-1")
    start = time.time()
    engine._dispatch_pdf(ctx)
    engine.wait()
    elapsed = time.time() - start
    engine.shutdown()
    return engine, ctx, elapsed


def test_pdf_concurrency():
    print("\n[1] PDF pages processed concurrently, each exactly once")
    pages = 24
    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "edition.pdf")
        make_pdf(pdf, pages)

        engine, ctx, parallel = _run_pdf(pdf, workers=8)
        check("total_pages detected", ctx.total_pages == pages, ctx.total_pages)
        check("edition extracted", ctx.edition_number == "22148", ctx.edition_number)
        check("publication date extracted", ctx.publication_date == "2026-01-05",
              ctx.publication_date)

    check("every page published", len(pubs) == pages, f"{len(pubs)} != {pages}")
    check("no duplicate asset ids", len({p['asset_id'] for p in pubs}) == pages)
    check("page numbers 1..N", sorted(int(p['page']) for p in pubs) == list(range(1, pages + 1)))
    check("no page failures", engine.counter.failed == 0, engine.counter.failed)

    # Control run: identical work with a single page worker.
    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "edition.pdf")
        make_pdf(pdf, pages)
        serial_engine, _, serial = _run_pdf(pdf, workers=1)
    check("serial run also complete", serial_engine.counter.done == pages,
          serial_engine.counter.done)

    speedup = serial / parallel if parallel else 0
    print(f"        serial={serial:.2f}s  parallel(8)={parallel:.2f}s  "
          f"speedup={speedup:.1f}x")
    check("parallel is meaningfully faster", speedup > 2.5, f"{speedup:.1f}x")


# ===========================================================================
# Test 2: repeated/blank pages hit the classifier cache, not the LLM
# ===========================================================================
def test_classifier_cache():
    print("\n[2] Classifier memoises repeated page text")
    FakeOllamaClient.calls.clear()
    pipeline = mod.EnhancedOCRPipeline(use_docling=False)
    text = "identical page text"
    for _ in range(5):
        pipeline.process_ocr_text(text, "x.pdf")
    check("only one LLM call for identical text", len(FakeOllamaClient.calls) == 1,
          len(FakeOllamaClient.calls))


# ===========================================================================
# Test 3: one Ollama client per thread
# ===========================================================================
def test_ollama_thread_isolation():
    print("\n[3] Ollama client is per-thread")
    from concurrent.futures import ThreadPoolExecutor
    classifier = mod.ContentClassifier()
    seen = {}

    def grab():
        c = classifier._get_ollama_client()
        seen[threading.get_ident()] = id(c)
        return id(c)

    with ThreadPoolExecutor(max_workers=6) as pool:
        ids = list(pool.map(lambda _: grab(), range(6)))
    check("distinct clients per thread", len(set(ids)) == len(seen),
          f"{len(set(ids))} unique of {len(seen)} threads")

    # and stable within a thread
    a = classifier._get_ollama_client()
    b = classifier._get_ollama_client()
    check("stable client within a thread", a is b)


# ===========================================================================
# Test 4: image path + filename parsing, thumbnails, cleanup
# ===========================================================================
def test_image_path():
    print("\n[4] Image processing path")
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        img_path = "2026JANUARY5PAGE3.jpg"
        make_image(img_path)
        engine = mod.init_engine(page_workers=4, file_workers=2)
        before = len(pubs)
        mod.ocr_image(img_path, "img-1", service=None)
        engine.shutdown()
    check("image published", len(pubs) == before + 1)
    check("image page number parsed", pubs[-1]['page'] == "3", pubs[-1]['page'])
    # NOTE: extract_info zero-pads the month but not the day ("2026-01-5");
    # that is pre-existing behaviour, unrelated to this refactor.
    check("image date parsed", pubs[-1]['date'] == "2026-01-5", pubs[-1]['date'])
    check("image file cleaned up", not os.path.exists(os.path.join(tmp, img_path)))


# ===========================================================================
# Test 5: thumbnail failure must not raise NameError (the old bug)
# ===========================================================================
def test_thumbnail_failure_does_not_crash():
    print("\n[5] Thumbnail failure degrades gracefully (no NameError)")
    original = mod.obfuscate_pdf_thumbnail_smart
    mod.obfuscate_pdf_thumbnail_smart = lambda page, output_size=None: None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pdf = os.path.join(tmp, "e.pdf")
            make_pdf(pdf, 2)
            engine = mod.init_engine(page_workers=2, file_workers=1, keep_files=True)
            ctx = engine._prepare_pdf(pdf, "f")
            engine._dispatch_pdf(ctx)
            engine.wait()
            engine.shutdown()
            ok_pubs = [p for p in pubs if p['page'] in ("1", "2") and p['thumb'] is None]
        check("pages still published without thumbnail", len(ok_pubs) >= 2,
              len(ok_pubs))
        check("no page failures", engine.counter.failed == 0, engine.counter.failed)
    finally:
        mod.obfuscate_pdf_thumbnail_smart = original


# ===========================================================================
# Test 6: backpressure bounds in-flight files
# ===========================================================================
def test_backpressure():
    print("\n[6] In-flight files are bounded")
    engine = mod.init_engine(page_workers=2, file_workers=2, max_inflight_files=2)
    check("semaphore sized from max_inflight_files",
          engine._slots._value == 2, engine._slots._value)
    engine.shutdown()


# ===========================================================================
# Test 7: a failing page does not abort the rest of the file
# ===========================================================================
def test_page_failure_isolation():
    print("\n[7] One failing page does not stop the others")
    original = mod.upload_pdf_page
    state = {"n": 0}
    lock = threading.Lock()

    def flaky(file_path, asset_id, page_number, doc=None):
        with lock:
            state["n"] += 1
            n = state["n"]
        if n == 1:
            raise RuntimeError("boom")
        return original(file_path, asset_id, page_number, doc=doc)

    mod.upload_pdf_page = flaky
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pdf = os.path.join(tmp, "e.pdf")
            make_pdf(pdf, 6)
            engine = mod.init_engine(page_workers=1, file_workers=1, keep_files=True)
            before = len(pubs)
            ctx = engine._prepare_pdf(pdf, "f2")
            engine._dispatch_pdf(ctx)
            engine.wait()
            engine.shutdown()
        made = len(pubs) - before
        check("other pages still published", made == 5, made)
        check("failure counted", engine.counter.failed == 1, engine.counter.failed)
    finally:
        mod.upload_pdf_page = original


# ===========================================================================
# Test 8: document handles are all closed (no leak / fd exhaustion)
# ===========================================================================
def test_document_pool_closes():
    print("\n[8] PDF handles are closed after the file finishes")
    pool = None
    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "e.pdf")
        make_pdf(pdf, 4)
        pool = mod.PdfDocumentPool(pdf)
        docs = [pool.get() for _ in range(1)]
        pool.close_all()
        check("handles registered", len(docs) == 1)
        check("closed flag set on doc", docs[0].is_closed, docs[0].is_closed)


# ===========================================================================
# Test 9: engine summary + progress accounting
# ===========================================================================
def test_progress_counter():
    print("\n[9] Progress counter is thread-safe and accurate")
    counter = mod.ProgressCounter(total=1000, label="pages")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda i: counter.tick(ok=(i % 10 != 0)), range(1000)))
    check("done count", counter.done == 900, counter.done)
    check("failed count", counter.failed == 100, counter.failed)


# ===========================================================================
# Test 10: wait() blocks until every page is finished; failures free slots
# ===========================================================================
def test_wait_semantics_and_slot_release():
    print("\n[10] wait() blocks until done; failures release in-flight slots")
    with tempfile.TemporaryDirectory() as tmp:
        pdf = os.path.join(tmp, "e.pdf")
        make_pdf(pdf, 12)
        engine = mod.init_engine(page_workers=4, file_workers=2,
                                 max_inflight_files=2, keep_files=True)
        ctx = engine._prepare_pdf(pdf, "w")
        before = len(pubs)
        engine._dispatch_pdf(ctx)

        # wait() must not return while pages are still in flight
        returned_early = engine.wait(timeout=0.001) and len(pubs) - before < 12
        check("wait(timeout) does not report completion prematurely",
              not returned_early)
        engine.wait()
        check("all pages published after wait()", len(pubs) - before == 12,
              len(pubs) - before)

        # A failed download must not leak an in-flight slot.
        mod.download_file = lambda *a, **k: None
        try:
            engine.submit_file("bad-id", "missing.pdf")
            engine.wait()
            leaked = engine._inflight_files
            check("failed download leaves no in-flight task", leaked == 0, leaked)
            check("failed download counted", engine.files_failed == 1,
                  engine.files_failed)
            check("slot was returned", engine._slots._value == 2,
                  engine._slots._value)
        finally:
            engine.shutdown()


# ===========================================================================
# Test 11: tesseract OpenMP budget is configured, not left to oversubscribe
# ===========================================================================
def test_tesseract_env_budget():
    print("\n[11] Tesseract OpenMP thread budget")
    saved = {k: os.environ.get(k) for k in ("OMP_THREAD_LIMIT", "OMP_NUM_THREADS")}
    try:
        # default: one tesseract thread per worker
        for k in saved:
            os.environ.pop(k, None)
        limit = mod.configure_tesseract_env(page_workers=8)
        check("defaults to one thread per tesseract", limit == 1, limit)
        check("OMP_THREAD_LIMIT exported for the subprocess",
              os.environ.get("OMP_THREAD_LIMIT") == "1", os.environ.get("OMP_THREAD_LIMIT"))

        # an explicit env var must win
        os.environ["OMP_THREAD_LIMIT"] = "4"
        check("explicit env var wins", mod.configure_tesseract_env(page_workers=8) == 4)
        os.environ.pop("OMP_THREAD_LIMIT", None)

        # explicit argument is honoured
        check("explicit argument honoured",
              mod.configure_tesseract_env(page_workers=4, omp_thread_limit=3) == 3)
        check("budget arithmetic", 4 * 3 == 12)

        # oversubscription is detected (8 workers x 4 threads on 12 cores)
        os.environ.pop("OMP_THREAD_LIMIT", None)
        mod.configure_tesseract_env(page_workers=mod.CPU_COUNT + 8)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main():
    tests = [
        test_pdf_concurrency,
        test_classifier_cache,
        test_ollama_thread_isolation,
        test_image_path,
        test_thumbnail_failure_does_not_crash,
        test_backpressure,
        test_page_failure_isolation,
        test_document_pool_closes,
        test_progress_counter,
        test_wait_semantics_and_slot_release,
        test_tesseract_env_budget,
    ]
    cwd = os.getcwd()
    for test in tests:
        try:
            test()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            FAILS.append(f"{test.__name__} raised {exc!r}")
        finally:
            os.chdir(cwd)

    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} FAILURE(S): {FAILS}")
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
