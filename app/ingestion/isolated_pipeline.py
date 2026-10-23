"""Run the frozen M1-M9 PDF pipeline in a throw-away child process with a wall-clock limit and a memory cap.

Why: M1 hands untrusted bytes to MuPDF, a C library. In-process, a crafted PDF can spin the CPU, exhaust memory or
crash the interpreter and take the whole API down with it. Here the parent only waits on a pipe: on timeout, crash or
memory exhaustion the child is killed and the request fails with a typed error while the server keeps running.

INTERIM MEASURE. Remaining risks: the memory cap is a Windows Job Object / POSIX RLIMIT_AS set by the child itself on
a best-effort basis; there is no CPU-share, file-system or network sandbox; each upload pays a process start (the
PDF libraries are imported again); and parsing is still serialized by the API's single heavy-job worker. A container
or a dedicated worker pool with OS-level limits is the real fix.
"""

import multiprocessing
import sys

from .pdf_resume_ingestion_service import PdfProcessingError

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MEMORY_BYTES = 1024 * 1024 * 1024
KILL_GRACE_SECONDS = 5


class PdfProcessingTimeoutError(PdfProcessingError):
    code = "PDF_PROCESSING_TIMEOUT"


class PdfResourceError(PdfProcessingError):
    """The child ran out of memory or died unexpectedly."""
    code = "PDF_RESOURCE_LIMIT"


def limit_process_memory(limit_bytes: int) -> bool:
    """Cap the CURRENT process's memory. Returns False if the platform refused (the caller still has the timeout)."""
    try:
        if sys.platform == "win32":
            return limit_windows(limit_bytes)
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        return True
    except Exception:
        return False


def limit_windows(limit_bytes: int) -> bool:
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes",
                                                         "WriteBytes", "OtherBytes")]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("Basic", BasicLimits), ("Io", IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return False
    limits = ExtendedLimits()
    limits.Basic.LimitFlags = 0x100          # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    limits.ProcessMemoryLimit = limit_bytes
    if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        return False
    return bool(kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()))


def worker_main(connection, pdf_bytes, source_filename, memory_bytes):
    """Child entry point. Sends exactly one tuple; only primitives and the canonical resume cross the pipe."""
    try:
        limit_process_memory(memory_bytes)                     # caps everything the parse allocates from here on
        from .pdf_resume_ingestion_service import DeterministicResumePipeline
        connection.send(("ok", DeterministicResumePipeline().build(pdf_bytes, source_filename)))
    except PdfProcessingError as error:
        connection.send(("pdf_error", error.stage, str(error)))    # already sanitized by the pipeline
    except MemoryError:
        connection.send(("memory",))
    except BaseException as error:                             # never send exception text: it can hold resume content
        connection.send(("error", type(error).__name__))
    finally:
        connection.close()


class IsolatedResumePipeline:
    """Same `build(pdf_bytes, source_filename) -> CanonicalResume` contract as DeterministicResumePipeline."""

    def __init__(self, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS, memory_bytes: int = DEFAULT_MEMORY_BYTES,
                 target=worker_main):
        if timeout_seconds <= 0 or memory_bytes < 1:
            raise ValueError("timeout and memory limit must be positive")
        self.timeout, self.memory_bytes, self.target = float(timeout_seconds), int(memory_bytes), target

    def build(self, pdf_bytes: bytes, source_filename: str):
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=self.target, args=(sender, bytes(pdf_bytes), source_filename,
                                                            self.memory_bytes), daemon=True)
        message = None
        try:
            try:
                process.start()
            except Exception:
                raise PdfProcessingError("PDF worker could not be started", "M1") from None
            sender.close()
            if receiver.poll(self.timeout):
                try:
                    message = receiver.recv()
                except (EOFError, OSError):
                    message = ("died",)
            else:
                message = ("timeout",)
        finally:
            if process.is_alive():
                process.kill()                                 # a hung or runaway parser cannot outlive its deadline
            process.join(KILL_GRACE_SECONDS)
            receiver.close()
        return self.interpret(message, process.exitcode)

    @staticmethod
    def interpret(message, exitcode):
        kind = message[0] if message else "died"
        if kind == "ok":
            return message[1]
        if kind == "pdf_error":
            raise PdfProcessingError(message[2], message[1])
        if kind == "timeout":
            raise PdfProcessingTimeoutError("PDF processing exceeded its time limit", "M1")
        if kind == "memory":
            raise PdfResourceError("PDF processing exceeded its memory limit", "M1")
        if kind == "died":
            raise PdfResourceError("PDF worker stopped unexpectedly", "M1")
        raise PdfProcessingError(f"PDF processing failed: {message[1] if len(message) > 1 else 'error'}", "M1")
