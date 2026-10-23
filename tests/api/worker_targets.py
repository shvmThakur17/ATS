"""Child-process entry points used by the process-isolation tests. Kept import-light: they run in a spawned process."""

import os
import time


def sleep_forever(connection, pdf_bytes, source_filename, memory_bytes):
    time.sleep(600)


def die_abruptly(connection, pdf_bytes, source_filename, memory_bytes):
    os._exit(3)


def exhaust_memory(connection, pdf_bytes, source_filename, memory_bytes):
    from app.ingestion.isolated_pipeline import limit_process_memory
    applied = limit_process_memory(memory_bytes)
    try:
        block = bytearray(memory_bytes * 2)               # twice the cap: must fail while the cap is in force
        connection.send(("ok", f"allocated {len(block)} (cap applied: {applied})"))
    except MemoryError:
        connection.send(("memory",))
    finally:
        connection.close()


def raise_with_secret(connection, pdf_bytes, source_filename, memory_bytes):
    try:
        raise RuntimeError("SECRET-TOKEN-123 person@example.com")
    except BaseException as error:
        connection.send(("error", type(error).__name__))
    finally:
        connection.close()
