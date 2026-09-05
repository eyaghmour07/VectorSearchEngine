from __future__ import annotations

from pathlib import Path

from codesearch.parser import parse_repo


def _write(tmp_path: Path, rel: str, source: str) -> Path:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_class_methods_get_qualified_names(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "sessions.py",
        '''
class Session:
    def request(self, method, url):
        prepared = self.prepare_request(method, url)
        return self.send(prepared)
        return prepared
''',
    )
    chunks = parse_repo(tmp_path)
    ids = {c.id for c in chunks}
    assert "sessions.py::Session.request" in ids
    chunk = next(c for c in chunks if c.qualname == "Session.request")
    assert chunk.signature.startswith("def request(")
    assert "self" in chunk.signature
    assert "method" in chunk.signature


def test_nested_functions_get_qualified_names(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "nested.py",
        '''
def outer(x):
    def inner(y):
        value = x + y
        doubled = value * 2
        return doubled
    return inner(x)
    return x
''',
    )
    chunks = parse_repo(tmp_path)
    qualnames = {c.qualname for c in chunks}
    assert "outer" in qualnames
    assert "outer.inner" in qualnames


def test_async_functions_are_extracted(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "async_mod.py",
        '''
async def fetch(url):
    response = await get(url)
    body = await response.text()
    return body
''',
    )
    chunks = parse_repo(tmp_path)
    assert len(chunks) == 1
    assert chunks[0].qualname == "fetch"
    assert chunks[0].signature.startswith("async def fetch(")


def test_decorators_do_not_break_parsing(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "decorated.py",
        '''
def deco(fn):
    return fn

@deco
def decorated(value):
    """Keep this function."""
    first = value + 1
    second = first + 1
    return second
''',
    )
    chunks = parse_repo(tmp_path)
    by_name = {c.qualname: c for c in chunks}
    assert "decorated" in by_name
    assert by_name["decorated"].start_line >= 1
    assert by_name["decorated"].docstring == "Keep this function."


def test_syntax_error_is_skipped(tmp_path: Path, caplog) -> None:
    _write(
        tmp_path,
        "good.py",
        '''
def ok(value):
    a = value + 1
    b = a + 1
    return b
''',
    )
    _write(tmp_path, "bad.py", "def broken(\n")
    with caplog.at_level("WARNING", logger="codesearch"):
        chunks = parse_repo(tmp_path)
    assert len(chunks) == 1
    assert chunks[0].qualname == "ok"
    assert any("SyntaxError" in rec.message for rec in caplog.records)


def test_skips_tests_setup_venv_and_short_bodies(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "pkg/real.py",
        '''
def long_enough(x):
    a = x + 1
    b = a + 1
    return b

def tiny():
    return 1
''',
    )
    _write(
        tmp_path,
        "pkg/test_real.py",
        '''
def test_something():
    assert True
    assert True
    assert True
''',
    )
    _write(
        tmp_path,
        "pkg/real_test.py",
        '''
def helper():
    a = 1
    b = 2
    return a + b
''',
    )
    _write(
        tmp_path,
        "setup.py",
        '''
def setup_helper():
    a = 1
    b = 2
    return a + b
''',
    )
    _write(
        tmp_path,
        ".venv/lib/fake.py",
        '''
def venv_fn():
    a = 1
    b = 2
    return a + b
''',
    )
    chunks = parse_repo(tmp_path)
    assert [c.id for c in chunks] == ["pkg/real.py::long_enough"]


def test_nested_class_method_keeps_outer_scope(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "nested_class.py",
        '''
class Outer:
    class Inner:
        def method(self, value):
            first = value + 1
            second = first + 1
            return second
''',
    )
    chunks = parse_repo(tmp_path)
    assert {c.qualname for c in chunks} == {"Outer.Inner.method"}
    assert chunks[0].id == "nested_class.py::Outer.Inner.method"


def test_property_setter_and_overload_are_disambiguated(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "props.py",
        '''
from typing import overload

class Box:
    @property
    def name(self):
        stored = self._name
        cleaned = stored.strip()
        return cleaned

    @name.setter
    def name(self, value):
        cleaned = value.strip()
        self._name = cleaned
        self._dirty = True

@overload
def parse(value: int) -> int:
    first = value + 1
    second = first + 1
    return second

def parse(value):
    text = str(value)
    cleaned = text.strip()
    return cleaned
''',
    )
    chunks = parse_repo(tmp_path)
    qualnames = {c.qualname for c in chunks}
    assert "Box.name" in qualnames
    assert "Box.name@setter" in qualnames
    assert "parse@overload" in qualnames
    assert "parse" in qualnames


def test_duplicate_ids_get_line_suffix(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "dupes.py",
        '''
def twin(value):
    first = value + 1
    second = first + 1
    return second

def twin(value):
    other = value * 2
    extra = other + 1
    return extra
''',
    )
    chunks = parse_repo(tmp_path)
    ids = {c.id for c in chunks}
    assert any(chunk_id.endswith("#L2") for chunk_id in ids)
    assert any("#L" in chunk_id for chunk_id in ids)
    assert len(ids) == 2


def test_encoding_declaration_is_honored(tmp_path: Path) -> None:
    source = (
        "# -*- coding: latin-1 -*-\n"
        "def decode_label(value):\n"
        "    label = 'caf\xe9'\n"
        "    extra = label + value\n"
        "    return extra\n"
    )
    path = tmp_path / "latin.py"
    path.write_bytes(source.encode("latin-1"))
    chunks = parse_repo(tmp_path)
    assert len(chunks) == 1
    assert "café" in chunks[0].body


def test_return_annotation_is_in_signature(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "typed.py",
        '''
def add(a: int, b: int = 0) -> int:
    total = a + b
    extra = total + 1
    return extra
''',
    )
    chunks = parse_repo(tmp_path)
    assert chunks[0].signature == "def add(a: int, b: int=0) -> int"
