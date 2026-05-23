from __future__ import annotations

import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from workspace import git_changed_files

_GLOBAL_SCORE_SCALE = 10_000

_TOKEN_RE = re.compile(
    r"""
    "(?:\\.|[^"\\])*"
    | '(?:\\.|[^'\\])*'
    | 0[xX][0-9a-fA-F]+
    | \d+(?:\.\d+)?
    | [A-Za-z_][A-Za-z0-9_]*
    | ==|!=|<=|>=|=>|->|::|\+\+|--|&&|\|\||<<|>>|\.\.\.|\.\.
    | \S
    """,
    re.VERBOSE,
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NUMBER_RE = re.compile(r"^(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?)$")

_KEYWORDS = frozenset(
    {
        "and", "as", "assert", "async", "await", "break", "case", "catch", "class",
        "const", "continue", "def", "default", "delete", "do", "elif", "else", "enum",
        "except", "export", "extends", "false", "finally", "fn", "for", "from", "func",
        "function", "if", "impl", "import", "in", "interface", "is", "let", "match",
        "module", "new", "nil", "none", "not", "null", "or", "package", "pass", "pub",
        "raise", "return", "self", "static", "struct", "switch", "this", "throw", "trait",
        "true", "try", "type", "var", "while", "with", "yield",
    }
)


@dataclass(slots=True)
class FileCompareResult:
    path: str
    changed_lines_a: int
    changed_lines_b: int
    matched_lines: int
    scored_positions: int
    similarity_ratio: float
    skipped_reason: str | None = None
    hunk_similarity: float = 0.0
    changed_hunks_a: int = 0
    changed_hunks_b: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "changed_lines_a": self.changed_lines_a,
            "changed_lines_b": self.changed_lines_b,
            "matched_lines": self.matched_lines,
            "scored_positions": self.scored_positions,
            "similarity_ratio": self.similarity_ratio,
            "skipped_reason": self.skipped_reason,
            "hunk_similarity": self.hunk_similarity,
            "changed_hunks_a": self.changed_hunks_a,
            "changed_hunks_b": self.changed_hunks_b,
        }


@dataclass(slots=True)
class CompareResult:
    matched_changed_lines: int
    scored_positions: int
    total_changed_lines_a: int
    total_changed_lines_b: int
    similarity_ratio: float
    per_file: list[FileCompareResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "matched_changed_lines": self.matched_changed_lines,
            "scored_positions": self.scored_positions,
            "total_changed_lines_a": self.total_changed_lines_a,
            "total_changed_lines_b": self.total_changed_lines_b,
            "similarity_ratio": self.similarity_ratio,
            "per_file": [item.to_dict() for item in self.per_file],
        }


@dataclass(frozen=True, slots=True)
class _EditHunk:
    old_start: int
    old_end: int
    new_start: int
    new_end: int
    deleted_lines: tuple[str, ...]
    added_lines: tuple[str, ...]

    @property
    def changed_line_count(self) -> int:
        return len(self.deleted_lines) + len(self.added_lines)

    @property
    def weight(self) -> int:
        return max(1, self.changed_line_count)

    @property
    def added_tokens(self) -> tuple[str, ...]:
        return tuple(_tokenize("\n".join(self.added_lines), shape=False))

    @property
    def added_shape_tokens(self) -> tuple[str, ...]:
        return tuple(_tokenize("\n".join(self.added_lines), shape=True))

    @property
    def deleted_tokens(self) -> tuple[str, ...]:
        return tuple(_tokenize("\n".join(self.deleted_lines), shape=False))

    @property
    def deleted_shape_tokens(self) -> tuple[str, ...]:
        return tuple(_tokenize("\n".join(self.deleted_lines), shape=True))


@dataclass(slots=True)
class _FileAnalysis:
    changed_sequence: list[str]
    hunks: list[_EditHunk]

    @property
    def changed_weight(self) -> int:
        return sum(hunk.weight for hunk in self.hunks)


def compare_solution_repos(*, original_dir: Path, repo_a_dir: Path, repo_b_dir: Path) -> CompareResult:
    """
    Compare two changed checkouts relative to the same original checkout.

    This intentionally does not run tests. It scores:
      - whether the same files/regions were changed,
      - whether the same original lines were removed/replaced,
      - whether inserted code has similar tokens and edit shape,
      - whether either side made many unrelated extra changes.

    For backward compatibility with the validator, matched_changed_lines is a
    fixed-scale score in [0, 10000], not a literal line count.
    """
    changed_files = sorted(
        _changed_files_against_original(original_dir=original_dir, repo_dir=repo_a_dir)
        | _changed_files_against_original(original_dir=original_dir, repo_dir=repo_b_dir)
    )

    per_file: list[FileCompareResult] = []
    total_changed_lines_a = 0
    total_changed_lines_b = 0
    weighted_similarity_sum = 0.0
    total_file_weight = 0

    for relative_path in changed_files:
        original_bytes = _read_file_bytes(original_dir / relative_path)
        file_a_bytes = _read_file_bytes(repo_a_dir / relative_path)
        file_b_bytes = _read_file_bytes(repo_b_dir / relative_path)

        if _is_binary_content(original_bytes) or _is_binary_content(file_a_bytes) or _is_binary_content(file_b_bytes):
            per_file.append(
                FileCompareResult(
                    path=relative_path,
                    changed_lines_a=0,
                    changed_lines_b=0,
                    matched_lines=0,
                    scored_positions=0,
                    similarity_ratio=0.0,
                    skipped_reason="binary_file",
                )
            )
            continue

        original_lines = _decode_lines(original_bytes)
        file_a_lines = _decode_lines(file_a_bytes)
        file_b_lines = _decode_lines(file_b_bytes)

        analysis_a = _analyze_file_change(original_lines, file_a_lines)
        analysis_b = _analyze_file_change(original_lines, file_b_lines)

        changed_lines_a = len(analysis_a.changed_sequence)
        changed_lines_b = len(analysis_b.changed_sequence)
        total_changed_lines_a += changed_lines_a
        total_changed_lines_b += changed_lines_b

        file_weight = max(analysis_a.changed_weight, analysis_b.changed_weight, 1)
        similarity_ratio = _file_similarity(analysis_a, analysis_b)
        matched_lines = int(round(similarity_ratio * file_weight))

        per_file.append(
            FileCompareResult(
                path=relative_path,
                changed_lines_a=changed_lines_a,
                changed_lines_b=changed_lines_b,
                matched_lines=matched_lines,
                scored_positions=file_weight,
                similarity_ratio=similarity_ratio,
                hunk_similarity=similarity_ratio,
                changed_hunks_a=len(analysis_a.hunks),
                changed_hunks_b=len(analysis_b.hunks),
            )
        )

        weighted_similarity_sum += similarity_ratio * file_weight
        total_file_weight += file_weight

    similarity_ratio = (weighted_similarity_sum / total_file_weight) if total_file_weight else 0.0

    return CompareResult(
        matched_changed_lines=int(round(similarity_ratio * _GLOBAL_SCORE_SCALE)) if total_file_weight else 0,
        scored_positions=_GLOBAL_SCORE_SCALE if total_file_weight else 0,
        total_changed_lines_a=total_changed_lines_a,
        total_changed_lines_b=total_changed_lines_b,
        similarity_ratio=similarity_ratio,
        per_file=per_file,
    )


def _changed_files_against_original(*, original_dir: Path, repo_dir: Path) -> set[str]:
    changed = set(git_changed_files(repo_dir))
    changed.update(_git_head_changed_files_against_original(original_dir=original_dir, repo_dir=repo_dir))
    return changed


def _git_head_changed_files_against_original(*, original_dir: Path, repo_dir: Path) -> set[str]:
    original_head = _git_rev_parse(original_dir, "HEAD")
    repo_head = _git_rev_parse(repo_dir, "HEAD")
    if not original_head or not repo_head or original_head == repo_head:
        return set()

    result = subprocess.run(
        ["git", "diff", "--name-only", "--relative", original_head, repo_head],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _git_rev_parse(repo_dir: Path, ref: str) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _read_file_bytes(path: Path) -> bytes | None:
    if not path.is_file():
        return None
    return path.read_bytes()


def _is_binary_content(raw_bytes: bytes | None) -> bool:
    return raw_bytes is not None and b"\0" in raw_bytes


def _decode_lines(raw_bytes: bytes | None) -> list[str]:
    if raw_bytes is None:
        return []
    return raw_bytes.decode("utf-8", errors="replace").splitlines()


def _analyze_file_change(original_lines: list[str], updated_lines: list[str]) -> _FileAnalysis:
    matcher = SequenceMatcher(a=original_lines, b=updated_lines, autojunk=False)
    changed_sequence: list[str] = []
    hunks: list[_EditHunk] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        deleted = tuple(original_lines[i1:i2]) if tag in {"replace", "delete"} else ()
        added = tuple(updated_lines[j1:j2]) if tag in {"replace", "insert"} else ()

        if deleted:
            changed_sequence.extend(f"-:{line}" for line in deleted)
        if added:
            changed_sequence.extend(f"+:{line}" for line in added)

        hunks.append(
            _EditHunk(
                old_start=i1,
                old_end=i2,
                new_start=j1,
                new_end=j2,
                deleted_lines=deleted,
                added_lines=added,
            )
        )

    return _FileAnalysis(changed_sequence=changed_sequence, hunks=hunks)


def _file_similarity(a: _FileAnalysis, b: _FileAnalysis) -> float:
    if not a.hunks and not b.hunks:
        return 0.0
    if not a.hunks or not b.hunks:
        return 0.0

    return _clamp01(
        0.5 * _directed_hunk_recall(a.hunks, b.hunks)
        + 0.5 * _directed_hunk_recall(b.hunks, a.hunks)
    )


def _directed_hunk_recall(source: list[_EditHunk], target: list[_EditHunk]) -> float:
    total_weight = sum(hunk.weight for hunk in source)
    if total_weight <= 0:
        return 0.0

    weighted = 0.0
    for source_hunk in source:
        best = 0.0
        for target_hunk in target:
            best = max(best, _hunk_similarity(source_hunk, target_hunk))
        weighted += best * source_hunk.weight

    return _clamp01(weighted / total_weight)


def _hunk_similarity(a: _EditHunk, b: _EditHunk) -> float:
    location = _span_similarity(a.old_start, a.old_end, b.old_start, b.old_end)

    deleted_line_f1 = _multiset_f1(
        _normalize_lines(a.deleted_lines),
        _normalize_lines(b.deleted_lines),
    )
    added_line_f1 = _multiset_f1(
        _normalize_lines(a.added_lines),
        _normalize_lines(b.added_lines),
    )

    added_token_f1 = _multiset_f1(a.added_tokens, b.added_tokens)
    added_shape_f1 = _multiset_f1(a.added_shape_tokens, b.added_shape_tokens)
    deleted_token_f1 = _multiset_f1(a.deleted_tokens, b.deleted_tokens)
    deleted_shape_f1 = _multiset_f1(a.deleted_shape_tokens, b.deleted_shape_tokens)

    operation_shape = _operation_shape_similarity(a, b)

    return _clamp01(
        0.22 * location
        + 0.17 * deleted_line_f1
        + 0.10 * deleted_token_f1
        + 0.05 * deleted_shape_f1
        + 0.08 * added_line_f1
        + 0.25 * added_token_f1
        + 0.08 * added_shape_f1
        + 0.05 * operation_shape
    )


def _span_similarity(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    a_len = max(0, a_end - a_start)
    b_len = max(0, b_end - b_start)

    if a_len == 0 and b_len == 0:
        return 1.0 / (1.0 + abs(a_start - b_start) / 3.0)

    if a_len > 0 and b_len > 0:
        overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
        union = max(a_end, b_end) - min(a_start, b_start)
        if union <= 0:
            return 0.0
        iou = overlap / union
        if iou > 0:
            return _clamp01(iou)

    a_mid = (a_start + a_end) / 2.0
    b_mid = (b_start + b_end) / 2.0
    scale = max(a_len, b_len, 1)
    return 1.0 / (1.0 + abs(a_mid - b_mid) / scale)


def _operation_shape_similarity(a: _EditHunk, b: _EditHunk) -> float:
    a_add = len(a.added_lines)
    a_del = len(a.deleted_lines)
    b_add = len(b.added_lines)
    b_del = len(b.deleted_lines)

    denom = max(a_add + a_del + b_add + b_del, 1)
    distance = abs(a_add - b_add) + abs(a_del - b_del)
    return _clamp01(1.0 - distance / denom)


def _normalize_lines(lines: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for line in lines:
        clean = " ".join(line.strip().split())
        if clean:
            normalized.append(clean)
    return tuple(normalized)


def _tokenize(text: str, *, shape: bool) -> list[str]:
    tokens = _TOKEN_RE.findall(text)
    if not shape:
        return tokens

    shaped: list[str] = []
    for token in tokens:
        lower = token.lower()
        if token.startswith(("'", '"')):
            shaped.append("STR")
        elif _NUMBER_RE.fullmatch(token):
            shaped.append("NUM")
        elif _IDENTIFIER_RE.fullmatch(token) and lower not in _KEYWORDS:
            shaped.append("ID")
        else:
            shaped.append(lower)
    return shaped


def _multiset_f1(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0

    left_counter = Counter(left)
    right_counter = Counter(right)
    overlap = sum((left_counter & right_counter).values())

    if overlap <= 0:
        return 0.0

    precision = overlap / sum(right_counter.values())
    recall = overlap / sum(left_counter.values())

    if precision + recall <= 0:
        return 0.0
    return _clamp01(2.0 * precision * recall / (precision + recall))


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
