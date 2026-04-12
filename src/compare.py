from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from workspace import git_changed_files


@dataclass(slots=True)
class FileCompareResult:
    path: str
    changed_lines_a: int
    changed_lines_b: int
    matched_lines: int
    scored_positions: int
    similarity_ratio: float
    skipped_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "changed_lines_a": self.changed_lines_a,
            "changed_lines_b": self.changed_lines_b,
            "matched_lines": self.matched_lines,
            "scored_positions": self.scored_positions,
            "similarity_ratio": self.similarity_ratio,
            "skipped_reason": self.skipped_reason,
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


def compare_solution_repos(*, original_dir: Path, repo_a_dir: Path, repo_b_dir: Path) -> CompareResult:
    changed_files = sorted(set(git_changed_files(repo_a_dir)) | set(git_changed_files(repo_b_dir)))
    per_file: list[FileCompareResult] = []
    matched_changed_lines = 0
    scored_positions = 0
    total_changed_lines_a = 0
    total_changed_lines_b = 0

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
        changed_sequence_a = _build_changed_line_sequence(original_lines, file_a_lines)
        changed_sequence_b = _build_changed_line_sequence(original_lines, file_b_lines)
        matched_lines, file_positions = _count_positional_matches(changed_sequence_a, changed_sequence_b)
        similarity_ratio = (matched_lines / file_positions) if file_positions else 0.0

        per_file.append(
            FileCompareResult(
                path=relative_path,
                changed_lines_a=len(changed_sequence_a),
                changed_lines_b=len(changed_sequence_b),
                matched_lines=matched_lines,
                scored_positions=file_positions,
                similarity_ratio=similarity_ratio,
            )
        )
        matched_changed_lines += matched_lines
        scored_positions += file_positions
        total_changed_lines_a += len(changed_sequence_a)
        total_changed_lines_b += len(changed_sequence_b)

    similarity_ratio = (matched_changed_lines / scored_positions) if scored_positions else 0.0
    return CompareResult(
        matched_changed_lines=matched_changed_lines,
        scored_positions=scored_positions,
        total_changed_lines_a=total_changed_lines_a,
        total_changed_lines_b=total_changed_lines_b,
        similarity_ratio=similarity_ratio,
        per_file=per_file,
    )


def _read_file_bytes(path: Path) -> bytes | None:
    if not path.exists():
        return None
    return path.read_bytes()


def _is_binary_content(raw_bytes: bytes | None) -> bool:
    return raw_bytes is not None and b"\0" in raw_bytes


def _decode_lines(raw_bytes: bytes | None) -> list[str]:
    if raw_bytes is None:
        return []
    return raw_bytes.decode("utf-8", errors="replace").splitlines()


def _build_changed_line_sequence(original_lines: list[str], updated_lines: list[str]) -> list[str]:
    matcher = SequenceMatcher(a=original_lines, b=updated_lines, autojunk=False)
    changed_lines: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag in {"replace", "delete"}:
            changed_lines.extend(f"-:{line}" for line in original_lines[i1:i2])
        if tag in {"replace", "insert"}:
            changed_lines.extend(f"+:{line}" for line in updated_lines[j1:j2])
    return changed_lines


def _count_positional_matches(sequence_a: list[str], sequence_b: list[str]) -> tuple[int, int]:
    scored_positions = max(len(sequence_a), len(sequence_b))
    if not scored_positions:
        return 0, 0
    matcher = SequenceMatcher(a=sequence_a, b=sequence_b, autojunk=False)
    matched_lines = sum(size for _, _, size in matcher.get_matching_blocks())
    return matched_lines, scored_positions
