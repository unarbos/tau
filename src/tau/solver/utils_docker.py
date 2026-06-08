import logging
import shlex
import subprocess
import tarfile
import tempfile
import textwrap
from pathlib import Path

from workspace import ensure_tree_has_no_symlinks

log = logging.getLogger(__name__)

_CONTAINER_ROOT = "/work"
_CONTAINER_REPO_DIR = f"{_CONTAINER_ROOT}/repo"


def _apply_patch_to_repo(*, repo_dir: Path, patch_text: str) -> None:
    if not patch_text.strip():
        return
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as temp_file:
        temp_file.write(patch_text)
        temp_file.write("\n")
        temp_path = Path(temp_file.name)
    try:
        _run(
            ["git", "apply", "--binary", "--whitespace=nowarn", str(temp_path)],
            cwd=repo_dir,
            timeout=120,
        )
        ensure_tree_has_no_symlinks(repo_dir, label="solver output tree")
    finally:
        temp_path.unlink(missing_ok=True)



def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    log.debug("Running command: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required command not found: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(cmd[:3])}") from exc

    if check and result.returncode != 0:
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        raise RuntimeError(f"Command failed ({' '.join(cmd[:3])}): {output[-500:]}")
    return result


def _collect_repo_patch_from_container(*, container_id: str) -> str:
    patch_cmd = (
        'cd "$TAU_REPO_DIR" && '
        "git diff --binary && "
        "while IFS= read -r -d '' path; do "
        'git diff --binary --no-index -- /dev/null "$path" || test $? -eq 1; '
        "done < <(git ls-files --others --exclude-standard -z)"
    )
    result = _run(
        [
            "docker",
            "exec",
            "-e",
            f"TAU_REPO_DIR={_CONTAINER_REPO_DIR}",
            container_id,
            "bash",
            "-lc",
            patch_cmd,
        ],
        timeout=120,
        check=False,
    )
    if result.returncode not in (0, 1):
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        raise RuntimeError(f"Failed to collect solver patch from container: {output[-500:]}")
    return result.stdout or ""


def _container_is_running(container_id: str) -> bool:
    result = _run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
        timeout=30,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip().lower() == "true"



def _copy_directory_to_container(
    *,
    source_dir: Path,
    container_id: str,
    target_dir: str,
    exclude_names: set[str] | None = None,
) -> None:
    if not source_dir.is_dir():
        raise RuntimeError(f"Directory to copy does not exist: {source_dir}")

    extract_script = textwrap.dedent(
        """\
        import sys
        import tarfile
        from pathlib import Path

        target = Path(sys.argv[1])
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=sys.stdin.buffer, mode="r|*") as archive:
            archive.extractall(target)
        """,
    ).strip()

    with tempfile.NamedTemporaryFile(suffix=".tar") as tar_file:
        with tarfile.open(fileobj=tar_file, mode="w") as archive:
            archive.add(source_dir, arcname=".", filter=_tar_filter(exclude_names))
        tar_file.flush()
        tar_file.seek(0)
        result = subprocess.run(
            ["docker", "exec", "-i", container_id, "python3", "-c", extract_script, target_dir],
            stdin=tar_file,
            capture_output=True,
            text=True,
            timeout=300,
        )
    if result.returncode != 0:
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        raise RuntimeError(f"Failed to copy directory into container: {output[-500:]}")


def _tar_filter(exclude_names: set[str] | None):
    def filter_member(tar_info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if not exclude_names:
            return tar_info
        parts = [part for part in Path(tar_info.name).parts if part not in {".", ""}]
        if any(part in exclude_names for part in parts):
            return None
        return tar_info

    return filter_member


def _find_repo_symlinks_in_container(*, container_id: str) -> str | None:
    check_script = textwrap.dedent(
        """\
        import os
        from pathlib import Path

        repo_dir = Path(os.environ["TAU_REPO_DIR"])
        symlinks = []
        for current_root, dirnames, filenames in os.walk(repo_dir, topdown=True, followlinks=False):
            current_dir = Path(current_root)
            for name in [*dirnames, *filenames]:
                candidate = current_dir / name
                if candidate.is_symlink():
                    symlinks.append(str(candidate.relative_to(repo_dir)))

        print("\\n".join(sorted(symlinks[:10])))
        """,
    ).strip()
    result = _run(
        [
            "docker",
            "exec",
            "-e",
            f"TAU_REPO_DIR={_CONTAINER_REPO_DIR}",
            container_id,
            "python3",
            "-c",
            check_script,
        ],
        timeout=120,
    )
    symlinks = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not symlinks:
        return None
    sample = ", ".join(symlinks)
    if len(symlinks) == 10:
        sample = f"{sample}, ..."
    return sample



def _kill_container(container_id: str) -> None:
    _run_best_effort(
        ["docker", "kill", container_id],
        timeout=30,
        action="kill container",
    )


def _run_best_effort(
    cmd: list[str],
    *,
    timeout: int,
    action: str,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return _run(cmd, cwd=cwd, timeout=timeout, check=False)
    except Exception as exc:
        log.warning(
            "Best-effort Docker %s failed (non-fatal): %s",
            action,
            exc,
        )
        return None



def _read_limited_output(path: Path, *, max_output_bytes: int | None = None) -> str:
    if not path.exists():
        return ""
    raw_bytes = path.read_bytes()
    if max_output_bytes is not None and len(raw_bytes) > max_output_bytes:
        raw_bytes = raw_bytes[-max_output_bytes:]
    return raw_bytes.decode("utf-8", errors="replace")


def _remove_container(container_id: str) -> None:
    _run_best_effort(
        ["docker", "rm", "-f", container_id],
        timeout=30,
        action="remove container",
    )



def _sanitize_repo_git_metadata_in_container(*, container_id: str, repo_dir: str) -> None:
    _run(
        ["docker", "exec", container_id, "bash", "-lc", _git_metadata_sanitize_script(repo_dir)],
        timeout=120,
    )



def _git_metadata_sanitize_script(repo_dir: str) -> str:
    quoted_repo = shlex.quote(repo_dir)
    return textwrap.dedent(
        f"""\
        set -euo pipefail
        repo={quoted_repo}
        git_dir="$repo/.git"
        if [ ! -d "$git_dir" ]; then
            exit 0
        fi

        head_sha="$(git -C "$repo" -c safe.directory="$repo" rev-parse --verify HEAD)"
        git -C "$repo" -c safe.directory="$repo" checkout --detach "$head_sha" >/dev/null 2>&1

        rm -f \
            "$git_dir/FETCH_HEAD" \
            "$git_dir/ORIG_HEAD" \
            "$git_dir/MERGE_HEAD" \
            "$git_dir/CHERRY_PICK_HEAD" \
            "$git_dir/REBASE_HEAD" \
            "$git_dir/packed-refs" \
            "$git_dir/objects/info/alternates"
        rm -rf "$git_dir/refs" "$git_dir/logs"
        mkdir -p "$git_dir/refs"

        git -C "$repo" -c safe.directory="$repo" reflog expire --expire=now --all >/dev/null 2>&1 || true
        git -C "$repo" -c safe.directory="$repo" gc --prune=now >/dev/null 2>&1 || true
        """
    ).strip()



def _write_text_to_container(*, container_id: str, target_path: str, content: str) -> None:
    parent_dir = str(Path(target_path).parent)
    quoted_parent_dir = shlex.quote(parent_dir)
    _run(
        ["docker", "exec", container_id, "bash", "-lc", f"mkdir -p {quoted_parent_dir}"],
        timeout=30,
    )
    write_script = textwrap.dedent(
        """\
        import sys
        from pathlib import Path

        target = Path(sys.argv[1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(sys.stdin.read())
        """,
    ).strip()
    result = subprocess.run(
        ["docker", "exec", "-i", container_id, "python3", "-c", write_script, target_path],
        input=content,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        raise RuntimeError(f"Failed to write file into container: {output[-500:]}")
