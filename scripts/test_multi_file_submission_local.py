#!/usr/bin/env python3
"""Isolated local smoke test for the multi-file submission pipeline."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from email.message import Message
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import RunConfig
from private_submission import private_submission_check_passed
from submission_api import SubmissionApiConfig, build_handler, handle_submission_request
from submission_bundle import canonical_bundle_sha256, default_ninja_repo_path
from validate import ValidatorSubmission, _cached_agent_source, _verify_hotkey_signature

HOTKEY = "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repoTitleHkey"
SIGNATURE = "local-test-signature"

MINIMAL_AGENT = """\
from typing import Optional

def solve(repo_path: str, issue: str, model: Optional[str] = None, api_base: Optional[str] = None, api_key: Optional[str] = None):
    return {"patch": "", "logs": "private submission", "steps": 1, "cost": None, "success": True}
"""

BASE_FILES = {
    "agent.py": MINIMAL_AGENT.encode("utf-8"),
}

def make_submitted_bundle() -> dict[str, bytes]:
    agent_text = MINIMAL_AGENT.replace(
        '"logs": "private submission"',
        '"logs": "private submission multi-file"',
        1,
    )
    return {
        "agent.py": agent_text.encode("utf-8"),
        "helper.py": b"BUNDLE_TEST_MARKER = 'multi-file-local-test'\n",
    }


def build_tar_gz(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path, content in sorted(files.items()):
            info = tarfile.TarInfo(name=path)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def encode_multipart(*, fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    boundary = f"----local-test-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )
    for name, (filename, content, content_type) in files.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                ).encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                content,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def judge_pass(_payload: dict) -> dict:
    return {
        "verdict": "pass",
        "overall_score": 90,
        "real_edit_score": 90,
        "safety_score": 95,
        "scope_score": 90,
        "contract_score": 95,
        "summary": "local test pass",
        "reasons": ["multi-file helper import"],
        "risks": [],
        "required_changes": [],
    }


def make_config(root: Path) -> SubmissionApiConfig:
    ninja = default_ninja_repo_path()
    return SubmissionApiConfig(
        private_submission_root=root,
        base_agent=ninja / "agent.py",
        run_config=RunConfig(validate_netuid=66),
        judge=judge_pass,
        judge_min_score=65,
        base_harness_git_repo=ninja,
        base_harness_git_ref="main",
        overwrite=True,
    )


def bundle_request_body(*, hotkey: str, submission_id: str, signature: str, archive: bytes) -> tuple[bytes, str]:
    return encode_multipart(
        fields={
            "hotkey": hotkey,
            "submission_id": submission_id,
            "signature": signature,
        },
        files={"bundle": ("harness.tar.gz", archive, "application/gzip")},
    )


def agent_request_body(*, hotkey: str, submission_id: str, signature: str, agent_py: str) -> tuple[bytes, str]:
    return encode_multipart(
        fields={
            "hotkey": hotkey,
            "submission_id": submission_id,
            "signature": signature,
        },
        files={"agent": ("agent.py", agent_py.encode("utf-8"), "text/x-python")},
    )


def post_local(url: str, body: bytes, content_type: str) -> dict:
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": content_type, "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read().decode("utf-8"))


def main() -> int:
    files = make_submitted_bundle()
    bundle_sha = canonical_bundle_sha256(files)
    archive = build_tar_gz(files)
    submission_id = f"local-test-{bundle_sha[:12]}"

    print(f"ninja repo: {default_ninja_repo_path()}")
    print(f"bundle_sha256: {bundle_sha}")
    print(f"archive_bytes: {len(archive)}")

    with tempfile.TemporaryDirectory(prefix="multi-file-submission-local-") as tmp:
        root = Path(tmp) / "private-submissions"
        root.mkdir()
        config = make_config(root)

        body, content_type = bundle_request_body(
            hotkey=HOTKEY,
            submission_id=submission_id,
            signature=SIGNATURE,
            archive=archive,
        )
        headers = Message()
        headers["Content-Type"] = content_type
        headers["Content-Length"] = str(len(body))

        with patch("submission_api._verify_hotkey_signature", return_value=True):
            with patch("submission_api.registration_context", return_value=(100, 42, None)):
                with patch("submission_api.publish_submissions_api_data"):
                    with patch("submission_api.read_base_harness_files", return_value=BASE_FILES):
                        status, payload = handle_submission_request(
                            headers=headers,
                            rfile=io.BytesIO(body),
                            config=config,
                        )

        print("\n[1] direct handle_submission_request (v2 bundle)")
        print(f"status={status} accepted={payload.get('accepted')} signature_version={payload.get('signature_version')}")
        if status != 200 or not payload.get("accepted"):
            print(json.dumps(payload, indent=2))
            return 1

        bundle_dir = Path(payload["bundle_path"])
        manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["bundle_sha256"] == bundle_sha
        assert (bundle_dir / "files" / "helper.py").is_file()
        print(f"bundle stored at {bundle_dir}")

        print("\n[2] private_submission_check_passed + validator agent source")
        assert private_submission_check_passed(
            root,
            submission_id,
            bundle_sha,
            hotkey=HOTKEY,
            signature_verifier=lambda *_args, **_kwargs: True,
        )
        submission = ValidatorSubmission(
            hotkey=HOTKEY,
            uid=42,
            repo_full_name=f"private-submission/{submission_id}",
            repo_url=f"private-submission://{submission_id}",
            commit_sha=bundle_sha,
            commitment=f"private-submission:{submission_id}:{bundle_sha}",
            commitment_block=100,
            source="private-submission",
        )
        run_config = RunConfig(
            workspace_root=ROOT,
            validate_netuid=66,
            validate_private_submission_root=root,
        )
        with patch("validate._verify_hotkey_signature", return_value=True):
            source = _cached_agent_source(run_config, submission)
        assert source.kind == "local_path"
        assert source.agent_file == "agent.py"
        helper_path = Path(source.local_path) / "helper.py"
        assert helper_path.is_file(), helper_path
        print(f"validator would load harness from {source.local_path}")

        print("\n[3] local HTTP server on 127.0.0.1:18066 (not pm2 :8066)")
        handler = build_handler(config)
        server = ThreadingHTTPServer(("127.0.0.1", 18066), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.2)

        with patch("submission_api._verify_hotkey_signature", return_value=True):
            with patch("submission_api.registration_context", return_value=(101, 42, None)):
                with patch("submission_api.publish_submissions_api_data"):
                    with patch("submission_api.read_base_harness_files", return_value=BASE_FILES):
                        with patch("submission_api.read_base_agent_py", return_value=MINIMAL_AGENT):
                            v2_id = f"local-v2-{bundle_sha[:12]}"
                            v2_body, v2_type = bundle_request_body(
                                hotkey=HOTKEY,
                                submission_id=v2_id,
                                signature=SIGNATURE,
                                archive=archive,
                            )
                            v2_resp = post_local("http://127.0.0.1:18066/api/submissions", v2_body, v2_type)
                            v1_id = f"local-v1-{hashlib.sha256(MINIMAL_AGENT.encode()).hexdigest()[:12]}"
                            v1_body, v1_type = agent_request_body(
                                hotkey=HOTKEY,
                                submission_id=v1_id,
                                signature=SIGNATURE,
                                agent_py=MINIMAL_AGENT,
                            )
                            with patch("submission_api.registration_context", return_value=(102, 42, None)):
                                v1_resp = post_local("http://127.0.0.1:18066/api/submissions", v1_body, v1_type)

        server.shutdown()
        print(f"v1 accepted={v1_resp.get('accepted')} signature_version={v1_resp.get('signature_version')}")
        print(f"v2 accepted={v2_resp.get('accepted')} signature_version={v2_resp.get('signature_version')}")
        if not v1_resp.get("accepted") or not v2_resp.get("accepted"):
            print("v1:", json.dumps(v1_resp, indent=2))
            print("v2:", json.dumps(v2_resp, indent=2))
            return 1

        harness_dir = Path(tmp) / "harness"
        harness_dir.mkdir()
        for rel_path, content in files.items():
            out = harness_dir / rel_path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(content)

        print("\n[4] ninja submit script dry-run payload shape")
        from submission_bundle import bundle_signature_payload as tau_bundle_signature_payload

        expected_payload = tau_bundle_signature_payload(
            hotkey=HOTKEY,
            submission_id=f"example-{bundle_sha[:12]}",
            bundle_sha256=bundle_sha,
        ).decode("utf-8")
        assert expected_payload.startswith("tau-private-submission-v2:")
        print(f"expected v2 payload prefix ok: {expected_payload.split(':', 3)[0]}:")
        ninja_script = ROOT.parent / "ninja" / "scripts" / "submit_private_submission.py"
        if ninja_script.is_file():
            import subprocess

            proc = subprocess.run(
                [sys.executable, str(ninja_script), "--bundle", str(harness_dir), "--dry-run"],
                text=True,
                capture_output=True,
            )
            if proc.returncode != 0 and "Keyfile" in (proc.stderr or ""):
                print("skip wallet-backed ninja dry-run; no local bittensor wallet in this environment")
            else:
                if proc.returncode != 0:
                    print(proc.stdout)
                    print(proc.stderr, file=sys.stderr)
                    return 1
                assert f"bundle_sha256: {bundle_sha}" in proc.stdout
                assert "tau-private-submission-v2:" in proc.stdout
                print("ninja --bundle --dry-run produced matching v2 payload")
        else:
            print(f"skip ninja script check; missing {ninja_script}")

    print("\nLOCAL MULTI-FILE SUBMISSION TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
