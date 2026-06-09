import hashlib
import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from submission_bundle import (
    build_manifest,
    bundle_files_from_archive,
    bundle_signature_payload,
    canonical_bundle_sha256,
    load_bundle_directory,
    normalize_bundle_files,
    unified_bundle_diff,
    validate_bundle_paths,
    write_bundle_directory,
)


def _sample_files() -> dict[str, bytes]:
    return {
        "agent.py": b"def solve(**kwargs):\n    return {}\n",
        "helper.py": b"VALUE = 1\n",
    }


class SubmissionBundleTest(unittest.TestCase):
    def test_canonical_hash_is_order_independent(self):
        files = _sample_files()
        reversed_items = dict(reversed(list(files.items())))
        self.assertEqual(
            canonical_bundle_sha256(files),
            canonical_bundle_sha256(reversed_items),
        )

    def test_build_manifest_requires_entrypoint(self):
        with self.assertRaises(ValueError):
            build_manifest(files={"helper.py": b"x = 1\n"})

    def test_validate_rejects_traversal_and_non_py(self):
        violations = validate_bundle_paths(["../agent.py", "agent.py", "notes.txt"])
        joined = "\n".join(violations)
        self.assertIn("agent.py", joined)
        self.assertTrue(any("notes.txt" in item for item in violations))

    def test_round_trip_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, manifest = write_bundle_directory(root=root, submission_id="sub-1", files=_sample_files())
            loaded_manifest, loaded_files = load_bundle_directory(target)
            self.assertEqual(manifest.bundle_sha256, loaded_manifest.bundle_sha256)
            self.assertEqual(loaded_files, _sample_files())
            self.assertTrue((target / "manifest.json").is_file())
            self.assertTrue((target / "files" / "agent.py").is_file())

    def test_bundle_signature_payload_v2(self):
        bundle_sha = canonical_bundle_sha256(_sample_files())
        payload = bundle_signature_payload(
            hotkey="5Hotkey",
            submission_id="sub-1",
            bundle_sha256=bundle_sha,
        )
        self.assertEqual(
            payload.decode("utf-8"),
            f"tau-private-submission-v2:5Hotkey:sub-1:{bundle_sha}",
        )

    def test_unified_bundle_diff_spans_files(self):
        base = _sample_files()
        submitted = dict(base)
        submitted["helper.py"] = b"VALUE = 2\n"
        submitted["new_module.py"] = b"X = 1\n"
        diff = unified_bundle_diff(base_files=base, submitted_files=submitted)
        self.assertIn("helper.py", diff)
        self.assertIn("new_module.py", diff)
        self.assertNotIn("agent.py", diff)

    def test_archive_parsers(self):
        files = _sample_files()
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w:gz") as archive:
            for path, content in files.items():
                info = tarfile.TarInfo(name=path)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        parsed_tar = bundle_files_from_archive(tar_buf.getvalue(), archive_name="bundle.tar.gz")
        self.assertEqual(parsed_tar, normalize_bundle_files(files))

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, mode="w") as archive:
            for path, content in files.items():
                archive.writestr(path, content)
        parsed_zip = bundle_files_from_archive(zip_buf.getvalue(), archive_name="bundle.zip")
        self.assertEqual(parsed_zip, normalize_bundle_files(files))

    def test_manifest_json_is_stable(self):
        manifest = build_manifest(files=_sample_files())
        payload = json.dumps(manifest.to_dict(), sort_keys=True)
        self.assertIn('"entrypoint": "agent.py"', payload)
        self.assertEqual(len(manifest.files), 2)
        for record in manifest.files:
            self.assertEqual(record.sha256, hashlib.sha256(_sample_files()[record.path]).hexdigest())


if __name__ == "__main__":
    unittest.main()
