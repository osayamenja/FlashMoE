import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "flashmoe_jit", Path(__file__).resolve().parents[1] / "flashmoe" / "jit.py"
)
jit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jit)


class JITCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "cache"
        self.builds = 0
        self.write("flashmoe/CMakeLists.txt", "set(NATIVE_REVISION v1)\n")
        self.write("flashmoe/cmake/CPM.cmake", "# CPM bootstrap\n")
        for patcher in (
            patch.object(jit, "__file__", str(self.root / "flashmoe" / "jit.py")),
            patch.dict("os.environ", {"FLASHMOE_CACHE_DIR": str(self.cache)}),
            patch("subprocess.run", side_effect=self.run_cmake),
            patch.object(jit, "_load_ext", side_effect=lambda name, path: path),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def write(self, relative, contents):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        return path

    def checkout(self):
        self.write("CMakeLists.txt", "project(flashmoe)\n")
        self.write("csrc/cmake/CPM.cmake", "# Native CPM bootstrap\n")
        return self.write("csrc/include/flashmoe/gate.cuh", "// Gate revision 1\n")

    def run_cmake(self, command, *, check):
        if command[1] == "--build":
            self.builds += 1
            (Path(command[2]) / "test_gate.so").write_bytes(b"compiled module")

    def compile(self):
        return jit._get_compiled(
            SimpleNamespace(gpu_arch=900), "generated bindings", "gate", "test_gate"
        )

    def test_checkout_header_change_rebuilds_and_unchanged_sources_reuse(self):
        header = self.checkout()
        original = self.compile()
        self.assertEqual(self.compile(), original)
        self.assertEqual(self.builds, 1)

        header.write_text("// Gate revision 2\n")
        changed = self.compile()
        self.assertNotEqual(changed, original)
        self.assertEqual(self.compile(), changed)
        self.assertEqual(self.builds, 2)

    def test_checkout_build_recipe_changes_rebuild(self):
        self.checkout()
        previous = self.compile()
        for relative in ("CMakeLists.txt", "csrc/cmake/CPM.cmake"):
            with self.subTest(path=relative):
                path = self.root / relative
                path.write_text(path.read_text() + "# Updated build recipe\n")
                changed = self.compile()
                self.assertNotEqual(changed, previous)
                previous = changed
        self.assertEqual(self.builds, 3)

    def test_packaged_revision_and_build_recipe_changes_rebuild(self):
        previous = self.compile()
        for relative, contents in (
            ("flashmoe/CMakeLists.txt", "set(NATIVE_REVISION v2)\n"),
            ("flashmoe/cmake/CPM.cmake", "# Updated CPM bootstrap\n"),
        ):
            with self.subTest(path=relative):
                self.write(relative, contents)
                changed = self.compile()
                self.assertNotEqual(changed, previous)
                self.assertEqual(self.compile(), changed)
                previous = changed
        self.assertEqual(self.builds, 3)

    def test_cache_tracks_selected_native_source(self):
        packaged = self.compile()
        self.write("csrc/include/unselected.cuh", "// Not used without gate.cuh\n")
        self.assertEqual(self.compile(), packaged)
        self.assertEqual(self.builds, 1)

        header = self.checkout()
        self.assertNotEqual(self.compile(), packaged)
        self.assertEqual(self.builds, 2)

        header.unlink()
        self.assertEqual(self.compile(), packaged)
        self.assertEqual(self.builds, 2)

    def test_legacy_cache_entry_is_not_loaded(self):
        key = hashlib.sha256(
            f"test_gate|py{sys.version_info[:2]}|generated bindings".encode()
        ).hexdigest()[:16]
        legacy = self.cache / f"test_gate_{key}" / "test_gate.so"
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"stale module")

        self.assertNotEqual(self.compile(), legacy)
        self.assertEqual(self.builds, 1)
        self.assertEqual(legacy.read_bytes(), b"stale module")


if __name__ == "__main__":
    unittest.main()
