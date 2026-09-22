# SPDX-License-Identifier: Apache-2.0
"""构建入口和产物校验的离线回归测试，不分配 GPU 或访问存储。"""

# Standard
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch
import os
import subprocess
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).with_name("build_lmcache_wheel.sh")
SOURCE = SCRIPT.read_text()
VALIDATOR = SOURCE.split("<<'PY_VALIDATE'\n", 1)[1].split("\nPY_VALIDATE", 1)[0]


class BuildWheelTests(unittest.TestCase):
    """验证配置边界与不完整产物的拒绝行为。"""

    def test_help(self) -> None:
        """帮助输出不需要构建工具链。"""
        result = subprocess.run(
            ["bash", str(SCRIPT), "--help"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("BUILD_WITH_MOONCAKE", result.stdout)
        self.assertIn("WHEEL_VERSION", result.stdout)
        self.assertIn("DEPENDENCY_PROXY_FALLBACK", result.stdout)

    def test_builder_has_no_environment_specific_proxy(self) -> None:
        """构建脚本不能默认绑定特定环境的内网代理。"""
        self.assertNotIn("192.168.10.6:3128", SOURCE)

    def test_invalid_flag(self) -> None:
        """不允许拼错的开关静默改变构建内容。"""
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            env={**os.environ, "BUILD_RUST": "maybe"},
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("BUILD_RUST", result.stderr)

    def validate(self, members: dict[str, bytes], mooncake: bool = False) -> list:
        """提供模拟 wheel，返回隔离导入子进程的调用。"""
        with tempfile.TemporaryDirectory() as folder:
            wheel = Path(folder) / "lmcache-test.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                for name, content in members.items():
                    archive.writestr("lmcache/" + name + ".so", content)
            with (
                patch("sys.argv", ["validate", folder, str(int(mooncake)), "0", "0"]),
                patch("subprocess.run") as run,
                redirect_stdout(StringIO()),
            ):
                exec(compile(VALIDATOR, str(SCRIPT), "exec"), {})
                return run.call_args_list

    def test_complete_wheel(self) -> None:
        """四个核心扩展齐全后必须执行隔离导入。"""
        calls = self.validate(
            dict.fromkeys(
                ["c_ops", "native_storage_ops", "lmcache_redis", "lmcache_fs"],
                b"fixture",
            )
        )
        self.assertEqual(len(calls), 1)
        self.assertIn("-I", calls[0].args[0])
        self.assertTrue(calls[0].kwargs["check"])

    def test_missing_cuda(self) -> None:
        """没有 CUDA 扩展的 wheel 不能报告成功。"""
        with self.assertRaisesRegex(SystemExit, "c_ops"):
            self.validate({"lmcache_fs": b"fixture"})

    def test_empty_extension(self) -> None:
        """空扩展文件不能充当编译产物。"""
        with self.assertRaisesRegex(SystemExit, "c_ops"):
            self.validate({"c_ops": b""})

    def test_requested_mooncake_missing(self) -> None:
        """用户请求的可选扩展也必须出现在产物内。"""
        with self.assertRaisesRegex(SystemExit, "lmcache_mooncake"):
            self.validate(
                dict.fromkeys(
                    ["c_ops", "native_storage_ops", "lmcache_redis", "lmcache_fs"],
                    b"fixture",
                ),
                mooncake=True,
            )

    def test_direct_mode_clears_proxies(self) -> None:
        """直连分支必须清除所有继承的代理。"""
        helper = SOURCE.split("with_proxy() {", 1)[1].split(
            "\ninstall_python_deps()", 1
        )[0]
        result = subprocess.run(
            ["bash", "-c", "with_proxy() {" + helper + "\nwith_proxy '' env"],
            env={**os.environ, "HTTP_PROXY": "bad", "ALL_PROXY": "bad"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertFalse(
            any(
                line.split("=", 1)[0].lower()
                in ("http_proxy", "https_proxy", "all_proxy")
                for line in result.stdout.splitlines()
            )
        )


if __name__ == "__main__":
    unittest.main()
