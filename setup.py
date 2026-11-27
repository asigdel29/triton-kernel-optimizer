"""Build the platform wheel with a Zig shared library for CPU inference."""
from pathlib import Path
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class ZigBuild(build_ext):
    """Compile a loadable library with the pinned build-time toolchain."""

    def build_extension(self, ext):
        """Emit the Zig C ABI under the platform extension filename."""
        dest = Path(self.get_ext_fullpath(ext.name)).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "ziglang", "build-lib", "native/kernels.zig",
                        "-dynamic", "-O", "ReleaseFast", "-mcpu", "baseline",
                        f"-femit-bin={dest}"], check=True)


setup(ext_modules=[Extension("kernel_optimizer._cpu", sources=["native/kernels.zig"])],
      cmdclass={"build_ext": ZigBuild})
