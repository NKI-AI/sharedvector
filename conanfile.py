"""Dependency manifest consumed by ``cmake/conan_provider.cmake``.

The provider runs ``conan install`` itself during the CMake configure step, so this file declares
requirements only: no toolchain, no layout, no ``generate()``.
"""

from conan import ConanFile


class SharedVectorConan(ConanFile):
    """Boost (header-only) for the extension; GoogleTest only when the C++ tests are built."""

    name = "sharedvector"
    settings = "os", "compiler", "build_type", "arch"
    options = {"with_tests": [True, False]}
    default_options = {"with_tests": False, "boost/*:header_only": True}
    generators = "CMakeConfigDeps"

    def requirements(self) -> None:
        """Declare the runtime dependencies of the extension."""
        self.requires("boost/1.87.0")

    def build_requirements(self) -> None:
        """Declare test-only dependencies."""
        if self.options.with_tests:
            self.test_requires("gtest/1.15.0")
