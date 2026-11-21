"""Ships ``docs/`` inside the wheel as ``mozo.docs``; pyproject.toml says why. pyproject.toml
can discover packages or list them, not both, so this keeps discovery and adds the one mapped
package. Everything else is in pyproject.toml."""

from setuptools import find_packages, setup

setup(
    packages=[*find_packages(include=["mozo*"]), "mozo.docs"],
    package_dir={"mozo.docs": "docs"},
)
