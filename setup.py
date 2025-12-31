#!/usr/bin/env python
import os

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

# Definição da extensão Cython (Caminho atualizado para flowmac)
exts = [
    Extension(
        name="flowmac.utils.monotonic_align.core",
        sources=["flowmac/utils/monotonic_align/core.pyx"],
    )
]

# Leitura do README para descrição longa
with open("README.md", encoding="utf-8") as readme_file:
    README = readme_file.read()

# Leitura da Versão (Lê do arquivo flowmac/VERSION)
cwd = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(cwd, "flowmac", "VERSION"), encoding="utf-8") as fin:
    version = fin.read().strip()

# Leitura dos requisitos do requirements.txt
def get_requires():
    requirements = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(requirements, encoding="utf-8") as reqfile:
        return [str(r).strip() for r in reqfile]

setup(
    name="flowmac",
    version=version,
    description="🌊 FlowMAC: Conditional Flow Matching for Audio Coding at Low Bit Rates",
    long_description=README,
    long_description_content_type="text/markdown",
    # Autores citados no README/Paper
    author="Nicola Pia, Martin Strauss, Markus Multrus, Bernd Edler",
    author_email="", 
    url="", # Adicione o link do repositório se houver
    install_requires=get_requires(),
    include_dirs=[numpy.get_include()],
    include_package_data=True,
    packages=find_packages(exclude=["tests", "tests/*", "examples", "examples/*"]),
    
    # Comandos que ficarão disponíveis no terminal após 'pip install -e .'
    entry_points={
        "console_scripts": [
            "flowmac-data-stats=flowmac.utils.generate_data_statistics:main",
            "flowmac-cli=flowmac.cli:cli",
            "flowmac-app=flowmac.app:main",
            "flowmac-get-durations=flowmac.utils.get_durations_from_trained_model:main",
        ]
    },
    ext_modules=cythonize(exts, language_level=3),
    python_requires=">=3.9.0",
)
