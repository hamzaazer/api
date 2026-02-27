from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="ecg_signal",                 # must match: from ecg_signal import ...
        sources=["ecg_signal.pyx"],         # your .pyx file
        include_dirs=[np.get_include()],
        language="c",
    )
]

setup(
    name="ecg_signal",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
