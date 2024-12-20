from setuptools import setup, find_packages

setup(
    name="pychladni",
    version="0.1.0",
    description="Chladni pattern generator",
    author="Clayton Rabideau",
    author_email="claytonrabideau@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "librosa>=0.9.2",
        "matplotlib>=3.5.0",
        "pillow>=9.0.0",
        "soundfile>=0.10.3",
        "p-tqdm>=1.3.3",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="chladni, physics, audio, visualization, sound, vibration, plate dynamics",
    project_urls={
        "Bug Reports": "https://github.com/clayton-r/pychladni/issues",
        "Source": "https://github.com/clayton-r/pychladni",
    }
)