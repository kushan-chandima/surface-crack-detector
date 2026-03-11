from setuptools import setup, find_packages

setup(
    name="surface-crack-detector",
    version="1.0.0",
    description="Surface Crack Detection using Deep Learning (CNN with Transfer Learning)",
    author="Crack Detector Team",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.15.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "scikit-learn>=1.4.0",
        "streamlit>=1.31.0",
        "Pillow>=10.2.0",
        "opendatasets>=0.1.22",
    ],
    entry_points={
        "console_scripts": [
            "crack-predict=src.predict:main",
        ],
    },
)
