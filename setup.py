from setuptools import setup, find_packages

setup(
    name="loom-video",
    version="0.2.0",  # Bumped from 0.1.0
    description="Latent video diffusion engine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.9.0",
        "scipy>=1.11.0",
        "transformers>=4.30.0",
        "diffusers>=0.20.0",
        "tqdm>=4.66.0",
        "Pillow>=9.0.0",
    ],
    entry_points={
        "console_scripts": [
            "loom=loom:main",  # Add a CLI entry if you make one
        ],
    },
)
