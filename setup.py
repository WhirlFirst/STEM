import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scSTEM",
    version="0.0.2",
    author="whirl",
    author_email="hmsh653@gmail.com",
    description="A Method for Mapping Single-cell and Spatial Transcriptomics Data with Transfer Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WhirlFirst/STEM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "scanpy>=1.8.2",
        "tqdm>=4.60.0",
        "seaborn>=0.11.1",
    ],
)