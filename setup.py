import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="meta-gp",
    version="0.0.1",
    author="Jonas Rothfuss",
    author_email="jonas.rothfuss@gmail.com",
    description="Meta-Learning Gausssian Proccess Priors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={'': 'src'},
    packages=setuptools.find_packages(),
    install_requires=[
        'torch',
        'numpy'
        'gpytorch'
    ],
)