from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='terrabyte_cube',
    version='0.1.0',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    include_package_data=True,
    description='terrabyte data cube spectral stacking',
    author='Baturalp Arisoy',
    author_email='baturalp.arisoy@uni-wuerzburg.de',
    # url='https://github.com/username/packagename',  # Optional
)