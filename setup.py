from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='stac2ardcube',
    version='1.0.0',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    include_package_data=True,
    description='STAC to Analysis-Ready-Data Cubes w/ terrabyte by DLR & LRZ',
    author='Baturalp Arisoy',
    author_email='baturalp.arisoy@uni-wuerzburg.de',
    url='https://github.com/BaturalpArisoy/stac2ardcube'
)