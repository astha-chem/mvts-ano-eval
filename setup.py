from setuptools import setup, find_packages

setup(
    name='mvts-eval',
    author='Astha Garg, Jules Samaran, Wenyu Zhang',
    description='An evaluation of multivariate time-series anomaly detection algorithms',
    long_description=open('README.md').read(),
    version='0.0',
    packages=find_packages(),
    scripts=[],
    # Requirements for executing the project (not development)
    # install_requires=parse_requirements('requirements.txt'),
    # url='github.com/KDD-OpenSource/DeepADoTS',
    # license='MIT License',
)
