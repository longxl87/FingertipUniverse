from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
with open(HERE / "requirements.txt", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="fingertip_universe",
    version="1.0.0",
    author="LXL",
    author_email="331942615@qq.com",
    description="a billion bits of ...",
    packages=find_packages(exclude=['test*', 'tests']),
    install_requires=requirements,
    python_requires='>=3.10',
)