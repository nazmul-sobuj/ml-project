from setuptools import setup, find_packages
from typing import List
HIPEN_E_DOT = "-e ."
def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements= [req.replace("\n", "") for req in requirements]
        if HIPEN_E_DOT in requirements:
            requirements.remove(HIPEN_E_DOT)
    return requirements

setup(
    name="ML PROJECT",
    version="0.0.1",
    author="Nazmul",
    author_email="sobujiiuc73@gmail.com",
    packages= find_packages(),
    install_requires = get_requirements("requirements.txt")
)