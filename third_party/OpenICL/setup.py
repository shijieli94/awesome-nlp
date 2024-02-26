from setuptools import find_packages, setup
from setuptools.command.install import install


class DownloadNLTK(install):
    def run(self):
        self.do_egg_install()
        import nltk

        nltk.download("punkt")


def do_setup():
    setup(
        name="openicl",
        version="0.1.7",
        description="An open source framework for in-context learning.",
        url="https://github.com/Shark-NLP/OpenICL",
        author="Zhenyu Wu, Yaoxiang Wang, Zhiyong Wu, Jiacheng Ye",
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        cmdclass={"download_nltk": DownloadNLTK},
        install_requires=open("requirements.txt", encoding="utf-8").read().splitlines(),
        python_requires=">=3.8.0",
        packages=find_packages(exclude=["test*", "paper_test*"]),
        keywords=["AI", "NLP", "in-context learning"],
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
        ],
    )


if __name__ == "__main__":
    do_setup()
