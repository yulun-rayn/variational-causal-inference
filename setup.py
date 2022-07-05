from distutils.core import setup

ext_modules = []
cmdclass = {}

setup(
    name="vci",
    version="1.0.0",
    description="",
    url="https://github.com/yulun-rayn/variational-causal-inference",
    author="Yulun Wu",
    author_email="yulun_wu@berkeley.edu",
    license="MIT",
    packages=["vci"],
    cmdclass=cmdclass,
    ext_modules=ext_modules
)
