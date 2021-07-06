from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("sloop.oopomdp.models.components",
              ["sloop/oopomdp/models/components/*.pyx"]),
]

setup(name='sloop',
      packages=['sloop'],
      version='0.1',
      description='Spatial Language OOPOMDP',
      install_requires=[
          "pomdp-py==1.2.4.5",
          "numpy>=1.20.3",
          "opencv-contrib-python>=4.5.1.48",
          "opencv-python>=4.1.1.26",
          "pandas>=1.1.2",
          "Pillow>=8.2.0",
          "pygame>=1.9.6",
          "scipy>=1.3.3",
          "seaborn>=0.11.0",
          "spacy==2.2.4",
          "torch==1.8.1",
          "torchvision==0.9.1",
          "allennlp>=2.5.0"
      ],
      ext_modules=cythonize(extensions,
                            build_dir="build",
                            compiler_directives={"language_level": "3"}))
