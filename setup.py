from setuptools import setup
from Cython.Build import cythonize

setup(name='sloop',
      packages=['sloop'],
      version='0.1',
      description='Spatial Language OOPOMDP',
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'pomdp_py==1.2.0',
          'opencv-python',  # for some tests
          'docutils',
          'gensim',
          'kiwisolver',
          'matplotlib',
          'numpy',
          'pandas',
          'scikit-learn',
          'scipy',
          'sklearn',
          'torch',
          'spacy',
          'nltk',
          'prettytable'
      ],
      ext_modules=cythonize(['sloop/oopomdp/models/transition_model.pyx',
                             'sloop/oopomdp/models/components/grid_map.pyx'],
                            build_dir="build",
                            compiler_directives={"language_level": "3"}))
