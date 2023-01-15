from setuptools import setup, find_packages
setup(name='brain2vec',
      version='0.1',
      description='brain2vec and related models',
      author='Morgan Stuart & Srdjan Lesaja',
      packages=find_packages(),
      install_requires=['numpy', 'pandas',
                'sklearn', 'torch', 'torchvision',
                'mat73',
                'attrs',
                'tqdm',
                'simple-parsing',
                'attrs'])