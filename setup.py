from setuptools import setup

exec(open('liset/version.py').read())

setup(
        name='liset',
        description= 'A simple clustering tool to assist the Lifecycle Screening of Emerging Technologies (LiSET)', 
        long_description=open('README.md').read(),
        url='https://github.com/majeau-bettez/LiSET',
        author='Guillaume Majeau-Bettez',
        author_email='guillaume.majeau-bettez@ntnu.no',
        version=__version__,
        packages=['liset',],
        install_requires = ['numpy >= 1.14.2',
                            'pandas >= 0.22.0',
                            'scipy >= 1.0.0',
                            'jenkspy >= 0.1.4',
                            'matplotlib >=2.1.2'],
        license='GPLv3',
    )
