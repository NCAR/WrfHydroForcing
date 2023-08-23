from setuptools import setup

setup(
        name='wrf_hydro_mfe',
        version='1.0',
        packages=['core'],
        url='',
        license='',
        author='Ishita Srivastava',
        author_email='ishitas@ucar.edu',
        description='',
        install_requires=['netCDF4', 'numpy', 'mpi4py','ESMPy']
)
