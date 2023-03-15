# WrfHydroForcing
### Python forcing engine for WRF-Hydro

The WRF-Hydro Forcing Engine is primarily used to prepare input forcing variables for WRF-Hydro model simulations. 
It is an object-oriented design (OOD) framework designed specifically for the WRF-Hydro modeling system. The framework has been written in Python version 3 and relies heavily on a few external software packages to properly read in native forcing products, while performing spatial interpolations in a parallel environment, along with downscaling and bias corrections. 
Some of the input forcing products include historical regional and global reanalysis products that can be used for retrospective WRF-Hydro studies. Other products include operational real-time numerical weather prediction (NWP) models that are run at the National Centers for Environmental Prediction (NCEP). These operational products are of interest to real-time forecasting and nowcasting applications of the WRF-Hydro modeling system. Other products include supplemental precipitation products that offer spatially distributed quantitative precipitation estimates from sensors such as doppler radar. The Multi-Radar/Multi-Sensor (MRMS) product from the National Severe Storms Laboratory (NSSL) is a good example of supplemental precipitation the user may wish to ingest into the WRF-Hydro modeling architecture in place of NWP estimated precipitation. Future development will continue to refine the list of NWP products, QPE products, or other forcing datasets of interest.

The following figure shows the overall flow of the Forcing Engine software package, along with the key components that lead to the eventual formation of forcing files that are ingested into the WRF- Hydro modeling system.
![image](https://user-images.githubusercontent.com/36771676/225337722-8275de94-f6ee-41e3-81e9-09906cd7f7c3.png)

The starting point for using this framework is preparation of config files. Sample configs are placed in the Config subdirectory. The earlier version of the code used JSON Configuration. The new code uses YAML configuration and an older version can be converted to the new version by using the following python utility placed under the Utils subdirectory.

Example Usage: 
```python
export PYTHONPATH=$PYTHONPATH:~/git/WrfHydroForcing/
export PYTHONPATH=$PYTHONPATH:~/git/WrfHydroForcing/core
./config2yaml.py ../Test/template_forcing_engine_AnA_v2.config ../Test/template_forcing_engine_AnA_v2_example.yaml
```
Example usage of running the framework:
```python
time mpiexec python3 -E ~/git/WrfHydroForcing/genForcing.py ~/git/WrfHydroForcing/Config/YAML/template_forcing_engine_PRVI_AnA.yaml 3.0 AnA
```

### System Level Requirements
1. The Forcing Engine code is written in Python, but it can only run on a Unix-based system. This is required as additional dependencies only run in this environment as well. For these reasons, the Forcing Engine cannot run in either Windows or Mac OS. The system will need to have Python 3 installed on the system. The current version of this software is not backwards compatible with Python 2. For larger WRF-Hydro modeling domains, it’s encouraged to make sure there is enough memory on the system to accommodate the interpolation from coarse grids to the higher resolution WRF-Hydro geogrid file. 
2. The user is required to have an installation of MPI on the machine they will be executing the Forcing Engine on. For many users, this requirement may already be fulfilled if WRF-Hydro is being ran in parallel using MPI as well. The Python code uses MPI-wrappers to split the processing amongst multiple processors. Even if the user specifies an execution on one processor, MPI software is still being invoked.
3. Installation of Earth System Modeling Framework (ESMF) Software and ESMPy.
4. Installation of wgrib2.
