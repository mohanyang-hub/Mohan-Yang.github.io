# LESS maize canopy hyperspectral radiative transfer simulation

This folder records a Python workflow generated from the LESS GUI and adapted for batch simulation with `pyLessSDK`.

## Project focus

The script constructs a maize canopy radiative transfer simulation scene in LESS. It imports maize stem and leaf OBJ geometry, places 28 maize plants in a regular canopy layout, defines soil and crop optical properties, configures solar illumination and orthographic observation, and runs hyperspectral image simulation from 400.50 nm to 899.50 nm.

## Main workflow

- Read leaf optical-property text files and calculate forward reflectance, backward reflectance, and transmittance values.
- Create or open a LESS simulation project.
- Clear existing scene elements and user-defined optical properties.
- Add maize stem and leaf geometry from OBJ files.
- Place maize plant instances with specified coordinates and rotations.
- Configure terrain, sun geometry, orthographic spectral sensor, image size, sampling parameters, and observation angle.
- Batch update optical properties and run LESS simulations for each optical-property file.

## Source file

- [`zc-ort19.py`](zc-ort19.py)

## Notes

The script keeps the original local paths used during the experiment, including the LESS installation directory, optical-property text folder, simulation folder, and maize OBJ files. Before running on another computer, these paths should be updated to match the local LESS environment and data locations.
