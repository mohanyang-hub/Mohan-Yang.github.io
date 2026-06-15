# Greenhouse Cucumber Canopy Simulation with Helios

This folder contains the C++ source code and CMake configuration for a Helios-based greenhouse cucumber canopy simulation project.

## Purpose

The project integrates cucumber canopy meshes, greenhouse structural geometry, meteorological input, optical properties, and Helios simulation modules to estimate microclimate-related variables and disease-risk indicators.

## Main Features

- Load cucumber canopy mesh from PLY files.
- Load greenhouse wall and roof geometry from PLY files.
- Assign greenhouse structure optical properties, including PAR reflectivity, PAR transmissivity, and longwave emissivity.
- Read meteorological data from CSV.
- Simulate or estimate:
  - PAR flux
  - Leaf temperature
  - Leaf relative humidity
  - Leaf wetness state
  - Disease-risk metrics
- Use the Helios Visualizer plugin for pseudocolor scene visualization.

## Files

- `main.cpp`: Main simulation workflow.
- `CMakeLists.txt`: Helios project CMake configuration.

## Helios Plugins Used

The CMake configuration uses the following Helios plugins:

- `radiation`
- `plantarchitecture`
- `energybalance`
- `visualizer`
- `voxelintersection`
- `canopygenerator`
- `solarposition`
- `photosynthesis`
- `stomatalconductance`
- `planthydraulics`

## Notes

This code is stored here as project source material for the personal homepage. It depends on a local Helios installation and local geometry/input data paths, so it is not intended to run directly on GitHub Pages.
