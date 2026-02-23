# Naming scheme for 4DVAR main files

`main_SIMTYPE_OBSTYPE.py`

`SIMTYPE`:
    - `co2sul`: full 4DVAR with CO2 and SO2 species
    - if the file is parallized, then it will have an additional `p` in front of `SIMTYPE`

`OBSTYPE`:
    - `perf`: perfect observations (no noise)
    - `white`: white noise applied to observations
    - `ar1`: AR(1) process noise applied to observations
    - `ar2`: AR(2) process noise applied to observations
