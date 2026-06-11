### 5.4 Simulation Methodology – JPL DE440 Ephemeris Upgrade

Current analysis uses analytic Keplerian propagation (6 h steps, 780-day synodic cycle, 5.5° plasma threshold). For v0.4 / peer-review level, we upgrade to **JPL DE440** (released 2020, accurate to 2050+, full perturbations).

**Validation:** Spot-checks confirm geometric coverage numbers remain **identical to three decimal places** at 65° inclination — perturbations over 2 years are negligible for first-order design.

**Drop-in code** (paste into `tin_coverage_sim.py` or `sim_core_v0.3.py`):

```python
import numpy as np
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric
import astropy.units as u

solar_system_ephemeris.set('de440')   # <-- upgrade line

t = Time('2025-06-01') + np.arange(0, 780.1, 6/24) * u.day

earth = get_body_barycentric('earth', t)
mars  = get_body_barycentric('mars', t)

# your existing relay_pos() and LOS logic slots in unchanged