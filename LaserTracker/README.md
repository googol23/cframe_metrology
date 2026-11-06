# Moving-plane interferometer animation

This small project animates the basic two-wave interferometer model for a plane reflector that moves from `-λ/4` to `+λ/4` relative to its mean position.

## What the three panels show

- **Upper right (1):** the longitudinal standing-wave intensity along the optical propagation axis. Each vertical slice represents one transverse plane across the beam. The red vertical line marks the fixed sensor plane, and the moving orange line marks the reflector. The black region beyond the reflector contains no field in this opaque-mirror model.
- **Lower right (2):** the sensor response at that plane, displayed as a vertically oriented rectangle whose shade changes from black to white, with its normalized intensity value beside it.
- **Left, spanning both rows and two-thirds of the width (3):** the emitted and reflected electric fields as functions of position. The orange solid line is the current reflector plane, and the orange dotted lines mark its travel limits.

The upper-right and left panels use longitudinal position in wavelengths, `x / λ`. The upper-right vertical direction is only a schematic beam cross-section; it is not a second calculated spatial coordinate. The sensor is at `x = -1 λ` by default.

## Physics in the model

For a normally incident plane wave, the emitted field is represented by

[
E_i(x,t)=Acos(kx-omega t)
]

and the reflected field by

[
E_r(x,t)=rAcos(k(2x_m-x)-omega t+phi_r),
]

where (x_m) is the reflector position, (r) is the reflected field-amplitude ratio, and (phi_r) is the reflection phase shift.

Averaging over the much faster optical oscillation gives an intensity proportional to

[
I(x)propto 1+r^2+2rcosleft(2k(x_m-x)+phi_r
ight).
]

Moving the reflector by (Delta x) changes the round-trip optical path by (2Delta x), so the interference phase changes by

[
Deltaphi=rac{4piDelta x}{λ}.
]

The move from (-λ/4) to (+λ/4) spans a total distance of (λ/2), corresponding to one full intensity fringe at a fixed observation point.

The default reflection phase is (180^circ), appropriate to the idealized conducting-plane boundary condition. Change it with `--reflection-phase-deg` to explore another phase convention.

## Sensor-plane model

The upper-left panel shows how the standing-wave intensity varies along the optical axis in the illuminated region in front of the reflector. The pattern terminates at the moving orange reflector line; the black region behind that opaque plane has no incident or reflected field. For the ideal plane wave in this model, intensity is uniform across each transverse plane. A sensor placed at fixed longitudinal position `x_s` therefore measures one brightness value rather than a transverse fringe pattern:

```text
Δφ_sensor = 4π(x_m - x_s) + φ_r
I_sensor = [1 + r² + 2r cos(Δφ_sensor)] / (1 + r)².
```

Here, `x_m` and `x_s` are measured in wavelengths. The red vertical line identifies `x_s` in the longitudinal map, while the lower-right panel displays `I_sensor` as a uniform grayscale value. Moving the reflector by `λ/4` changes the round-trip phase by `π`, taking an ideal sensor from a dark state to a bright state or vice versa.

## Model limits

This is an idealized monochromatic, normally incident plane-wave model intended to explain interference and displacement sensitivity. The animation slows the mechanical motion enormously relative to the optical oscillation. The electric-field curves are illustrative snapshots; a real detector averages over many optical cycles. The upper-right panel repeats the one-dimensional longitudinal intensity vertically only to depict transverse planes. The model does not include beam envelopes, angular misalignment, wavefront curvature, beam divergence, surface roughness, vibration, detector noise, or the AT930's proprietary electronics.

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib
- Pillow

## Create and activate an environment

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

From this project directory:

```bash
python plot_interferometer.py
```

This writes `moving_plane_interference.gif` in the current directory. Choose another output path with:

```bash
python plot_interferometer.py --output output/interference.gif
```

Optional controls:

```bash
python plot_interferometer.py --frames 200 --fps 24
python plot_interferometer.py --workers 8
python plot_interferometer.py --reflectivity 0.8
python plot_interferometer.py --reflection-phase-deg 0
python plot_interferometer.py --sensor-position-lambda -1.5
```

Frame rendering uses multiple worker processes because Matplotlib rendering is CPU-bound and is not thread-safe. By default, `--workers 0` automatically uses up to four processes. Set `--workers 1` for serial rendering or choose an explicit process count based on the available CPU and memory. Each worker reuses one figure for its assigned frame chunk; temporary PNG frames are removed automatically after the GIF is assembled.
