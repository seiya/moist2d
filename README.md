# moist2d
Two-dimensional moist simulation model with fully compressible non‑hydrostatic equations

## Running tests

Execute the following commands from the project directory. Activate the
project with `--project` so that the bundled dependencies are used:

```bash

# Run tests in single precision (Float32)
julia --project test/runtests.jl

# Run tests in double precision (Float64)
julia --project test/runtests.jl Float64
```

## Installing dependencies

All required packages are listed in `Project.toml`. Before running the
simulation or the tests, activate the project and instantiate the
environment:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

Running Julia with `--project` ensures that the correct versions of all
dependencies, including `MPI`, are available.
