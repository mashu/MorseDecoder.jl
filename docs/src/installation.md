# Installation

## Layout

Place [MorseSimulator.jl](https://github.com/mashu/MorseSimulator.jl) and [CTCLoss.jl](https://github.com/mashu/CTCLoss.jl) next to MorseDecoder.jl (e.g. all three in the same parent folder).

## From the terminal

From the MorseDecoder.jl directory:

```bash
julia --project=. -e 'using Pkg; Pkg.add(path="../MorseSimulator.jl"); Pkg.develop(path="../CTCLoss.jl"); Pkg.instantiate()'
```

## From the Julia REPL

```julia
using Pkg
Pkg.activate(".")
Pkg.add(path="../MorseSimulator.jl")
Pkg.develop(path="../CTCLoss.jl")
Pkg.instantiate()
```

## Troubleshooting

- **Julia 1.11 + path dependency:** If `Pkg.instantiate()` hits `TypeError: in typeassert, expected String, got a value of type Dict{String, Any}` (Julia bug in `get_uuid_name`), work around by instantiating once with MorseSimulator removed from `[deps]`, then add it back and run without calling `instantiate()` again; or use Julia 1.10 and add MorseSimulator via `Pkg.develop(path="...")` from the REPL.
