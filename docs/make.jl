using Documenter
using MorseDecoder

makedocs(
    sitename = "MorseDecoder.jl",
    authors = "Mateusz Kaduk <mateusz.kaduk@gmail.com>",
    repo = "https://github.com/mashu/MorseDecoder.jl/blob/{commit}{path}#{line}",
    format = Documenter.HTML(
        prettyurls = !isempty(get(ENV, "CI", "")),
        canonical = "https://mashu.github.io/MorseDecoder.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Quick start" => "quickstart.md",
        "API" => "api.md",
    ],
    strict = true,
    checkdocs = :exports,
)

# Docs are deployed via GitHub Actions (upload-pages-artifact + deploy-pages).
# To deploy from local: uncomment and run deploydocs() after makedocs().
# deploydocs(repo = "github.com/mashu/MorseDecoder.jl.git", devbranch = "main", push_preview = true)
