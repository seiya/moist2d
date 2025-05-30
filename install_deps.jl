import Pkg

code = read("moist.jl", String)

mods = Set{String}()

for line in eachline(IOBuffer(code))
    s = strip(line)
    if startswith(s, "using ") || startswith(s, "import ")
        parts = split(s, r"[ ,]+")
        for part in parts[2:end]
            m = match(r"^([A-Za-z_]\w*)", part)
            if m !== nothing
                push!(mods, m.captures[1])
            end
        end
    end
end

for pkg in mods
    fp = Base.find_package(pkg)
    if fp === nothing || (isa(fp, String) && isempty(fp))
        println("Adding ", pkg)
        Pkg.add(pkg)
    end
end

Pkg.resolve()

