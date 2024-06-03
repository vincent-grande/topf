using Distances
#using Plots
using Ripserer
using CSV
using DataFrames
using Base
using Base

input_string = read(Base.stdin, String)

csv_reader = CSV.File(IOBuffer(input_string), header=false)
ambient_dim = length(csv_reader[1])
points = NTuple{ambient_dim,Float64}[]
thresh = parse(Float64, ARGS[1])
dim_max = parse(Float64, ARGS[2])
type = ARGS[3]
dim_max = Int(dim_max)
for row in csv_reader
    push!(points, NTuple{ambient_dim,Float64}([entry for entry in row]))
end
if thresh == 0
    if type == "rips"
        RipsComplex = Ripserer.Rips(points)
    else
        RipsComplex = Ripserer.Alpha(points; threshold=10000000)
    end
else
    if type == "rips"
        RipsComplex = Ripserer.Rips(points; threshold=thresh, sparse=true)
    else
        RipsComplex = Ripserer.Alpha(points; threshold=thresh)#, sparse=true)
    end
end
RipsResults = Ripserer.ripserer(RipsComplex; modulus=:3, reps=true, dim_max=dim_max + 1, alg=:involuted)
CSV.write(Base.stdout, permutedims(DataFrame([[HomClass[1], HomClass[2], [vertices(rep) for rep in HomClass.representative], [1 for rep in HomClass.representative]] for HomClass in RipsResults[1]], :auto)), header=false)
print("newfile")
CSV.write(Base.stdout, permutedims(DataFrame([[HomClass[1], HomClass[2], [vertices(rep.simplex) for rep in HomClass.representative], [Int(rep[2]) for rep in HomClass.representative]] for HomClass in RipsResults[2]], :auto)), header=false)
if dim_max == 2
    print("newfile")
    CSV.write(Base.stdout, permutedims(DataFrame([[HomClass[1], HomClass[2], [vertices(rep.simplex) for rep in HomClass.representative], [Int(rep[2]) for rep in HomClass.representative]] for HomClass in RipsResults[3]], :auto)), header=false)
end