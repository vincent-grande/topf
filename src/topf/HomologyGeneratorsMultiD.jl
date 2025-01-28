start_time = time()
using Distances
#using Plots
using Ripserer
using CSV
using DataFrames
using Base
end_time = time()
import_time = end_time - start_time
start_time = time()
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
end_time = time()
reading_time = end_time - start_time
start_time = time()
if thresh == 0
    if type == "rips"
        RipsComplex = Ripserer.Rips{Int128}(points)
    else
        RipsComplex = Ripserer.Alpha{Int128}(points; threshold=10000000)
    end
else
    if type == "rips"
        RipsComplex = Ripserer.Rips{Int128}(points; threshold=thresh, sparse=true)
    else
        RipsComplex = Ripserer.Alpha{Int128}(points; threshold=thresh)#, sparse=true)
    end
end
end_time = time()
complex_construction_time = end_time - start_time
RipsResults = Ripserer.ripserer(RipsComplex; modulus=:3, reps=true, dim_max=dim_max + 1, alg=:involuted)
start_time = end_time
end_time = time()
homology_time = end_time - start_time
num_entries = length(RipsResults[1])
subset_size = min(1000, max(min(50, num_entries), Int(ceil(0.1 * num_entries))))
subset = RipsResults[1][end-subset_size+1:end]
CSV.write(Base.stdout, permutedims(DataFrame([[HomClass[1], HomClass[2], [vertices(rep) for rep in HomClass.representative], [1 for rep in HomClass.representative]] for HomClass in subset], :auto)), header=false)
print("newfile")
#only consider first ten percent of homology classes
num_entries = length(RipsResults[2])
subset_size = min(1000, max(min(50, num_entries), Int(ceil(0.1 * num_entries))))
subset = RipsResults[2][end-subset_size+1:end]
CSV.write(Base.stdout, permutedims(DataFrame([[HomClass[1], HomClass[2], [vertices(rep.simplex) for rep in HomClass.representative], [Int(rep[2]) for rep in HomClass.representative]] for HomClass in subset], :auto)), header=false)
if dim_max == 2
    print("newfile")
    num_entries = length(RipsResults[3])
    subset_size = min(1000, max(min(50, num_entries), Int(ceil(0.1 * num_entries))))
    subset = RipsResults[3][end-subset_size+1:end]
    CSV.write(Base.stdout, permutedims(DataFrame([[HomClass[1], HomClass[2], [vertices(rep.simplex) for rep in HomClass.representative], [Int(rep[2]) for rep in HomClass.representative]] for HomClass in subset], :auto)), header=false)
end
start_time = end_time
end_time = time()
print("newfile")
print("import_time: ", import_time, " reading time: ", reading_time, " complex_construction_time: ", complex_construction_time, " homology_time: ", homology_time, " output_time: ", end_time - start_time)