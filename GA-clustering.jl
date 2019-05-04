using Plots
using Distributions
using LinearAlgebra
using Random
using Dates
using Clustering

mutable struct Individual
    chrom::Matrix{Float64}
    fitness::Float64
end

function metric(clst::Vector{Matrix{Float64}}, center::Matrix{Float64})
    metric = 0.0
    for clst_idx = 1:length(clst)
        for data_idx = 1:size(clst[clst_idx], 2)
            metric += norm(clst[clst_idx][:,data_idx] - center[:,clst_idx])
        end
    end

    return metric
end

function fitness(clst::Vector{Matrix{Float64}}, cent::Matrix{Float64})
    return 1/metric(clst, cent)
end

function crossover!(chrom_a::Matrix{Float64}, chrom_b::Matrix{Float64})
    r = rand(1:size(chrom_a,2)-1)

    for i = r:size(chrom_a,1)
        chrom_a[:,i], chrom_b[:,i] = chrom_b[:,i], chrom_a[:,i]
    end
end

function mutateat!(chrom::Matrix{Float64}, point::Int)
    δ = rand()
    if rand() < 0.5
        chrom[1,point] += 2*δ*chrom[1,point]
    else
        chrom[1,point] += 2*δ*chrom[1,point]
    end

    if rand() < 0.5
        chrom[2,point] += 2*δ*chrom[2,point]
    else
        chrom[2,point] += 2*δ*chrom[2,point]
    end
end

function roulette_selection(population::Vector{Individual}; rev=false)
    fitness_sum = 0

    for i = 1:length(population)
        fitness_sum += population[i].fitness
    end

    p = [ population[i].fitness for i = 1:length(population) ]
    p /= fitness_sum

    new_population = [Individual(zeros(size(population[1].chrom,1),2), 0.0) for i = 1:length(population)] 

    for i = 1:length(population)
        r = rand()
        for j = 1:length(population)
            if r <= p[j]
                new_population[i].chrom = copy(population[j].chrom)
                new_population[i].fitness = population[j].fitness
                break
            else
                r -= p[j]
            end
        end
    end

    return new_population
end

function assign!(clst::Vector{Matrix{Float64}}, data::Matrix{Float64}, assignments::Vector{Int})
    idx = collect(1:length(assignments))

    for i =  1:length(clst)
        c = data[:,idx[assignments .== i]]
        clst[i] = c
    end
end

function gac(data::Matrix{Float64}, clstnum::Int64; popsize=200, gennum=200, p_c=0.9, p_m=0.001)
    datanum = size(data,2)

    # 初期集団
    pop = Vector{Individual}(undef, popsize)
    idx = collect(1:datanum)
    clst = Vector{Matrix{Float64}}(undef, clstnum)
    for i = 1:popsize
        r = kmeans(data, clstnum, maxiter=0)
        assign!(clst, data, shuffle!(r.assignments))
        pop[i] = Individual(r.centers, fitness(clst, data))
    end


    for gen = 1:gennum

        pop = roulette_selection(pop)

        for i = 1:2:popsize
            if rand() < p_c
                crossover!(pop[i].chrom, pop[i+1].chrom)
            end
        end

        for i = 1:popsize, j = 1:clstnum
            if rand() < p_m
                mutateat!(pop[i].chrom, j)
            end
        end

        for i = 1:popsize
            r = kmeans!(data, pop[i].chrom, maxiter=1)
            assign!(clst, data, r.assignments)
            pop[i].chrom = r.centers
            pop[i].fitness = fitness(clst, pop[i].chrom)
        end
    end

    # 最良個体の抽出
    best = sort(pop, rev=true, by=x->x.fitness)[1]
    r = kmeans!(data, pop[1].chrom, maxiter=1)
    assign!(clst, data, r.assignments)

    return clst, r.centers
end


function main()
    clstnum = 7
    popsize = 50
    chromlen = clstnum
    gennum = 100

    # 適当にデータつくる
    Random.seed!(114)
    _clst1 = rand(Normal(0,1), 2, 50)
    _clst2 = [rand(Normal( 4,1), 1, 50); rand(Normal( 5,1), 1, 50)]
    _clst3 = [rand(Normal(-4,1), 1, 50); rand(Normal( 5,1), 1, 50)]
    _clst4 = [rand(Normal( 4,1), 1, 50); rand(Normal(-5,1), 1, 50)]
    _clst5 = [rand(Normal(-4,1), 1, 50); rand(Normal(-5,1), 1, 50)]
    _clst6 = [rand(Normal( 6,1), 1, 50); rand(Normal( 0,1), 1, 50)]
    _clst7 = [rand(Normal(-6,1), 1, 50); rand(Normal( 0,1), 1, 50)]

    data = hcat(_clst1, _clst2, _clst3, _clst4, _clst5, _clst6, _clst7)

    clst, centers = gac(data, clstnum)

    # クラスタリングなし
    p1 = scatter(data[1,:], data[2,:], legend=:none, markerstrokewidth=0)
    
    # GA-clustering
    p2 = scatter(clst[1][1,:], clst[1][2,:], legend=:none, markerstrokewidth=0)
    for i = 2:clstnum
        p2 = scatter!(clst[i][1,:], clst[i][2,:], legend=:none, markerstrokewidth=0)
    end

    # k-means
    r = kmeans(data,clstnum)
    assign!(clst, data, r.assignments)

    p3 = scatter(clst[1][1,:], clst[1][2,:], legend=:none, markerstrokewidth=0)
    for i = 2:clstnum
        p3 = scatter!(clst[i][1,:], clst[i][2,:], legend=:none, markerstrokewidth=0)
    end
    

    plot(p1,p2, p3, size=(1200,400), layout=(1,3))
    savefig("clustering.svg")
end

main()
