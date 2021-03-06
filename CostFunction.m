function Population = CostFunction(a, start_xd_min, individuals)

    popsize = length(individuals);
    % COST FUNCTION
    parfor i = 1:popsize
        result = simulate(a, start_xd_min, individuals(:, :, i));
        tmp = [];
        tmp.cost = result.fitness;
        tmp.chrom = result.LA_nom;
        tmp.BE = result.w4;
        Population(i) = tmp;
    end

    % Sort the population members from best to worst
    Cost = zeros(1, popsize, 'gpuArray');

    for i = 1:popsize
        Cost(i) = Population(i).cost;
    end

    [Cost, indices] = sort(Cost, 2, 'ascend');
    sizes = size(Population(1).chrom);

    Chroms = zeros(sizes(1), sizes(2), popsize);

    for i = 1:popsize
        Chroms(:, :, i) = Population(indices(i)).chrom;
    end

    BE = zeros(1, popsize);

    for i = 1:popsize
        BE(i) = Population(i).BE;
    end

    for i = 1:popsize
        Population(i).chrom = Chroms(:, :, i);
        Population(i).cost = Cost(i);
        Population(i).BE = BE(i);
    end

end
