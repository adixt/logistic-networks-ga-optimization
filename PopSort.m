function [Population, indices] = PopSort(Population)

    % Sort the population members from best to worst
    popsize = length(Population);
    Cost = zeros(1, popsize);

    for i = 1:popsize
        Cost(i) = Population(i).cost;
    end

    [Cost, indices] = sort(Cost, 2, 'descend');
    sizes = size(Population(1).chrom);

    Chroms = zeros(sizes(1), sizes(2), popsize);

    for i = 1:popsize
        tmp = Population(indices(i)).chrom;
        Chroms(:, :, i) = tmp;
    end

    for i = 1:popsize
        Population(i).chrom = Chroms(:, :, i);
        Population(i).cost = Cost(i);
    end
