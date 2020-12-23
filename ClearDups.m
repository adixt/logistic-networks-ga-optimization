function [Population] = ClearDups(Population)

    % Make sure there are no duplicate individuals in the population.
    % This logic does not make 100% sure that no duplicates exist, but any duplicates that are found are
    % randomly mutated, so there should be a good chance that there are no duplicates after this procedure.
    for i = 1:length(Population)
        Chrom1 = Population(i).chrom;

        for j = i + 1:length(Population)
            Chrom2 = Population(j).chrom;

            if isequal(Chrom1, Chrom2)
                % MUTATE RANDOM COLUMN
                max = size(Population(j).chrom, 2);

                columnNo = ceil(max * rand);
                LAcolumn = Population(j).chrom(:, columnNo);
                indiciesWithNonzeroLA = find(LAcolumn);
                sizeLA = size(indiciesWithNonzeroLA, 1);

                if (sizeLA == 1)
                    sizeOne = true;

                    while (sizeOne == true)
                        columnNo = ceil(max * rand);
                        LAcolumn = Population(j).chrom(:, columnNo);
                        indiciesWithNonzeroLA = find(LAcolumn);
                        sizeLA = size(indiciesWithNonzeroLA, 1);

                        if (sizeLA ~= 1)
                            sizeOne = false;
                        end

                    end

                end

                LAcolumnMutated = MutateColumn(Population(j).chrom, columnNo);
                Population(j).chrom(:, columnNo) = LAcolumnMutated;

                % % OR MUTATE ENTIRE CHROM
                % LAMutated = MutateIndividuals(Population(j).chrom);
                % Population(j).chrom = LAMutated;
                % %
            end

        end

    end

    return;
end
