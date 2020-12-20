function [Population] = ClearDups(Population)

    % Make sure there are no duplicate individuals in the population.
    % This logic does not make 100% sure that no duplicates exist, but any duplicates that are found are
    % randomly mutated, so there should be a good chance that there are no duplicates after this procedure.
    for i = 1:length(Population)
        Chrom1 = Population(i).chrom;

        for j = i + 1:length(Population)
            Chrom2 = Population(j).chrom;

            % if isequal(Chrom1, Chrom2)
            max = size(Population(j).chrom, 2);

            column = ceil(max * rand);
            LAcolumn = Population(j).chrom(:, column);
            indiciesWithNonzeroLA = find(LAcolumn);
            sizeLA = size(indiciesWithNonzeroLA, 1);

            LAcolumnMutated = zeros(size(Population(j).chrom, 1), 1);
            indiciesWithNonzeroLA = shuffle(indiciesWithNonzeroLA);

            if (sizeLA == 1)
                sizeOne = true;

                while (sizeOne == true)
                    column = ceil(max * rand);
                    LAcolumn = Population(j).chrom(:, column);
                    indiciesWithNonzeroLA = find(LAcolumn);
                    sizeLA = size(indiciesWithNonzeroLA, 1);

                    if (sizeLA ~= 1)
                        sizeOne = false;
                    end

                end

            end

            for s = 1:sizeLA
                idx = indiciesWithNonzeroLA(s);
                valueToMutate = LAcolumn(idx);

                if (s ~= sizeLA)
                    valueMutated = rand(1) * valueToMutate;
                    LAcolumnMutated(idx) = valueMutated;
                else
                    allAboveThis = sum(LAcolumnMutated);
                    last = 1 - allAboveThis;
                    LAcolumnMutated(idx) = last;
                end

            end

            Population(j).chrom(:, column) = LAcolumnMutated;
            %end

        end

    end

    return;
end

function v = shuffle(v)
    v = v(randperm(length(v)));
end
