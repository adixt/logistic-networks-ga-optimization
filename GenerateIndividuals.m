function individuals = GenerateIndividuals(n, m, LA_nom, generationSize)
    individuals = zeros(n + m, n, generationSize);

    for individualNo = 1:generationSize
        individuals(:, :, individualNo) = MutateIndividuals(n, m, LA_nom);
    end

end

function LAmutated = MutateIndividuals(n, m, LAToMutate)
    % THIS CODE MUTATES LA
    LAmutated = zeros(n + m, n);

    for node = 1:n
        LAcolumn = LAToMutate(:, node);
        LAcolumnMutated = zeros(n + m, 1);
        indiciesWithNonzeroLA = find(LAcolumn);
        indiciesWithNonzeroLA = shuffle(indiciesWithNonzeroLA);
        sizeLA = size(indiciesWithNonzeroLA, 1);

        if (sizeLA == 1)
            LAmutated(:, node) = LAcolumn;
            continue
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

        LAmutated(:, node) = LAcolumnMutated;

    end

    ValidateAllocation(LAmutated);
end

function v = shuffle(v)
    v = v(randperm(length(v)));
end

function ValidateAllocation(LAmutated)
    % Verify if allocation correct - elements in each column should sum up to 1 or 0
    columns = size(LAmutated, 2);
    rows = size(LAmutated, 1);

    for j = 1:columns
        temp = 0;

        for i = 1:rows
            temp = temp + LAmutated(i, j);
        end

        if (temp == 0) || (temp == 1)
            %fprintf('Proper allocation in column: %d\n', j);
        else
            error('Improper allocation in column: %d', j);
        end

    end

end
