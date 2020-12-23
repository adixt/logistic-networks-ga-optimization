function LAcolumnMutated = MutateColumn(LAToMutate, columnNo)
    rows = size(LAToMutate,1);
    LAcolumn = LAToMutate(:, columnNo);
    LAcolumnMutated = zeros(rows, 1);
    indiciesWithNonzeroLA = find(LAcolumn);
    indiciesWithNonzeroLA = ShuffleVector(indiciesWithNonzeroLA);
    sizeLA = size(indiciesWithNonzeroLA, 1);

    if (sizeLA == 1)
        LAcolumnMutated = LAcolumn;
        return;
    end

    for s = 1:sizeLA
        idx = indiciesWithNonzeroLA(s);
        valueToMutate = LAcolumn(idx);

        if (s ~= sizeLA)
            valueMutated = rand(1) * valueToMutate;

            if valueMutated > 0.99
                valueMutated = 0.99;
            end

            if valueMutated < 0.001
                valueMutated = 0.001;
            end

            LAcolumnMutated(idx) = valueMutated;
        else
            allAboveThis = sum(LAcolumnMutated);
            last = 1 - allAboveThis;
            LAcolumnMutated(idx) = last;
        end

    end

    return;
end
