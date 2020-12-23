function LAmutated = MutateIndividuals(LAToMutate)
    % THIS CODE MUTATES LA
    columns = size(LAToMutate, 2);
    rows = size(LAToMutate, 1);
    LAmutated = zeros(rows, columns);

    for columnNo = 1:columns
        LAmutated(:, columnNo) = MutateColumn(LAToMutate, columnNo);
    end

    ValidateAllocation(LAmutated);
    return;
end
