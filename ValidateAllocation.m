function ValidateAllocation(LAmutated)
    % Verify if allocation correct - elements in each column should sum up to 1 or 0
    columns = size(LAmutated, 2);
    rows = size(LAmutated, 1);

    for j = 1:columns
        temp = 0;

        for i = 1:rows
            temp = temp + LAmutated(i, j);
        end

        columnValueOne = 1.0000; % in Matlab 1.000 ~= 1.000
        tol = 5 * eps(columnValueOne); % A very small value.
        areEssentiallyEqual = ismembertol(columnValueOne, temp, tol);

        if (temp == 0) || (areEssentiallyEqual)
            %fprintf('Proper allocation in column: %d\n', j);
        else
            error('Improper allocation in column: %d', j);
        end

    end

end
