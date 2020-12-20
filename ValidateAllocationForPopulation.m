function ValidateAllocationForPopulation(ppp)

    for z = 1:length(ppp)
        la_pop = ppp(z).chrom;

        % Verify if allocation correct - elements in each column should sum up to 1 or 0
        columns = size(la_pop, 2);
        rows = size(la_pop, 1);

        for j = 1:columns
            temp = 0;

            for i = 1:rows
                temp = temp + la_pop(i, j);
            end

            if (temp == 0) || (temp == 1)
                %fprintf('Proper allocation in column: %d\n', j);
            else
                error('Improper allocation in column: %d', j);
            end

        end

    end

end