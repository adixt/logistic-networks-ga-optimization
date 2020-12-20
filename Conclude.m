function Conclude(DisplayFlag, popsize, Maxgen, Population, nLegal, MinCost, BE)

    % Output results of population-based optimization algorithm.

    if DisplayFlag == true
        % Count the number of duplicates
        NumDups = 0;

        for i = 1:popsize
            Chrom1 = sort(Population(i).chrom);

            for j = i + 1:popsize
                Chrom2 = sort(Population(j).chrom);

                if isequal(Chrom1, Chrom2)
                    NumDups = NumDups + 1;
                end

            end

        end

        disp([num2str(NumDups), ' duplicates in final population.']);
        disp([num2str(nLegal), ' legal individuals in final population.']);
        % Display the best solution
        Chrom = sort(Population(1).chrom);
        disp('Best chromosome = ');
        fprintf([repmat('%f\t', 1, size(Chrom, 2)) '\n'], Chrom');
        % Plot some results
        close all;
        figure(9);
        plot([0:Maxgen], MinCost, 'r');
        xlabel('Generation');
        ylabel('Minimum Cost');

        figure(10);
        plot([0:Maxgen], BE, 'g');
        xlabel('Generation');
        ylabel('Bullwhip Effect (w4)');
    end

    return;
end
