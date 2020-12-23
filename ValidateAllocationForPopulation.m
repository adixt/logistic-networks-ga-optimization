function ValidateAllocationForPopulation(population)

    for z = 1:length(population)
        la_pop = population(z).chrom;
        ValidateAllocation(la_pop);
    end

end
