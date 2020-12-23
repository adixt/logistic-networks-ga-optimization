function individuals = GenerateIndividuals(LA_nom, generationSize)

    columns = size(LA_nom, 2);
    rows = size(LA_nom, 1);
    individuals = zeros(rows, columns, generationSize);

    for individualNo = 1:generationSize
        individuals(:, :, individualNo) = MutateIndividuals(LA_nom);
    end

    return;
end
