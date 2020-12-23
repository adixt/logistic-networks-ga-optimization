function v = ShuffleVector(v)
    v = v(randperm(length(v)));
end
