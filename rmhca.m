% logistic-networks-ga-optimization

clc;
clear all;

beginApplicationTic = tic();

chartVisibility = true;
fileGenerator = true;
generations = 5000;
stepRange = 10;
learningBreak = 500;

methodType = 'bbo';
topologyFilename = 'topologies/2_100_a.mat';

% GA parameters
generationSize = 10;
mutationProbability = 0.01;
alpha = 1;
beta = 2;
resultGenerationNo = 0;

x_inf = 100000; % infinite stock at external sources
generationsDone = 0;

% TOPOLOGY GENERATOR

% network = TopologyGenerator(simTime, m);

filename = 'test3.mat';
network = FileTopologyGenerator(filename);
%tutaj inicjujemy siec 1
network.n = 8;
network.TC = zeros(network.n + network.m, network.n);
TC_UNIT = 0.0415;
% %tutaj Ivanow 1
% network.TC(6, 1) = 32 * TC_UNIT;
% network.TC(1, 3) = 77 * TC_UNIT;
% network.TC(3, 4) = 58 * TC_UNIT;
% network.TC(7, 2) = 50 * TC_UNIT;
% network.TC(2, 3) = 59 * TC_UNIT;
% network.TC(3, 5) = 60 * TC_UNIT;

%tutaj Ivanow 2
network.TC(9, 1) = 32 * TC_UNIT;
network.TC(1, 3) = 77 * TC_UNIT;
network.TC(3, 4) = 58 * TC_UNIT;
network.TC(10, 2) = 50 * TC_UNIT;
network.TC(2, 3) = 59 * TC_UNIT;
network.TC(3, 5) = 60 * TC_UNIT;

network.TC(9, 2) = 85 * TC_UNIT;
network.TC(9, 6) = 124 * TC_UNIT;
network.TC(10, 1) = 75 * TC_UNIT;
network.TC(10, 6) = 42 * TC_UNIT;

network.TC(1, 7) = 92 * TC_UNIT;
network.TC(2, 7) = 88 * TC_UNIT;
network.TC(6, 3) = 82 * TC_UNIT;
network.TC(6, 7) = 47 * TC_UNIT;

network.TC(3, 8) = 88 * TC_UNIT;
network.TC(7, 4) = 69 * TC_UNIT;
network.TC(7, 5) = 54 * TC_UNIT;
network.TC(7, 8) = 53 * TC_UNIT;
%tutaj Ivanow 3
network.TC(1, 2) = 75 * TC_UNIT;
network.TC(2, 6) = 42 * TC_UNIT;
network.TC(3, 7) = 35 * TC_UNIT;
network.TC(4, 5) = 30 * TC_UNIT;
network.TC(5, 8) = 38 * TC_UNIT;

% %tutaj 3
% network.TC(6, 1) = 20 * TC_UNIT;
% network.TC(7, 2) = 30 * TC_UNIT;
% network.TC(2, 3) = 30 * TC_UNIT;
% network.TC(1, 3) = 50 * TC_UNIT;
% network.TC(1, 2) = 20 * TC_UNIT;
% network.TC(2, 4) = 30 * TC_UNIT;
% network.TC(3, 4) = 50 * TC_UNIT;
% network.TC(4, 5) = 70 * TC_UNIT;
% network.TC(3, 5) = 40 * TC_UNIT;
% network.TC(5, 1) = 80 * TC_UNIT;

% %tutaj 4
% network.TC(6, 1) = 20 * TC_UNIT;
% network.TC(7, 2) = 30 * TC_UNIT;
% network.TC(2, 3) = 30 * TC_UNIT;
% network.TC(1, 3) = 50 * TC_UNIT;
% network.TC(1, 2) = 20 * TC_UNIT;
% network.TC(2, 4) = 30 * TC_UNIT;
% network.TC(3, 4) = 50 * TC_UNIT;
% network.TC(4, 5) = 70 * TC_UNIT;
% network.TC(3, 5) = 40 * TC_UNIT;
% network.TC(5, 1) = 80 * TC_UNIT;
% network.TC(4, 1) = 90 * TC_UNIT;
% network.TC(5, 2) = 70 * TC_UNIT;

% % tutaj 1 new
% network.TC(4, 1) = 20 * TC_UNIT;
% network.TC(5, 2) = 30 * TC_UNIT;
% network.TC(2, 3) = 30 * TC_UNIT;
% network.TC(1, 3) = 50 * TC_UNIT;
% network.TC(1, 2) = 20 * TC_UNIT;

% % tutaj 2
% network.TC(6, 1) = 20 * TC_UNIT;
% network.TC(7, 1) = 30 * TC_UNIT;
% network.TC(1, 4) = 40 * TC_UNIT;
% network.TC(1, 3) = 50 * TC_UNIT;
% network.TC(1, 2) = 20 * TC_UNIT;
% network.TC(1, 5) = 30 * TC_UNIT;

%%

network.LT = zeros(network.n + network.m, network.n);
% %tutaj Ivanow 1
% network.LT(6, 1) = 2;
% network.LT(1, 3) = 4;
% network.LT(3, 4) = 4;
% network.LT(7, 2) = 4;
% network.LT(2, 3) = 4;
% network.LT(3, 5) = 3;

%tutaj Ivanow 2
network.LT(9, 1) = 2;
network.LT(1, 3) = 4;
network.LT(3, 4) = 4;
network.LT(10, 2) = 4;
network.LT(2, 3) = 4;
network.LT(3, 5) = 3;

network.LT(9, 2) = 5;
network.LT(9, 6) = 7;
network.LT(10, 1) = 5;
network.LT(10, 6) = 3;

network.LT(1, 7) = 6;
network.LT(2, 7) = 5;
network.LT(6, 3) = 6;
network.LT(6, 7) = 3;

network.LT(3, 8) = 6;
network.LT(7, 4) = 5;
network.LT(7, 5) = 3;
network.LT(7, 8) = 3;
%tutaj Ivanow 3
network.LT(1, 2) = 4;
network.LT(2, 6) = 4;
network.LT(3, 7) = 3;
network.LT(4, 5) = 2;
network.LT(5, 8) = 2;

% % tutaj 3
% network.LT(6, 1) = 2;
% network.LT(7, 2) = 4;
% network.LT(2, 3) = 3;
% network.LT(1, 3) = 3;
% network.LT(1, 2) = 1;
% network.LT(2, 4) = 3;
% network.LT(3, 4) = 2;
% network.LT(4, 5) = 1;
% network.LT(3, 5) = 4;
% network.LT(5, 1) = 1;

% % tutaj 4
% network.LT(6, 1) = 2;
% network.LT(7, 2) = 4;
% network.LT(2, 3) = 3;
% network.LT(1, 3) = 3;
% network.LT(1, 2) = 1;
% network.LT(2, 4) = 3;
% network.LT(3, 4) = 2;
% network.LT(4, 5) = 1;
% network.LT(3, 5) = 4;
% network.LT(5, 1) = 1;
% network.LT(4, 1) = 3;
% network.LT(5, 2) = 2;

% tutaj 2
% network.LT(6, 1) = 2;
% network.LT(7, 1) = 4;
% network.LT(1, 4) = 1;
% network.LT(1, 3) = 2;
% network.LT(1, 2) = 3;
% network.LT(1, 5) = 4;

% % tutaj 1
% network.LT(4, 1) = 2;
% network.LT(5, 2) = 4;
% network.LT(2, 3) = 3;
% network.LT(1, 3) = 3;
% network.LT(1, 2) = 1;

network.L = max(network.LT(:)); % max lead time
network.simTime = 1000;
network.LA_nom = zeros(network.n + network.m, network.n);
network.LA = zeros(network.n + network.m, network.n, network.simTime + 1);
% %tutaj Ivanow 1
% network.LA_nom(6, 1) = 1;
% network.LA_nom(1, 3) = 0.5;
% network.LA_nom(3, 4) = 1;
% network.LA_nom(7, 2) = 1;
% network.LA_nom(2, 3) = 0.5;
% network.LA_nom(3, 5) = 1;

% %tutaj Ivanow 2
% network.LA_nom(9, 1) = 0.5;
% network.LA_nom(1, 3) = 0.334;
% network.LA_nom(3, 4) = 0.5;
% network.LA_nom(10, 2) = 0.5;
% network.LA_nom(2, 3) = 0.333;
% network.LA_nom(3, 5) = 0.5;

% network.LA_nom(9, 2) = 0.5;
% network.LA_nom(9, 6) = 0.5;
% network.LA_nom(10, 1) = 0.5;
% network.LA_nom(10, 6) = 0.5;

% network.LA_nom(1, 7) = 0.334;
% network.LA_nom(2, 7) = 0.333;
% network.LA_nom(6, 3) = 0.333;
% network.LA_nom(6, 7) = 0.333;

% network.LA_nom(3, 8) = 0.5;
% network.LA_nom(7, 4) = 0.5;
% network.LA_nom(7, 5) = 0.5;
% network.LA_nom(7, 8) = 0.5;
%tutaj Ivanow 3
network.LA_nom(9, 1) = 0.5;
network.LA_nom(1, 3) = 0.334;
network.LA_nom(3, 4) = 0.5;
network.LA_nom(10, 2) = 0.334;
network.LA_nom(2, 3) = 0.333;
network.LA_nom(3, 5) = 0.334;

network.LA_nom(9, 2) = 0.333;
network.LA_nom(9, 6) = 0.334;
network.LA_nom(10, 1) = 0.5;
network.LA_nom(10, 6) = 0.333;

network.LA_nom(1, 7) = 0.25;
network.LA_nom(2, 7) = 0.25;
network.LA_nom(6, 3) = 0.333;
network.LA_nom(6, 7) = 0.25;

network.LA_nom(3, 8) = 0.334;
network.LA_nom(7, 4) = 0.5;
network.LA_nom(7, 5) = 0.333;
network.LA_nom(7, 8) = 0.333;

network.LA_nom(1, 2) = 0.333;
network.LA_nom(2, 6) = 0.333;
network.LA_nom(3, 7) = 0.25;
network.LA_nom(4, 5) = 0.333;
network.LA_nom(5, 8) = 0.333;

% %tutaj 3
% network.LA_nom(6, 1) = 0.5;
% network.LA_nom(7, 2) = 0.5;
% network.LA_nom(2, 3) = 0.5;
% network.LA_nom(1, 3) = 0.5;
% network.LA_nom(1, 2) = 0.5;
% network.LA_nom(2, 4) = 0.5;
% network.LA_nom(3, 4) = 0.5;
% network.LA_nom(4, 5) = 0.5;
% network.LA_nom(3, 5) = 0.5;
% network.LA_nom(5, 1) = 0.5;

% %tutaj 4
% network.LA_nom(6, 1) = 0.334;
% network.LA_nom(7, 2) = 0.334;
% network.LA_nom(2, 3) = 0.5;
% network.LA_nom(1, 3) = 0.5;
% network.LA_nom(1, 2) = 0.333;
% network.LA_nom(2, 4) = 0.5;
% network.LA_nom(3, 4) = 0.5;
% network.LA_nom(4, 5) = 0.5;
% network.LA_nom(3, 5) = 0.5;
% network.LA_nom(5, 1) = 0.333;
% network.LA_nom(4, 1) = 0.333;
% network.LA_nom(5, 2) = 0.333;

% %tutaj 1
% network.LA_nom(4, 1) = 1;
% network.LA_nom(5, 2) = 0.5;
% network.LA_nom(2, 3) = 0.5;
% network.LA_nom(1, 3) = 0.5;
% network.LA_nom(1, 2) = 0.5;

% %tutaj 2
% network.LA_nom(6, 1) = 0.5;
% network.LA_nom(7, 1) = 0.5;
% network.LA_nom(1, 4) = 1;
% network.LA_nom(1, 3) = 1;
% network.LA_nom(1, 2) = 1;
% network.LA_nom(1, 5) = 1;

network.LA(:, :, 1) = network.LA_nom;
network.d = zeros(network.n, network.simTime + 1);
%
%
%dmax = [10; 15; 20; ]; %tutaj 1
dmax = [10; 15; 20; 17; 13; 18; 16; 19]; %tutaj 2, 3, 4

for j = 1:network.simTime + 1
    multiplier = 0.6;

    for k = 1:network.n
        randomDemand = poissrnd(round(multiplier * dmax(k)));

        % check if randomDemand is smaller than dmax
        if (randomDemand > dmax(k)); network.d(k, j) = dmax(k); else; network.d(k, j) = randomDemand; end
    end

end

save(filename, 'network');
network2 = FileTopologyGenerator(filename);
TC = network.TC;
simTime = network.simTime;
n = network.n;
m = network.m;
LT = network.LT;
L = network.L;
LA_nom = network.LA_nom;
LA = network.LA;
d = network.d;

% Initial conditions
time = linspace(0, simTime - 1, simTime); % from 0 to simTime-1
u = zeros(n, simTime + 1);
u_hist = zeros(n, simTime + 1); % order history
x = zeros(n + m, simTime + 1);
y = zeros(n + m, simTime + 1);

dmax = max(d, [], 2); % take the biggest value of each row

% State-space description
% System matrices
B_nom = zeros(n, n, L);
B = zeros(n, n, L, simTime + 1);

% Assuming zero order processing time
B_0 = -LA(1:n, 1:n, 1);

for k = 1:L% index k corresponds to delay k

    for j = 1:n
        t_sum = 0;

        for i = 1:n + m

            if LT(i, j) == k
                t_sum = t_sum + LA(i, j, 1);
            end

        end

        B_nom(j, j, k) = t_sum;
    end

end

B(:, :, :, 1) = B_nom;

% Sum of delay matrices
Lambda = zeros(n);

for k = 1:L% table index k corresponds to delay k
    Lambda = Lambda + B(:, :, k, 1);
end

Lambda = Lambda + B_0;

% Reference stock level for full demand satisfaction
temp = zeros(n);

for k = 1:L% table index k corresponds to delay k
    temp = temp + k * B(:, :, k, 1);
end

dmean = mean(d, 2);
dstd = std(d, 0, 2);
xd_min = ceil((eye(n) + temp) * inv(Lambda) * dmax) + 1;

xd = [xd_min; 0; 0]; % reference stock level (see calculation of xd_min below)

xd_min_source(1:m) = x_inf;
x(1:n + m, 1) = [xd_min; xd_min_source']; % initial stock level

start_xd_min = xd_min;
best_simulation = 0;
a = NetworkSimulator(simTime, n, m, L, LT, LA, d, LA_nom, B_nom, TC);

% Main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a = simulate(a, start_xd_min, LA_nom);
fprintf([repmat('%f\t', 1, size(LA_nom, 2)) '\n'], LA_nom');
fprintf('Mean Bullwhip Effect = %d', a.w4);
initial_HC = a.HC;
a.initialBE = a.w4;
initialBE1 = a.w1;
initialBE2 = a.w2;
initialBE3 = a.w3;
initialBE4 = a.w4;
initialBE5 = a.w5;
a.initialTTC = a.TTC;
fprintf('\nTTC = %d', a.TTC);
a.initialHC = initial_HC;
a.alpha = alpha;
a.beta = beta;

best_fitness = a.fitness;
best_HC = a.HC;
best_TTC = a.TTC;
best_BE1 = a.w1;
best_BE2 = a.w2;
best_BE3 = a.w3;
best_BE4 = a.w4;
best_BE5 = a.w5;
best_LA = a.LA_nom;
best_xd_min = a.xd(1:n);
last_good_xdmin = best_xd_min;

methodTypeChars = char(methodType);

switch methodTypeChars
    case char('brute-force')

        for xd_min1 = start_xd_min(1):-stepRange:0
            fprintf('Current: %d', xd_min1);
            tic;

            for xd_min2 = start_xd_min(2):-stepRange:0
                temp_res = zeros(start_xd_min(3), 1);
                idx = 1:stepRange:start_xd_min(3);

                parfor (xd_min3 = 1:numel(idx), 16)
                    b = a;
                    temp_xd_min = xd_min;
                    temp_xd_min(1) = xd_min1;
                    temp_xd_min(2) = xd_min2;
                    temp_xd_min(3) = idx(xd_min3);

                    b = simulate(b, temp_xd_min);

                    if (b.isCrashed == true)
                        continue
                    end

                    temp_res(xd_min3) = b.HC;
                end

                if (isempty(temp_res(temp_res > 0)))
                    continue
                end

                if (best_HC > min(best_HC, min(temp_res(temp_res > 0))))
                    best_HC = min(best_HC, min(temp_res(temp_res > 0)));
                end

                if (best_TTC > min(best_TTC, min(temp_res(temp_res > 0))))
                    best_TTC = min(best_TTC, min(temp_res(temp_res > 0)));
                end

            end

            best_TTC
            fprintf(' in %d seconds\n', toc);
        end

    case char('rmhca')
        stop = 0;

        for k = 1:generations
            q = round(rand(1) * (n - 1)) + 1;
            param = round(rand(1) * stepRange) + 1;
            xd_min(q) = best_xd_min(q) - param;
            a = simulate(a, xd_min);

            if (a.isCrashed ~= true && a.HC < best_HC)
                best_HC = a.HC;
                best_TTC = a.TTC;
                best_xd_min = a.xd(1:n);
                last_good_xdmin = xd_min;
                stop = 0;
            else
                stop = stop + 1;
                xd_min = last_good_xdmin;
            end

            if stop == learningBreak
                generationsDone = k;
                break;
            end

        end

    case char('ga')
        individuals = GenerateIndividuals(LA_nom, generationSize);

        best_set = zeros(n, 1);
        stop = 0;

        gaProcessArchive = GAProcessArchive(topologyFilename);

        while generationsDone < generations
            generationsDone = generationsDone + 1;

            gaProcessArchive.bestHCCourse(generationsDone) = best_HC;
            gaProcessArchive.bestTTCourse(generationsDone) = best_TTC;
            gaProcessArchive.bestFitnessCourse(generationsDone) = best_fitness;

            individualFitnesses = zeros(1, generationSize);
            individualUsed = zeros(1, generationSize);

            unproductivity = 0;

            for individualNo = 1:generationSize
                LAmutated = individuals(:, :, individualNo);
                a = simulate(a, start_xd_min, LAmutated);
                individualFitnesses(individualNo) = a.fitness;

                gaProcessArchive.fitnessCourse(:, end + 1) = [generationsDone; a.fitness];
                gaProcessArchive.HCCourse(:, end + 1) = [generationsDone; a.HC];
                gaProcessArchive.TTCCourse(:, end + 1) = [generationsDone; a.TTC];

                if individualFitnesses(individualNo) > best_fitness
                    best_fitness = individualFitnesses(individualNo);
                    best_xd_min = xd_min;
                    best_HC = a.HC;
                    best_LA = a.LA_nom;
                    best_TTC = a.TTC
                    best_BE1 = a.w1;
                    best_BE2 = a.w2;
                    best_BE3 = a.w3;
                    best_BE4 = a.w4
                    best_BE5 = a.w5;
                    resultGenerationNo = generationsDone

                    gaProcessArchive.bestHCFixes(:, end + 1) = [generationsDone; best_HC];
                    gaProcessArchive.bestTTCFixes(:, end + 1) = [generationsDone; best_TTC];
                    gaProcessArchive.bestFitnessFixes(:, end + 1) = [generationsDone; best_fitness];

                    stop = 0;
                else
                    unproductivity = unproductivity + 1;

                    if unproductivity == generationSize
                        stop = stop + 1;
                    end

                end

            end

            pie((individualFitnesses - 0.99) / (sum(individualFitnesses) - generationSize))
            drawnow

            if stop == learningBreak
                break;
            end

            fitSum = sum(individualFitnesses);
            pairs = zeros(n + m, n, 1);
            addedFirst = false;

            while true
                fitRandom = rand;

                for individualNo = 1:generationSize
                    summ = (sum(individualFitnesses(1:individualNo)));
                    sumdiv = summ / fitSum;

                    if fitRandom < (sum(individualFitnesses(1:individualNo)) / fitSum)

                        if (individualUsed(individualNo) == 1)
                            break;
                        end

                        ttt = size(pairs, 3);

                        if addedFirst == true
                            pairs(:, :, ttt + 1) = individuals(:, :, individualNo);
                        else
                            pairs(:, :, ttt) = individuals(:, :, individualNo);
                            addedFirst = true;
                        end

                        individualUsed(individualNo) = 1;
                        break;
                    end

                end

                if size(pairs, 3) == generationSize
                    break;
                elseif size(pairs, 3) == generationSize - 1

                    for individualNo = 1:individualNo

                        if (individualUsed(individualNo) == 0)
                            ttt = size(pairs, 3);
                            pairs(:, :, ttt + 1) = individuals(:, :, individualNo);
                            individualUsed(individualNo) = 1;
                            break;
                        end

                    end

                end

            end

            i = round(rand(1) * (n - 1)) + 1;

            for index = 1:2:generationSize
                individuals(:, 1:i, index) = pairs(:, 1:i, index);
                individuals(:, i + 1:end, index) = pairs(:, i + 1:end, index + 1);
                individuals(:, 1:i, index + 1) = pairs(:, 1:i, index + 1);
                individuals(:, i + 1:end, index + 1) = pairs(:, i + 1:end, index);
            end

            for individualNo = 1:generationSize

                for node = 1:n
                    randomMutation = rand;

                    if randomMutation < mutationProbability
                        laToMutate = individuals(:, :, individualNo);
                        muteated = GenerateIndividuals(laToMutate, 1);
                        individuals(:, :, individualNo) = muteated;
                    end

                end

            end

        end

    case char('bbo')
        ProbFlag = false; %true or false, whether or not to use probabilities to update emigration rates.
        DisplayFlag = true;
        pmodify = 1; % habitat modification probability
        pmutate = 0.005; % initial mutation probability
        popsize = 10; % total population size
        Maxgen = 50; % generation count limit
        individuals = GenerateIndividuals(LA_nom, popsize);
        sizes = size(individuals(:, :, 1));
        numVar = sizes(2); % number of genes in each population member (equal column size in LA)

        Population = CostFunction(a, start_xd_min, individuals);

        if Population(1).cost > best_fitness
            best_fitness = Population(1).cost;
            best_LA = Population(1).chrom;
            a = simulate(a, start_xd_min, best_LA);
            best_TTC = a.TTC;
            best_BE1 = a.w1;
            best_BE2 = a.w2;
            best_BE3 = a.w3;
            best_BE4 = a.w4;
            best_BE5 = a.w5;
        end

        MinCost = [Population(1).cost];
        BE = [Population(1).BE];
        AvgCost = ComputeAveCost(Population);
        Keep = 2; % elitism parameter: how many of the best habitats to keep from one generation to the next
        lambdaLower = 0.0; % lower bound for immigration probabilty per gene
        lambdaUpper = 1; % upper bound for immigration probabilty per gene
        dt = 1; % step size used for numerical integration of probabilities
        I = 1; % max immigration rate for each island
        E = 1; % max emigration rate, for each island
        P = popsize; % max species count, for each island

        % Initialize the species count probability of each habitat
        % Later we might want to initialize probabilities based on cost
        Prob = zeros(popsize, 1);

        for j = 1:length(Population)
            Prob(j) = 1 / length(Population);
        end

        % Begin the optimization loop
        for GenIndex = 1:Maxgen
            chromKeep = zeros(sizes(1), sizes(2), Keep);
            costKeep = zeros(Keep, 1);
            % Save the best habitats in a temporary array.
            for j = 1:Keep
                chromKeep(:, :, j) = Population(j).chrom;
                costKeep(j) = Population(j).cost;
            end

            % Map cost values to species counts.
            for i = 1:length(Population)

                if Population(i).cost < inf
                    Population(i).SpeciesCount = P - i;
                else
                    Population(i).SpeciesCount = 0;
                end

            end

            % Compute immigration rate and emigration rate for each species count.
            % lambda(i) is the immigration rate for habitat i.
            % mu(i) is the emigration rate for habitat i.
            lambda = zeros(length(Population), 1);
            mu = zeros(length(Population), 1);

            for i = 1:length(Population)
                lambda(i) = I * (1 - Population(i).SpeciesCount / P);
                mu(i) = E * Population(i).SpeciesCount / P;
            end

            if ProbFlag == true
                ProbDot = zeros(length(Population), 1);
                % Compute the time derivative of Prob(i) for each habitat i.
                for j = 1:length(Population)
                    % Compute lambda for one less than the species count of habitat i.
                    lambdaMinus = I * (1 - (Population(j).SpeciesCount - 1) / P);
                    % Compute mu for one more than the species count of habitat i.
                    muPlus = E * (Population(j).SpeciesCount + 1) / P;
                    % Compute Prob for one less than and one more than the species count of habitat i.
                    % Note that species counts are arranged in an order opposite to that presented in
                    % MacArthur and Wilson's book - that is, the most fit
                    % habitat has index 1, which has the highest species count.
                    if j < length(Population)
                        ProbMinus = Prob(j + 1);
                    else
                        ProbMinus = 0;
                    end

                    if j > 1
                        ProbPlus = Prob(j - 1);
                    else
                        ProbPlus = 0;
                    end

                    ProbDot(j) = -(lambda(j) + mu(j)) * Prob(j) + lambdaMinus * ProbMinus + muPlus * ProbPlus;
                end

                % Compute the new probabilities for each species count.
                Prob = Prob + ProbDot * dt;
                Prob = max(Prob, 0);
                Prob = Prob / sum(Prob);
            end

            % Now use lambda and mu to decide how much information to share between habitats.
            lambdaMin = min(lambda);
            lambdaMax = max(lambda);

            Island = zeros(sizes(1), numVar, Maxgen);

            for k = 1:length(Population)

                if rand > pmodify
                    continue;
                end

                % Normalize the immigration rate.
                lambdaScale = lambdaLower + (lambdaUpper - lambdaLower) * (lambda(k) - lambdaMin) / (lambdaMax - lambdaMin);
                % Probabilistically input new information into habitat i
                for j = 1:numVar

                    if rand < lambdaScale
                        % Pick a habitat from which to obtain a feature
                        RandomNum = rand * sum(mu);
                        Select = mu(1);
                        SelectIndex = 1;

                        while (RandomNum > Select) && (SelectIndex < popsize)
                            SelectIndex = SelectIndex + 1;
                            Select = Select + mu(SelectIndex);
                        end

                        tmp = Population(SelectIndex).chrom(:, j);
                        Island(:, j, k) = Population(SelectIndex).chrom(:, j);
                    else
                        tmp = Population(k).chrom(:, j);
                        Island(:, j, k) = Population(k).chrom(:, j);
                    end

                end

            end

            if ProbFlag == true
                % Mutation
                Pmax = max(Prob);
                MutationRate = pmutate * (1 - Prob / Pmax);
                % Mutate only the worst half of the solutions
                Population = PopSort(Population);

                for k = round(length(Population) / 2):length(Population)

                    for parnum = 1:numVar

                        if MutationRate(k) > rand
                            Island(k, parnum) = floor(MinParValue + (MaxParValue - MinParValue + 1) * rand);
                        end

                    end

                end

            end

            % Replace the habitats with their new versions.
            for k = 1:length(Population)
                tmp = Island(:, :, k);
                Population(k).chrom = tmp;
            end

            % Make sure each individual is legal.
            ValidateAllocationForPopulation(Population);
            % Calculate cost
            individualsTmp = zeros(sizes(1), sizes(2), length(Population));

            for z = 1:length(Population)
                individualsTmp(:, :, z) = Population(z).chrom;
            end

            Population = CostFunction(a, start_xd_min, individualsTmp);
            % Sort from best to worst
            Population = PopSort(Population);

            if Population(1).cost > best_fitness
                best_fitness = Population(1).cost;
                best_LA = Population(1).chrom;
                a = simulate(a, start_xd_min, best_LA);
                best_TTC = a.TTC;
                best_BE1 = a.w1;
                best_BE2 = a.w2;
                best_BE3 = a.w3;
                best_BE4 = a.w4;
                best_BE5 = a.w5;
            end

            % Replace the worst with the previous generation's elites.
            popsize = length(Population);

            for k = 1:Keep
                Population(n - k + 1).chrom = chromKeep(:, :, k);
                Population(n - k + 1).cost = costKeep(k);
            end

            % Make sure the population does not have duplicates.
            Population = ClearDups(Population);
            % Sort from best to worst
            Population = PopSort(Population);
            % Compute the average cost
            [AverageCost, nLegal] = ComputeAveCost(Population);
            % Display info to screen
            MinCost = [MinCost Population(1).cost];
            BE = [BE Population(1).BE];
            AvgCost = [AvgCost AverageCost];

            if DisplayFlag == true
                disp(['The best and mean of Generation # ', num2str(GenIndex), ' are ', ...
                        num2str(MinCost(end)), ' and ', num2str(AvgCost(end))]);
            end

        end

        Conclude(DisplayFlag, popsize, Maxgen, Population, nLegal, MinCost, BE);
        MinParValue = 0.001;
        MaxParValue = 1;
        %         % Obtain a measure of population diversity
        %         for k = 1:length(Population)
        %             Chrom = Population(k).chrom;
        %
        %             for j = MinParValue:MaxParValue
        %                 %iterate each column
        %                 col = size(Chrom,2);
        %                 indices = [];
        %                 for z =1:col
        %                         colValues = Chrom(:, z);
        %                         tmpIndices =  find(colValues == j);
        %                         in = [indices tmpIndices] ;
        %                       indices = in;
        %                 end
        %
        %                 CountArr(k, j) = length(indices); % array containing gene counts of each habitat
        %             end
        %
        %         end
        %
        %         Hamming = 0;
        %
        %         for m = 1:length(Population)
        %
        %             for j = m + 1:length(Population)
        %
        %                 for k = MinParValue:MaxParValue
        %                     Hamming = Hamming + abs(CountArr(m, k) - CountArr(j, k));
        %                 end
        %
        %             end
        %
        %         end
        %
        %         if DisplayFlag == true
        %             disp(['Diversity measure = ', num2str(Hamming)]);
        %         end

    otherwise
        disp('Optimization method not found! Try again!\n')
end

a = simulate(a, best_xd_min, best_LA);

TimeSpent = toc(beginApplicationTic);

% DEBUG HOLDING COST
a.xd(1:n)
round(a.initialTTC)
round(a.TTC)
a.satisfiedRate
fprintf('Mean Bullwhip Effect = %d after %d sec', a.w4, TimeSpent);
fprintf('\nTTC = %d', a.TTC);

TimeSpent;

if fileGenerator == true
    directory = 'reports';
    filename = sprintf('report_%d_%d_', simTime, m);
    extension = 'txt';

    filepath = [directory '/' filename methodType datestr(now, '_mmdd_HHMMSS') '.' extension];
    fid = fopen(filepath, 'w');

    print_initial_HC = round(initial_HC);
    print_best_HC = round(best_HC);
    print_initial_mf = ((inv((eye(n) + temp) * inv(Lambda)) * (start_xd_min - 1)) - dmean) ./ dstd;
    print_best_mf = ((inv((eye(n) + temp) * inv(Lambda)) * (best_xd_min - 1)) - dmean) ./ dstd;
    print_satisfied_rate = a.satisfiedRate;

    if ~isempty(topologyFilename)
        fprintf(fid, [topologyFilename '\n\n']);
    end

    fprintf(fid, 'Generations\n%d/%d (break: %d)\n', generationsDone, generations, learningBreak);
    fprintf(fid, 'Step range\n%d\n', stepRange);
    fprintf(fid, 'Computing time\n%f\n', TimeSpent);
    fprintf(fid, 'Holding cost\n%d -> %d\n', print_initial_HC, print_best_HC);
    fprintf(fid, 'Satisfied rate\n%f\n', print_satisfied_rate);

    if strcmp(methodType, 'ga')
        fprintf(fid, '-> Genetic algorithm parameters\n');
        fprintf(fid, 'Alpha\n%f\n', a.alpha);
        fprintf(fid, 'Beta\n%f\n', a.beta);
        fprintf(fid, 'Fitness\n%f\n', a.fitness);
        fprintf(fid, 'Generation size\n%f\n', generationSize);
        fprintf(fid, 'Mutation probability\n%f\n', mutationProbability);
        fprintf(fid, 'Result generation number\n%f\n', resultGenerationNo);

        archiveFilepath = [directory '/' filename methodType datestr(now, '_mmdd_HHMMSS') '.mat'];
        save(archiveFilepath, 'gaProcessArchive');
    end

    fprintf(fid, 'Reference stock levels\nFrom To\n');
    fprintf(fid, '%d\t\t%d \n', [start_xd_min best_xd_min]');
    fprintf(fid, 'Transportation Cost\nFrom To\n');
    fprintf(fid, '%d\t\t%d \n', [a.initialTTC best_TTC]');
    fprintf(fid, 'Bullwhip Effect 1\nFrom To\n');
    fprintf(fid, '%d\t\t%d \n', [initialBE1 best_BE1]');
    fprintf(fid, 'Bullwhip Effect 2\nFrom To\n');
    fprintf(fid, '%d\t\t%d \n', [initialBE2 best_BE2]');
    fprintf(fid, 'Bullwhip Effect 3\nFrom To\n');
    fprintf(fid, '%d\t\t%d \n', [initialBE3 best_BE3]');
    fprintf(fid, 'Bullwhip Effect 4\nFrom To\n');
    fprintf(fid, '%d\t\t%d \n', [initialBE4 best_BE4]');
    fprintf(fid, 'Bullwhip Effect 5\nFrom To\n');
    fprintf(fid, '%d\t\t%d \n', [initialBE5 best_BE5]');

    for node = 1:n
        fprintf(fid, 'LA changed at node%d \nFrom To\n', node);
        fprintf(fid, '%d\t\t%d \n', [LA_nom(:, node) best_LA(:, node)]');
    end

    fprintf(fid, 'Magic factors\nFrom To\n');
    fprintf(fid, '%d\t\t%d \n', [print_initial_mf print_best_mf]');
    fclose(fid);
end

if chartVisibility == true
    a = simulate(a, start_xd_min, LA_nom);
    plotter = ChartGenerator(time, simTime, n, a.x, a.u_hist, a.d, a.h);
    stock_level(plotter, 5);
    order_quantity(plotter, 6);
    demand(plotter, 7);
    satisfied_demand(plotter, 8);

    a = simulate(a, start_xd_min, best_LA);

    plotter = ChartGenerator(time, simTime, n, a.x, a.u_hist, a.d, a.h);
    stock_level(plotter, 1);
    order_quantity(plotter, 2);
    demand(plotter, 3);
    satisfied_demand(plotter, 4);
end
