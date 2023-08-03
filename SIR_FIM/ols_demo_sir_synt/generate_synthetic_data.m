clear all
%%% true parameters
    beta0=2.2;
    gamma0=1.1;
    N=1e3;
    perc1 = 0.01;
    theta0=[beta0;gamma0];
    sigma0 = sqrt(0.1);
    
%%% initial conditions
    initial_vector = zeros(12,1);
    initial_vector(1:2) = [(1-perc1)*N;perc1*N]; 
    tspan = linspace(0,15,50);
 
%%% numerical solutions
    [t x] = ode45(@sir_singleoutbreak_sensitivity_eqns,tspan,initial_vector,[],theta0(1),theta0(2),N);

%%% synthetic data
    yobs1 = x(:,2) + sigma0.*randn(size(x(:,2),1),1);
    
    figure;
    plot(t,x(:,2),'-b',t,yobs1,'rx')
     
%%% save output files

name1 = 'synthetic_data_sir1.txt';
name3 = 'synthetic_dataset1_sir.mat';

m1 = [t yobs1];

% % save(name1,'-ascii','m1');
% % save(name3,'beta0','gamma0','N','perc1','initial_vector','tspan','m1');
