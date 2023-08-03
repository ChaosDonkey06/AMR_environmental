clear all

%% active parameters
    beta0 = 1;
    gamma0 =  1;
    
%% fixed parameters
    n0 = 1e+3;

%% initial conditions
    vec_ini=zeros(12,1);
    vec_ini(1:2)=[990;10];

%% initial guess

    theta0 = [beta0;gamma0];
    
%% data

    x = load('synthetic_data_sir1June19_2013.txt','-ascii');
    timewindow = x(:,1);
    
%% optimization
    
    options    = optimset('Display','iter','MaxFunEvals',1e4,'MaxIter',1e4,'TolFun',1e-30,'TolX',1e-30);
    [thetahat] = fminsearch(@(theta)costfunction1(x(:,2),timewindow,vec_ini,n0,theta),theta0,options);
    
%% standard errors: Fisher

    [t1,y1] = ode45(@sir_singleoutbreak_sensitivity_eqns,timewindow,vec_ini,[],thetahat(1),thetahat(2),n0);
    ws = y1(:,2);
    sigma0 = sqrt((1/(length(x(:,2))-2))*sum((x(:,2)-ws).^2));
    ChiM = y1(:,7:8);
    cov_mat = sigma0^2*inv(ChiM'*ChiM);
    sterrvec = sqrt(diag(cov_mat));
    
    
    
%% standard errors: Bootstrap
    bootstrap_size = 25;
    thetahat_matrix = zeros(bootstrap_size,length(thetahat));
    rsdl = sqrt(size(x,1)/(size(x,1)-length(thetahat)))*(x(:,2) - ws);  % residuals
        
    tic
    figure
    for m=1:bootstrap_size
            disp(sprintf('bootstrap iteration = %d',m))
        %%% create SRS with replacement from residuals
            srs_residuals = randsample(rsdl,length(rsdl),true);
        %%% create bootstrap sample points
            y_srs = ws + srs_residuals;
        %%% compute OLS estimate
            options = [];
            [thetahat_m] =  fminsearch(@(theta)costfunction1(y_srs,timewindow,vec_ini,n0,theta),theta0,options); 
            thetahat_matrix(m,:) = thetahat_m; 
            [tm,ym] = ode45(@sir_singleoutbreak_sensitivity_eqns,timewindow,vec_ini,[],thetahat_m(1),thetahat_m(2),n0);
            hold on
            plot(tm,ym(:,2),'-','Color',[0.65 0.65 0.65])
    end
    toc
    hold on 
    plot(x(:,1),x(:,2),'xk')
    hold off

    %%% computing bootstrap estimates

        q_boot = (1/size(thetahat_matrix,1))*sum(thetahat_matrix)';
    
    %%% computing covariance matrix
    
        cov_q = zeros(size(thetahat_matrix,2));

        for m=1:size(thetahat_matrix,1)
            q = thetahat_matrix(m,:)';
            cov_q = cov_q + (q - q_boot)*(q - q_boot)';
        end
        cov_q = (1/(size(thetahat_matrix,1)-1))*cov_q;

%% display estimates
    disp('Fisher Estimates')
    thetahat
    sterrvec
    %
    disp('Bootstrap Estimates')
    q_boot
    se_vec = sqrt(diag(cov_q))
        
  
 %% sensitivity functions       

    %%% tradiational 
    figure
    plot(t1,y1(:,7:8))
    title('Traditional Sensitivity Functions')
    legend('w.r.t. \beta','w.r.t. \gamma')
    
    
    %%% relative
    figure
    xbeta = (y1(:,7)*thetahat(1))./(y1(:,2));
    xgamma = (y1(:,8)*thetahat(2))./(y1(:,2));
    plot(t1,[xbeta xgamma])
    title('Relative Sensitivity Functions')
    legend('w.r.t. \beta','w.r.t. \gamma')
    
    %%% generalized
    
    F1 = sigma0^2*inv(ChiM'*ChiM);
    gs_mtrx = zeros(size(x,1),2);
    
    for k1 = 1:size(x,1)
        gsf = zeros(2,1);
        for i=1:k1
            gsf = gsf + (1/sigma0^2)*F1*ChiM(i,:)'.*ChiM(i,:)';
        end
        gs_mtrx(k1,:) = gsf';
    end
    
    figure
    plot(x(:,1),gs_mtrx)
    title('Generalized Sensitivity Functions')
    legend('w.r.t. \beta','w.r.t. \gamma')
    



    