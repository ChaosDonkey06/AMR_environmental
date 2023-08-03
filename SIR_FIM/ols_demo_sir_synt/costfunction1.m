function val = costfunction1(data,timepoints,initc_vec,n,q)

    [t1,y1]=ode45(@sir_singleoutbreak_sensitivity_eqns,timepoints,initc_vec,[],q(1),q(2),n);
    
    w = y1(:,2);
    
    val = sum((data-w).^2);