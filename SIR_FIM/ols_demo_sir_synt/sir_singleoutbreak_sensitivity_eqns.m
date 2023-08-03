function w = sir_singleoutbreak_sensitivity_eqns(t,x,b,c,n)

w = zeros(12,1);
%
%%% model equations
%

w(1) = -b*x(1)*x(2)/n;
w(2) = b*x(1)*x(2)/n-c*x(2);
w(3) = c*x(2);

%
%%% matrices
%

%
dfdx      = zeros(3,3);
dfdx(1,1) = -b*x(2)/n;
dfdx(1,2) = -b*x(1)/n;
%

%
dfdx(2,1) = b*x(2)/n;
dfdx(2,2) = b*x(1)/n-c;
%

%
dfdx(3,2) = c;
%

%
dfdtheta      = zeros(3,3);
dfdtheta(1,1) = -x(1)*x(2)/n;
dfdtheta(1,3) = b*x(1)*x(2)/(n^2);
%

%
dfdtheta(2,1) = x(1)*x(2)/n;
dfdtheta(2,2) = -x(2);
dfdtheta(2,3) = -b*x(1)*x(2)/(n^2);
%

%
dfdtheta(3,2)= x(2);
%

%
%%% sensitivity equations
%
mtrx = dfdx *[x(4:6)';x(7:9)';x(10:12)'] + dfdtheta;
w(4:6) = mtrx(1,:);
w(7:9) = mtrx(2,:);
w(10:12)= mtrx(3,:);



