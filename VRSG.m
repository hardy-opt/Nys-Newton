% min f(x)=\|Ax-b\|^2   A\in R^{m \times n}
% m is the number of data
m=100;
n=1000;

A=rand(m,n);
b=rand(m,1);

T=1000;
K=m;
Ts=100;

t0=0.001;
x0=zeros(n,1);

% (normal) variance reduction stochastic gradient method with diminishing
% step length
rng(1);
e1=zeros(T,1);
x=x0;
t1=t0/m;
full_nf=A'*(A*x-b);
bx=x0;
for i=1:T
    if  mod(i,K)==0
        bx=x;
        full_nf=A'*(A*x-b);

        fprintf(' %d: i=%d,  obj= %f \n', i, tmp, error);
    end
    tmp=floor(rand(1)*m)+1;
    nfi=(A(tmp,:)*x-b(tmp))*A(tmp,:)';
    if i<Ts
        g=nfi;
        tk=t0/sqrt(i);
    else
        g=nfi-(A(tmp,:)*bx-b(tmp))*A(tmp,:)'+full_nf;
        t1=2*t0/m;
        tk=t1/sqrt(i);
        %tk=0.001;
    end
    
 
    %tk=0.01;
    x=x-tk*g;
    res=A*x-b;
    error=0.5*res'*res;
    %fprintf(' %d: i=%d,  obj= %f \n', i, tmp, error);
    e1(i)=error;
end

fprintf('\n');

% variance reduction stochstic gradient with adaptive step length
rng(1);
e2=zeros(T,1);
x=x0;
t1=t0/m;
full_nf=A'*(A*x-b);
bx=x0;
tdata=zeros(T,1);
tdatamin=zeros(T,1);
for i=1:T
    if  mod(i,K)==0
        bx=x;
        full_nf=A'*(A*x-b);

        fprintf(' %d: i=%d,  obj= %f, t=%f \n', i, tmp, error,tk);
    end
    tmp=floor(rand(1)*m)+1;
    nfi=(A(tmp,:)*x-b(tmp))*A(tmp,:)';
    if i<Ts*10
        g=nfi;
        tk=t0/sqrt(i);
    else
        g=nfi-(A(tmp,:)*bx-b(tmp))*A(tmp,:)'+full_nf;
        %t1=2*t0/m;
        alpha=0.000000001/(sqrt(i)*norm(bx-x,inf));
        tk=min(max(m*t0/(sqrt(i)*m),10^(-5)), max(max(2*t0/(sqrt(i)*m),10^(-7)),alpha));
    end
    tdata(i)=tk;
    tdatamin(i)=2*t0/(sqrt(i)*m);
    %tk=t1/sqrt(i);
    %tk=0.01;
    x=x-tk*g;
    res=A*x-b;
    error=0.5*res'*res;

    e2(i)=error;
end
size(e2)
%plot(1000:T,tdata(1000:T),1000:T,tdatamin(1000:T)); % plot step length
semilogy(100:T, e1(100:T),'b', 100:T, e2(100:T), 'r'); % plot error (proposed one is red)
    