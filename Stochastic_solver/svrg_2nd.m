% This file is the part of Emp_risk library
% Tankaria Hardik,
% PhD student, Kyoto University (2020-July)

function [w,infos] = svrg_2nd(problem,options)

% Stochastic gradient descent (SGD) algorithm

% Set dimensions and samples
d = problem.dim();
n = problem.samples();

% store details

% initialize

epoch = 0;

grad_calc_count = 0; % number of gradient calculation

w = options.w_init; % initial point

test_cost = 0; %problem.test_cost(w);

m = 2*n; % number of iteration per epoch

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(options.step_alg,'decay-4')
    iter = 1;
else
    iter = 0;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% store first details
clear infos;
[infos, f_val,~,gnorm,optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0,0,randn(d,1),0,test_cost);

% set start time
start_time = tic();
    
    % display infos
    if options.verbose > 0
       fprintf('SVRG 2nd: Epoch = %03d, cost = %.5e,time = %.4e, ||g||=%.5e\n', epoch, f_val, 0, gnorm);
    end  
    
 LT = {'r*-','bs-','ko:','g+:','cx:','ro-','bo-','mo-','go-','co-','yo-','y*-','r+--','b+--','m+--','g+--','c+--','y+--','rs:','bs:','ms:','gs:','cs:','ys:','r.','b.','c.','g.','m.','y.'};
 fontsize = 22;
 markersize = 14;
 
    
stpbb=0;
T = options.max_epoch; %(Max no of epoch)== Termination criteria

NONE   = zeros(T,1);
APTA = zeros(T+1,1);
OTA = zeros(T+1,1);
NTWO   = zeros(T,1); 
APTB = zeros(T+1,1); 
OTB = zeros(T+1,1);
NTHREE = zeros(T,1);
APTC = zeros(T+1,1);
OTC = zeros(T+1,1);
NFOUR  = zeros(T,1);
APTD = zeros(T+1,1);
OTD = zeros(T+1,1);

    for t=1:T
    
    
    full_grad = problem.full_grad(w);
    
%     GN = abs(full_grad);
%     set = zeros(1,10);
%     for p = 1:10
%         [a,b] = max(GN);
%         
%         set(p) = b;    %[set b];
%         
%         GN(b) = GN(b) - a;
%         fprintf('max(G)=%d, indices=%d\n',a,b);
%         GN1 = GN;
%         GN = GN1;
%         
%     end
%     set
    grad_calc_count = grad_calc_count + n;
    
    w0=w;
    
    options.w=w; % For Nakai's step_size;
    options.n = n;
    
         %   if epoch ==  0
                
         %       perm_idx = rngperm(n) ; 
                
         %       for i = 1:n
                
         %       ind = perm_idx(i); %% ind denotes the batch indices but we take 1 index per iterations so our batch size is 1.
                %To perform batch svrgbb then change appropriate indices by changing
                %batch size and no of samples % ind = perm_idx(st_i):perm_idx(st_i+batch_size)
                % i:i so that full gradient can be calculate in the same definitions
                
         %       g = problem.grad(w,ind);
                
                % update step-size
              
         %       stp = options.stepsize_alg(iter, options);
              
                
         %       iter = iter + 1;
         %       
         %       v =  stp*g;
                
                %update
         %       w = w - v;
         %       end
                
                % count gradient evaluations
         %       grad_calc_count = grad_calc_count + i;
                
         %       bb=0;
                
          %  else
                
                perm_idx = rngperm(n);  %%% Take 2n rand permutation for next 2n iterations.
                
                for i=1:m  %%% m = 2n Number of iterations per epoch
                
                ind = perm_idx(i); %% ind denotes the batch indices but we take 1 index per iterations so our batch size is 1.
                %To perform batch svrg then change appropriate indices by changing
                %batch size and no of samples % ind = perm_idx(st_i):perm_idx(st_i+batch_size)
                % i:i so that full gradient can be calculate in the same definitions
                
                grad_w = problem.grad(w,ind);
                grad_0 = problem.grad(w0,ind);
                vec = w - w0; % x^k - x^m
                
                if i==1
                    
%                     E = set(1:7);
%                     F = set(1:5);
%                     G = set(1:3);
%                     
%                     
%                     [ah,fn1,apta,orta] = problem.app_hess(w0,1:n,set);
%                     [ah,fn2,aptb,ortb] = problem.app_hess(w0,1:n,E);
%                     [ah,fn3,aptc,ortc] = problem.app_hess(w0,1:n,F);
%                     [ah,fn4,aptd,ortd] = problem.app_hess(w0,1:n,G);
%                     
%                     NONE(t) = fn1;   % k = 10
%                     APTA(t+1) = APTA(t) + apta; 
%                     OTA(t+1) = OTA(t) + orta; 
%                     
%                     
%                     NTWO(t) = fn2; % k = 7
%                     APTB(t+1) = APTB(t) + aptb; 
%                     OTB(t+1) = OTB(t) + ortb; 
%                     
%                     NTHREE(t) = fn3; % k = 5;
%                     APTC(t+1) = APTC(t) + aptc; 
%                     OTC(t+1) = OTC(t) + ortc; 
%                     
%                     
%                     NFOUR(t) = fn4;   % k = 3;
%                     APTD(t+1) = APTD(t) + aptd; 
%                     OTD(t+1) = OTD(t) + ortd; 
%                     
                       %grad_m = problem.grad(w_m,ind); 
                    if n<d % Since Hessian is same for the loop, we calculate Hessian at the beginning of the loop.
                        
                        H = problem.partial_hess(w0,1:n); %% Full Hessian at w0 (x^m)               
                        
                    
                    elseif d<n
                        
                        H = problem.full_hess(w0);
                        
                       % fprintf('set = [%d, %d]\n',set(1),set(2));
                        
                        
                    end
                    
                end
                
                if n<d
                        Hv = problem.partial_hess_vec(vec,1:n,H);
                        Ht = problem.partial_hess(w0,ind);  %% Stochastic Hessian at w0 (x^m)
                        Htk = problem.partial_hess_vec(vec,ind,Ht);
                elseif d<n
                        Hv = H*vec;
                        Ht = problem.hess(w0,ind); 
                        Htk = Ht*vec;
                end
                %g = problem.grad(w,ind) - problem.grad(w0,ind) + full_grad + problem.hess_vec() + problem.hess_vec();
                
                bb = Hv-Htk;
                
                
                g = grad_w - grad_0 + full_grad + bb;
                
                % update step-size
                stp = options.stepsize_alg(iter, options);
            
               % stpbb=norm(stp*bb);
                
                v =  stp*g;
                
                w = w - v;
                
                
                
                iter = iter + 1;
                end
                
                % count gradient evaluations
                grad_calc_count = grad_calc_count + 2*m;
                
                
           % end
           
           test_cost = problem.test_cost(w);
         
        %Total number of epoch
        epoch=epoch+1;
           
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + i;        
        
        
       
        
        % store infos
        [infos, f_val,~,gnorm,optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time,iter,(v),stp,test_cost);        

        % display infos
        if options.verbos > 0
            fprintf('SVRG 2nd: Epoch = %03d, cost = %.5e, time = %.4e, ||bb|| =%.5e, ||g|| =%.5e, optgap = %.5e\n', epoch, f_val, elapsed_time,norm(bb), gnorm, optgap);
        end
          
        
        if epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
        end
    end
%     figure;
%     semilogy(1:T,NONE(1:T),LT{1},1:T,NTWO(1:T),LT{2},1:T,NTHREE(1:T),LT{3},1:T,NFOUR(1:T),LT{4},'MarkerSize', markersize, 'Linewidth', 2); hold on;
%     xlabel('  Epoch ','FontSize',30);
%     ylabel(' Frobenious Norm with k ','FontSize',30);
%     legend('k=10','k=7','k=5','k=3');
%    % set(gca,'yscale','log');
%     %set(gca, 'FontSize', 30);   
%     hold off;
%     
%     
%     
%     figure; semilogy(1:T+1,APTA(1:T+1),LT{1},1:T+1,APTB(1:T+1),LT{2},1:T+1,APTC(1:T+1),LT{3},1:T+1,APTD(1:T+1),LT{4},1:T+1,OTA(1:T+1),LT{5},'MarkerSize', markersize, 'Linewidth', 2); 
%     xlabel('  Epoch ','FontSize',30);
%     ylabel('  Time: original Hess. & Approx. Hess ','FontSize',30);
%     legend('k=10 Approx','k=7 Approx','k=5 Approx','k=3 Approx','Orig Hess');
%     hold off;
%   
end
