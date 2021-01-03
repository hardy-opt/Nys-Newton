% This file is the part of Emp_risk library
% Tankaria Hardik,
% PhD student, Kyoto University (2020-July)

function [w,infos] = svrg(problem,options)

% Stochastic gradient descent (SGD) algorithm

% Set dimensions and samples
d = problem.dim();
n = problem.samples();

% store details

% initialize

epoch = 0;

grad_calc_count = 0; % number of gradient calculation

w = options.w_init; % initial point

test_cost = 0;%problem.test_cost(w);

m = 2*n; % number of iteration per epoch

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(options.step_alg,'decay-4')
    iter = 1;
else
    iter = 0;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%                 ful1= problem.full_grad(w);
%                 [o,e]=size(w);
%                 w2 = randn(o,e);
%                 ful2 = problem.full_grad(w2);
%                 L = norm(ful1 - ful2)/norm(w - w2);
%                 fprintf('Lipschitz constant L=%.5e\n',L);


% store first details
clear infos;
[infos, f_val,~,gnorm,optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0,0,randn(d,1),0,test_cost);

% set start time
start_time = tic();
    
    % display infos
    if options.verbose > 0
      fprintf('SVRG: Epoch = %03d, cost = %.5e,elapsed_time = %.4e, gnorm=%.3e\n', epoch, f_val, 0, gnorm);
    end  
    
t=1;
T = options.max_epoch; %(Max no of epoch)== Termination criteria


f_valn=f_val+1;
    while (t<=T) || f_val - f_valn>10^-22
    %while f_valn - f_val>10^-1
    %if mod(t,K)==0
    w0=w;
    full_grad = problem.full_grad(w0);
    grad_calc_count = grad_calc_count + n;
    %end
    
    options.w = w; % For Nakai's step_size;
    options.n = n;

      %      if epoch ==  0
                
      %          perm_idx = rngperm(n);
                
      %          for i = 1:n
                
      %          ind = perm_idx(i); % i:i so that full gradient can be calculate in the same definitions
                
      %          g = problem.grad(w,ind);
                
                % update step-size
                
                
      %          stp = options.stepsize_alg(iter, options);
                
      %          v =  stp*g;
                
                %update
      %          w = w - v;
                
      %          iter = iter + 1;
      %          end
                
                 % count gradient evaluations
      %           grad_calc_count = grad_calc_count + i;
        
      %      else
                perm_idx = rngperm(n);  %%% Take 2n rand permutation for next 2n iterations.
                for i=1:m  %%% m = 2n Number of iterations per epoch
                
                ind = perm_idx(i); %% ind denotes the batch indices but we take 1 index per iterations so our batch size is 1.
                %To perform batch svrg then change appropriate indices by changing
                %batch size and no of samples % ind = perm_idx(st_i):perm_idx(st_i+batch_size)
                % i:i so that full gradient can be calculate in the same definitions
                
                g = problem.grad(w,ind) - problem.grad(w0,ind) + full_grad;
                
                
                % update step-size
                stp = options.stepsize_alg(iter, options);
            
                v =  stp*g;
                
                w = w - v;
               
                iter = iter + 1;
               
                end
                
                % count gradient evaluations
                grad_calc_count = grad_calc_count + 2*m;
                
     %       end
     
      test_cost = problem.test_cost(w);
         
        %Total number of epoch
        epoch=epoch+1;
           
        % measure elapsed time
        elapsed_time = toc(start_time);
        
               
        f_valn=f_val;
        % store infos
        [infos, f_val,~,gnorm,optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time,iter,v, stp,test_cost);        

        % display infos
        if options.verbos > 0
            fprintf('SVRG: Epoch = %03d, cost = %.5e, time = %.4e, ||g|| =%.5e, optgap = %.5e\n', epoch, f_val, elapsed_time, gnorm, optgap);
        end
           
        
        if epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
        end
         
         t = t + 1;
    end
end
