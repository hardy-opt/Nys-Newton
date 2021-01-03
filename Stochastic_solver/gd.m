% This file is the part of Emp_risk library
% Tankaria Hardik,
% PhD student, Kyoto University (2020-July)

function [w,infos] = gd(problem,options)

% Stochastic gradient descent (SGD) algorithm

% Set dimensions and samples
d = problem.dim();
n = problem.samples();

% store details

% initialize

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(options.step_alg,'decay-4')
    iter = 1;
else
    iter = 0;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



epoch = 0;

grad_calc_count = 0; % number of gradient calculation

w = options.w_init; % initial point

test_cost = 0; %problem.test_cost(w);

m = n; % number of iteration per epoch

% store first details
clear infos;
[infos, f_val,~,gnorm,optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0,0,randn(d,1),0,test_cost);

% display details

if options.verbose > 0
    fprintf('GD: Epoch = %03d, cost = %.5e,elapsed_time = %.4e, gnorm=%.5e, optgap = %.4e\n', epoch, f_val,0, gnorm,optgap);
end


% set start time
start_time = tic();

            % Main loop
       while epoch < options.max_epoch %|| gnorm<10^-12
           
            perm_idx = rngperm(n);

            for i=1:n
            
            %ind = perm_idx(i):perm_idx(i); %% ind denotes the batch indices but we take 1 index per iterations so our batch size is 1.
            %To perform batch sgd then change appropriate indices by changing
            %batch size and no of samples % ind = perm_idx(st_i):perm_idx(st_i+batch_size)
            % i:i so that full gradient can be calculate in the same definitions
            
            ind = perm_idx(i);
            
            g = problem.grad(w,ind);
            
            % update step-size

               
            stp = options.stepsize_alg(iter, options);
            
            v =  stp*g;   
            
            w = w - v;
            
            test_cost = 0; %problem.test_cost(w);
            
            iter = iter +1;
           
            end
            
            
        %Total number of epoch
        epoch=epoch+1;
           
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + i;        
        

        % store infos
        [infos, f_val,~,gnorm,optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time,iter,v,stp,test_cost);        

        % display infos
        if options.verbose > 0
            fprintf('GD: Epoch = %03d, cost = %.5e, time = %.4e, ||g||=%.5e, optgap = %.4e\n', epoch, f_val, elapsed_time, gnorm,optgap);
        end
       end
        
       
        if epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
        end
end