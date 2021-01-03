% This file is the part of Emp_risk library
% Tankaria Hardik,
% PhD student, Kyoto University (2020-July)

function [w,infos] = svrg_abs_bb(problem,options)

% Stochastic gradient descent (SGD) algorithm

% Set dimensions and samples
d = problem.dim();
n = problem.samples();

% store details

% initialize

epoch = 0;

grad_calc_count = 0; % number of gradient calculation

w = options.w_init; % initial point

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
[infos, f_val,~,gnorm,optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0,0,randn(d,1),0,0);

% set start time
start_time = tic();
    
    % display infos
    if options.verbose > 0
       fprintf('SVRG ABS BB: Epoch = %03d, cost = %.5e, elapsed_time = %.4e, gnorm=%.5e\n', epoch, f_val, 0, gnorm);
    end  
    
    
A_new = 0;   
Atk_new = 0;
T = options.max_epoch; %(Max no of epoch)== Termination criteria

    for t=1:T
    
         if epoch == 0
             
             full_grad = problem.full_grad(w);
             
         else
             
            % store full gradient
            full_grad_old = full_grad;

            % compute full gradient
            full_grad = problem.full_grad(w); %x^m
  
            % automatic step size selection based on Barzilai-Borwein (BB)
            s_diff = w - w0; %x^m - x^(m-1)
            s2 = s_diff'*s_diff;
            %w_m_1 = w;
            w_m = w0; %x^(m-1)
            y_diff = full_grad - full_grad_old;
            
            A_new = ((s_diff' * y_diff)/s2); %% B^m
            
             %A_new = 0.0*A_new+1*A;
            
         end
         
    %Store w
    w0=w; 
    grad_calc_count = grad_calc_count + n;
    
     options.w=w; % For Nakai's step_size;
     options.n = n;
            

            if epoch ==  0
                
                perm_idx = rngperm(n);
                
                for i = 1:m
                
                ind = perm_idx(i); %% ind denotes the batch indices but we take 1 index per iterations so our batch size is 1.
                %To perform batch svrgbb then change appropriate indices by changing
                %batch size and no of samples % ind = perm_idx(st_i):perm_idx(st_i+batch_size)
                % i:i so that full gradient can be calculate in the same definitions
                
                g = problem.grad(w,ind)-problem.grad(w0,ind)+full_grad;
                
                % update step-size
                
                stp = options.stepsize_alg(iter, options);
                
                
                iter = iter + 1;
                
                v =  stp*g;
                
                %update
                w = w - v;
                end
                
                bb=0; % For first comment/fprintf
        
            else
                
                perm_idx = rngperm(n);  %%% Take 2n rand permutation for next 2n iterations.
                
                for i=1:m  %%% m = 2n Number of iterations per epoch
                
                ind = perm_idx(i); %% ind denotes the batch indices but we take 1 index per iterations so our batch size is 1.
                %To perform batch svrg then change appropriate indices by changing
                %batch size and no of samples % ind = perm_idx(st_i):perm_idx(st_i+batch_size)
                % i:i so that full gradient can be calculate in the same definitions
                
                
                % update step-size
                stp = options.stepsize_alg(iter, options);
                
                if A_new==0
                    z=1;
                else
                z = inv(A_new);
                end
                step  = (1/m)*(z);
                
                stp = stp*step;
                
                grad_w = problem.grad(w,ind);
                grad_0 = problem.grad(w0,ind);
                grad_m = problem.grad(w_m,ind); 
                
                yt_diff = grad_0 - grad_m;  %x^m - x^m-1
                
                vec = (w - w0); % x^k - x^m
                
                Atk_new = (s_diff' * yt_diff)/s2; %%% a= s'*s/s'*y
                
                 %Atk_new = ((i-1)/i)*Atk_new+(1/i)*Atk;
                
                %g = problem.grad(w,ind) - problem.grad(w0,ind) + full_grad + problem.hess_vec() + problem.hess_vec();
                bb = abs(A_new)*vec -abs(Atk_new)*vec;
                
                c = stp/(1+stp);
                
                
                g = grad_w - grad_0 + full_grad + bb;
                
                
                
            	v =  step*g;	
            	    
                w = w - v;
                
                iter = iter + 1;
                end
                
            end
         
        %Total number of epoch
        epoch=epoch+1;
           
        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + i;        
        
       
        % store infos
        [infos, f_val,~,gnorm,optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time,iter,(v),stp,0);        

        % display infos
        if options.verbose > 0
            fprintf('SVRG ABS BB: Epoch = %03d, cost = %.5e,time = %.4e, ||bb||=%.5e, ||g||=%.5e, optgap=%.5e\n', epoch, f_val, elapsed_time,norm(bb), gnorm, optgap);
        end
         
        if epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
        end
        
    end
end
