function perform()


    clc;
    clear;
    close all;


   
   %Choose data (Give integer between 1 to 9)
    
   %    REGRESSION                         CLASSIFICATION
   
   % 1) Arcene                             6) Airfoil
   % 2) Gisette                            7) Combined_cycle_pp 
   % 3) Madelon                            8)  UJIndoorLoc                       
   % 4) RCV1_binary
   % 5) COVtype
   % 9) Synthetic_logistic_data_generator
   
    fprintf('###                  Select the Data set               ###\n'); 
    fprintf('##########################################################\n');
    fprintf('### REGRESSION                        CLASSIFICATION   ###\n');
    fprintf('###                                                    ###\n'); 
    fprintf('### 1) Arcene                         6) Airfoil       ###\n');
    fprintf('### 2) Gisette                        7) Combined_CPP  ###\n');
    fprintf('### 3) Madelon                        8) UJIndoorLoc   ###\n'); 
    fprintf('### 4) RCV1_binary                                     ###\n'); 
    fprintf('### 5) COVtype                                         ###\n'); 
    fprintf('### 8) Synthetic_logistic_data_generator               ###\n'); 
    fprintf('##########################################################\n');
    fprintf('###                                                    ###\n'); 
   
   in = input('Which data set you want to perfrom on?\nPlease enter the integer between 1 to 7: ');
   
   seed = input('Give the random seed to select the random data samples:');
   isaninteger = @(x)isfinite(x) & x==floor(x);
   if isaninteger(seed)
       
   else
       seed = 'default';
   end
   
   
   data = data_input(in,seed);
   
%    if in==9 || in<=5
       
       %%% LOGISTIC_REGRESSION (CLASSIFICATION)
       fprintf('Performing logistic regerssion: ');
       
       problem = logistic_regression(data.x_train, data.y_train, data.x_test, data.y_test,0.0001);
%        p=1;
%    elseif in<9 || in>5
%       
%        %%% LINEAR_REGRESSION/ LEAST_SQUARE (REGRESSION)
%        
%        fprintf('Performing linear regression: ');
%        
%        problem = least_square(data.x_train,data.y_train, data.x_test, data.y_test,0.0001);
%        
%        p=2;
%    end
             
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             There are three step size
%   1) 'fix'
%   2) 'decay-2' ---> a0/1+sqrt(k)
%   3) 'decay-4' ---> a0/sqrt(k)

% NOTE: 'decay' and 'decay-3' depends on the regularized parameter which
% may vary in each case that's why I only want to use the step size which
% doesn't depend on the regularized parameter 'lambda'.

% Therfore, I am taking the defult 'lambda' = 10^-4.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   


    
   
    options.w_init = data.w_init;
    options.store_w = true;
    options.verbose = 1;
    options.stepsize_alg= @stepsize_alg;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    [w_o,info_o] = optimum(problem,options);
    
    options.f_opt = problem.cost(w_o);
    
    fprintf('f_opt/Min_cost f(w*) = %.4e', options.f_opt);
    
    %options.g_opt = problem.full_grad(w_o);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    options.max_epoch=50;
    
%SGD   

        %%%%%%%%%% Initial Step Size %%%%%%%%%%%
        
        a0 = 0.1;
        
        alg = 'decay-2';
        
        options.step_init = a0;
        
        options.step_alg = alg;
    
        [w_sgd, info_sgd] = sgd(problem, options); 
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%SVRG 
            options.step_init = 01;
            
            options.step_alg = 'fix';
    
         [w_svrg, info_svrg] = svrg(problem, options); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%SVRG-BB 
        
       
        options.step_init = a0;
        
        options.step_alg = alg;
    
        [w_svrg_bb, info_svrg_bb] = svrg_bb(problem, options);  
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%SVRG_dh % diagonal_hession
    
        options.step_init = a0;
        
        options.step_alg = alg;
    
        [w_svrg_dh, info_svrg_dh] = svrg_dh(problem, options);  


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%SVRG_2nd

        
        options.step_init = a0;
        
        options.step_alg = alg;
    
        [w_svrg_2nd, info_svrg_2nd] = svrg_adagrad(problem, options);  
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
   %algorithms = {'SGD:0.005','SGD:0.0025','SVRG:0.025','SVRG:0.085', 'SVRG-BB:0.035','SVRG-BB:0.075'};
   %w_list = {w_sgd_1,w_sgd,w_svrg_1,w_svrg,w_svrg_bb_1,w_svrg_bb};
   %info_list = {info_sgd_1,info_sgd,info_svrg_1,info_svrg,info_svrg_bb_1,info_svrg_bb};     
         
         
     algorithms = {'SGD','SVRG','SVRG-BB','SVRG-DH','SVRG-2nd'};
     w_list = {w_sgd,w_svrg,w_svrg_bb,w_svrg_dh,w_svrg_2nd};
     info_list = {info_sgd,info_svrg,info_svrg_bb,info_svrg_dh,info_svrg_2nd};     
    
         

%       
%      algorithms = {'SGD','SVRG','SVRG-BB'};
%      w_list = {w_sgd,w_svrg,w_svrg_bb};
%      info_list = {info_sgd,info_svrg,info_svrg_bb};   
    
    display_graph('epoch','variance', algorithms, w_list, info_list);
    
    display_graph('epoch','cost', algorithms, w_list, info_list);
    
  %  display_graph('grad_calc_count','gnorm', algorithms, w_list, info_list);
    
    display_graph('epoch','optgap', algorithms, w_list, info_list);
    display_graph('grad_calc_count','optgap', algorithms, w_list, info_list);
    
    
   
    
end

