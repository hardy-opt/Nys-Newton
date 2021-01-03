function SVRGBB_test()


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
   
   
   %
           
     fprintf('###                  Select the Data set               ###\n'); 
    fprintf('##########################################################\n');
    fprintf('### Logistic REGRESSION                                ###\n');
    fprintf('###                                                    ###\n'); 
    fprintf('### 1) Arcene                                          ###\n');
    fprintf('### 2) Gisette                                         ###\n');
    fprintf('### 3) Madelon                                         ###\n'); 
    fprintf('### 4) RCV1_binary                                     ###\n'); 
    fprintf('### 5) COVtype                                         ###\n'); 
    fprintf('### 6) sido0                                           ###\n');
    fprintf('### 7) a9a                                             ###\n');
    fprintf('### 8) w8a                                             ###\n'); 
    fprintf('### 9) ijcnn1                                          ###\n'); 
    fprintf('### 10) MNIST38 (binary 3 = 1, 8 = -1                  ###\n'); 
    fprintf('### 11) Synthetic_logistic_data_generator              ###\n'); 
    fprintf('##########################################################\n');
    fprintf('###                                                    ###\n');

   
  in = input('Which data set you want to perfrom on?\nPlease enter the integer between 1 to 10: ');
   
  
   
   data = data_input(in);
   
  
options.verbos = 1;    
       %%% LOGISTIC_REGRESSION (CLASSIFICATION)
       fprintf('Performing logistic regerssion: \n');
       
      problem = logistic_regression(data.x_train, data.y_train, data.x_test, data.y_test,0.0001);
  
%   
%       
       %%% LINEAR_REGRESSION/ LEAST_SQUARE (REGRESSION)
       
%        fprintf('Performing linear regression: \n');
%        
%        problem = least_square(data.x_train,data.y_train, data.x_test, data.y_test,0.0001);
% %        
%        p=2;
%    
             
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

    
    
    
    options.f_opt = f_opt(in);
   
    fprintf('f_opt/Min_cost f(w*) = %.4e\n', options.f_opt);
    
    %options.g_opt = problem.full_grad(w_o);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    options.max_epoch=100;
    
    options.eta_1 = 01;     % should be less than 1.
    options.eta_2 = 0.01;    % Should be greater than 1.
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
        if in == 1 % Arcene
        
        c = 10^0;
        step = {0.0025*c, 1*c, 10*c};  %For data 1) Arcene
        alg = 'fix';
        
        
        
        elseif  in== 2 %% Gisette
        
        c = 1;
        step = {0.009*c, 0.02*c, 0.025};  %For data 2) Gisette
        alg = 'fix';
        
        elseif in == 3 %Madelon
        
       % c = 10^-3; For fix step size
       % step = {55*c, 60*c, 65*c}; %For data 3) Madelon
       d= 10^-1;
       step={1*d,5*d,10*d};
        alg = 'fix';
        
        
        elseif in == 4  %%%RCV1 binary
         
        c = 10^0;
        step = {0.1*c, 1*c, 3.5*c};  
        alg = 'fix';
        
        
        elseif in == 5  %%% COVtype
        
            
        c = 10^0;
        step = {0.05*c, 0.1*c, 0.15*c};  
        alg = 'fix';
%         step = {0.95*c, 0.75*c, 0.55*c, 0.35*10^-5, 0.15*10^-5};
%         alg= 'decay-2'
        
        elseif in == 6 %%% sido0        
            
        c = 10^0;
        step = {0.001*c, 0.01*c, 1*c}; %For data 4) Airfoil
        alg = 'fix';
        
        elseif in== 7 %%%% a9a
            
        elseif in ==8   %%% w8a
        
        c = 10^0;
        step = {0.001*c, 0.025*c, 0.05*c};  
        alg = 'fix';
        
        elseif in ==9 %%% ijcnn1
            
            
        c = 10^0;
        step = {0.01*c, 0.1*c, 1.5*c};  
        alg = 'fix';
        
        elseif in==10 %% MNIST38
          
        c = 10^0;
        step = {0.001*c, 0.025*c, 0.05*c};  
        alg = 'fix';
            
        end
        
        options.step_alg = alg;
        
        for i=1:3
      
        options.step_init = step{i};
        %options.step_alg = step_alg{i}; 
        
        
        [w_svrg_bb_{i}, info_svrg_bb_{i}] = svrg_bb(problem, options);
        
        [w_svrg_{i}, info_svrg_{i}] = svrg1(problem, options);
        [w_svrg_dh_{i}, info_svrg_dh_{i}] = svrg_dh(problem, options);
        [w_svrg_2nd_{i}, info_svrg_2nd_{i}] = svrg_2nd(problem, options);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        cost(i,:) = [info_svrg_{i}.cost(end)   info_svrg_bb_{i}.cost(end) info_svrg_dh_{i}.cost(end) info_svrg_2nd_{i}.cost(end) ]; % COST
        optgap(i,:) = [info_svrg_{i}.optgap(end) info_svrg_bb_{i}.optgap(end) info_svrg_dh_{i}.optgap(end) info_svrg_2nd_{i}.optgap(end)]; % Optgap
        gnorm(i,:) = [info_svrg_{i}.gnorm(end)  info_svrg_bb_{i}.gnorm(end) info_svrg_dh_{i}.gnorm(end) info_svrg_2nd_{i}.gnorm(end)]; % ||G||
        time(i,:) = [info_svrg_{i}.time(end)   info_svrg_bb_{i}.time(end) info_svrg_dh_{i}.time(end) info_svrg_2nd_{i}.time(end) ]; % Time 
        variance(i,:) = [info_svrg_{i}.var(end)   info_svrg_bb_{i}.var(end) info_svrg_dh_{i}.var(end) info_svrg_2nd_{i}.var(end) ]; % Variance

        M{i} =   [cost(i,:)' optgap(i,:)' gnorm(i,:)' time(i,:)' variance(i,:)'] ;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%
        
            svrg_cost= info_svrg_{i}.cost';
            svrg_optgap = info_svrg_{i}.optgap';
            svrg_gnorm = info_svrg_{i}.gnorm';
            svrg_time = info_svrg_{i}.time';
            svrg_var = info_svrg_{i}.var';
            
            SVRG{i} = table(svrg_cost,svrg_optgap,svrg_gnorm,svrg_time,svrg_var);
            
            svrg_bb_cost= info_svrg_bb_{i}.cost';
            svrg_bb_optgap = info_svrg_bb_{i}.optgap';
            svrg_bb_gnorm = info_svrg_bb_{i}.gnorm';
            svrg_bb_time = info_svrg_bb_{i}.time';
            svrg_bb_var = info_svrg_bb_{i}.var';
            
            SVRG_bb{i} = table(svrg_bb_cost,svrg_bb_optgap,svrg_bb_gnorm,svrg_bb_time,svrg_bb_var);
            
            svrg_dh_cost= info_svrg_dh_{i}.cost';
            svrg_dh_optgap = info_svrg_dh_{i}.optgap';
            svrg_dh_gnorm = info_svrg_dh_{i}.gnorm';
            svrg_dh_time = info_svrg_dh_{i}.time';
            svrg_dh_var = info_svrg_dh_{i}.var';
            
            SVRG_dh{i} = table(svrg_dh_cost,svrg_dh_optgap,svrg_dh_gnorm,svrg_dh_time,svrg_dh_var);
            
            
            svrg_2nd_cost= info_svrg_2nd_{i}.cost';
            svrg_2nd_optgap = info_svrg_2nd_{i}.optgap';
            svrg_2nd_gnorm = info_svrg_2nd_{i}.gnorm';
            svrg_2nd_time = info_svrg_2nd_{i}.time';
            svrg_2nd_var = info_svrg_2nd_{i}.var';
            
            SVRG_2nd{i} =table(svrg_2nd_cost,svrg_2nd_optgap,svrg_2nd_gnorm,svrg_2nd_time,svrg_2nd_var);
        
        
            %N{i} = [SVRG,SVRG_bb,SVRG_dh,SVRG_2nd];
        %%%%%%
            
        end
        
         
        
        
        algorithms_1 = {sprintf('SVRG-%.2e',step{1}),sprintf('SVRG-BB-%.2e',step{1}),sprintf('SVRG-DH-%.2e',step{1}),sprintf('SVRG-2nd-%.2e',step{1})};
        w_list_1 = {w_svrg_{1}, w_svrg_bb_{1}, w_svrg_dh_{1}, w_svrg_2nd_{1}};
        info_list_1 = {info_svrg_{1}, info_svrg_bb_{1}, info_svrg_dh_{1}, info_svrg_2nd_{1}};     
         
        %display_graph('grad_calc_count','cost', algorithms_1, w_list_1, info_list_1);
        display_graph('grad_calc_count','Test_cost', algorithms_1, w_list_1, info_list_1);
        display_graph('grad_calc_count','gnorm', algorithms_1, w_list_1, info_list_1);
        display_graph('grad_calc_count','optgap', algorithms_1, w_list_1, info_list_1);
        display_graph('grad_calc_count','variance', algorithms_1, w_list_1, info_list_1);
        display_graph('time','optgap',algorithms_1,w_list_1,info_list_1);

        
        algorithms_2 = {sprintf('SVRG-%.2e',step{2}),sprintf('SVRG-BB-%.2e',step{2}),sprintf('SVRG-DH-%.2e',step{2}),sprintf('SVRG-2nd-%.2e',step{2})};
        w_list_2 = {w_svrg_{2}, w_svrg_bb_{2}, w_svrg_dh_{2}, w_svrg_2nd_{2}};
        info_list_2 = {info_svrg_{2}, info_svrg_bb_{2}, info_svrg_dh_{2}, info_svrg_2nd_{2}};     
         
        display_graph('grad_calc_count','Test_cost', algorithms_2, w_list_2, info_list_2);
        display_graph('grad_calc_count','gnorm', algorithms_2, w_list_2, info_list_2);
        display_graph('grad_calc_count','optgap', algorithms_2, w_list_2, info_list_2);
        display_graph('grad_calc_count','variance', algorithms_2, w_list_2, info_list_2);
        display_graph('time','optgap',algorithms_2,w_list_2,info_list_2);
        
        
        
        algorithms_3 = {sprintf('SVRG-%.2e',step{3}),sprintf('SVRG-BB-%.2e',step{3}),sprintf('SVRG-DH-%.2e',step{3}),sprintf('SVRG-2nd-%.2e',step{3})};
        w_list_3 = {w_svrg_{3}, w_svrg_bb_{3}, w_svrg_dh_{3}, w_svrg_2nd_{3}};
        info_list_3 = {info_svrg_{3}, info_svrg_bb_{3}, info_svrg_dh_{3}, info_svrg_2nd_{3}};     
         
        display_graph('grad_calc_count','Test_cost', algorithms_3, w_list_3, info_list_3);
        display_graph('grad_calc_count','gnorm', algorithms_3, w_list_3, info_list_3);
        display_graph('grad_calc_count','optgap', algorithms_3, w_list_3, info_list_3);
        display_graph('grad_calc_count','variance', algorithms_3, w_list_3, info_list_3);
        display_graph('time','optgap',algorithms_3,w_list_3,info_list_3);     
   

        
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                TABLE
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
       for i=1:3
       
       Method = {sprintf('svrg_%.2e',step{i}); sprintf('SVRG-BB-%.2e',step{i}); sprintf('SVRG-dh-%.2e',step{i}); sprintf('SVRG-2nd-%.2e',step{i})};
           
       P = M{i};
       
       F = P';
       
       cost = F(1,:)';
        
       optgap = F(2,:)';
      
       gnorm = F(3,:)';
       
       time = F(4,:)';
       
       variance = F(5,:)';
    
       t = table(cost,optgap,variance,gnorm, time,'RowNames', Method);
       
        head(t)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%        Q = N{i};
%        
%        E = Q';
%        
%        Cost = E(1,:)';
%         
%        Optgap = E(2,:)';
%       
%        Gnorm = E(3,:)';
%        
%        Time = E(4,:)';
%        
%        Variance = E(5,:)';

       Variables = matlab.lang.makeValidName(Method');
    
       T = table(SVRG{i},SVRG_bb{i},SVRG_dh{i},SVRG_2nd{i},'VariableNames', Variables);
       
       T2 = splitvars(T);
       if in == 4 || in ==1
          
           writetable(T2,'RCV1_calc.xls','Sheet',i);
           
       elseif in == 8
           
           writetable(T2,'w8a_calc.xls','Sheet',i);
           
           
       elseif in == 9
           
           writetable(T2,'ijcann1_calc.xls','Sheet',i);
           
           
      elseif in == 10
           
           writetable(T2,'MNIST38_calc.xls','Sheet',i);
       end
       
       end
       
       
  
end