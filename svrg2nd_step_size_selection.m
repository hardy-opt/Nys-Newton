function svrg2nd_step_size_selection()



    clc;
    clear;
    close all;

    
      
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
   
   
       
       %%% LOGISTIC_REGRESSION (CLASSIFICATION)
       fprintf('Performing logistic regerssion: ');
       
       problem = logistic_regression(data.x_train, data.y_train, data.x_test, data.y_test,10^-4);
       p=1;
       
   
        
      options.max_epoch=10;
    
    %In order to choose decay-3 and decay-4 we need to choose one more
    %parameter for each
    options.eta_1 = 01;     % should be less than 1.
    options.eta_2 = 0.01;    % Should be greater than 1.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    

%%%%%%%%%%%%%%%%%%%%%%%%%      SVRG-BB     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    options.w_init = data.w_init;
    options.store_w = true;
    options.verbose = 1;
    options.stepsize_alg= @stepsize_alg;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    
    options.f_opt = 0;%optimum(problem,options,in);
    fprintf('f_opt/Min_cost f(w*) = %.4e\n', options.f_opt);
    
    %options.g_opt = problem.full_grad(w_o);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
  
    
    options.eta_1 = 01;     % should be less than 1.
    options.eta_2 = 0.01;    % Should be greater than 1.
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
    
        if in == 1 % Arcene
        
        c = 10^0;
        step = {0.01*c, 0.05*c, 10*c};  %For data 1) Arcene
        alg = 'fix';
        
        
        
        elseif  in== 2 %% Gisette
        
        c = 1;
        step = {0.009*c, 0.02*c, 0.025};  %For data 2) Gisette
        alg = 'decay-2';
        
        elseif in == 3 %Madelon
        
       % c = 10^-3; For fix step size
       % step = {55*c, 60*c, 65*c}; %For data 3) Madelon
       d= 10^-1;
       step={1*d,5*d,10*d};
        alg = 'fix';
        
        
        elseif in == 4  %%%RCV1 binary
         
        c = 10^0;
        step = {1*c, 2.5*c, 5*c};  
        alg = 'fix';
        
        
        elseif in == 5  %%% COVtype
        
            
        c = 10^0;
        step = {0.01*c, 0.1*c, 0.005*c};  
        alg = 'fix';

        
        elseif in == 6 %%% sido0        
            
        c = 10^0;
        step = {0.4*c, 0.6*c, 0.2*c}; 
        alg = 'fix';
        
        elseif in== 7 %%%% a9a
            
        c = 10^0;
        step = {0.04*c, 0.6*c, 0.2*c}; 
        alg = 'fix';
            
        elseif in ==8   %%% a8a
        
        c = 10^0;
        step = {0.5*c, 0.1*c, 0.5*c};  
        alg = 'fix';
        
        elseif in ==9 %%% ijcnn1
            
            
        c = 10^0;
        step = {0.01*c, 0.01*c, 0.25*c};  
        alg = 'fix';
        
        elseif in==10 %% MNIST38
            
        c = 10^0;
        step = {0.01*c, 0.005*c, 0.0025*c};  
        alg = 'fix';
        
            
        end
        
        options.step_alg = alg;
%         
%         for i=1:3
%             
%         options.step_init = step{i};
%         %options.step_alg = step_alg{i}; 
%         
%         
%         [w_svrg_bb_{i}, info_svrg_bb_{i}] = svrg_bb(problem, options);  
% 
%         end
%         
%           
%         algorithms_1 = {'SVRG-BB-1','SVRG-BB-2','SVRG-BB-3'};
%        w_list_1 = {w_svrg_bb_{1}, w_svrg_bb_{2}, w_svrg_bb_{3}};
%         info_list_1 = {info_svrg_bb_{1}, info_svrg_bb_{2}, info_svrg_bb_{3}};     
%          
%         display_graph('epoch','cost', algorithms_1, w_list_1, info_list_1);
%         display_graph('grad_calc_count','gnorm', algorithms_1, w_list_1, info_list_1);
%         
%         
        %%%%%%%%%%%%%%%%%%%%%%%%%   svrg-dh    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
        
        %step = {0.5*c, 0.55*c, 0.475*c, 0.45*c, 0.425*c};
    
        for i=1:3
            
        options.step_init = step{i};
        %options.step_alg = step_alg{i}; 
        
    
        
        [w_svrg_dh_{i}, info_svrg_dh_{i}] = svrg(problem, options);  

        end
        
         algorithms_2 = {'SVRG-dh-1','SVRG-dh-2','SVRG-dh-3'};
         w_list_2 = {w_svrg_dh_{1}, w_svrg_dh_{2}, w_svrg_dh_{3}};
        info_list_2 = {info_svrg_dh_{1}, info_svrg_dh_{2}, info_svrg_dh_{3}};     
%          
        %display_graph('epoch','Test_cost', algorithms_2, w_list_2, info_list_2);
        display_graph('grad_calc_count','gnorm', algorithms_2, w_list_2, info_list_2);
        
        %%%%%%%%%%%%%%%%%%%%%%%%     svrg-2nd      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
        for i=1:3
            
        options.step_init = step{i};
        %options.step_alg = step_alg{i}; 
        
        
        [w_svrg_2nd_{i}, info_svrg_2nd_{i}] = svrg_2nd(problem, options);  

        end
        
        
        algorithms_3 = {'SVRG-2nd-1','SVRG-2nd-2','SVRG-2nd-3'};
        w_list_3 = {w_svrg_2nd_{1}, w_svrg_2nd_{2}, w_svrg_2nd_{3}};
        info_list_3 = {info_svrg_2nd_{1}, info_svrg_2nd_{2}, info_svrg_2nd_{3}};     
         
        display_graph('epoch','cost', algorithms_3, w_list_3, info_list_3);
        display_graph('grad_calc_count','gnorm', algorithms_3, w_list_3, info_list_3);
        
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
end