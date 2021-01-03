function  optimumpoint()


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
   
   seed = input('Give the random seed to select the random data samples:');
   isaninteger = @(x)isfinite(x) & x==floor(x);
   
   if isaninteger(seed)
       
   else
       seed = 'default';
   end
   
% for j=5:5
%    

%   
   data = data_input(in,seed);
   
   
       
       %%% LOGISTIC_REGRESSION (CLASSIFICATION)
       fprintf('Performing logistic regerssion: ');
       if in ==4 || in==5
           lambda = 10^-5;
       else
           lambda =10^-4;
       end
       
       problem = logistic_regression(data.x_train, data.y_train, data.x_test, data.y_test,lambda);
       
       

        
   

    
   
    options.w_init = data.w_init;
    options.store_w = true;
    options.verbos = 1;
    options.stepsize_alg = @stepsize_alg;
    
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

    
    
%    options.f_opt = 0;%optimum(problem,options,in);
 %   fprintf('f_opt/Min_cost f(w*) = %.4e\n', options.f_opt);
    
    %options.g_opt = problem.full_grad(w_o);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    options.eta_1 = 01;     % should be less than 1.
    options.eta_2 = 0.01;    % Should be greater than 1.
    alg = 'fix';
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
    
        if in == 1 % Arcene
        
        c = 10^0;
        step_svrg = {0.001*c, 0.01*c, 0.1*c,1,2.5};  %For data 1) Arcene
        alg = 'fix';
        
        
        
        elseif  in== 2 %% Gisette
        
        c = 1;
        step_svrg = {0.001*c, 0.01*c, 0.1,1,2.5};  %For data 2) Gisette
        alg = 'fix';
        
        elseif in == 3 %Madelon
        
       d= 10^0;
       step_svrg={0.01*d,0.1*d,0.5*d,1,2.5};
        alg = 'fix';
        
        
        elseif in == 4  %%%RCV1 binary
         
        c = 10^0;
        step_svrg = {0.01,0.1*c, 1,2.5, 5*c};  
        alg = 'fix';
        
        %step_sgd = {1,5,20};
        
        elseif in == 5  %%% COVtype
        
            
        c = 10^0;
        step_svrg ={2.5};% {0.05*c, 0.5*c,1,2.5,5};  
        alg = 'fix';
        

        
        elseif in == 6 %%% sido0        
            
        c = 10^0;
        step_svrg = {0.001,0.01,0.05*c, 0.1*c,1}; 
        alg = 'fix';
        
            
        elseif in ==8   %%% w8a
        
        c = 10^0;
        step_svrg = {0.001,0.01, 0.025*c, 0.1*c,1};  
        alg = 'fix';
        
        %step_sgd = {0.01,0.1,0.5};
        
        elseif in ==9 %%% ijcnn1
            
            
        c = 10^0;
        step_svrg = {0.01*c, 0.1*c, 1*c,2.5,5};  
        alg = 'fix';
        
        elseif in==10 %% MNIST38
            
        c = 10^0;
        %step_svrg = {0.001*c,0.01, 0.025*c, 0.05,0.1};  
        step_svrg = {0.05,0.1,0.5,1,5};%for svrg-2nd only
        alg = 'fix';
        
       % step_sgd = {0.1,0.5,1};
            
        end
        
        options.step_alg = alg;




     %options.max_epoch = 300;
     options.max_epoch=5; 
     o = options.max_epoch;
    
   
     
% We have selected the 2nd step size for RCV1 to find the f_opt by svrg
% i.e : 2.5
%
% We have selected the 1st step size for IJCNN1 to find the f_opt by svrg
% i.e. : 0.02 or 0.25
     
    %options.w_init = data.w_init;
    options.store_w = true;
    options.verbose = 1;
   
    options.method='NORM';
    options.f_opt  = f_opt(in);
    
        
       for i = 1:5
        options.stepsize_alg= @stepsize_alg;
%         options.step_init = step_sgd{i};
%         
%         options.step_alg = 'decay-2';
%         
%         fprintf('SGD %d\n',i);
%         
%         [w_gd{i}, info_gd{i}] = gd(problem, options); 
        
        options.step_init = step_svrg{i};
        
        options.step_alg = 'fix';
        
        fprintf('SVRG %d\n',i);
    
        %if q==1
            [w_svrg{i}, info_svrg{i}] = svrg_app(problem, options); 
        %elseif q==2
%             [w_svrg{i}, info_svrg{i}] = svrg_bb(problem, options); 
%         elseif q==3
%             [w_svrg{i}, info_svrg{i}] = svrg_dh(problem, options); 
%         elseif q==4
%             [w_svrg{i}, info_svrg{i}] = svrg_2nd(problem, options); 
%         end
%         
        end
        
     
        c1 = info_svrg{1}.cost(end);
        c2 = info_svrg{2}.cost(end);
        c3 = info_svrg{3}.cost(end);
        c4 = info_svrg{4}.cost(end);
        c5 = info_svrg{5}.cost(end);
        %c6 = info_svrg{3}.cost(o);
         C = [c1 c2 c3 c4 c5];
        
       % C = round(V*10^13)/10^13;
        
        
        
        
        [~,mn] = min(C);
        
        if c1 == min(C)
            
            w = w_svrg{1};
            info = info_svrg{1};
            stp = step_svrg{1};
            fprintf('###    SVRG 1 is selected for optimum for input=%d with step size=%.3e  ####\n \n',in,stp);
            
        elseif c2 ==min(C)
            
            w = w_svrg{2};
            info = info_svrg{2};
            stp = step_svrg{2};
            fprintf('###    SVRG 2 is selected for optimum  for input=%d  with step size=%.3e  ####\n \n',in,stp);
            
        elseif c3 == min(C)
            
            
            w = w_svrg{3};
            info = info_svrg{3};
            stp = step_svrg{3};
            fprintf('###    SVRG 3 is selected for optimumfor input=%d  with step size=%.3e  ####\n \n',in,stp);
            
        elseif c4 ==min(C)
            
            w = w_svrg{4};
            info = (info_svrg{4});
            stp = step_svrg{4};
            fprintf('###    SVRG 4 is selected for optimum   for input=%d  with step size=%.3e  ####\n \n',in,stp);
            
        
        elseif c5 ==min(C)
            
            w = w_svrg{5};
            info = (info_svrg{5});
            stp = step_svrg{5};
            fprintf('###    SVRG 5 is selected for optimum  for input=%d  with step size=%.3e  ####\n \n',in,stp);
            
        
%         elseif ((c6<c1) && (c6<c2) && (c6<c3) && (c6<c4) && (c6<c5))
%             
%             w = w_svrg{3};
%             info = (info_svrg{3});
%             fprintf('SVRG 3 is selected for optimum\n');

        elseif min(C)==max(C)
            
            fprintf(' All have attained the minimum value \n ');
%           
            w=w_svrg{5};
            info = info_svrg{5};
            stp = step_svrg{5};
            

        end
        
       
    algorithms_1 = {'SVRG1','SVRG2','SVRG3','SVRG4','SVRG5'};
    w_list_1 = {w_svrg{1},w_svrg{2}, w_svrg{3},w_svrg{4},w_svrg{5}};
    info_list_1 = {info_svrg{1},info_svrg{2},info_svrg{3},info_svrg{4},info_svrg{5}};
        
    
    display_graph('grad_calc_count','cost', algorithms_1, w_list_1, info_list_1);    
    display_graph('epoch','Test_cost', algorithms_1, w_list_1, info_list_1);
    display_graph('grad_calc_count','variance', algorithms_1, w_list_1, info_list_1);
    display_graph('grad_calc_count','optgap',algorithms_1,w_list_1,info_list_1);
        
         fprintf('Min value is %.25e attains by SVRG-%s',min(C),mn);
        %fprintf('w_opt = %.4e\n',w_opt);
%         
        variables = {'epoch';'cost';'gnorm';'time';'Stepsize'};
        
        A = info.epoch;
        B = info.cost;
        C = info.gnorm;
        D = info.time;
        [y,z]=size(D);
        S = stp*ones(y,z);
        T = table(A',B',C',D',S','VariableNames',variables);
        
        q=1; %SVRG
        Table(T,in,q);
%         q ==1 SVRG
%         q ==2 SVRG-BB
%         q ==3 SVRG-DH
%         q ==4 SVRG-2nd
%        
 
%end

end
