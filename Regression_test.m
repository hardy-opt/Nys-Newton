function Regression_test

clear;
clc;
close all;

 fprintf('###                  Select the Data set               ###\n'); 
    fprintf('##########################################################\n');
    fprintf('### Linear REGRESSION                                  ###\n');
    fprintf('###                                                    ###\n');
    fprintf('### 12) Bodyfat                                        ###\n');
    fprintf('### 13) Abalone                                        ###\n'); 
    fprintf('### 14) Slice                                          ###\n'); 
    fprintf('### 15) E2006_tfidf                                    ###\n'); 
    fprintf('### 16) Year_Million_Song                              ###\n');
    fprintf('##########################################################\n');
    fprintf('###                                                    ###\n');

   
  in = input('Which data set you want to perfrom on?\nPlease enter the integer between 12 to 16: ');
  
  seed = input('Give the random seed to select the random data samples:');
   isaninteger = @(x)isfinite(x) & x==floor(x);
   if isaninteger(seed)
       
   else
       seed = 'default';
   end
  
  data = data_input(in,seed);
  
  
  %problem = least_square(data.x_train,data.y_train,data.x_test,data.y_test, 0.0001);
  problem = logistic_regression(data.x_train,data.y_train,data.x_test,data.y_test, 0.0001);
    options.w_init = data.w_init;
    options.store_w = true;
    options.verbose = 1;
    options.stepsize_alg= @stepsize_alg;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    options.verbos = 1;
    
    
    options.f_opt = f_opt(in);
   
    fprintf('f_opt/Min_cost f(w*) = %.4e\n', options.f_opt);
    
    %options.g_opt = problem.full_grad(w_o);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    options.max_epoch=5;
    c=1;
    step = {0.001*c, 0.1*c, 2*c};  
    alg = 'fix';
        
    options.step_alg = alg;
    options.method = 'av';
    options.step_init = step{2};
    [w_s1,info_s1] = svrg_app(problem,options);
    
    options.step_init = step{2};
    [w_s2,info_s2] = svrg_bb(problem,options);
         
    options.step_init = step{2};
    [w_s3,info_s3] = svrg_dh(problem,options); 
    
    options.method = 'avg';
    options.step_init = 0.5;%step{2};
    %[w_bb,info_bb] = svrg_bb(problem,options);
    [w_bb,info_bb] = svrg_abs_bb(problem,options);
    
    options.method = 'avg';
    options.step_init = step{2};
    %[w_dh,info_dh] = svrg_dh(problem,options);
    [w_dh,info_dh] = svrg_2nd(problem,options);
    
    options.method = 'avg';
    options.step_init = 0.5;%step{2};
    %[w_2nd,info_2nd] = svrg_2nd(problem,options);
    [w_2nd,info_2nd] = svrg_abs_dh(problem,options);
    
    algorithms_1 = {'SVRG1','SVRG2','SVRG3','BB','DH','2nd'};
    w_list_1 = {w_s1,w_s2,w_s3,w_bb,w_dh,w_2nd};
    info_list_1 = {info_s1,info_s2,info_s3,info_bb,info_dh,info_2nd};
    
    C = [info_s1.cost(end), info_s2.cost(end),info_s3.cost(end), info_bb.cost(end), info_dh.cost(end), info_2nd.cost(end)];
    
    [minv,d] = min(C);
    fprintf('Cost = %.4e\n',d);
    if d==1 
        m = 'SVRG - 1';
    elseif d ==2
        m = 'SVRG - 2';
    elseif d==3
        m = 'SVRG - 3';
    elseif d==4
        m = 'SVRG - BB';
    elseif d==5 
        m = 'SVRG - DH';
    elseif d==6
        m = 'SVRG - 2nd';
    end
    fprintf('Min value is %.22e attains by %s',minv,m);
    
    display_graph('grad_calc_count','cost', algorithms_1, w_list_1, info_list_1);    
    %display_graph('grad_calc_count','optgap', algorithms_1, w_list_1, info_list_1);
    display_graph('grad_calc_count','variance', algorithms_1, w_list_1, info_list_1);
    display_graph('grad_calc_count','optgap',algorithms_1,w_list_1,info_list_1);
    %display_graph('grad_calc_count','Test_cost', algorithms_1, w_list_1, info_list_1);
   %display_graph('grad_calc_count','gnorm', algorithms_1, w_list_1, info_list_1);
   %costbb0.05=2.4563511861114180057086e+00 e100
  % 2.4563511861114162293518e+00 0.01 svrg 400

end