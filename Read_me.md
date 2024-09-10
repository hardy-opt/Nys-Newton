This is a library to solve Empirical risk minimization

Tankaria Hardik
PhD student, Kyoto University (2020,July).

This library includes four stochastic methods to solve logistic regression minimization problem


%% Look at the PNG file to see the "site_map"



									EMP_risk_library
										||
										||
		---------------------------------------------------------------------------------------------------------------------
		=====================================================================================================================
		/					/					/					/
		/					/					/					/
		/					/					/					/
		/					/					/					/
	       DATA				   EMP_risk_prob			Stochastic_solver			1) Read_me.text
	        |					|					|				2) Run_me_first.m
	        |					|					|				3) perform.m %% To perform all the method at once.
   ===========================		===============	       	==============				    | (Choose required arguments)
   1) /Arcene_data % Real_data		1) logistic_regression.m		1) SGD					4)VSRG %% svrg_nakai (N. Yamashita)
   |						|					|	
   2) /Gisette_data % Real data		2) min_square.m 			2) SVRG
   |						%% Not working yet			|
   3) /Madelon_data % Real data							3) SVRG-BB
   |											|
   4) /Synthetic_data_generator							4) SVRG-2nd
   		|
   		1) logistic_regression_data_generator
   		|
		2) sigmoid.m
		|
		3) min_square_data_generator %% Not working yet
   |
   5) display_graph %% Graph information  
   |
   6) stepsize_alg %% Diffrent step size defined(Including Nakai's stepsize)
   	%% New step size can be added by giving new name
   |	
   7) Store_infos %% stores all of the infos of current numerical performance 







%% Usage

1) Stochastic Gradient Method = sgd(problem, options)  %% SGD method

2) Stochastic Variance reduced gradient method = svrg(problem,options)  %% SVRG method

3) SVRG-BB = svrg_bb(problem,options) %% SVRG-bb method

4) SVRG_2nd_order = svrg_2nd(problem,options) %% SVRG with 2nd order method

Initial parameters stored in "options"  with struct. vector and "problem" is our objective function definition.

options.max_epoch ===> Denotes maximum epoch

options.step_init ===> Denotes initial step size

options.w_init ===> Denotes initial point

options.stepsize_alg ===> Denotes different stepsize

	1) "fix" denotes the fix step size of an initial step size

	2) " decay" denotes the ===> t0/(1+t0*lambda*iter)
	
	3) "decay-2" denotes the ===> t0/(1+sqrt(iter))
	
	4) "decay-3" denotes the ===> t0/(lambda+iter)
	
	5) "Nakai" denotes the Nakai's step size.
