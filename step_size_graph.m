function step_size_graph

clear;
clc;

initial_step = 0.1;

iter = 0;

lambda = 01;

limit = 10^5;

a = 1;

b= 0.01;

for i = 1 : limit

r = floor(i/100);
 
step_1(i) = initial_step*a^(r); %%% blue

step_2(i) = initial_step/(1+initial_step*(i)); %%% Red

step_3(i) = initial_step/(1+b*r); %%% Black



end



%plot(1:limit,step_1, 'b-',1:limit, step_2,'r:',1:limit, step_3,'k','MarkerSize', 6 , 'Linewidth', 8); 
semilogy(1:limit,step_1, 'b-',1:limit, step_2,'r:',1:limit, step_3,'k','MarkerSize', 6 , 'Linewidth', 5); 
legend('Step-1', 'Step-2', 'Step-3');
xlabel('Iteration', 'FontSize', 18); 
ylabel('Step-size', 'FontSize', 18); 
set(gca, 'FontSize', 18);
end