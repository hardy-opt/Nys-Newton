%This file is the part of Emp_risk_library

%Tankaria Hardik,
%PhD student, Kyoto University-2020(July).

% min f(x)= 1/n sum_i^n (a'*x-y)^2  

classdef least_square
   
    
    properties
    samples;
    dim;
    name;
    lambda;
    classes;
    d;
    n_train;
    n_test;
    x_train;
    x_test;
    y_train;
    y_test;
    x;
    x_norm;
    end
    
    methods
        
        function obj = least_square(x_train,y_train,x_test,y_test,varargin)
            
            
            obj.x_train = x_train;
            obj.y_train = y_train;
            obj.x_test = x_test;
            obj.y_test = y_test;            

            if nargin < 5
                obj.lambda = 0.000;
            else
                obj.lambda = varargin{1};
            end

            obj.d = size(obj.x_train, 1);
            obj.n_train = length(y_train);
            obj.n_test = length(y_test);      
            obj.name = 'least_square';    
            obj.dim = obj.d;
            obj.samples = obj.n_train;
            obj.classes = 2;  
            obj.x_norm = sum(obj.x_train.^2,1);
            obj.x = obj.x_train;
        end
        
        function f = cost(obj,w)
        
           f = sum((w'*obj.x_train-obj.y_train).^2)/(2*obj.n_train)+ (obj.lambda/2)*(w'*w);  
           
        end
        
        function g = grad(obj,w,ind)
            ru = (w'*obj.x_train(:,ind));
%             if isnan(ru)
%                 fprintf('w^t*objx is NaN')
%                 return;
%             end
            res = ru - obj.y_train(ind);
%            
%             if isnan(res)
%                 fprintf('res is NaN')
%                 return;
%             end    
            
            g =  (obj.x_train(:,ind)*res')/length(ind)+ obj.lambda*w;
%             
%              if isnan(g)
%                 fprintf('grad is NaN')
%                 return;
%             end
        
        end
        
        function g = full_grad(obj,w)
            
            g = obj.grad(w,1:obj.n_train);
            
        end
        
        
        function h = hess(obj,w,ind)
            
            h = (obj.x_train(:,ind)*obj.x_train(:,ind)')./length(ind) + obj.lambda*eye(obj.d);
        end
        
        
        function h = full_hess(obj,w)
            
            h = obj.hess(w,1:obj.n_train) ;
        end
        
        
        function hv = hess_vec(obj,w,v,ind)
            
            hv = 1/length(ind)* obj.x_train(:,ind)*(obj.x_train(:,ind)'*v) + obj.lambda*v;
        end
        
        function fhv = fullhess_vec(obj,w,v)
            
            fhv = hess_vec(obj,w,v,1:obj.n_train);
        end
        
        function dh = diag_hess(obj,w,ind)
            
            dh = sum((obj.x_train(:,ind)).^2,2)/length(ind);
        end
        
        
        function dh = full_diag_hess(obj,w)
           
            dh = diag_hess(obj,w,1:obj.n_train);
        end
        
        
        function p = prediction(obj, w)
            p = w' * obj.x_test;        
        end

        function e = mse(obj, y_pred)

            e = sum((y_pred-obj.y_test).^2)/ (2 * obj.n_test);

        end
        
        function tc = test_cost(obj,w)
            tc = sum((w'*obj.x_test-obj.y_test).^2)/(2*obj.n_test)+ (obj.lambda/2)*(w'*w);  
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [G,fn,apptime,origtime] =  app_hess(obj,w,indices,set)
            tic;
            sigm_val = sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices)));
            c = sigm_val .* (ones(1,length(indices))-sigm_val); 
            
            h1 = (obj.x_train(:,indices) .* (obj.y_train(indices).^2 .* c));
            
            h2 = obj.x_train(set,indices)';
            %fprintf('Approx size of h1 %dx%d and h2 = %dx%d\n',size(h1),size(h2));
           
            a = (obj.x_train(:,indices)*obj.x_train(set,indices)')./length(indices); 
            %a = 1/length(indices)* h1 * h2;
            %h3 = obj.x_train(:,indices)';
            %fprintf('Orig size of h1 %dx%d and h2 = %dx%d\n',size(h1),size(h3));
            l = length(set);
            for i=1:l
                a(set(i),i) = a(set(i),i)+obj.lambda;
            end
             aN = a;
             a = aN;
            
            C = a;
            A = a(set,:);
            [U,W,V] = svds(A);
            I = inv(W);
            v = sqrt(diag(I));
            B = U.*v';
            Z = C*B;
            G = Z*Z';
            
            apptime = toc;
            
            %%%%%%%%%%%%%%%%%%%%%%
            tic;
           % H = full_hess(obj,w);
            origtime = toc;
            %%%%%%%%%%%%%%%%%%%%%%
            
          %  M = H - G;
          %  fn = norm(M,'fro');
          fn = 0; 
            
%             if all(H(set,set)==a(set,:))
%                 if all(H(:,set)==a)
%                 fprintf('both Hessian are the same\n');
%                 end
%             end
%             
            
        end
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
end
