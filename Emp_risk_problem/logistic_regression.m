

classdef logistic_regression
    
   
    
    properties
       name;    
        dim;
        samples;
        lambda;
        classes;  
        hessain_w_independent;
        d;
        n_train;
        n_test;
        x_train;
        y_train;
        x_test;
        y_test;
        x_norm;
        x;  
    end
    
    methods
        
        function obj = logistic_regression(x_train,y_train,x_test,y_test,varargin)
            obj.x_train = x_train;
            obj.y_train = y_train;
            obj.x_test = x_test;
            obj.y_test = y_test;
            
            if nargin < 5
                obj.lambda = 0.001;
            else
                obj.lambda = varargin{1};
            end
            
            obj.d = size(obj.x_train,1);
            obj.n_train = length(y_train);
            obj.n_test = length(y_test);
            obj.name = 'logistic_regression';
            obj.dim = obj.d;
            obj.samples = obj.n_train;
            obj.classes = 2;
            obj.x_norm = sum(obj.x_train.^2,1);
            obj.x = obj.x_train;
        end
        
        function f = cost(obj,w)
            
            %f = sum(log(1+exp(-obj.y_train.*(w'*obj.x_train)))/obj.n_train,2) + obj.lambda*(w'*w)/2;
            
            sigmod_result = sigmoid(obj.y_train.*(w'*obj.x_train));
            sigmod_result = sigmod_result + (sigmod_result<eps).*eps;
            f = -sum(log(sigmod_result),2)/obj.n_train + obj.lambda * (w'*w) / 2;
            
        end
        
         function f = cost_batch(obj, w, indices)

            f = -sum(log(sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices))))/obj.n_train,2) + obj.lambda * (w'*w) / 2;

        end
        
      %  function g = sgrad(obj,w,ind)  %%% Stochastic gradient for "one" single index
            
            %g = y_train.*x_train * (1-sigmoid(y_train.*(w'*x_train)));
            
       %     g = -sum(ones(obj.d,1) * obj.y_train(ind).*obj.x_train(:,ind) * (ones(1,length(ind))-sigmoid(obj.y_train(ind).*(w'*obj.x_train(:,ind))))',2)/length(ind)+ obj.lambda * w;
       % end
        
        function g = grad(obj,w,ind) %%% sum of stochastic gradient to get full gradient
            
            %g = y_train.*x_train * (1-sigmoid(y_train.*(w'*x_train)));
            
            xy = obj.y_train(ind).*obj.x_train(:,ind); 
            g = -sum(((ones(length(ind),1)-sigmoid(xy'*w)).*xy')',2)/length(ind)+obj.lambda*w;
            %g =-sum(ones(obj.d,1) * xy * (ones(1,length(indices))-sigmoid(w'*xy))',2)/length(indices)+ obj.lambda * w;
            %g = -sum(ones(obj.d,1) * obj.y_train(indices).*obj.x_train(:,indices) * (ones(1,length(indices))-sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices))))',2)/length(indices)+ obj.lambda * w;
        end
        
        function g = full_grad(obj,w)
            
            g = grad(obj, w, 1:obj.n_train);
            
        end
        
        function h = hess(obj, w, indices)

            %org code
            %temp = exp(-1*(y_train(indices)').*(x_train(:,indices)'*w));
            %b = temp ./ (1+temp);
            %h = 1/length(indices)*x_train(:,indices)*(diag(b-b.^2)*(x_train(:,indices)'))+lambda*eye(d); 

            sigm_val = sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices)));
            c = sigm_val .* (ones(1,length(indices))-sigm_val); 
            
            h1 = (obj.x_train(:,indices) .* (obj.y_train(indices).^2 .* c));
            
            h = 1/length(indices)* h1 * obj.x_train(:,indices)'+obj.lambda*eye(obj.d);
        end
        
        function h = full_hess(obj, w)

            h = hess(obj, w, 1:obj.n_train);

        end
       
        
        function hv = hess_vec(obj, w, v, indices)  %%% Hessian - vector multiplication

            sigm_val = sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices)));
            c = sigm_val .* (ones(1,length(indices))-sigm_val); 
            
            
            
            h1 = (obj.x_train(:,indices) .* (obj.y_train(indices).^2 .* c));
            
            hv = 1/length(indices)* h1 * (obj.x_train(:,indices)' * v) +obj.lambda*v;

        end
        
        function hv = fullhess_vec(obj, w, v)   %%% Hessian - vector multiplication
            
                
            hv = hess_vec(obj,w,v,1:obj.n_train);

        end
        
        function ph = partial_hess(obj,w,indices)
            sigm_val = sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices)));
            c = sigm_val .* (ones(1,length(indices))-sigm_val); 
            
            
            
            h1 = (obj.x_train(:,indices) .* (obj.y_train(indices).^2 .* c));
           
            ph = 1/length(indices)* h1;

        end
        
        function phv = partial_hess_vec(obj,v,indices,ph)
            
            phv = ph* (obj.x_train(:,indices)' * v) +obj.lambda*v;
        end
        
        
          
        function dh = diag_hess(obj,w,indices)
            
            sigm_val = sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices)));
            thd2 = sigm_val.* (ones(1,length(indices))-sigm_val);
            %sqtthd = sqrt(thd2);
            %xthd2 = obj.x_train(:,indices).^2*thd2';
            xy = obj.x_train(:,indices).*obj.y_train(indices);
            xythd2 = xy.^2*thd2';
            dh = (1/length(indices))*xythd2 + obj.lambda*ones(obj.dim,1);
            %dh = (1/length(indices))*xthd2 + obj.lambda*ones(obj.dim,1);
            %dh = sum((obj.x_train(:,indices)*sqtthd').^2,2)/length(indices);
            %[n1,d1]= size(dh)
            %if any(isnan(sqtthd)) || any(isinf(sqtthd))
            %   fprintf(' sqt is naninf= \n');
            %end
            
%              h = hess(obj,w,indices);
%              H = diag(h);
%             
%              if all(dh == H)
% %                 
%               fprintf('digaonal elements are the same\n');
%              end
           % dh = dh + obj.lambda*ones(n1,d1);
         
        end
        
        
        function dh = full_diag_hess(obj,w)
           
            dh = diag_hess(obj,w,1:obj.n_train);
        end
        
        
        %%%%%%% Test cost
        function f = test_cost(obj,w)
            
            %f = sum(log(1+exp(-obj.y_train.*(w'*obj.x_train)))/obj.n_train,2) + obj.lambda*(w'*w)/2;
            
            sigmod_result = sigmoid(obj.y_test.*(w'*obj.x_test));
            sigmod_result = sigmod_result + (sigmod_result<eps).*eps;
            f = -sum(log(sigmod_result),2)/obj.n_test + obj.lambda * (w'*w) / 2;
            
        end
        
        %%%%%%%
        function p = prediction(obj, w)

            p = sigmoid(w' * obj.x_test);

            class1_idx = p>0.5;
            class2_idx = p<=0.5;         
            p(class1_idx) = 1;
            p(class2_idx) = -1;         

        end

        function a = accuracy(obj, y_pred)

            a = sum(y_pred == obj.y_test) / obj.n_test; 

        end
        
        
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        
         function [G,fn,apptime,origtime] =  app_hess(obj,w,indices,set)
            tic;
            fprintf('Approximate Hessian Started\n');
            sigm_val = sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices)));
            c = sigm_val .* (ones(1,length(indices))-sigm_val); 
            
            h1 = (obj.x_train(:,indices) .* (obj.y_train(indices).^2 .* c));
            
            h2 = obj.x_train(set,indices)';
            %fprintf('Approx size of h1 %dx%d and h2 = %dx%d\n',size(h1),size(h2));
            a = 1/length(indices)* h1 * h2;
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
            fprintf('Approximate Hessian Has been computed\n');
            %%%%%%%%%%%%%%%%%%%%%%
            tic;
            fprintf('Original Hessian Started\n');
            H = full_hess(obj,w);
            fprintf('Original Hessian Has been computed\n');
            origtime = toc;
            %%%%%%%%%%%%%%%%%%%%%%
            
            M = H - G;
            fn = norm(M,'fro');
          
            
%             if all(H(set,set)==a(set,:))
%                 if all(H(:,set)==a)
%                 fprintf('both Hessian are the same\n');
%                 end
%             end
%             
            
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        
    end
end
