function data=data_input(n,seed)

    switch n 
        
%         
%          fprintf('###                  Select the Data set               ###\n'); 
%     fprintf('##########################################################\n');
%     fprintf('### Logistic REGRESSION                                ###\n');
%     fprintf('###                                                    ###\n'); 
%     fprintf('### 1) Arcene                                          ###\n');
%     fprintf('### 2) Gisette                                         ###\n');
%     fprintf('### 3) Madelon                                         ###\n'); 
%     fprintf('### 4) RCV1_binary                                     ###\n'); 
%     fprintf('### 5) COVtype                                         ###\n'); 
%     fprintf('### 6) sido0                                           ###\n');
%     fprintf('### 7) a9a                                             ###\n');
%     fprintf('### 8) w8a                                             ###\n'); 
%     fprintf('### 9) ijcnn1                                          ###\n'); 
%     fprintf('### 10) MNIST38 (binary 3 = 1, 8 = -1                  ###\n'); 
%     fprintf('### 11) Synthetic_logistic_data_generator              ###\n'); 
%     fprintf('##########################################################\n');
%     fprintf('###                                                    ###\n'); 
        
        
        case 1
            
            data = Arcene();
            
        case 2
            
            data = Gisette();
            
        case 3
            
            data = Madelon();
            
        case 4 
            
            data = RCV1_binary(seed);
            
        case 5 
            
            data = covtype(seed);
            
        case 6 
            
            data = sido0(seed);
            
        case 7 
            
            data = a9a();
            
        case 8
            
            data =  w8a(seed);       %W8a
            
        case 9 
            
            data = ijcnn1(seed);    %IJCNN1
            
            
        case 10
            
            data = MNIST38(seed);    %MNIST38
            
        case 11
            
            fprintf('This is synthetic logistic data generator \n');
            
            no=input('Enter the number of samples: ');
            
            dim=input('Enter the number of dimensions:  ');
            
            
            data = logistic_regression_data_generator(no, dim); 
        
            
        case 12 % bodyfat 
            
            data = bodyfat();
            
        case 13
            
            data = abalone();
            
        case 14
            
            data = Slice();
            
        case 15 
            
            data = E2006_tfidf();
            
        case 16 
            
            data = YearPMSD;
    
    end     
   

end


% case 6
%             
%             data = Airfoil();
%             
%         case 7
%             
%             data = Combined_Cycle_pp();
%             
%         case 8
%             
%             data = UJIndoorLoc();