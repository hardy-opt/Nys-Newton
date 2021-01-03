function f_opt = f_opt(in)

switch in

    case 1  %Arcene
        
        f_opt = 0.013822750288641; %with step size 1.
        
    case 2 %Gisette
            f_opt = 0.007179380742494; %Gnorm =  9.12527158768233E-10, step size 0.01 and 1300 epoch
            %f_opt = 0.00717938138661 
    case 3 % Madelon
        
    case 4 %RCV1
    	  
    	  f_opt = 0.10850491835192; %Gnorm=6.36564885559253E-15; and step size is 1;
    	  
    case 5 % Covtype
                
         f_opt = 2.0330364703420990366705690e-04;
         %2.0330364703418030494774804e-04; overall opt
        % 0.000203303647034; %Gnorm=3.08466450191631E-16 and step size is 0.5;
         
    case 6 % Sido0
            
        f_opt = 0.009822683143711; % GNorm = 6.4642708902803E-10 and step size 0.01, 621 epoch
    case 7 %a9a
        
    case 8 % W8A
         %
         f_opt = 1.4239859065451695996351589e-01; %Gnorm=4.51250059035171E-14 and step size= 0.025
          
    
    case 9 %IJCNN1
         
         f_opt = 0.221263128349259; %Gnorm=2.76248645207146E-15 and step size = 0.1
         
    case 10 %MNIST38
                %8.5555069940706635067684260e-02 
                
        f_opt = 8.5555069940706635067684260e-02;%0.085555069940707; % Gnorm= 6.76149410620798E-14 and step size = 0.025 svrg

    case 12 %Bodyfat
        
        f_opt = 6.0106173106121888285221e-05;
        
    case 13 % Abalone 
        
        f_opt = 2.4563511861114162293518;


end
