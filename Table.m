function Table(T,in,q)


 
        if in == 1
            

            if q==1
                writetable(T,'arcene.xlsx','Sheet','SVRG');
            elseif q==2
                writetable(T,'arcene.xlsx','Sheet','SVRG_bb');
            elseif q==3
                writetable(T,'arcene.xlsx','Sheet','SVRG_dh');
            elseif q==4
                writetable(T,'arcene.xlsx','Sheet','SVRG_2nd');
            end
            
            
            
        elseif in == 2
            
            if q==1
                writetable(T,'Gisette.xlsx','Sheet','SVRG');
            elseif q==2
                writetable(T,'Gisette.xlsx','Sheet','SVRG_bb');
            elseif q==3
                writetable(T,'Gisette.xlsx','Sheet','SVRG_dh');
            elseif q==4
                writetable(T,'Gisette.xlsx','Sheet','SVRG_2nd');
            end
            
        elseif in == 3
            
            if q==1
                writetable(T,'Madelon.xlsx','Sheet','SVRG');
            elseif q==2
                writetable(T,'Madelon.xlsx','Sheet','SVRG_bb');
            elseif q==3
                writetable(T,'Madelon.xlsx','Sheet','SVRG_dh');
            elseif q==4
                writetable(T,'Madelon.xlsx','Sheet','SVRG_2nd');
            end
            
        elseif in == 4
            
            if q==1
                writetable(T,'RCV1.xlsx','Sheet','SVRG');
            elseif q==2
                writetable(T,'RCV1.xlsx','Sheet','SVRG_bb');
            elseif q==3
                writetable(T,'RCV1.xlsx','Sheet','SVRG_dh');
            elseif q==4
                writetable(T,'RCV1.xlsx','Sheet','SVRG_2nd');
            end
            
        elseif in == 5
            
            if q==1
                writetable(T,'Covtype.xlsx','Sheet','SVRG');
            elseif q==2
                writetable(T,'Covtype.xlsx','Sheet','SVRG_bb');
            elseif q==3
                writetable(T,'Covtype.xlsx','Sheet','SVRG_dh');
            elseif q==4
                writetable(T,'Covtype.xlsx','Sheet','SVRG_2nd');
            end
            
            
        elseif in == 6
            
            if q==1
                writetable(T,'Sido0.xlsx','Sheet','SVRG');
            elseif q==2
                writetable(T,'Sido0.xlsx','Sheet','SVRG_bb');
            elseif q==3
                writetable(T,'Sido0.xlsx','Sheet','SVRG_dh');
            elseif q==4
                writetable(T,'Sido0.xlsx','Sheet','SVRG_2nd');
            end
            
          
        elseif in == 7
            
            if q==1
                writetable(T,'a9a.xlsx','Sheet','SVRG');
            elseif q==2
                writetable(T,'a9a.xlsx','Sheet','SVRG_bb');
            elseif q==3
                writetable(T,'a9a.xlsx','Sheet','SVRG_dh');
            elseif q==4
                writetable(T,'a9a.xlsx','Sheet','SVRG_2nd');
            end
          
            
        elseif in ==8
         
            if q==1
                writetable(T,'w8a.xlsx','Sheet','SVRG');
            elseif q==2
                writetable(T,'w8a.xlsx','Sheet','SVRG_bb');
            elseif q==3
                writetable(T,'w8a.xlsx','Sheet','SVRG_dh');
            elseif q==4
                writetable(T,'w8a.xlsx','Sheet','SVRG_2nd');
            end
            
        elseif in ==9
            
            if q==1
                writetable(T,'Ijcnn1.xlsx','Sheet','SVRG');
            elseif q==2
                writetable(T,'Ijcnn1.xlsx','Sheet','SVRG_bb');
            elseif q==3
                writetable(T,'Ijcnn1.xlsx','Sheet','SVRG_dh');
            elseif q==4
                writetable(T,'Ijcnn1.xlsx','Sheet','SVRG_2nd');
            end
            
        elseif in==10
            
            if q==1
                writetable(T,'MNIST38.xlsx','Sheet','SVRG');
            elseif q==2
                writetable(T,'MNIST38.xlsx','Sheet','SVRG_bb');
            elseif q==3
                writetable(T,'MNIST38.xlsx','Sheet','SVRG_dh');
            elseif q==4
                writetable(T,'MNIST38.xlsx','Sheet','SVRG_2nd');
            end
        end

end