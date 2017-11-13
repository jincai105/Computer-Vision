finalVector=[];
for i=1:26
    finalVector=[finalVector;final_result(i+1,1)-final_result(i,1),final_result(i+1,2)-final_result(i,2),final_result(i+1,3)-final_result(i,3)]
end
finalVector=[finalVector;0,0,0]