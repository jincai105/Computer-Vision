x= P1(:,1);
y= P1(:,2);
x2 = P2(:,1);
y2 = P2 (:,2);
FC=zeros(1,9);
for i = 1:length(x)-1
     L=[x(i)*x2(i), x(i)*y2(i), x(i), y(i)*x2(i), y(i)*y2(i), y(i), x2(i), y2(i),1];
     FC=[FC;L];
end
[U,S,V]=svd(FC);
F=reshape(V(:,end),3,3);
F3 = F