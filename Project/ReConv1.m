
%Center Points of ball from Camera 1
BallN=[653,495,13.0081;
    696,450,13.0415;
    736,409,13.0748;
    778,372,13.1082;
    820,335,13.1415;
    860,306,13.1749;
    899,275,13.2082;
    940,251,13.2416;
    978,227,13.2749;
    1017,211,13.3083;
    1053,192,13.3417;
    1093,179,13.375;
    1129,170,13.4084;
    1166,160,13.4417;
    1203,154,13.4751;
    1242,153,13.5084;
    1278,152,13.5418;
    1317,154,13.5751;
    1353,155,13.6085;
    1391,165,13.6419;
    1428,171,13.6751;
    1465,181,13.7084;
    1503,197,13.7418;
    1541,214,13.7752;
    1581,235,13.8085;
    1619,256,13.8418;
    1661,282,13.8752];

%Center point of Ball from camera 2 
BallW=[365,149,4.3917;
    389,130,4.4246;
    413,109,4.4575;
    432,95,4.4904;
    453,83,4.5234;
    470,72,4.5563;
    490,65,4.5892;
    506,57,4.6221;
    522,53,4.6551;
    539,52,4.688;
    554,53,4.7209;
    568,52,4.7538;
    580,56,4.7868;
    597,63,4.8197;
    607,68,4.8526;
    620,76,4.8855;
    631,83,4.9185;
    643,90,4.9514;
    657,103,4.9843;
    666,115,5.0172;
    678,128,5.0502;
    687,143,5.0831;
    698,156,5.116;
    707,171,5.1489;
    719,188,5.1819;
    729,204,5.2148;
    739,224,5.2477];
%camera 2 calibration
p1=[0,34,0,1]';
p2=[0,94,0,1]';
p3=[0,154,0,1]';
p4=[0,34,90,1]';
p5=[0,94,90,1]';
p6=[0,154,90,1]';
p7=[0,34,180,1]';
p8=[0,94,180,1]';
p9=[0,154,180,1]';
p10=[34,0,0,1]';
p11=[94,0,0,1]';
p12=[154,0,0,1]';
p13=[34,0,90,1]';
p14=[94,0,90,1]';
p15=[154,0,90,1]';
p16=[34,0,180,1]';
p17=[94,0,180,1]';
p18=[154,0,180,1]';
zeroes=[0,0,0,0]';
u1=587;
u2=653;
u3=732;
u4=585;
u5=655;
u6=734;
u7=584;
u8=655;
u9=738;
u10=505;
u11=417;
u12=317;
u13=504;
u14=414;
u15=310;
u16=502;
u17=408;
u18=301;

v1=543;
v2=571;
v3=604;
v4=397;
v5=416;
v6=435;
v7=245;
v8=250;
v9=259;
v10=543;
v11=565;
v12=595;
v13=395;
v14=409;
v15=429;
v16=241;
v17=246;
v18=252;

subu1=horzcat(p1',zeroes',-u1*p1');
subv1=horzcat(zeroes',p1',-v1*p1');
subu2=horzcat(p2',zeroes',-u2*p2');
subv2=horzcat(zeroes',p2',-v2*p2');
subu3=horzcat(p3',zeroes',-u3*p3');
subv3=horzcat(zeroes',p3',-v3*p3');
subu4=horzcat(p4',zeroes',-u4*p4');
subv4=horzcat(zeroes',p4',-v4*p4');
subu5=horzcat(p5',zeroes',-u5*p5');
subv5=horzcat(zeroes',p5',-v5*p5');
subu6=horzcat(p6',zeroes',-u6*p6');
subv6=horzcat(zeroes',p6',-v6*p6');
subu7=horzcat(p7',zeroes',-u7*p7');
subv7=horzcat(zeroes',p7',-v7*p7');
subu8=horzcat(p8',zeroes',-u8*p8');
subv8=horzcat(zeroes',p8',-v8*p8');
subu9=horzcat(p9',zeroes',-u9*p9');
subv9=horzcat(zeroes',p9',-v9*p9');
subu10=horzcat(p10',zeroes',-u10*p10');
subv10=horzcat(zeroes',p10',-v10*p10');
subu11=horzcat(p11',zeroes',-u11*p11');
subv11=horzcat(zeroes',p11',-v11*p11');
subu12=horzcat(p12',zeroes',-u12*p12');
subv12=horzcat(zeroes',p12',-v12*p12');
subu13=horzcat(p13',zeroes',-u13*p13');
subv13=horzcat(zeroes',p13',-v13*p13');
subu14=horzcat(p14',zeroes',-u14*p14');
subv14=horzcat(zeroes',p14',-v14*p14');
subu15=horzcat(p15',zeroes',-u15*p15');
subv15=horzcat(zeroes',p15',-v15*p15');
subu16=horzcat(p16',zeroes',-u16*p16');
subv16=horzcat(zeroes',p16',-v16*p16');
subu17=horzcat(p17',zeroes',-u17*p17');
subv17=horzcat(zeroes',p17',-v17*p17');
subu18=horzcat(p18',zeroes',-u18*p18');
subv18=horzcat(zeroes',p18',-v18*p18');
p = vertcat(subu1, subv1, subu2, subv2, subu3, subv3, subu4, subv4, subu5, subv5, subu6, subv6, subu7, subv7, subu8, subv8, subu9, subv9, subu10, subv10, subu11, subv11, subu12, subv12, subu13, subv13, subu14, subv14, subu15, subv15, subu16, subv16, subu17, subv17, subu18, subv18);
[u,s,v]=svd(p);
[min_val, min_index] = min(diag(s(1:12,1:12)));
m = v(1:12, min_index);
a1=m(1:3);
a2=m(5:7);
a3=m(9:11);
rho=1.0/norm(a3);
r3=rho*a3;
u0=rho *rho*dot(a1, a3);
v0=rho*rho*dot(a2, a3);
cross13 = cross(a1, a3);
cross23 = cross(a2, a3);
cos_theta = -dot(cross13, cross23)/(norm(cross13)*norm(cross23));
sin_theta = sqrt(1-cos_theta*cos_theta);
alpha = rho*rho*norm(cross13)*sin_theta;
beta = rho*rho*norm(cross23)*sin_theta;
r1 = cross23/norm(cross23);
r2 = cross(r3, r1);
b = [m(4),m(8),m(12)]';
k_wei = [alpha, -alpha*cos_theta/sin_theta, u0;0,beta/sin_theta,v0; 0,0,1];

%Camera 1 calibration
p1=[0,34,0,1]';
p2=[0,94,0,1]';
p3=[0,154,0,1]';
p4=[0,34,90,1]';
p5=[0,94,90,1]';
p6=[0,154,90,1]';
p7=[0,34,180,1]';
p8=[0,94,180,1]';
p9=[0,154,180,1]';
p10=[34,0,0,1]';
p11=[94,0,0,1]';
p12=[154,0,0,1]';
p13=[34,0,90,1]';
p14=[94,0,90,1]';
p15=[154,0,90,1]';
p16=[34,0,180,1]';
p17=[94,0,180,1]';
p18=[154,0,180,1]';
u1=943;
u2=1099;
u3=1255;
u4=944;
u5=1105;
u6=1287;
u7=946;
u8=1112;
u9=1301;
u10=798;
u11=670;
u12=520;
u13=795;
u14=663;
u15=504;
u16=793;
u17=654;
u18=489;

v1=802;
v2=846;
v3=887;
v4=540;
v5=570;
v6=604;
v7=261;
v8=274;
v9=287;
v10=809;
v11=872;
v12=944;
v13=544;
v14=583;
v15=632;
v16=261;
v17=275;
v18=295;

subu1=horzcat(p1',zeroes',-u1*p1');
subv1=horzcat(zeroes',p1',-v1*p1');
subu2=horzcat(p2',zeroes',-u2*p2');
subv2=horzcat(zeroes',p2',-v2*p2');
subu3=horzcat(p3',zeroes',-u3*p3');
subv3=horzcat(zeroes',p3',-v3*p3');
subu4=horzcat(p4',zeroes',-u4*p4');
subv4=horzcat(zeroes',p4',-v4*p4');
subu5=horzcat(p5',zeroes',-u5*p5');
subv5=horzcat(zeroes',p5',-v5*p5');
subu6=horzcat(p6',zeroes',-u6*p6');
subv6=horzcat(zeroes',p6',-v6*p6');
subu7=horzcat(p7',zeroes',-u7*p7');
subv7=horzcat(zeroes',p7',-v7*p7');
subu8=horzcat(p8',zeroes',-u8*p8');
subv8=horzcat(zeroes',p8',-v8*p8');
subu9=horzcat(p9',zeroes',-u9*p9');
subv9=horzcat(zeroes',p9',-v9*p9');
subu10=horzcat(p10',zeroes',-u10*p10');
subv10=horzcat(zeroes',p10',-v10*p10');
subu11=horzcat(p11',zeroes',-u11*p11');
subv11=horzcat(zeroes',p11',-v11*p11');
subu12=horzcat(p12',zeroes',-u12*p12');
subv12=horzcat(zeroes',p12',-v12*p12');
subu13=horzcat(p13',zeroes',-u13*p13');
subv13=horzcat(zeroes',p13',-v13*p13');
subu14=horzcat(p14',zeroes',-u14*p14');
subv14=horzcat(zeroes',p14',-v14*p14');
subu15=horzcat(p15',zeroes',-u15*p15');
subv15=horzcat(zeroes',p15',-v15*p15');
subu16=horzcat(p16',zeroes',-u16*p16');
subv16=horzcat(zeroes',p16',-v16*p16');
subu17=horzcat(p17',zeroes',-u17*p17');
subv17=horzcat(zeroes',p17',-v17*p17');
subu18=horzcat(p18',zeroes',-u18*p18');
subv18=horzcat(zeroes',p18',-v18*p18');
p = vertcat(subu1, subv1, subu2, subv2, subu3, subv3, subu4, subv4, subu5, subv5, subu6, subv6, subu7, subv7, subu8, subv8, subu9, subv9, subu10, subv10, subu11, subv11, subu12, subv12, subu13, subv13, subu14, subv14, subu15, subv15, subu16, subv16, subu17, subv17, subu18, subv18);
[u,s,v]=svd(p);
[min_val, min_index] = min(diag(s(1:12,1:12)));
m = v(1:12, min_index);
a1=m(1:3);
a2=m(5:7);
a3=m(9:11);
rho=1.0/norm(a3);
r3=rho*a3;
u0=rho *rho*dot(a1, a3);
v0=rho*rho*dot(a2, a3);
cross13 = cross(a1, a3);
cross23 = cross(a2, a3);
cos_theta = -dot(cross13, cross23)/(norm(cross13)*norm(cross23));
sin_theta = sqrt(1-cos_theta*cos_theta);
alpha = rho*rho*norm(cross13)*sin_theta;
beta = rho*rho*norm(cross23)*sin_theta;
r1 = cross23/norm(cross23);
r2 = cross(r3, r1);
b = [m(4),m(8),m(12)]';
k_nan = [alpha, -alpha*cos_theta/sin_theta, u0;0,beta/sin_theta,v0; 0,0,1];

%calculate essential matrix
%fixed corresponding points in two figures
P1=[641,179;648,314;708,205;776,253;771,189;438,426;438,479;541,411;542,462;437,574;71,454;69,479;64,537;35,70;25,124;300,263;376,183;319,201;];
P2=[1643,217;1680,466;1726,237;1823,302;1800,180;1732,681;1743,770;1867,627;1883,729;1769,946;1364,766;1368,804;1372,885;1465,168;1469,248;536,655;638,518;620,558];
P1x=[];
P1y=[];
P2x=[];
P2y=[];
kn = inv(k_nan);
kw = inv(k_wei);
point = double([1,1,1]);
cpoint = double([1,1,1]);
for i=1:15
    point(1)=P1(i,1);
    point(2)=P1(i,2);
    cpoint = kw * point';
    P1x = [P1x,cpoint(1)];
    P1y = [P1y,cpoint(2)];
end

for i=1:15
    point(1)=P2(i,1);
    point(2)=P2(i,2);
    cpoint = kn * point';
    P2x = [P2x,cpoint(1)];
    P2y = [P2y,cpoint(2)];
end

subcoef = [0,0,0,0,0,0,0,0];
coef = subcoef;
for i=1:1:14 coef = vertcat(coef, subcoef);end;
for i = 1:1:15 coef(i,1) = P1x(i)*P2x(i); coef(i,2) = P1x(i)*P2y(i); coef(i,3) = P1x(i); coef(i,4) = P1y(i)*P2x(i); coef(i,5) = P1y(i)*P2y(i);  coef(i,6) = P1y(i); coef(i,7) = P2x(i); coef(i,8) = P2y(i); end;
%using svd calculate the fundamental matrix
f_solution = coef\[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]';
%E = [f_solution(1), f_solution(2), f_solution(3); f_solution(4), f_solution(5), f_solution(6); f_solution(7), f_solution(8), 1];

%using matlab function estimateFundamentalMatrix to calculate
F = estimateFundamentalMatrix(P1,P2);
%the result function is different each time, choose the most reasonal one
F2 = ...
  [6.8532322583998537E-8 8.471385712696026E-7 -0.00060091911689349672;
   4.2465891047268738E-7 1.2975400457221454E-7 -0.0017013409158305832;
   -0.00038036597730768131 0.001210062561952217 0.99999756770306469];
F = F2;
%calculate the essential matrix
E= k_nan'*F*k_wei;

%reconstruction
[ue,de,ve] = svd(E);
W = [0,-1,0;1,0,0;0,0,1];
Z = [0,1,0;-1,0,0;0,0,0];
s = ue*Z*ue';
r = ue*W'*ve';

%triangulation
mp1 = [k_wei,zeros(3,1)];
mp2 = k_nan * [r,-[-s(2,3);s(1,3);-s(1,2)]];
uv1 = [P1;BallW(:,1:2)];
uv2 = [P2;BallN(:,1:2)];
[paircount,tmp] = size(uv1);
result = zeros(paircount,4);

vp11 = mp1(1,1:4);
vp12 = mp1(2,1:4);
vp13 = mp1(3,1:4);

vp21 = mp2(1,1:4);
vp22 = mp2(2,1:4);
vp23 = mp2(3,1:4);

A = zeros(4,4);
for i = 1:paircount
  A(1,1:4) = uv1(i,1)*vp13-vp11;
  A(2,1:4) = uv1(i,2)*vp13-vp12;
  A(3,1:4) = uv2(i,1)*vp23-vp21;
  A(4,1:4) = uv2(i,2)*vp23-vp22;
  [ua,ea,va]=svd(A'*A);
  [min_val, min_index] = min(diag(ea(1:4,1:4)));
  m = va(1:4, min_index);
  m = m /m(4);
  result(i,1:4) = m';
  end;

 figure;
 result3=result(:,1:3);
 %rotate the result so the board will be vertical to the horizon
 x= result(2,1:3)-result(1,1:3);
 %r=vrrotvec2mat(vrrotvec([0,0,-1],x/(x*x')));
 r = [0.99926892228815045, -0.028821792328848978, 0.025118225176763775;
   -0.028821792328848978, -0.13626184957269305, 0.99025350927829658;
   -0.025118225176763775, -0.99025350927829658, -0.13699292728454238];


 result3x = result3*r;

 %rotate the result so the board will be parrel to the player
 ax = [1,-2.8,0];
 %r2 = vrrotvec2mat(vrrotvec([1,0,0],ax/(ax*x'))); 
 r2 = [0.33633639699815626, 0.94174191159483744, 0;
   -0.94174191159483744, 0.33633639699815626, -0;
   -0, 0, 1];


 result3x = result3x*r2;

 %plot the fixed reference point
 plot3(-result3x(1:5,1), result3x(1:5,2),result3x(1:5,3),'b');
 hold on;
 plot3(-result3x(6:10,1),result3x(6:10,2),result3x(6:10,3),'b');
 hold on;
 plot3(-result3x(11:13,1),result3x(11:13,2),result3x(11:13,3),'b');
 hold on;
 plot3(-result3x(16:18,1),result3x(16:18,2),result3x(16:18,3),'b');
 hold on;
 
 %plot the result
 plot3(-result3x(19:paircount,1), result3x(19:paircount,2),result3x(19:paircount,3),'r');
 
 
 %
 xlabel('X');
 ylabel('Y');
 zlabel('Z');
 axis equal;
 axis([0 3 0 3 -1 2]);
 grid on;
 CameraPosition= [-23.553 -4.238 4.296];
 view(CameraPosition);
 
 figure 
  %plot the fixed reference point
 plot3(-result3x(1:5,1), result3x(1:5,2),result3x(1:5,3),'b');
 hold on;
 plot3(-result3x(6:10,1),result3x(6:10,2),result3x(6:10,3),'b');
 hold on;
 plot3(-result3x(11:13,1),result3x(11:13,2),result3x(11:13,3),'b');
 hold on;
 plot3(-result3x(16:18,1),result3x(16:18,2),result3x(16:18,3),'b');
 hold on;
 %plot the result
 plot3(-result3x(19:paircount,1), result3x(19:paircount,2),result3x(19:paircount,3),'r');

 xlabel('X');
 ylabel('Y');
 zlabel('Z');
 axis equal;
 axis([0 3 0 3 -1 2]);
 grid on;
 CameraPosition= [-1 0 0];
 view(CameraPosition)
 
 figure 
  %plot the fixed reference point
 plot3(-result3x(1:5,1), result3x(1:5,2),result3x(1:5,3),'b');
 hold on;
 plot3(-result3x(6:10,1),result3x(6:10,2),result3x(6:10,3),'b');
 hold on;
 plot3(-result3x(11:13,1),result3x(11:13,2),result3x(11:13,3),'b');
 hold on;
 plot3(-result3x(16:18,1),result3x(16:18,2),result3x(16:18,3),'b');
 hold on;
 %plot the result
 plot3(-result3x(19:paircount,1), result3x(19:paircount,2),result3x(19:paircount,3),'r');

 xlabel('X');
 ylabel('Y');
 zlabel('Z');
 axis equal;
 axis([0 3 0 3 -1 2]);
 grid on;
 CameraPosition= [0 -1 0];
 view(CameraPosition)
 
 figure 
 %plot the fixed reference point
 plot3(-result3x(1:5,1), result3x(1:5,2),result3x(1:5,3),'b');
 hold on;
 plot3(-result3x(6:10,1),result3x(6:10,2),result3x(6:10,3),'b');
 hold on;
 plot3(-result3x(11:13,1),result3x(11:13,2),result3x(11:13,3),'b');
 hold on;
 plot3(-result3x(16:18,1),result3x(16:18,2),result3x(16:18,3),'b');
 hold on;
 %plot the result
 plot3(-result3x(19:paircount,1), result3x(19:paircount,2),result3x(19:paircount,3),'r');

 xlabel('X');
 ylabel('Y');
 zlabel('Z');
 axis equal;
 axis([0 3 0 3 -1 2]);
 grid on;
 CameraPosition= [0 0 1];
 view(CameraPosition)

