function [xHatUpdated r S Ppost]=UKFp(t,y,u,xinit,dt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% UNSCENTED KF ROUTINE %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%   Tuning Parameters  %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%parameters of UKF
k=0; %usually zero
alpha=1; %spread around the mean
beta=2; %2 is optimal for gaussian

%parameters of process noise
Cbn=11000; %current bias noise (Force) (Main effect of residual oscillations)
Wdn=.03; %wave disturbance noise (Position) (Main effect of residual & covariance matching)
Pn=.001; %Position noise (Direct effect on estimated Position covariance)
Vn=.0001; %Velocity noise (Direct effect on Heading error and covariance)
wn=.001; %Omega noise
sn=1.5; %Sigma noise

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%   Initialize memory  %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
persistent P xHat

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%   Initialize filter  %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(P)
    P=zeros(21);
    xHat=xinit;
end

L=21;
m=length(y);

Lambda=(alpha^2)*(L+k)-L;

Wm=[Lambda/(L+Lambda) 1/(2*(L+Lambda))+zeros(1,(2*L))];
Wc=Wm;
Wc(1)=Wm(1)+(1-alpha^2+beta);

XhatPRE_UKF=zeros(size(xHat));
XhatSIGMA=zeros(L,(2*L)+1);
YhatSIGMA=zeros(m,(2*L)+1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%   Initialize system  %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C=[eye(3) zeros(3,9) eye(3);
    zeros(3,3) eye(3) zeros(3,9)];

R=1.1*[0.1^2 0 0 0 0 0;
    0 0.1^2 0 0 0 0;
    0 0 (.01/180*pi)^2 0 0 0;
    0 0 0 0.1^2 0 0;
    0 0 0 0 0.1^2 0;
    0 0 0 0 0 (.01/180*pi)^2];

sig1=xHat(19); sig2=xHat(20); sig3=xHat(21);
Sigma=diag([sig1 sig2 sig3]);
Sigma2=[zeros(3,3); 
        Sigma];

Psi=eye(3);
    
E=[zeros(6,6); 
    Psi zeros(3,3);
    zeros(6,3) Sigma2;
    zeros(6,6)];

Qstd2=diag([Cbn,Cbn,Cbn, Wdn,Wdn, Wdn/180*pi]);
Q2=Qstd2.^2;
Qstd1=diag([Pn,Pn,0,Vn,Vn,Vn/180*pi,0,0,0,0,0,0,0,0,0,wn,wn,wn,sn,sn,sn]);
Q1=Qstd1.^2;
Q=(E*Q2*E'+Q1)*dt;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  Propagate state and estimate %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[var varRES]=sqrtm((L+Lambda)*P);
var=real(var);
for i=0:2*L
    if i==0
        xSQUIGGLY=zeros(L,1);
    elseif i<=L
        xSQUIGGLY=var(i,:)';
    else
        xSQUIGGLY=-var(i-L,:)';
    end
    sigmaPOINTS=xHat(1:21)+xSQUIGGLY;
    sigmaPOINTS(22:24)=xHat(22:24);
    XhatSIGMAdot=SimVesselNLmodel(t,sigmaPOINTS,u,dt,0);
    XhatSIGMA(1:15,i+1)=sigmaPOINTS(1:15)+XhatSIGMAdot(1:15)*dt;
    XhatSIGMA(16:21,i+1)=sigmaPOINTS(16:21)+(Q(16:end,16:end)).^.5*randn(6,1);
    XhatPRE_UKF(1:21)=XhatPRE_UKF(1:21)+Wm(i+1)*XhatSIGMA(1:21,i+1);
end
XhatPRE_UKF(22:24)=xHat(22:24);
Xresidual_UKF=(XhatSIGMA-XhatPRE_UKF(1:21,ones(1,(2*L)+1)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%  Propagate Apriori covar.  %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PPRE_UKF=Xresidual_UKF*diag(Wc)*Xresidual_UKF'+Q;
PPRE_UKF=(PPRE_UKF+PPRE_UKF')/2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%   Propagate meas.   %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isnan(y)
    xHat=XhatPRE_UKF;
    P=PPRE_UKF;
    S=C*PPRE_UKF(1:15,1:15)*C'+R;
    r=NaN*ones(size(y));
else
    Y=y;
    Yhat_UKF=zeros(size(Y));

    [var varRes] = sqrtm((L+Lambda)*PPRE_UKF);
    var=real(var);
    for i=0:2*L
        if i==0
            xSQUIGGLY=zeros(L,1);
        elseif i<=L
            xSQUIGGLY=var(i,:)';
        else
            xSQUIGGLY=-var(i-L,:)';
        end
    sigmaPOINTS=XhatPRE_UKF(1:21)+xSQUIGGLY;
    YhatSIGMA(:,i+1)=C*sigmaPOINTS(1:15);
    Yhat_UKF=Yhat_UKF+Wm(i+1)*YhatSIGMA(:,i+1);
    end

    Yresidual_UKF=(YhatSIGMA-Yhat_UKF(:,ones(1,(2*L)+1)));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%  Kalman Gain   %%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    S=Yresidual_UKF*diag(Wc)*Yresidual_UKF'+R;
    PXY_UKF=Xresidual_UKF*diag(Wc)*Yresidual_UKF';
    K_UKF=PXY_UKF/S;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%  State Update  %%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    r=Y-Yhat_UKF;
    xHat=XhatPRE_UKF(1:21)+K_UKF*(r);
    xHat(22:24)=XhatPRE_UKF(22:24);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%  Propagate Posteriori covar. %%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    P=PPRE_UKF-K_UKF*S*K_UKF';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%   output  %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P=(P+P')/2;
xHatUpdated=xHat;
Ppost=P;

end
