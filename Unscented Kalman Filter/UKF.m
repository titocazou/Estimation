function [xHatUpdated r S Ppost]=UKF(t,y,u,xinit,dt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% UNSCENTED KF ROUTINE %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%   Tuning Parameters  %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%parameters of UKF
k=0; %usually zero
alpha=1; %spread around the mean
beta=0; %2 is optimal for gaussian

%parameters of process noise
Cbn=11000; %current bias noise (Force) (Main effect of residual oscillations)
Wdn=.01; %wave disturbance noise (Position) (Main effect of residual & covariance matching)
Pn=.001; %Position noise (Direct effect on estimated Position covariance)
Vn=.00001; %Velocity noise (Direct effect on Heading error and covariance)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%   Initialize memory  %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
persistent PPOST_UKF XhatPOST_UKF

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%   Initialize system  %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C=[eye(3) zeros(3,9) eye(3);
    zeros(3,3) eye(3) zeros(3,9)];

R=1*[0.1^2 0 0 0 0 0;
    0 0.1^2 0 0 0 0;
    0 0 (.01/180*pi)^2 0 0 0;
    0 0 0 0.1^2 0 0;
    0 0 0 0 0.1^2 0;
    0 0 0 0 0 (.01/180*pi)^2];

sig1=50; sig2=50; sig3=50;

Sigma=diag([sig1 sig2 sig3]);

Sigma2=[zeros(3,3); 
        Sigma];

Psi=eye(3);
    
E=[zeros(6,6); 
    Psi zeros(3,3);
    zeros(6,3) Sigma2];

Qstd2=diag([Cbn,Cbn,Cbn, Wdn,Wdn, Wdn/180*pi]);
Q2=Qstd2.^2;
Qstd1=diag([Pn,Pn,0,Vn,Vn,Vn/180*pi,0,0,0,0,0,0,0,0,0]);
Q1=Qstd1.^2;

Q=(E*Q2*E'+Q1)*dt;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%   Initialize filter  %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(PPOST_UKF)
    PPOST_UKF=zeros(15);
    XhatPOST_UKF=xinit;
end

L=length(XhatPOST_UKF);
m=length(y);

Lambda=(alpha^2)*(L+k)-L;

Wm=[Lambda/(L+Lambda) 1/(2*(L+Lambda))+zeros(1,(2*L))];
Wc=Wm;
Wc(1)=Wm(1)+(1-alpha^2+beta);

XhatPRE_UKF=zeros(size(XhatPOST_UKF));
XhatSIGMA=zeros(L,(2*L)+1);
YhatSIGMA=zeros(m,(2*L)+1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  Propagate state and estimate %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[var varRES]=sqrtm((L+Lambda)*PPOST_UKF);
var=real(var);
for i=0:2*L
    if i==0
        xSQUIGGLY=zeros(L,1);
    elseif i<=L
        xSQUIGGLY=var(i,:)';
    else
        xSQUIGGLY=-var(i-L,:)';
    end
    sigmaPOINTS=XhatPOST_UKF+xSQUIGGLY;
    XhatSIGMAdot=SimVesselNLmodel(t,sigmaPOINTS,u,dt,0);
    XhatSIGMA(:,i+1)=sigmaPOINTS+XhatSIGMAdot(1:15)*dt;
    XhatPRE_UKF=XhatPRE_UKF+Wm(i+1)*XhatSIGMA(:,i+1);
end

Xresidual_UKF=(XhatSIGMA-XhatPRE_UKF(:,ones(1,(2*L)+1)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%  Propagate Apriori covar.  %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PPRE_UKF=Xresidual_UKF*diag(Wc)*Xresidual_UKF'+Q;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%   Propagate meas.   %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isnan(y)
    XhatPOST_UKF=XhatPRE_UKF;
    PPOST_UKF=PPRE_UKF;
    S=C*PPRE_UKF*C'+R;
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
    sigmaPOINTS=XhatPRE_UKF+xSQUIGGLY;
    YhatSIGMA(:,i+1)=C*sigmaPOINTS;
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
    XhatPOST_UKF=XhatPRE_UKF+K_UKF*(r);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%  Propagate Posteriori covar. %%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    PPOST_UKF=PPRE_UKF-K_UKF*S*K_UKF';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%   output  %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xHatUpdated=XhatPOST_UKF;
Ppost=PPOST_UKF;

end