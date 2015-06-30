function [tpxgz]=testPLSA(pxy,Z,tpygz,tpz,maxit)
[X Y]=size(pxy);
tpxgz=rand(X,Z); tpxgz=tpxgz./repmat(sum(tpxgz),X,1);

for emloop=1:maxit
	% E-step:
	for z=1:Z
		qzgxy(z,:,:)=tpxgz(:,z)*tpygz(:,z)'.*tpz(z)+eps;
	end
	for z=1:Z
		qzgxy(z,:,:)=squeeze(qzgxy(z,:,:))./repmat(sum(sum(qzgxy(z,:,:),3),2),X,Y);
	end
	qzgxy=qzgxy./repmat(sum(qzgxy),Z,1); 

	% M-step:
	for z=1:Z
		tpxgz(:,z)=sum(pxy.*squeeze(qzgxy(z,:,:)),2);
	end
	tpxgz=tpxgz./repmat(sum(tpxgz),X,1);

end