function demoPLSA

prom=0;

for i=1:100

	X=10; Y=10; Z=4;
	pz=rand(Z,1); pz=pz/sum(pz);
	pxgz=rand(X,Z); pxgz=pxgz./repmat(sum(pxgz),X,1);
	pygz=rand(Y,Z); pygz=pygz./repmat(sum(pygz),Y,1);
	pxy=zeros(X,Y);
	for z=1:Z
		pxy =pxy+pxgz(:,z)*pygz(:,z)'.*pz(z);
	end

	opts.maxit = 100;
	opts.tol=0.000001;
	opts.plotprogress=0;
	opts.randinit=1;
	[tpxgz,tpygz,tpz,tpxy]=plsa(pxy,Z,opts);

	[ttpxgz]=testPLSA(pxy,Z,tpygz,tpz,100);
	% [ttpxgz2]=testPLSAinv(pxy,tpygz);

	% disp('----pxy vs tpxy ---')
	% mean(mean(abs(pxy-tpxy)))
	% disp ('----pxgz vs pxgz ---')
	% mean(mean(abs(pxgz-tpxgz)))
	% disp ('----pygz vs tpygz ---')
	% mean(mean(abs(pygz-tpygz)))
	% disp ('----pz vs tpz ---')
	% mean(mean(abs(pz-tpz)))

	% disp('*******************')
	% disp('----testPLSA----')
	prom = prom +mean(mean(abs(tpxgz-ttpxgz)));
end

prom/100