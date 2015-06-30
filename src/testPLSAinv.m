function [tpxgz]=testPLSAinv(pxy,tpygz)
[X Y]=size(pxy);

tpxgz=pxy*(inv(tpygz*transpose(tpygz))*tpygz)