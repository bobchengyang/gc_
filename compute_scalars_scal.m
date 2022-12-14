function [ M_current_eigenvector0,...
    scaled_M,...
    scaled_factors,lmin ] = ...
    compute_scalars_scal( ...
    M_updated_current,...
    M_current_eigenvector0,...
    first_or_not)
H_dim=length(M_current_eigenvector0);
if first_or_not~=0
[M_current_eigenvector0,lmin] = ...
    lobpcg_fv(...
    M_current_eigenvector0,...
    M_updated_current*1/1,...
    1e-16,...
    1e3);
else
M_current_eigenvector0=ones(H_dim,1);  
M_current_eigenvector0(end)=-1;
end
%     1e-16,...
%     1e3
% 1e-4
% 200

a=M_current_eigenvector0(:,1);

scaled_M = (1./a) .* M_updated_current .* a';
scaled_factors = (1./a) .* ones(H_dim) .* a';

end

