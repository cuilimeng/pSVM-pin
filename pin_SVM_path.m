function [alphas, offset] = pin_SVM_path(X, Y, C, tau_Set)
%Solution path traversal algorithm for pin-SVM 
%   [ALPHAS OFFSET] = PIN_SVM_PATH(X,Y,C,TAU_Set)
%   trains pin-SVM for all tau by traversing the solution path. X is a numeric
%   matrix of predictor data. Y is a column vector of labels. Each values of 
%   Y is +1 or -1 representing which group the corresponding data
%   point belongs to. C gives the boxconstraints for each dual variable.
%   TAU_SET defines the set of the output (the optimal alphas and offset)
%
% References:
% [1] X. Huang, L. Shi, and J.A.K. Suykens: Support Vector Machine Classifier with Pinball Loss, 
%     IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(2014), 984-997.
% [2] X. Huang, L. Shi, and J.A.K. Suykens: Solution Path for pin-SVM Classifiers with Positive 
%     and Negative tau Values, ESAT-STADIUS, internal report 14-123.

global rbf_sigma
tol = 10e-5;

if nargin == 3
    tau_Set = -0.99 : 0.01 : 1; % default
end


tau_Set_cursor = 1;

alphas = zeros(length(Y), length(tau_Set));
offset = zeros(size(tau_Set));

tau = -1;

%author: clm
%update: change to linear kernal
KK = rbf_kernel2(X, X);
% KK = kernel_f(X, 'linear');

alphas_0 = -tau * C;
offset_0 = 0;
tau_0 = -1;

% offset_0 = 4.8282;

value_0 = Y.*(KK * (Y.*alphas_0) + offset_0);
L = find(value_0 < 1 - tol);
R = find(value_0 > 1 + tol);
E = find(abs(value_0 - 1) <= tol);
F = [];
Index_tried = [];


while isempty(R)
    offset_0 = 10*(rand - 0.5);
    value_0 = Y.*(KK * (Y.*alphas_0) + offset_0);
    L = find(value_0 < 1 - tol);
    R = find(value_0 > 1 + tol);
    E = find(abs(value_0 - 1) <= tol);    
end

initial_offset = offset_0;

delta_offset_last = 0;
delta_alphas_last = 0;

tau_cursor = 1;
initial_adjust = 1;
ppp = 0;
while tau_0 < 1  && tau_Set_cursor <= length(tau_Set)
  
    removed = 0;
    ppp = ppp + 1;
    if ppp > 10
        break;
%         sss = 1;
    end
    
    delta_alphas = zeros(size(Y));
    delta_alphas(L) = 0;
    delta_alphas(R) = -C(R);
    delta_value_part = Y.*(KK * (Y.*delta_alphas));
    
    
    if isempty(E)   
        
        L_0 = L;
        R_0 = R;
        E_0 = E;
        
        ii = 0;
        
        vio = sum(Y.* delta_alphas);
        R_move_indx = [];
        
        
        [offset_candidate, sort_indx] = sort((1 - value_0)./Y);
        sort_indx_R = ismember(sort_indx, R);   
        sort_indx_L = ismember(sort_indx, L);
        offset_least = min(find(offset_candidate > tol));
        if isempty(offset_least)
            offset_least = length(sort_indx) + 1;
        end
        offset_new = offset_0;
%         [~, offset_least] = min(abs(offset_candidate));
        
        R_L = [];
        L_R = [];
        R_E = [];
        L_E = [];
        if abs(vio) > tol
            for ii = offset_least : length(sort_indx)
                vio;
                offset_new = offset_0 + offset_candidate(ii);
                indx = sort_indx(ii);
  
                if sort_indx_R(ii)  % R -- > L
                    if Y(indx)*sign(vio) < 0        % vio decreses
                        if abs(vio) + tau_0 * C(indx) < tol
                            alpha_E = tau_0 * C(indx) + abs(vio);
                            vio = 0;
                            R_E = indx;
                            break;
                        else
                            if alphas_0( indx ) > C(indx) - tol
                                vio = vio - tau_0*Y(indx)*C(indx);
                                R_L = [R_L; indx];
                            else
                                break;
                            end
                        end
                    else
                        if alphas_0( indx ) > C(indx) - tol
                            vio = vio - tau_0*Y(indx)*C(indx);
                            R_L = [R_L; indx];
                        else
                            break;
                        end
                    end
                else                %  L -- > R
                    if Y(indx)*sign(vio) > 0    % vio decreses
                        if abs(vio) - C(indx) < tol
                            alpha_E = -abs(vio);
                            vio = 0;
                            L_E = indx;
                            break;
                        else
                            if alphas_0( indx ) < -tau_0 * C(indx) + tol
                                vio = vio + tau_0* Y(indx)*C(indx);
                                L_R = [L_R; indx];
                            else
                                break;
                            end
                        end
                    else
                        if alphas_0( indx ) < -tau_0 * C(indx) + tol
                            vio = vio - Y(indx)*C(indx);
                            L_R = [L_R; indx];
                        else
                            break;
                        end
                    end
                end
            end
        end
        if abs(vio) > tol
            vio = sum(Y.* delta_alphas);
            R_L = [];
            L_R = [];
            R_E = [];
            L_E = [];
            offset_least = max(find(offset_candidate < tol));
            if isempty(offset_least)
                offset_least = 0;
            end
            for ii = offset_least : -1 : 1
                offset_new = offset_0 + offset_candidate(ii);
                indx = sort_indx(ii);

                if sort_indx_R(ii)  % R -- > L
                    if Y(indx)*sign(vio) < 0        % vio decreses
                        if abs(vio) + tau_0 * C(indx) < tol
                            alpha_E = tau_0 * C(indx)+abs(vio);
                            vio = 0;
                            R_E = indx;
                            break;
                        else
                            vio = vio - tau_0*Y(indx)*C(indx);
                            R_L = [R_L; indx];
                        end
                    else
                        vio = vio + C(indx);
                        R_L = [R_L; indx];
                    end
                else                %  L -- > R
                    if Y(indx)*sign(vio) > 0    % vio decreses
                        if abs(vio) - C(indx) < tol
                            alpha_E = -abs(vio);
                            vio = 0;
                            L_E = indx;
                            break;
                        else
                            vio = vio + tau_0*Y(indx)*C(indx);
                            L_R = [L_R; indx];
                        end
                    else
                        vio = vio + tau_0*Y(indx)*C(indx);
                        L_R = [L_R; indx];
                    end
                end
            end
        end
        E = [R_E; L_E];
        R = [L_R; setdiff(R_0, [R_L; R_E])];
        L = [R_L; setdiff(L_0, [L_R; L_E])];
        
               
        delta_alphas(L) = 0;
        delta_alphas(R) = -C(R);
        delta_alphas(E) = 0;       
        
              
        delta_value_part = Y.*(KK * (Y.*delta_alphas));
        value_0 = value_0 + Y * (offset_new - offset_0);
        offset_0 = offset_new;
        
        if initial_adjust
            initial_offset = offset_0;
        end
        
%         delta_alphas(E) = alpha_E;
        
        
%         sum(delta_alphas.*Y);        
    end
    
    
    delta_offset = 0;
    E_step = Inf;
    E_unchanged = 1;
    possible_enter = 0;
    E_out = [];
    while E_unchanged && ~isempty(E)
        A_temp = [0 Y(E)'; Y(E), (Y(E) * Y(E)').*KK(E, E)];
        R_temp = [ -sum(Y([R;L]).*delta_alphas([R;L])); ...
                zeros(size(E)) - delta_value_part(E)];
        degenerated = 0;
        
%         if max((isnan(b_temp))) || (max(b_temp) > 10^10)
%             degenerated = 1;
%         end
%         if min(abs(eig(A_temp))) < 10^-6
%             degenerated = 1;
%         end
        degenerated = size(A_temp,1) - rank(A_temp, 10^-6);

        if  degenerated  > 0           
            
            A_reduced = [A_temp( 1 : end - degenerated, :); ...
                zeros(degenerated, length(E) - degenerated + 1), eye(degenerated)];
            b_reduced = alphas_0(E(end - degenerated + 1 : end)) > C(E(end - degenerated + 1)) - tol;
            b_reduced = 1 - 2 * b_reduced;
            R_reduced = [R_temp( 1 : end - degenerated); b_reduced];

  
            b_temp = A_reduced^-1 * R_reduced;  
            
            removed = 1;

%             A_extended = [A_temp; ...
%                 zeros(degenerated, length(E) - degenerated + 1), eye(degenerated)];
%             R_extended = alphas_0(E(end - degenerated + 1 : end)) > C(E(end - degenerated + 1)) - tol;
%             R_extended = 1 - 2 * R_extended;
%             R_extended = [R_temp; R_extended];
%             b_temp = R_extended\A_extended; 
%             b_temp = (A_extended' * A_extended)^-1 * A_extended' * R_extended;
            
        else
            b_temp = A_temp^-1 * R_temp;
        end
        delta_offset = b_temp(1);
        delta_alphas(E) = b_temp(2:end);
        
        if ~initial_adjust
            E_step = Inf;
            for ii = 1 : length(E)
                if delta_alphas(E(ii)) > tol
                    temp = (C(E(ii)) - alphas_0(E(ii)))/delta_alphas(E(ii));
                    if  E_step > temp
                        E_step = temp;
                        E_type = 1;% E -> L
                        E_indx = E(ii);
                        change_indx = ii;
                    end
                else
                    if delta_alphas(E(ii)) < -C(E(ii)) - tol
                        temp = (-tau_0*C(E(ii)) - alphas_0(E(ii)))/(delta_alphas(E(ii))+C(E(ii)));
                        if E_step > temp
                            E_step = temp;
                            E_type = 2;% E -> R
                            E_indx = E(ii);
                            change_indx = ii;
                        end
                    end
                end
            end
        end
        if E_step > 10^-7
            E_unchanged = 0;
        else
            if possible_enter == 0
                possible_R_E = find(abs(value_0(R) - 1) < tol);
                possible_R_E = intersect(possible_R_E, find( abs(alphas_0(R) + tau_0*C(R)) < tol));
                possible_R_E = R(possible_R_E);
                possible_L_E = find(abs(value_0(L) - 1) < tol);
                possible_L_E = intersect(possible_L_E, find( abs(alphas_0(L) - C(L)) < tol));
                possible_L_E = L(possible_L_E);
            else
                E = setdiff(E, possible_R_E);
                E = setdiff(E, possible_L_E);
                R = [R; possible_R_E];
                L = [L; possible_L_E];
                possible_R_E = [];
                possible_L_E = [];
            end
            
            if isempty(possible_R_E) && isempty(possible_L_E)
                E = setdiff(E, E_indx);
                if E_type == 1
                    L = [L; E_indx];
                    delta_alphas(E_indx) = 0;
                    E_out = [E_out, E_indx];
                    E_indx = [];                   
                else
                    R = [R; E_indx];
                    delta_alphas(E_indx) = -C(E_indx);
                    E_out = [E_out, E_indx];
                    E_indx = [];
                end
                delta_alphas(E) = 0;
                delta_value_part = Y.*(KK * (Y.*delta_alphas));
            else
                possible_enter = 1;
                E = [E; possible_R_E; possible_L_E];
                R = setdiff(R, possible_R_E);
                L = setdiff(L, possible_L_E);
            end
            
        end
    end
    
    delta_value = Y.*(KK * (Y.*delta_alphas) + delta_offset);
%     delta_value(E)
%     sum(Y.* delta_alphas)
    
%     if norm(delta_value(E)) > tol
%         sss = 1
%     end

    L_step_indx = find(delta_value(L) > tol);
    L_step = (1 - value_0(L(L_step_indx)))./delta_value(L(L_step_indx));
    [L_step, indx_temp] = min(L_step);
    L_step = max(L_step, 0);
    L_step_indx = L(L_step_indx(indx_temp));
    if isempty(L_step)
        L_step = Inf;
    end
    
    
    R_step_indx = find(delta_value(R) < -tol);
    R_step = (1 - value_0(R(R_step_indx)))./delta_value(R(R_step_indx));
    [R_step, indx_temp] = min(R_step);
    R_step_indx = R(R_step_indx(indx_temp)); 
%     if R_step < 0
%         sss = 1
%     end
    
    R_step = max(R_step, 0);
    if isempty(R_step)
        R_step = Inf;
    end
   

    
    
    step = min(E_step, min(R_step, L_step));
    
    if step > tol
        ppp = 0;
        Index_tried = [];
    end
    
    if step < 1
        alphas_new = alphas_0 + step * delta_alphas;
        offset_new = offset_0 + step * delta_offset;
%         delta_offset
        tau_new = tau_0 + step;
        value_new = Y.*(KK * (Y.*alphas_new) + offset_new);
        delta_alphas_last = delta_alphas;
        delta_offset_last = delta_offset;
    else
        delta_alphas = delta_alphas_last;
        delta_offset = delta_offset_last;
        break;
    end
    
    
    while tau_new > tau_Set(tau_Set_cursor)
        step_zero = tau_Set(tau_Set_cursor) - tau_0;
        alphas(:, tau_Set_cursor) = alphas_0 + step_zero * delta_alphas;
        offset(tau_Set_cursor) = offset_0 + step_zero * delta_offset;
        tau_Set_cursor = tau_Set_cursor + 1;
        if tau_Set_cursor > length(tau_Set)
            break;
        end
    end
    if tau_new > 1
        break;
    end
    
    if removed %&& ~E_unchanged
        if ~isempty(E)
            F = [F; E(end)];
            E = E( 1 : end - 1);
        end
    end


%     alphas_new(R(end));

    
    if step == E_step
        E = setdiff(E, E_indx);
        if E_type == 1            
            L = [L; E_indx];
        else
            R = [R; E_indx];            
        end        
    end
    if step == R_step  
        if ~ismember( R_step_indx, E_out);
            R = setdiff(R, R_step_indx);
            E = [E; R_step_indx];
        end
    end
    if step == L_step
        if ~ismember( L_step_indx, E_out);
            L = setdiff(L, L_step_indx);
            E = [E; L_step_indx];
        end
    end
    
    tau_0 = tau_new;
    
    Index_current = ones(size(Y));
    Index_current(L) = -1;
    Index_current(E) = 0;
    been_tried = 0;
    for ttt = 1 : size(Index_tried, 2)
        been_tried = been_tried + (norm(Index_current - Index_tried(:,ttt), 1) < 0.01);
    end
    if been_tried > 0
        ppp = ppp + 1;
    else
        ppp = 0;
        Index_tried = [Index_tried, Index_current];
    end
    

%         ppp
%     correct = 1;
%     if ~isempty(R)
%         correct = correct * (min(value_new(R)) > 1 - tol);
%         correct = correct * (norm(alphas_new(R) + tau_new * C(R)) < tol);
%     end
%     if ~isempty(E)        
%         correct = correct * (max(abs(value_new(E) - 1)) < tol);
%     end
%     if ~isempty(L)
%         correct = correct * (max(value_new(L)) < 1 + tol);
%         correct = correct * (norm(alphas_new(L) - C(L)) < tol);
%     end
%     correct = correct * (sum(Y.*alphas_new) < tol);     
%     if ~correct
%         sss = 1
%     end
    
    value_0 = value_new; 
    alphas_0 = alphas_new;
    offset_0 = offset_new;
    
    if (max(value_0) - min(value_0)) < tol
        break;
    end
    
%     str = '------------'
% %     ppp
% %     max(abs(alphas_0(R) + tau_0 * C(R)))
%     
%     tau_0
%     min(value_0(R))
%     value_0(E)'
%     max(value_0(L))
    
    initial_adjust = 0;
    
    E;
    
    if isempty(L) && isempty(R)
        break;
    end
    
    
%     if step < tol
%         initial_adjust = 1;
%     else
%         initial_adjust = 0;
%     end
    
    %     tau_0

    %     L = find(value_0 < 1 - tol);
    %     R = find(value_0 > 1 + tol);
    %     E = find(abs(value_0 - 1) <= tol);
  
end
delta_alphas(F) = 0;

for ii = tau_Set_cursor : length(tau_Set)
    step_zero = tau_Set(ii) - tau_0;
    alphas(:, ii) = alphas_0 + step_zero * delta_alphas;
    offset(ii) = offset_0 + step_zero * delta_offset;
end

alphas = [-tau * C, alphas];
offset = [ initial_offset, offset];
