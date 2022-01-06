function main()
    load('dataset4.mat');
    step = 4;
    tolerance = 0.01;
    iter_count = 0;
    max_iter = 10000;
    % generate the initial theta randomly
    theta = rand(size(X,2),1);
    theta_previous = theta + 2*tolerance;
    classification_errors = [];
    empirical_risks = [];
    while (iter_count < max_iter)
        iter_count = iter_count + 1;
        fx = (1+exp(-X*theta));
        risks = (Y-1).*log(1-fx)-Y.*log(fx);
        risk = sum(risks) / length(risks);
        % concatenate the matrixs
        empirical_risks = cat(1,empirical_risks, risk);
        if (norm(theta - theta_previous) < tolerance)
            break;
        end
        fx(fx<=0.5) = 0;
        fx(fx>0.5) = 1;
        error = sum(fx~=Y)/ size(X,1);
        classification_errors = cat(1,classification_errors,error);
        theta_previous = theta;
        
        
        % Gradient Descent
        y = repmat(Y,1,size(X,2));
        fx2 = repmat(fx,1,size(X,2));
        d = X.*repmat(exp(-X*theta),1,size(X,2));
        gradient = sum((1-Y).*(X-d.*fx2) - Y.*d.*fx2) / size(X,1);
   
        theta = theta - step*gradient';
    end
    
    % plot the figure
    close all;
    figure
    disp(classification_errors);
    disp(empirical_risks);
    plot(1:iter_count, classification_errors, "classification errors", empirical_risks, "empirical risks");
    
end

