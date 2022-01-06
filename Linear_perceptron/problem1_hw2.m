function main
    load('data3.mat');
    x = data(:,1:2);
    y = data(:,3);
    perceptron(x,y);
end

function perceptron(trainX, trainY)
    % initialize the weights
    theta = [0,0];
    bias = 0.8;
    N = length(trainY);
    learning_rate = 1;
    convergence = false;
    iter = 0;
    classification_loss = [];
    perceptron_loss = [];
    while ~convergence
        iter = iter + 1;
        classification_loss(iter) = 0.0;
        perceptron_loss(iter) = 0.0;
        for i = 1:length(trainY)
            % response
            y_value = trainX(i,:) * theta' + bias;
            if y_value >= 0
                y_i = 1;
            else
                y_i = -1;
            end
            if y_i ~= trainY(i)
                theta = theta + learning_rate * trainY(i) * trainX(i,:);
                classification_loss(iter) = classification_loss(iter) + step(-trainY(i)*y_value) / N;
                perceptron_loss(iter) = perceptron_loss(iter) - trainY(i)*y_value / N;
            end   
        end
        
        disp(classification_loss(iter));
        if (classification_loss(iter) == 0.0) | iter > 1000
            convergence = true
        end
    end
    
    % plot the resulting linear decision boundary on the 2d-x data
    
    figure
    for i = 1:N
        if trainY(i) == 1
            plot(trainX(i,1),trainX(i,2),'ro');
        else
            plot(trainX(i,1),trainX(i,2),'bo');
        end
        hold on
    end
    % the boundary: x1 * theta(1) + x2 * theta(2) + bias = 0
    x1 = trainX(:,1);
    x2 = (-x1*theta(1)-bias)/theta(2);
    plot(x1,x2);
    saveas(gcf,'boundary.png')
    
    
    
    % plot the evolution of binary classification error and the perceptron error
    figure
    plot(1:iter, classification_loss, 1:iter, perceptron_loss), legend('classification loss', 'perceptron loss')
    xlabel('iteration');
    saveas(gcf,'loss.png')
    
    
end

function z = step(x)
    if (x >= 0)
        z = 1;
    else
        z = -1;
    end
end



