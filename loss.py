import torch.nn.functional as F
import torch as t

def one_hot(y):
    '''Return (batch_size x num_classes) shaped 
    one hot torch tensor'''
    
    y_ = torch.zeros((y.shape[0], hparams.num_classes))
    y_[torch.arange(y.shape[0]), y] = 1
    return y_


def focal_loss(inputs, targets, weights, gamma = 2, reduction = 'mean'):
    '''
    Focal Loss for Multi-Class Classification
    
    input           : torch.Tensor
                        Class estimations, shape:(batch_size, num_classes) 

    target          : torch.Tensor
                        Ground true classes, shape:(batch_size,)

    weight          : torch.Tensor
                        Class weights, shape:(num_classes,)
    
    gamma           : int
                        Exponent of the modulating factor (1 - p_t) to
                        balance easy vs hard examples. Default: 2
    
    reduction       : string
                        See torch loss functions
                        options: 'none', 'mean', 'sum'
    '''
    p = F.softmax(inputs, dim = 1) #(batch_size, n_class)
    log_p = F.log_softmax(inputs, dim = 1) #(batch_size, n_class)

    ce_loss = F.nll_loss(input = log_p, target = targets, 
        weight = weights, ignore_index = 8, reduction = 'none') #(batch_size,)

    targets_ = one_hot(targets) # (batch_size,n_class)
    
    p_t = t.sum(targets_ * p, dim = 1) #(batch_size,)

    focal_loss = ce_loss * ((1 - p_t) ** gamma)

    if reduction == 'mean':
        focal_loss = focal_loss.mean() #(1)
    if reduction == 'sum':
        focal_loss = focal_loss.sum() #(1)
    if reduction == 'none':
        pass
    
    return focal_loss