import torch
import torch.nn as nn
import torch.nn.functional as F



# Temperature Scaling 적용 Focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, temperature=1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # self.size_average = size_average
        self.reduction = reduction
        self.temperature = temperature

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
            
        target = target.view(-1,1)

        logpt = F.log_softmax(input/self.temperature)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        
        loss = -1 * (1-pt)**self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
    		
        elif self.reduction == 'sum':
            return loss.sum()
    
        elif self.reduction == 'none':
            return loss


#num_classes = 6
#model_output = torch.randn((32, num_classes), requires_grad = True)
#true_labels = torch.randint(0, num_classes, (32,))

#criterion = FocalLoss(gamma=2)
#loss = criterion(model_output, true_labels)

#print("Focal Loss:", loss.item())

# 바꾸기 전 Focal loss
# class FocalLoss(nn.Module):
# 	def __init__(self, alpha = None, gamma=2, reduction='mean'):
# 		super(FocalLoss, self).__init__()
# 		self.alpha = alpha # class addweights
# 		self.gamma = gamma # Focal loss gamma
# 		self.reduction = reduction

# 	def forward(self, inputs, targets):
# 		ce_loss =  F.cross_entropy(inputs, targets, reduction='none')

# 		pt = torch.exp(-ce_loss)
# 		focal_loss = (1-pt) ** self.gamma * ce_loss

# 		if self.alpha is not None:
# 			assert len(self.alpha) == inputs.shape[1]
# 			alpha = torch.tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)
			
# 		if self.reduction == 'mean':
# 			return focal_loss.mean()
		
# 		elif self.reduction == 'sum':
# 			return focal_loss.sum()

# 		elif self.reduction == 'none':
# 			return focal_loss
	