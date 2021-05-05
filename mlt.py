import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    import apex
except ImportError:
    pass

BERT_DIM = 768
BERT_LAYERS = 13 # (emb + 12 hidden from transformers)
IGNORED_INDEX = -100

'''
def trim_input(bert_ids, bert_mask, bert_labels=None):
    max_length = (bert_mask !=0).max(0)[0].nonzero().numel()
    
    if max_length < bert_ids.shape[1]:
        bert_ids = bert_ids[:, :max_length]
        bert_mask = bert_mask[:, :max_length]
        if bert_labels is not None:
            bert_labels = bert_labels[:, :max_length]

    if bert_labels is not None:
        return bert_ids, bert_mask, bert_labels
    else:
        return bert_ids, bert_mask
'''

def permute_aug(data, mask, labels, ncopy):
    # permute token order in seqs 
    # bs, seqlen, dim
    bs, max_seqlen = data.size()
    # make sure [CLS] and [SEP] is not modified?
    seqlens = (mask!=0).sum(-1)-2

    auged_data = [data]
    auged_mask = [mask]
    auged_labels = [labels]

    for _ in range(ncopy):
        for i in range(bs):
            seqlen = int(seqlens[i].cpu().item())
            perm = np.random.permutation(seqlen) + 1
            new_idx = [0] + list(perm) + [seqlen+1] + list(range(seqlen+2, max_seqlen))
            auged_data.append(data[i, new_idx].unsqueeze(0))
            auged_mask.append(mask[i].unsqueeze(0))
            auged_labels.append(labels[i, new_idx].unsqueeze(0))

    return torch.cat(auged_data, 0), torch.cat(auged_mask, 0), torch.cat(auged_labels, 0)

def _dot(grad_a, grad_b):
    return sum([torch.dot(gv[0].view(-1), gv[1].view(-1)) for gv in zip(grad_a, grad_b) if gv[0] is not None and gv[1] is not None])

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])
    
def sync_backward(loss, opt, args, retain_graph=False): # DDP and AMP compatible backward
    if args.amp == -1: # no amp
        loss.backward(retain_graph=retain_graph)
    else:
        with apex.amp.scale_loss(loss, opt) as scaled_loss:
            scaled_loss.backward(retain_graph=retain_graph)

def sync_autograd(loss, net, opt, args, retain_graph=False): # DDP and AMP compatible autograd
    if args.local_rank == -1: # single GPU
        grads = torch.autograd.grad(loss, net.parameters(), allow_unused=True)
    else:
        # distributed, with AMP optionally
        net.zero_grad()
        if args.amp == -1: # PyTorch DDP
            loss.backward(retain_graph=retain_graph)
        else: # Apex DDP
            with apex.amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward(retain_graph=retain_graph)

        # this assumed loss scale is 1 as when it's scaled p.grad might not be the valid grad values!
        grads = [p.grad.clone() for p in net.parameters()] 

    return grads

def modify_parameters(net, deltas, eps):
    for param, delta in zip(net.parameters(), deltas):
        if delta is not None:
            param.data.add_(eps, delta)
            
        #    for i, param in enumerate(net.parameters()):
        #param.data += eps * grads[i]

# logit is a 3d tensor and labels is 2d
def masked_cross_entropy_longvector(logit, labels):
    bs, seqlen, _ = logit.size()
    loss_vector = F.cross_entropy(logit.view(-1, logit.size(-1)),
                                  labels.reshape(-1),
                                  ignore_index=IGNORED_INDEX,
                                  reduction='none')

    # filter out IGNORED_INDEX
    loss_vector = loss_vector[labels.reshape(-1)!=IGNORED_INDEX]
    return loss_vector.unsqueeze(-1)
        

# logit is a 3d tensor and labels is 2d
def masked_cross_entropy_matrix(logit, labels):
    bs, seqlen, _ = logit.size()
    loss_vector = F.cross_entropy(logit.view(-1, logit.size(-1)),
                                  labels.reshape(-1),
                                  ignore_index=IGNORED_INDEX,
                                  reduction='none')
    loss_matrix = loss_vector.view(bs, seqlen).sum(-1) / ((labels!=IGNORED_INDEX).sum(-1))
    return loss_matrix.unsqueeze(-1)

def masked_cross_entropy(logit, labels, weights=None):
    # loss_sum = F.cross_entropy(logit.view(-1, logit.size(-1)),
    #                            labels.reshape(-1),
    #                            ignore_index=IGNORED_INDEX,
    #                            reduction='sum')
    loss = F.cross_entropy(logit.view(-1, logit.size(-1)),
                               labels.reshape(-1),
                               ignore_index=IGNORED_INDEX,
                               reduction='none')
    if weights is not None:
        # print(loss.shape)
        # print(weights.shape)
        loss_sum = torch.sum(loss * weights)
        # print(loss_sum)
    else:
        loss_sum = torch.sum(loss)
    loss = loss_sum / (labels!=IGNORED_INDEX).sum()
    return loss


# this only apply meta_net to the transformer in layerid
def forward_last(model, raptor, data, mask, ext_mask, layerid): # only insert meta_net to the last layer of transformer
    _, h = model(data, attention_mask=mask)
    new_h = raptor(h[layerid]) # h from last transformer
    logit = model.forward_tail(layerid+1, new_h, attention_mask=ext_mask) 

    return logit

def step_mlt_multi(main_net, main_opt, meta_net, meta_opt,
                   data_s, mask_s, target_s,
                   data_t, mask_t, target_t, 
                   eta, args):
    # META NET START
    # given current meta net, get transformed features
    _, h_s = main_net(data_s, attention_mask=mask_s)
    ext_mask_s = main_net.get_ext_mask(mask_s)
    
    alpha = meta_net.get_alpha()

    loss_s = 0
    for i in range(BERT_LAYERS):
        new_h = meta_net(h_s[i])
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)

        loss_s += alpha[i] * masked_cross_entropy(logit_s, target_s)

    # retain graph as for DDP it uses backward to get the gradients, is not set when using single GPU
    #f_param_grads = sync_autograd(loss_s, main_net, main_opt, args, retain_graph=True)
    f_param_grads = torch.autograd.grad(loss_train, main_net.parameters(), allow_unused=True, create_graph=True)    

    # /////////// NEW WAY ////////////

    # or just use SGD as in Algorithm 1, this works best for now
    f_param = [param.data.clone() for param in main_net.parameters()]
    for i, param in enumerate(main_net.parameters()):
        if f_param_grads[i] is not None:
            #f_param.append(param.data.clone())
            param.data.sub_(args.magic*eta*f_param_grads[i]) # SGD update

    # 3. compute d_w' L_{D}(w')
    logit_t, _ = main_net(data_t, attention_mask=mask_t)
    loss_t  = masked_cross_entropy(logit_t, target_t)

    f_param_grads_prime = torch.autograd.grad(loss_eval, main_net.parameters(), allow_unused=True)
    #f_param_grads_prime = sync_autograd(loss_t, main_net, main_opt, args)

    # 4. revert from w' to w for main net
    for i, param in enumerate(main_net.parameters()):
        param.data = f_param[i]
        
    proxy_g = -args.magic * eta * _dot(f_param_grads, f_param_grads_prime)    


    # back prop on alphas
    meta_opt.zero_grad()
    # backward on proxy_g as proxy_g shares the same gradient as loss_g
    sync_backward(proxy_g, meta_opt, args)
    meta_opt.step()
    # META NET END

    # MAIN NET START
    #main_net.train()

    # loss on data_s
    _, h_s = main_net(data_s, attention_mask=mask_s)
    alpha = meta_net.get_alpha().detach()
    loss_s = 0
    for i in range(BERT_LAYERS):
        new_h = meta_net(h_s_p[i]).detach()
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)
        loss_s += alpha[i] * masked_cross_entropy(logit_s, target_s)

    # update classifier weights
    main_opt.zero_grad()
    # backward on loss_s
    sync_backward(loss_s, main_opt, args)
    main_opt.step()
    # MAIN NET END

    return loss_t, loss_s

# ============== mlt step procedure debug ===================
# NOTE: main_net is a BERT-like model
#       meta_net is implemented as nn.Module as usual
def step_metaw_mix(main_net, main_opt, meta_net, meta_opt,
                   data_s, mask_s, target_s, # data from other languages
                   data_t, mask_t, target_t, # train data for target lang
                   data_g, mask_g, target_g, # validation data for target lang
                   eta, args):
    # META NET START
    # given current meta net, get transformed features
    logit_s, h_s = main_net(data_s, attention_mask=mask_s)
    logit_t, _ = main_net(data_t, attention_mask=mask_t)
    
    loss_t = masked_cross_entropy(logit_t, target_t)

    loss_s = masked_cross_entropy_matrix(logit_s, target_s)
    w = meta_net(loss_s.detach()) # (bs, 1)

    loss_s_w = w * loss_s

    bs_t = (target_t!=IGNORED_INDEX).sum()
    bs_s = (target_s!=IGNORED_INDEX).sum()

    loss_train = (loss_t * bs_t + loss_s_w.sum()) / (bs_t + bs_s) # bs_s or w.sum()????
    
    # retain graph as for DDP it uses backward to get the gradients, is not set when using single GPU
    #f_param_grads = sync_autograd(loss_train, main_net, main_opt, args, retain_graph=True)
    f_param_grads = torch.autograd.grad(loss_train, main_net.parameters(), allow_unused=True, create_graph=True)

    # /////////// NEW WAY ////////////

    # or just use SGD as in Algorithm 1, this works best for now
    f_param = [param.data.clone() for param in main_net.parameters()]
    for i, param in enumerate(main_net.parameters()):
        if f_param_grads[i] is not None:
            param.data.sub_(args.magic*eta*f_param_grads[i]) # SGD update

    # 3. compute d_w' L_{D}(w')
    logit_g, _ = main_net(data_g, attention_mask=mask_g)
    loss_eval  = masked_cross_entropy(logit_g, target_g)

    f_param_grads_prime = torch.autograd.grad(loss_eval, main_net.parameters(), allow_unused=True)
    #f_param_grads_prime = sync_autograd(loss_eval, main_net, main_opt, args)

    # 4. revert from w' to w for main net
    for i, param in enumerate(main_net.parameters()):
        param.data = f_param[i]

    proxy_g = -args.magic * eta * _dot(f_param_grads, f_param_grads_prime)
    
    # back prop on alphas
    meta_opt.zero_grad()
    # backward on proxy_g as proxy_g shares the same gradient as loss_eval
    sync_backward(proxy_g, meta_opt, args)#, retain_graph=True)
    meta_opt.step()
    # META NET END

    # MAIN NET START
    # loss on data_s
    logit_t, _ = main_net(data_t, attention_mask=mask_t)
    logit_s, h_s = main_net(data_s, attention_mask=mask_s)

    loss_t = masked_cross_entropy(logit_t, target_t)
    loss_s = masked_cross_entropy_matrix(logit_s, target_s)
    w = meta_net(loss_s.detach()).detach() # note the detach here

    loss_s_w = w * loss_s
    loss_train = (loss_t * bs_t + loss_s_w.sum()) / (bs_t + bs_s) # bs_s or w.sum()????
    
    # update classifier weights
    main_opt.zero_grad()
    # backward on loss_train
    sync_backward(loss_train, main_opt, args)
    main_opt.step()
    # MAIN NET END

    return loss_eval, loss_train

# ============== mlt step procedure debug ===================
# NOTE: main_net is a BERT-like model
#       meta_net is implemented as nn.Module as usual
def step_metawt_mix(main_net, main_opt, meta_net, meta_opt,
                    data_s, mask_s, target_s, # data from other languages
                    data_t, mask_t, target_t, # train data for target lang
                    data_g, mask_g, target_g, # validation data for target lang
                    eta, args):
    # META NET START
    # given current meta net, get transformed features
    logit_s, h_s = main_net(data_s, attention_mask=mask_s)
    logit_t, _ = main_net(data_t, attention_mask=mask_t)

    loss_t = masked_cross_entropy(logit_t, target_t)

    loss_s = masked_cross_entropy_longvector(logit_s, target_s)
    w = meta_net(loss_s.detach())
    print(w[0])

    loss_s_w = w * loss_s

    bs_t = (target_t!=IGNORED_INDEX).sum()
    bs_s = (target_s!=IGNORED_INDEX).sum()

    loss_train = (loss_t * bs_t + loss_s_w.sum()) / (bs_t + bs_s) # bs_s or w.sum()????

    # retain graph as for DDP it uses backward to get the gradients, is not set when using single GPU
    #f_param_grads = sync_autograd(loss_train, main_net, main_opt, args, retain_graph=True)
    f_param_grads = torch.autograd.grad(loss_train, main_net.parameters(), allow_unused=True, create_graph=True)

    # /////////// NEW WAY ////////////

    # or just use SGD as in Algorithm 1, this works best for now
    f_param = [param.data.clone() for param in main_net.parameters()]
    for i, param in enumerate(main_net.parameters()):
        if f_param_grads[i] is not None:
            param.data.sub_(args.magic*eta*f_param_grads[i]) # SGD update

    # 3. compute d_w' L_{D}(w')
    logit_g, _ = main_net(data_g, attention_mask=mask_g)
    loss_eval  = masked_cross_entropy(logit_g, target_g)

    f_param_grads_prime = torch.autograd.grad(loss_eval, main_net.parameters(), allow_unused=True)
    #f_param_grads_prime = sync_autograd(loss_eval, main_net, main_opt, args)

    # 4. revert from w' to w for main net
    for i, param in enumerate(main_net.parameters()):
        param.data = f_param[i]

    proxy_g = -args.magic * eta * _dot(f_param_grads, f_param_grads_prime)

    # back prop on alphas
    meta_opt.zero_grad()
    # backward on proxy_g as proxy_g shares the same gradient as loss_eval
    sync_backward(proxy_g, meta_opt, args)#, retain_graph=True)
    meta_opt.step()
    # META NET END

    # MAIN NET START
    # loss on data_s
    logit_t, _ = main_net(data_t, attention_mask=mask_t)
    logit_s, h_s = main_net(data_s, attention_mask=mask_s)

    loss_t = masked_cross_entropy(logit_t, target_t)
    loss_s = masked_cross_entropy_longvector(logit_s, target_s)
    w = meta_net(loss_s.detach()).detach() # note the detach here

    loss_s_w = w * loss_s
    loss_train = (loss_t * bs_t + loss_s_w.sum()) / (bs_t + bs_s) # bs_s or w.sum()????

    # update classifier weights
    main_opt.zero_grad()
    # backward on loss_train
    sync_backward(loss_train, main_opt, args)
    main_opt.step()
    # MAIN NET END

    return loss_eval, loss_train

# ============== mlt step procedure debug ===================
# NOTE: main_net is a BERT-like model
#       meta_net is implemented as nn.Module as usual
def step_metawt_multi_mix(main_net, main_opt, meta_net, meta_opt,
                          data_s, mask_s, target_s, # data from other languages
                          data_t, mask_t, target_t, # train data for target lang
                          data_g, mask_g, target_g, # validation data for target lang
                          eta, args, idx):# idx is id for lang
    # META NET START
    # given current meta net, get transformed features
    logit_s, h_s = main_net(data_s, attention_mask=mask_s)
    logit_t, _ = main_net(data_t, attention_mask=mask_t)
    
    loss_t = masked_cross_entropy(logit_t, target_t)

    loss_s = masked_cross_entropy_longvector(logit_s, target_s)
    w = meta_net(idx, loss_s.detach())

    loss_s_w = w * loss_s
    
    bs_t = (target_t!=IGNORED_INDEX).sum()
    bs_s = (target_s!=IGNORED_INDEX).sum()

    loss_train = (loss_t * bs_t + loss_s_w.sum()) / (bs_t + bs_s) # bs_s or w.sum()????
    
    # retain graph as for DDP it uses backward to get the gradients, is not set when using single GPU
    #f_param_grads = sync_autograd(loss_train, main_net, main_opt, args, retain_graph=True)
    f_param_grads = torch.autograd.grad(loss_train, main_net.parameters(), allow_unused=True, create_graph=True)

    # /////////// NEW WAY ////////////

    # or just use SGD as in Algorithm 1, this works best for now
    f_param = [param.data.clone() for param in main_net.parameters()]
    for i, param in enumerate(main_net.parameters()):
        if f_param_grads[i] is not None:
            param.data.sub_(args.magic*eta*f_param_grads[i]) # SGD update

    # 3. compute d_w' L_{D}(w')
    logit_g, _ = main_net(data_g, attention_mask=mask_g)
    loss_eval  = masked_cross_entropy(logit_g, target_g)

    f_param_grads_prime = torch.autograd.grad(loss_eval, main_net.parameters(), allow_unused=True)
    #f_param_grads_prime = sync_autograd(loss_eval, main_net, main_opt, args)

    # 4. revert from w' to w for main net
    for i, param in enumerate(main_net.parameters()):
        param.data = f_param[i]

    proxy_g = -args.magic * eta * _dot(f_param_grads, f_param_grads_prime)
    
    # back prop on alphas
    meta_opt.zero_grad()
    # backward on proxy_g as proxy_g shares the same gradient as loss_eval
    sync_backward(proxy_g, meta_opt, args)#, retain_graph=True)
    meta_opt.step()
    # META NET END

    # MAIN NET START
    # loss on data_s
    logit_t, _ = main_net(data_t, attention_mask=mask_t)
    logit_s, h_s = main_net(data_s, attention_mask=mask_s)

    loss_t = masked_cross_entropy(logit_t, target_t)
    loss_s = masked_cross_entropy_longvector(logit_s, target_s)
    w = meta_net(idx, loss_s.detach()).detach() # note the detach here

    loss_s_w = w * loss_s
    loss_train = (loss_t * bs_t + loss_s_w.sum()) / (bs_t + bs_s) # bs_s or w.sum()????
    
    # update classifier weights
    main_opt.zero_grad()
    # backward on loss_train
    sync_backward(loss_train, main_opt, args)
    main_opt.step()
    # MAIN NET END

    return loss_eval, loss_train





# ============== mlt step procedure debug ===================
# NOTE: main_net is a BERT-like model
#       meta_net is implemented as nn.Module as usual
def step_mlt_multi_mix(main_net, main_opt, meta_net, meta_opt,
                   data_s, mask_s, target_s, # data from other languages
                   data_t, mask_t, target_t, # train data for target lang
                   data_g, mask_g, target_g, # validation data for target lang
                   eta, args):
    # META NET START
    # given current meta net, get transformed features
    _, h_s = main_net(data_s, attention_mask=mask_s)
    logit_t, _ = main_net(data_t, attention_mask=mask_t)
    
    ext_mask_s = main_net.get_ext_mask(mask_s)
    
    alpha = meta_net.get_alpha()

    loss_train = masked_cross_entropy(logit_t, target_t)

    loss_train2 = 0
    
    for i in range(BERT_LAYERS):
        new_h = meta_net(i, h_s[i])
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)
        loss_train2 += alpha[i] * masked_cross_entropy(logit_s, target_s)

    loss_train = (loss_train + loss_train2) / 2

    # retain graph as for DDP it uses backward to get the gradients, is not set when using single GPU
    #f_param_grads = sync_autograd(loss_train, main_net, main_opt, args, retain_graph=True)
    f_param_grads = torch.autograd.grad(loss_train, main_net.parameters(), allow_unused=True, create_graph=True)

    # /////////// NEW WAY ////////////

    # or just use SGD as in Algorithm 1, this works best for now
    f_param = [param.data.clone() for param in main_net.parameters()]
    for i, param in enumerate(main_net.parameters()):
        if f_param_grads[i] is not None:
            #f_param.append(param.data.clone())
            param.data.sub_(args.magic*eta*f_param_grads[i]) # SGD update

    # 3. compute d_w' L_{D}(w')
    logit_g, _ = main_net(data_g, attention_mask=mask_g)
    loss_eval  = masked_cross_entropy(logit_g, target_g)

    f_param_grads_prime = torch.autograd.grad(loss_eval, main_net.parameters(), allow_unused=True)
    #f_param_grads_prime = sync_autograd(loss_eval, main_net, main_opt, args)

    # 4. revert from w' to w for main net
    for i, param in enumerate(main_net.parameters()):
        param.data = f_param[i]

    proxy_g = -args.magic * eta * _dot(f_param_grads, f_param_grads_prime)
    
    # back prop on alphas
    meta_opt.zero_grad()
    # backward on proxy_g as proxy_g shares the same gradient as loss_eval
    sync_backward(proxy_g, meta_opt, args, retain_graph=True)
    meta_opt.step()
    # META NET END

    # MAIN NET START
    # loss on data_s
    #_, h_s = main_net(data_s, attention_mask=mask_s)
    #logit_t, _ = main_net(data_t, attention_mask=mask_t)
    
    alpha = meta_net.get_alpha().detach()
    loss_train = masked_cross_entropy(logit_t, target_t)
    loss_train2 = 0
    for i in range(BERT_LAYERS):
        new_h = meta_net(i, h_s[i]).detach()
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)
        loss_train2 += alpha[i] * masked_cross_entropy(logit_s, target_s)

    loss_train = (loss_train + loss_train2) / 2

    # update classifier weights
    main_opt.zero_grad()
    # backward on loss_train
    sync_backward(loss_train, main_opt, args)
    main_opt.step()
    # MAIN NET END

    return loss_eval, loss_train


# ============== mlt step procedure debug ===================
# NOTE: main_net is a BERT-like model
#       meta_net is implemented as nn.Module as usual
def step_mlt_multi_fd(main_net, main_opt, meta_net, meta_opt,
                   data_s, mask_s, target_s, # data from other languages
                   data_t, mask_t, target_t, # train data for target lang
                   data_g, mask_g, target_g, # validation data for target lang
                   eta, args):
    # META NET START
    # given current meta net, get transformed features
    _, h_s = main_net(data_s, attention_mask=mask_s)
    logit_t, _ = main_net(data_t, attention_mask=mask_t)
    
    ext_mask_s = main_net.get_ext_mask(mask_s)
    
    alpha = meta_net.get_alpha()

    loss_train = masked_cross_entropy(logit_t, target_t)

    loss_train2 = 0
    
    for i in range(BERT_LAYERS):
        new_h = meta_net(i, h_s[i])
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)
        loss_train2 += alpha[i] * masked_cross_entropy(logit_s, target_s)

    loss_train = (loss_train + loss_train2) / 2

    # retain graph as for DDP it uses backward to get the gradients, is not set when using single GPU
    f_param_grads = sync_autograd(loss_train, main_net, main_opt, args, retain_graph=True)

    # /////////// NEW WAY ////////////

    # or just use SGD as in Algorithm 1, this works best for now
    f_param = [param.data.clone() for param in main_net.parameters()]
    for i, param in enumerate(main_net.parameters()):
        if f_param_grads[i] is not None:
            #f_param.append(param.data.clone())
            param.data.sub_(args.magic*eta*f_param_grads[i]) # SGD update

    # 3. compute d_w' L_{D}(w')
    logit_g = main_net(data_g, attention_mask=mask_g)[0]
    loss_eval  = masked_cross_entropy(logit_g, target_g)

    f_param_grads_prime = sync_autograd(loss_eval, main_net, main_opt, args)

    # 4. revert from w' to w for main net
    for i, param in enumerate(main_net.parameters()):
        param.data = f_param[i]
    
    # change main_net parameter 
    eps = 1e-3 # #1e-2 / _concat(f_param_grads_prime).norm()# eta 1e-6 before
    # modify w to w+
    modify_parameters(main_net, f_param_grads_prime, eps)
    _, h_s_p  = main_net(data_s, attention_mask=mask_s)
    logit_t_p, _ = main_net(data_t, attention_mask=mask_t)
    loss_train_p = masked_cross_entropy(logit_t_p, target_t)
    loss_train_p2 = 0
    for i in range(BERT_LAYERS):
        new_h = meta_net(i, h_s_p[i])
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)
        loss_train_p2 += alpha[i] * masked_cross_entropy(logit_s, target_s)

    loss_train_p = (loss_train_p + loss_train_p2) / 2

    # modify w to w- (w is w+ now)
    modify_parameters(main_net, f_param_grads_prime, -2*eps)
    _, h_s_n  = main_net(data_s, attention_mask=mask_s)
    logit_t_n, _ = main_net(data_t, attention_mask=mask_t)
    loss_train_n = masked_cross_entropy(logit_t_n, target_t)
    loss_train_n2 = 0
    for i in range(BERT_LAYERS):
        new_h = meta_net(i, h_s_n[i])
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)
        loss_train_n2 += alpha[i] * masked_cross_entropy(logit_s, target_s)
    loss_train_n = (loss_train_n + loss_train_n2) / 2

    proxy_g = -args.magic * eta * (loss_train_p - loss_train_n) / (2.*eps)

    # modify to original w
    modify_parameters(main_net, f_param_grads_prime, eps)

    # back prop on alphas
    meta_opt.zero_grad()
    # backward on proxy_g as proxy_g shares the same gradient as loss_eval
    sync_backward(proxy_g, meta_opt, args)
    meta_opt.step()
    # META NET END

    # MAIN NET START
    # loss on data_s
    _, h_s = main_net(data_s, attention_mask=mask_s)
    logit_t, _ = main_net(data_t, attention_mask=mask_t)
    
    alpha = meta_net.get_alpha().detach()
    loss_train = masked_cross_entropy(logit_t, target_t)
    loss_train2 = 0
    for i in range(BERT_LAYERS):
        new_h = meta_net(i, h_s[i]).detach()
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)
        loss_train2 += alpha[i] * masked_cross_entropy(logit_s, target_s)

    loss_train = (loss_train + loss_train2) / 2

    # update classifier weights
    main_opt.zero_grad()
    # backward on loss_train
    sync_backward(loss_train, main_opt, args)
    main_opt.step()
    # MAIN NET END

    return loss_eval, loss_train


# ============== mlt step procedure debug ===================
# NOTE: main_net is a BERT-like model
#       meta_net is implemented as nn.Module as usual
def step_mlt(main_net, main_opt, meta_net, meta_opt,
             data_s, mask_s, target_s,
             data_t, mask_t, target_t, 
             eta, args):
    # META NET START
    # given current meta net, get transformed features
    _, h_s = main_net(data_s, attention_mask=mask_s)
    ext_mask_s = main_net.get_ext_mask(mask_s)
    
    alpha = meta_net.get_alpha()

    loss_s = 0
    for i in range(BERT_LAYERS):
        new_h = meta_net(h_s[i])
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)
        loss_s += alpha[i] * masked_cross_entropy(logit_s, target_s)

    # retain graph as for DDP it uses backward to get the gradients, is not set when using single GPU
    f_param_grads = sync_autograd(loss_s, main_net, main_opt, args, retain_graph=True)

    # /////////// NEW WAY ////////////

    # or just use SGD as in Algorithm 1, this works best for now
    f_param = [param.data.clone() for param in main_net.parameters()]
    for i, param in enumerate(main_net.parameters()):
        if f_param_grads[i] is not None:
            #f_param.append(param.data.clone())
            param.data.sub_(args.magic*eta*f_param_grads[i]) # SGD update

    # 3. compute d_w' L_{D}(w')
    logit_t = main_net(data_t, attention_mask=mask_t)[0]
    loss_t  = masked_cross_entropy(logit_t, target_t)

    f_param_grads_prime = sync_autograd(loss_t, main_net, main_opt, args)

    # 4. revert from w' to w for main net
    for i, param in enumerate(main_net.parameters()):
        param.data = f_param[i]
    
    # change main_net parameter 
    eps = 1e-6 # 1e-3 / _concat(f_param_grads_prime).norm()# eta 1e-6 before
    # modify w to w+
    modify_parameters(main_net, f_param_grads_prime, eps)
    _, h_s_p  = main_net(data_s, attention_mask=mask_s)
    loss_s_p = 0
    for i in range(BERT_LAYERS):
        new_h = meta_net(h_s_p[i])
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)
        loss_s_p += alpha[i] * masked_cross_entropy(logit_s, target_s)

    # modify w to w- (w is w+ now)
    modify_parameters(main_net, f_param_grads_prime, -2*eps)
    _, h_s_n  = main_net(data_s, attention_mask=mask_s)
    loss_s_n = 0
    for i in range(BERT_LAYERS):
        new_h = meta_net(h_s_n[i])
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)
        loss_s_n += alpha[i] * masked_cross_entropy(logit_s, target_s)

    proxy_g = -args.magic * eta * (loss_s_p - loss_s_n) / (2.*eps)

    # modify to original w
    modify_parameters(main_net, f_param_grads_prime, eps)

    # back prop on alphas
    meta_opt.zero_grad()
    # backward on proxy_g as proxy_g shares the same gradient as loss_g
    sync_backward(proxy_g, meta_opt, args)
    meta_opt.step()
    # META NET END

    # MAIN NET START
    #main_net.train()

    # loss on data_s
    _, h_s = main_net(data_s, attention_mask=mask_s)
    alpha = meta_net.get_alpha().detach()
    loss_s = 0
    for i in range(BERT_LAYERS):
        new_h = meta_net(h_s_p[i]).detach()
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)
        loss_s += alpha[i] * masked_cross_entropy(logit_s, target_s)

    # update classifier weights
    main_opt.zero_grad()
    # backward on loss_s
    sync_backward(loss_s, main_opt, args)
    main_opt.step()
    # MAIN NET END

    return loss_t, loss_s

# ============== mlt step procedure debug ===================
# NOTE: main_net is a BERT-like model
#       meta_net is implemented as nn.Module as usual
def step_mlt_mix(main_net, main_opt, meta_net, meta_opt,
                 data_s, mask_s, target_s,
                 data_g, mask_g, target_g,
                 data_t, mask_t, target_t, 
                 eta, args):
    # META NET START
    # given current meta net, get transformed features
    _, h_s = main_net(data_s, attention_mask=mask_s)
    logit_g, _ = main_net(data_g, attention_mask=mask_g)
    
    ext_mask_s = main_net.get_ext_mask(mask_s)
    
    alpha = meta_net.get_alpha()

    loss_s = masked_cross_entropy(logit_g, target_g)
    
    for i in range(BERT_LAYERS):
        new_h = meta_net(h_s[i])
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)
        loss_s += alpha[i] * masked_cross_entropy(logit_s, target_s)

    loss_s /= 2

    # retain graph as for DDP it uses backward to get the gradients, is not set when using single GPU
    f_param_grads = sync_autograd(loss_s, main_net, main_opt, args, retain_graph=True)

    # /////////// NEW WAY ////////////

    # or just use SGD as in Algorithm 1, this works best for now
    f_param = [param.data.clone() for param in main_net.parameters()]
    for i, param in enumerate(main_net.parameters()):
        if f_param_grads[i] is not None:
            #f_param.append(param.data.clone())
            param.data.sub_(args.magic*eta*f_param_grads[i]) # SGD update

    # 3. compute d_w' L_{D}(w')
    logit_t = main_net(data_t, attention_mask=mask_t)[0]
    loss_t  = masked_cross_entropy(logit_t, target_t)

    f_param_grads_prime = sync_autograd(loss_t, main_net, main_opt, args)

    # 4. revert from w' to w for main net
    for i, param in enumerate(main_net.parameters()):
        param.data = f_param[i]
    
    # change main_net parameter 
    eps = 1e-6 # 1e-3 / _concat(f_param_grads_prime).norm()# eta 1e-6 before
    # modify w to w+
    modify_parameters(main_net, f_param_grads_prime, eps)
    _, h_s_p  = main_net(data_s, attention_mask=mask_s)
    logit_g_p, _ = main_net(data_g, attention_mask=mask_g)
    loss_s_p = masked_cross_entropy(logit_g_p, target_g)
    for i in range(BERT_LAYERS):
        new_h = meta_net(h_s_p[i])
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)
        loss_s_p += alpha[i] * masked_cross_entropy(logit_s, target_s)

    loss_s_p /= 2

    # modify w to w- (w is w+ now)
    modify_parameters(main_net, f_param_grads_prime, -2*eps)
    _, h_s_n  = main_net(data_s, attention_mask=mask_s)
    logit_g_n, _ = main_net(data_g, attention_mask=mask_g)
    loss_s_n = masked_cross_entropy(logit_g_n, target_g)
    
    for i in range(BERT_LAYERS):
        new_h = meta_net(h_s_n[i])
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)
        loss_s_n += alpha[i] * masked_cross_entropy(logit_s, target_s)
    loss_s_n /= 2

    proxy_g = -args.magic * eta * (loss_s_p - loss_s_n) / (2.*eps)

    # modify to original w
    modify_parameters(main_net, f_param_grads_prime, eps)

    # back prop on alphas
    meta_opt.zero_grad()
    # backward on proxy_g as proxy_g shares the same gradient as loss_g
    sync_backward(proxy_g, meta_opt, args)
    meta_opt.step()
    # META NET END

    # MAIN NET START
    # loss on data_s
    _, h_s = main_net(data_s, attention_mask=mask_s)
    logit_g, _ = main_net(data_g, attention_mask=mask_g)
    
    alpha = meta_net.get_alpha().detach()
    loss_s = masked_cross_entropy(logit_g, target_g)
    for i in range(BERT_LAYERS):
        new_h = meta_net(h_s_p[i]).detach()
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)
        loss_s += alpha[i] * masked_cross_entropy(logit_s, target_s)

    loss_s /=2

    # update classifier weights
    main_opt.zero_grad()
    # backward on loss_s
    sync_backward(loss_s, main_opt, args)
    main_opt.step()
    # MAIN NET END

    return loss_t, loss_s

# ============== gold only (supervised method) ===================
# NOTE: main_net is a BERT-like model
def step_gold_only(main_net, main_opt, 
                   data_g, mask_g, target_g,
                   args):
    # MAIN NET START
    logit_g, _ = main_net(data_g, attention_mask=mask_g, for_classification=(args.task_name=="sent"))
    loss_s = masked_cross_entropy(logit_g, target_g)

    # update classifier weights
    main_opt.zero_grad()
    # backward on loss_s
    sync_backward(loss_s, main_opt, args)
    main_opt.step()
    # MAIN NET END

    loss_t = torch.tensor(-1).type_as(loss_s)

    return loss_t, loss_s

# ============== gold mix (supervised method) ===================
# NOTE: main_net is a BERT-like model
def step_gold_mix(main_net, main_opt,
                  data_s, mask_s, target_s,
                  data_g, mask_g, target_g,
                  args):
    # MAIN NET START
    logit_g, _ = main_net(data_g, attention_mask=mask_g, for_classification= (args.task_name=="sent"))
    loss_s = masked_cross_entropy(logit_g, target_g)

    logit_s, _ = main_net(data_s, attention_mask=mask_s, for_classification= (args.task_name=="sent"))
    loss_s += masked_cross_entropy(logit_s, target_s)
    loss_s /= 2

    # update classifier weights
    main_opt.zero_grad()
    # backward on loss_s
    sync_backward(loss_s, main_opt, args)
    main_opt.step()
    # MAIN NET END

    loss_t = torch.tensor(-1).type_as(loss_s)

    return loss_t, loss_s


# ============== mlt zero resource transfer debug ===================
# NOTE: main_net is a BERT-like model
#       meta_net is implemented as nn.Module as usual
# target_g shouldn't be used
def step_zero_mix(main_net, main_opt, meta_net, meta_opt,
                  data_s, mask_s, target_s,
                  data_g, mask_g, target_g,
                  data_t, mask_t, target_t, 
                  eta, args):
    # META NET START
    # given current meta net, get transformed features
    _, h_s = main_net(data_s, attention_mask=mask_s)
    logit_g, _ = main_net(data_g, attention_mask=mask_g)
    
    ext_mask_s = main_net.get_ext_mask(mask_s)
    
    alpha = meta_net.get_alpha()

    loss_s = masked_cross_entropy(logit_g, target_g)
    
    for i in range(BERT_LAYERS):
        new_h = meta_net(h_s[i])
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)
        loss_s += alpha[i] * masked_cross_entropy(logit_s, target_s)

    loss_s /= 2

    # retain graph as for DDP it uses backward to get the gradients, is not set when using single GPU
    f_param_grads = sync_autograd(loss_s, main_net, main_opt, args, retain_graph=True)

    # /////////// NEW WAY ////////////

    # or just use SGD as in Algorithm 1, this works best for now
    f_param = [param.data.clone() for param in main_net.parameters()]
    for i, param in enumerate(main_net.parameters()):
        if f_param_grads[i] is not None:
            #f_param.append(param.data.clone())
            param.data.sub_(args.magic*eta*f_param_grads[i]) # SGD update

    # 3. compute d_w' L_{D}(w')
    logit_t = main_net(data_t, attention_mask=mask_t)[0]
    loss_t  = masked_cross_entropy(logit_t, target_t)

    f_param_grads_prime = sync_autograd(loss_t, main_net, main_opt, args)

    # 4. revert from w' to w for main net
    for i, param in enumerate(main_net.parameters()):
        param.data = f_param[i]
    
    # change main_net parameter 
    eps = 1e-6 # 1e-3 / _concat(f_param_grads_prime).norm()# eta 1e-6 before
    # modify w to w+
    modify_parameters(main_net, f_param_grads_prime, eps)
    _, h_s_p  = main_net(data_s, attention_mask=mask_s)
    logit_g_p, _ = main_net(data_g, attention_mask=mask_g)
    loss_s_p = masked_cross_entropy(logit_g_p, target_g)
    for i in range(BERT_LAYERS):
        new_h = meta_net(h_s_p[i])
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)
        loss_s_p += alpha[i] * masked_cross_entropy(logit_s, target_s)

    loss_s_p /= 2

    # modify w to w- (w is w+ now)
    modify_parameters(main_net, f_param_grads_prime, -2*eps)
    _, h_s_n = main_net(data_s, attention_mask=mask_s)
    logit_g_n, _ = main_net(data_g, attention_mask=mask_g)
    loss_s_n = masked_cross_entropy(logit_g_n, target_g)
    
    for i in range(BERT_LAYERS):
        new_h = meta_net(h_s_n[i])
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)
        loss_s_n += alpha[i] * masked_cross_entropy(logit_s, target_s)
    loss_s_n /= 2

    proxy_g = -args.magic * eta * (loss_s_p - loss_s_n) / (2.*eps)

    # modify to original w
    modify_parameters(main_net, f_param_grads_prime, eps)

    # back prop on alphas
    meta_opt.zero_grad()
    # backward on proxy_g as proxy_g shares the same gradient as loss_g
    sync_backward(proxy_g, meta_opt, args)
    meta_opt.step()
    # META NET END

    # MAIN NET START
    # loss on data_s
    _, h_s = main_net(data_s, attention_mask=mask_s)
    logit_g, _ = main_net(data_g, attention_mask=mask_g)
    
    alpha = meta_net.get_alpha().detach()
    loss_s = masked_cross_entropy(logit_g, target_g)
    for i in range(BERT_LAYERS):
        new_h = meta_net(h_s_p[i]).detach()
        logit_s = main_net.forward_tail(i+1, new_h, attention_mask=ext_mask_s)
        loss_s += alpha[i] * masked_cross_entropy(logit_s, target_s)

    loss_s /=2

    # update classifier weights
    main_opt.zero_grad()
    # backward on loss_s
    sync_backward(loss_s, main_opt, args)
    main_opt.step()
    # MAIN NET END

    return loss_t, loss_s


def get_mask(mask):
    src_lengths = torch.sum(mask, dim=1)
    max_length = mask.shape[1]

    permutation_mask = torch.stack([F.pad(torch.ones(src_length - 1, src_length - 1),
                                          (0, max_length - src_length, 0, max_length - src_length))
                                    for src_length in src_lengths])
    permutation_minimal = torch.ones_like(permutation_mask) * (-100000)
    permutation_mask = permutation_mask.to(mask.device)
    permutation_minimal = permutation_minimal.to(mask.device)
    return permutation_mask, permutation_minimal

def detached(x, detach):
    if detach:
        return x.detach()
    else:
        return x


# ============== metaxl dynamic language and layer debug  ===================
def step_metaxl(main_net, main_opt,
                    meta_net, meta_opt,
                    reweighting_model, reweighting_opt,
                    data_s, mask_s, target_s,
                    data_g, mask_g, target_g,
                    data_t, mask_t, target_t,
                    source_language_id, transfer_layers, eta, args):

    print(type(main_net))
    bs_s = (target_s != IGNORED_INDEX).sum()
    bs_g = (target_g != IGNORED_INDEX).sum()

    logits_s, h_s = main_net(data_s, attention_mask=mask_s, for_classification=(args.task_name=="sent"))

    logits_g, _ = main_net(data_g, mask_g, for_classification=(args.task_name=="sent"))
    loss_g = masked_cross_entropy(logits_g, target_g)

    ext_mask_s = main_net.get_ext_mask(mask_s)
    alpha = meta_net.get_alpha(i = source_language_id)
    loss_s = 0
    for j, layer_id in enumerate(transfer_layers):
        new_h = meta_net(source_language_id, j, h_s[layer_id].detach()) + h_s[layer_id]
        sequence_output = main_net.forward_tail(layer_id + 1, new_h, attention_mask=ext_mask_s)
        logits_s = main_net.forward_classifier(sequence_output, for_classification=(args.task_name=="sent"))

        if args.add_instance_weights:
            if args.weights_from == "features":
                weights = reweighting_model(sequence_output.detach())  # batch * token * 1
                weights = weights.squeeze(-1).view(-1)
                # weights = reweighting_model(sequence_output[:, 0].detach()) # batch * 1 * 1
                # weights = weights.repeat(1, sequence_output.shape[1]).view(-1)
                loss_s += alpha[j] * masked_cross_entropy(logits_s, target_s, weights)
            elif args.weights_from == "loss":
                loss_ = masked_cross_entropy_longvector(logits_s, target_s)
                weights = reweighting_model(loss_.detach())
                loss_s += alpha[j] * (loss_ * weights).sum()
        else:
            loss_s += alpha[j] * masked_cross_entropy(logits_s, target_s)

    if len(transfer_layers) == 0:
        if args.add_instance_weights:
            sequence_output = h_s[-1]
            if args.weights_from == "features":
                weights = reweighting_model(sequence_output.detach()) # batch * token * 1
                weights = weights.squeeze(-1).view(-1)
                loss_s += masked_cross_entropy(logits_s, target_s, weights)
            elif args.weights_from == "loss":
                loss_ = masked_cross_entropy_longvector(logits_s, target_s)
                weights = reweighting_model(loss_.detach())
                loss_s += (loss_ * weights).sum()
        else:
            loss_s += masked_cross_entropy(logits_s, target_s)

    if args.add_instance_weights and args.weights_from == "loss":
        loss_train = (loss_s + loss_g * bs_g) / (bs_s + bs_g)
    else:
        loss_train = loss_s + loss_g

            # retain graph as for DDP it uses backward to get the gradients, is not set when using single GPU
    # 1. calculate gradient
    f_param_grads = torch.autograd.grad(loss_train, main_net.parameters(), allow_unused=True, create_graph=True)

    # 2. update model parameters with the gradient
    f_param = [param.data.clone() for param in main_net.parameters()]
    for i, param in enumerate(main_net.parameters()):
        if f_param_grads[i] is not None:
            param.data.sub_(args.magic * eta * f_param_grads[i])  # SGD update

    # 3. compute d_w' L_{D}(w')
    logits_t, h_t = main_net(data_t, mask_t, for_classification=(args.task_name=="sent"))
    loss_t = masked_cross_entropy(logits_t, target_t)
    f_param_grads_prime = torch.autograd.grad(loss_t, main_net.parameters(), allow_unused=True)

    # 4. revert from w' to w for main net
    for i, param in enumerate(main_net.parameters()):
        param.data = f_param[i]

    proxy_g = -args.magic * eta * _dot(f_param_grads, f_param_grads_prime)

    # back prop on alphas and extra structures
    if meta_opt is not None:
        meta_opt.zero_grad()
        # print("before permute", permutate_net.permute_net[2].weight)
    if args.add_instance_weights:
        reweighting_opt.zero_grad()

    # backward on proxy_g as proxy_g shares the same gradient as loss_g
    # sync_backward(proxy_g, meta_opt, args)
    # sync_backward(proxy_g, sinkhorn_opt, args)
    proxy_g.backward(retain_graph=False)
    # torch.nn.utils.clip_grad_norm_(meta_net.parameters(), 10)
    if reweighting_model is not None:
        torch.nn.utils.clip_grad_norm_(reweighting_model.parameters(), 10)


    if meta_opt is not None:
        meta_opt.step()
    if args.add_instance_weights:
        reweighting_opt.step()

    # loss on data_s
    logits_s, h_s = main_net(data_s, attention_mask=mask_s, for_classification=(args.task_name=="sent"))
    logits_g, _ = main_net(data_g, attention_mask=mask_g, for_classification=(args.task_name=="sent"))
    loss_g = masked_cross_entropy(logits_g, target_g)

    ext_mask_s = main_net.get_ext_mask(mask_s)
    alpha = meta_net.get_alpha(i=source_language_id)
    loss_s = 0
    for j, layer_id in enumerate(transfer_layers):
        new_h = meta_net(source_language_id, j, h_s[layer_id].detach()) + h_s[layer_id]
        sequence_output = main_net.forward_tail(layer_id + 1, new_h, attention_mask=ext_mask_s)
        logits_s = main_net.forward_classifier(sequence_output, for_classification=(args.task_name=="sent"))

        if args.add_instance_weights:
            if args.weights_from == "features":
                weights = reweighting_model(sequence_output.detach())  # batch * token * 1
                weights = weights.squeeze(-1).view(-1)
                loss_s += alpha[j] * masked_cross_entropy(logits_s, target_s, weights)
            elif args.weights_from == "loss":
                loss_ = masked_cross_entropy_longvector(logits_s, target_s)
                weights = reweighting_model(loss_.detach())
                print(weights[0])
                loss_s += alpha[j] * (loss_ * weights).sum()
        else:
            loss_s += alpha[j] * masked_cross_entropy(logits_s, target_s)

    if len(transfer_layers) == 0:
        if args.add_instance_weights:
            sequence_output = h_s[-1]
            if args.weights_from == "features":
                weights = reweighting_model(sequence_output.detach()) # batch * token * 1
                weights = weights.squeeze(-1).view(-1)
                loss_s += masked_cross_entropy(logits_s, target_s, weights)
            elif args.weights_from == "loss":
                loss_ = masked_cross_entropy_longvector(logits_s, target_s)
                weights = reweighting_model(loss_.detach())
                loss_s += (loss_ * weights).sum()
        else:
            loss_s += masked_cross_entropy(logits_s, target_s)

    if args.add_instance_weights and args.weights_from == "loss":
        loss_train = (loss_s + loss_g * bs_g) / (bs_s + bs_g)
    else:
        loss_train = loss_s + loss_g

    # update classifier weights
    main_opt.zero_grad()
    # backward on loss_s
    sync_backward(loss_train, main_opt, args)
    torch.nn.utils.clip_grad_norm_(main_net.parameters(), 10)
    main_opt.step()
    # MAIN NET END

    return loss_t, loss_train

def step_jt_metaxl(main_net, main_opt,
                  meta_net, meta_opt,
                  reweighting_model, reweighting_opt,
                  data_s, mask_s, target_s,
                  data_g, mask_g, target_g,
                  source_language_id, transfer_layers, eta, args):
    bs_s = (target_s != IGNORED_INDEX).sum()
    bs_g = (target_g != IGNORED_INDEX).sum()

    logits_s, h_s = main_net(data_s, attention_mask=mask_s, for_classification=(args.task_name=="sent"))

    logits_g, _ = main_net(data_g, mask_g, for_classification=(args.task_name=="sent"))
    loss_g = masked_cross_entropy(logits_g, target_g)

    ext_mask_s = main_net.get_ext_mask(mask_s)
    alpha = meta_net.get_alpha(i = source_language_id)
    loss_s = 0
    for j, layer_id in enumerate(transfer_layers):
        new_h = meta_net(source_language_id, j, h_s[layer_id])
        sequence_output = main_net.forward_tail(layer_id + 1, new_h, attention_mask=ext_mask_s)
        logits_s = main_net.forward_classifier(sequence_output, for_classification=(args.task_name=="sent"))

        if args.add_instance_weights:
            if args.weights_from == "features":
                weights = reweighting_model(sequence_output[:, 0].detach()) # batch * 1 * 1
                weights = weights.repeat(1, sequence_output.shape[1]).view(-1)
                loss_s += alpha[j] * masked_cross_entropy(logits_s, target_s, weights)
            elif args.weights_from == "loss":
                loss_ = masked_cross_entropy_longvector(logits_s, target_s)
                weights = reweighting_model(loss_.detach())
                loss_s += alpha[j] * (loss_ * weights).sum()
        else:
            loss_s += alpha[j] * masked_cross_entropy(logits_s, target_s)

    if len(transfer_layers) == 0:
        if args.add_instance_weights:
            sequence_output = h_s[-1]
            if args.weights_from == "features":
                weights = reweighting_model(sequence_output.detach()) # batch * token * 1
                weights = weights.squeeze(-1).view(-1)
                loss_s += masked_cross_entropy(logits_s, target_s, weights)
            elif args.weights_from == "loss":
                loss_ = masked_cross_entropy_longvector(logits_s, target_s)
                weights = reweighting_model(loss_.detach())
                loss_s += (loss_ * weights).sum()
        else:
            loss_s += masked_cross_entropy(logits_s, target_s)

    if args.add_instance_weights and args.weights_from == "loss":
        loss_train = (loss_s + loss_g * bs_g) / (bs_s + bs_g)
    else:
        loss_train = loss_s + loss_g

            # retain graph as for DDP it uses backward to get the gradients, is not set when using single GPU

    # update classifier weights
    main_opt.zero_grad()

    # back prop on alphas and extra structures
    meta_opt.zero_grad()

    if args.add_instance_weights:
        reweighting_opt.zero_grad()

    # backward on proxy_g as proxy_g shares the same gradient as loss_g
    # sync_backward(proxy_g, meta_opt, args)
    # sync_backward(proxy_g, sinkhorn_opt, args)
    loss_train.backward(retain_graph=False)
    # torch.nn.utils.clip_grad_norm_(meta_net.parameters(), 10)
    if reweighting_model is not None:
        torch.nn.utils.clip_grad_norm_(reweighting_model.parameters(), 10)


    main_opt.step()
    meta_opt.step()
    if args.add_instance_weights:
        reweighting_opt.step()

    return torch.tensor(-1).type_as(loss_train), loss_train

# ============== metaxl finetune dynamic language and layer debug  ===================
def step_metaxl_finetune(main_net, main_opt, meta_net,
                    data_s, mask_s, target_s,
                    data_t, mask_t, target_t,
                    source_language_id, transfer_layers, args):
    _, h_s = main_net(data_s, attention_mask=mask_s, for_classfication=args.for_classification)

    logits_t, h_t = main_net(data_t, attention_mask= mask_t, for_classfication=args.for_classification)
    loss_t = masked_cross_entropy(logits_t, target_t)

    ext_mask_s = main_net.get_ext_mask(mask_s)
    alpha = meta_net.get_alpha(i = source_language_id)

    loss_s = 0
    for j, layer_id in enumerate(transfer_layers):
        new_h = meta_net(source_language_id, j, h_s[layer_id])
        logit_s = main_net.forward_tail(layer_id + 1, new_h, attention_mask=ext_mask_s)
        loss_s += alpha[j] * masked_cross_entropy(logit_s, target_s)

    # update classifier weights
    main_opt.zero_grad()
    # backward on loss_s
    sync_backward(loss_s + loss_t, main_opt, args)
    main_opt.step()
    # MAIN NET END

    return loss_t, loss_s

