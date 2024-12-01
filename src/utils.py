import torch


def add_punctuation_to_decoded(decoded_texts, punctuation="."):
    processed_texts = []
    for text in decoded_texts:
        if not text.endswith(punctuation):
            text += " " + punctuation
        processed_texts.append(text)
    return processed_texts


def preprocess_sentence(sentence, max_length, tokenizer, punctuation="."):
    sentence = add_punctuation_to_decoded(sentence, punctuation)
    tokenized = tokenizer(sentence, padding="max_length", truncation=True,
                          max_length=max_length, return_tensors="pt")
    return tokenized


def adding_punctuation_to_tokenization(samples, tokenizer, max_length):
    # adding punctuation in tokens.
    tokenized_batch = preprocess_sentence(samples, max_length=max_length - 2, tokenizer=tokenizer)
    decoded_batch = [tokenizer.decode(tb, skip_special_tokens=True) for tb in tokenized_batch["input_ids"]]
    retokenized_batch = preprocess_sentence(decoded_batch, max_length=max_length, tokenizer=tokenizer)
    return retokenized_batch


def get_device():
    """Determine the best available device for macOS"""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def sinkhorn(b, a, C, reg=1e-1, method='sinkhorn', maxIter=1000,
             stopThr=1e-9, verbose=False, log=True, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    if method.lower() == 'sinkhorn':
        # Changed to the batch version
        return sinkhorn_knopp_batched(
                a, b, C, reg, maxIter=maxIter, stopThr=stopThr,
                verbose=verbose, log=log, warm_start=warm_start,
                eval_freq=eval_freq, print_freq=print_freq, **kwargs
            )

        # return sinkhorn_knopp(a, b, C, reg, maxIter=maxIter,
        #                       stopThr=stopThr, verbose=verbose, log=log,
        #                       warm_start=warm_start, eval_freq=eval_freq, print_freq=print_freq,
        #                       **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def sinkhorn_knopp(a, b, C, reg=1e-1, maxIter=1000, stopThr=1e-9,
                   verbose=False, log=False, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    C = C.t()
    device = a.device
    na, nb = C.shape

    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b does't match that of C"
    assert reg > 0, 'reg should be greater than 0'
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'

    if log:
        log = {'err': []}

    if warm_start is not None:
        u = warm_start['u']
        v = warm_start['v']
    else:
        u = torch.ones(na, dtype=a.dtype).to(device) / na
        v = torch.ones(nb, dtype=b.dtype).to(device) / nb

    K = torch.exp(-reg * C)

    it = 1
    err = 1

    M_EPS = torch.tensor(1e-16).to(device)
    while (err > stopThr and it <= maxIter):

        u = u.to(device)
        v = v.to(device)
        b = b.to(device)
        K = K.to(device)
        upre, vpre = u, v

        KTu = torch.matmul(u.to(torch.float32), K.to(torch.float32))
        v = torch.div(b, KTu + M_EPS)
        Kv = torch.matmul(K, v)
        u = torch.div(a, Kv + M_EPS)

        if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or \
                torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)):
            print('Warning: numerical errors at iteration', it)
            u, v = upre, vpre
            break

        if log and it % eval_freq == 0:
            b_hat = torch.matmul(u, K) * v
            err = (b - b_hat).pow(2).sum().item()

            log['err'].append(err)

        if verbose and it % print_freq == 0:
            print('iteration {:5d}, constraint error {:5e}'.format(it, err))

        it += 1

    if log:
        log['u'] = u
        log['v'] = v

        log['alpha'] = reg * torch.log(u + M_EPS)
        log['beta'] = reg * torch.log(v + M_EPS)

    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    if log:
        return P, log
    else:
        return P


def sinkhorn_knopp_batched(a, b, C, reg=1e-1, maxIter=1000, stopThr=1e-9,
                           verbose=False, log=False, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    """
    Batched Sinkhorn-Knopp algorithm for solving Optimal Transport.

    Args:
        a (torch.Tensor): Source distributions of shape (batch_size, na).
        b (torch.Tensor): Target distributions of shape (batch_size, nb).
        C (torch.Tensor): Cost matrices of shape (batch_size, na, nb).
        reg (float): Regularization term.
        maxIter (int): Maximum number of iterations.
        stopThr (float): Stopping threshold.
        verbose (bool): Whether to print progress.
        log (bool): Whether to log intermediate errors.
        warm_start (dict): Initial values for u and v (optional).

    Returns:
        P (torch.Tensor): Optimal transport plans of shape (batch_size, na, nb).
        log (dict): Log containing intermediate values (if log=True).
    """
    device = a.device
    batch_size, na, nb = C.shape

    assert reg > 0, 'reg should be greater than 0'
    assert (a.min() >= 0).all() and (b.min() >= 0).all(), 'Distributions must be non-negative'
    assert C.shape == (batch_size, na, nb), "Cost matrix shape must match batch size and distributions"

    if log:
        logs = {'err': []}

    if warm_start is not None:
        u = warm_start['u']
        v = warm_start['v']
    else:
        u = torch.ones(batch_size, na, device=device) / na
        v = torch.ones(batch_size, nb, device=device) / nb

    K = torch.exp(-reg * C)

    M_EPS = torch.tensor(1e-16, device=device)
    it = 1
    err = 1

    while err > stopThr and it <= maxIter:
        u_prev, v_prev = u.clone(), v.clone()

        # Update v
        KTu = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)
        v = b / (KTu + M_EPS)

        # Update u
        Kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1)
        u = a / (Kv + M_EPS)

        # Check for numerical issues
        if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or \
                torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)):
            print(f'Warning: numerical errors at iteration {it}')
            u, v = u_prev, v_prev
            break

        # Compute error if logging
        if log and it % eval_freq == 0:
            b_hat = (u.unsqueeze(-1) * K * v.unsqueeze(1)).sum(dim=1)
            err = ((b - b_hat) ** 2).sum(dim=1).mean().item()
            logs['err'].append(err)

        if verbose and it % print_freq == 0:
            print(f'Iteration {it}, constraint error: {err:e}')

        it += 1

    # Compute transport plans
    P = u.unsqueeze(-1) * K * v.unsqueeze(1)

    if log:
        logs['u'] = u
        logs['v'] = v
        logs['alpha'] = reg * torch.log(u + M_EPS)
        logs['beta'] = reg * torch.log(v + M_EPS)
        return P, logs
    else:
        return P


if __name__ == '__main__':
    batch_size = 32
    na, nb = 80, 712
    C = torch.rand(batch_size, na, nb, device='cpu')  # Random cost matrices
    a = torch.full((batch_size, na), 1 / na, device='cpu')  # Uniform source distributions
    b = torch.full((batch_size, nb), 1 / nb, device='cpu')  # Uniform target distributions

    P = sinkhorn_knopp_batched(a, b, C, reg=0.1, maxIter=100)
    print(P.shape)  # Should be (batch_size, na, nb)
