import torch


def get_lang_file_dict():
    # for the folder directory in the server
    # folderdir = "dataset/Morphology-Matters-corpus"
    return {'afr': 'afr-1953', 'aln': 'aln-aln',
            'als': 'als', 'amh': 'amh', 'arb': 'arb-arb', 'arz': 'arz-arz',
            'ayr': 'ayr-2011', 'bba': 'bba-bba', 'ben': 'ben-mussolmani',
            'bqc': 'bqc-bqc', 'bul': 'bul-veren', 'cac': 'cac-ixtatan',
            'cak': 'cak-central2003', 'ceb': 'ceb-pinadayag', 'ces': 'ces-kralicka',
            'cmn': 'cmn-sf_ncv-zefania', 'cnh': 'cnh-cnh', 'crk': 'crk-mason',
            'cym': 'cym-morgan1804', 'dan': 'dan-1931', 'deu': 'deu-luther1912',
            'dje': 'dje', 'ell': 'ell-modern2009', 'eng': 'eng-literal',
            'esu': 'esu', 'fin': 'fin-1992', 'fra': 'fra-pirotclamer',
            'gug': 'gug', 'gui': 'gui', 'guj': 'guj-guj', 'gur': 'gur-frafra',
            'hat': 'hat-1999', 'heb': 'heb',
            'hin': 'hin_5', 'hrv': 'hrv-hrv', 'hun': 'hun-2005', 'ike':
            'ike', 'ikt': 'ikt', 'ind': 'ind-terjemahanbaru', 'isl':
            'isl', 'ita': 'ita-diodati', 'jpn_tok': 'jpn_tok', 'kal': 'kal',
            'kan': 'kan', 'kek': 'kek-1988', 'kjb': 'kjb-kjb', 'kor': 'kor',
            'lat': 'lat-novavulgata', 'lit': 'lit-lit', 'mah': 'mah-mah',
            'mal': 'mal-malirv', 'mam': 'mam-northern', 'mar': 'mar-marirv',
            'mri': 'mri-mri', 'mya': 'mya-mya', 'nch': 'nch', 'nep': 'nep',
            'nhe': 'nhe', 'nld': 'nld-nld', 'nor': 'nor-student',
            'pck': 'pck', 'pes': 'pes', 'plt': 'plt-romancatholic', 'poh': 'poh-eastern',
            'pol': 'pol-ubg', 'por': 'por-almeidaatualizada', 'qub': 'qub-qub', 'quh':
            'quh-1993', 'quy': 'quy-quy', 'quz': 'quz-quz', 'ron': 'ron-cornilescu',
            'rus': 'rus-synodal', 'slk': 'slk', 'slv': 'slv', 'sna': 'sna-sna2002',
            'som': 'som-som', 'spa': 'spa-gug', 'swe': 'swe', 'tbz': 'tbz-tbz',
            'tel': 'tel-tel', 'tgl': 'tgl-1905', 'tha_tok': 'tha_tok', 'tob': 'tob',
            'tpi': 'tpi-tpi', 'tpm': 'tpm-tpm', 'tur': 'tur', 'ukr': 'ukr-2009',
            'vie': 'vie-2002', 'wal': 'wal-wal', 'wbm': 'wbm-wbm', 'xho': 'xho-xho',
            'zom': 'zom-zom'}


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


def pairwise_cosine(m1, m2=None, eps=1e-6):
    if m2 is None:
        m2 = m1
    w1 = m1.norm(p=2, dim=1, keepdim=True)
    w2 = m2.norm(p=2, dim=1, keepdim=True)

    return torch.mm(m1, m2.t()) / (w1 * w2.t()).clamp(eps)


def get_device():
    """Determine the best available device for macOS"""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def sinkhorn(b, a, C, reg=1e-1, method='sinkhorn', maxIter=1000,
             stopThr=1e-9, verbose=False, log=True, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    if method.lower() == 'sinkhorn':

        return sinkhorn_knopp(a, b, C, reg, maxIter=maxIter,
                              stopThr=stopThr, verbose=verbose, log=log,
                              warm_start=warm_start, eval_freq=eval_freq, print_freq=print_freq,
                              **kwargs)
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

