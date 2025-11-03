def _sense_id(s):
    # turn a Synset or string into a synset-id string
    return s if isinstance(s, str) else s.name()

def bert_1nn(batch: T.List[T.List[WSDToken]],
             indices: T.Iterable[T.Iterable[int]],
             sense_vecs: T.Mapping[str, Tensor]) -> T.List[T.List[Synset]]:
    # 1) Run BERT on lemmas (pre-tokenized)






    embeddings, offset_mappings = run_bert(batch_token)   # [B, T_bert, H]
    B, T, H = embeddings.shape

    pred: T.List[T.List[Synset]] = []

    for b_idx, (original, offsets, target_idxs) in enumerate(zip(batch, offset_mappings, indices)):
        word_vector_pieces: T.List[Tensor] = []
        curr_sentence: T.List[Tensor] = []

        for t_idx, (start, end) in enumerate(offsets):
            if start == 0 and end == 0:
                continue 

            sub_vec = embeddings[b_idx, t_idx] 

            if start == 0:
                # new word starts: flush previous word if exists
                if word_vector_pieces:
                    curr_sentence.append(torch.mean(torch.stack(word_vector_pieces, dim=0), dim=0))
                    word_vector_pieces = []
                word_vector_pieces.append(sub_vec)
            else:
                # continuation piece of current word
                word_vector_pieces.append(sub_vec)

        # flush last word
        if word_vector_pieces:
            curr_sentence.append(torch.mean(torch.stack(word_vector_pieces, dim=0), dim=0))

        # align lengths safely
        length_token = min(len(original), len(curr_sentence))

        # 3) For each target token, pick 1-NN sense by cosine
        sent_preds: T.List[Synset] = []

        for idx in target_idxs:
            # out-of-range after alignment → fallback to MFS
            if idx >= length_token:
                sent_preds.append(mfs(original, idx))
                continue

            tok = original[idx]
            tok_vec = curr_sentence[idx].detach()  # keep on GPU, drop graph

            synsets = getattr(tok, "synsets", None)
            if not synsets:
                sent_preds.append(mfs(original, idx))
                continue

            # build candidate matrix S (only senses we have vectors for)
            S_list: T.List[Tensor] = []
            kept_syns: T.List[Synset] = []
            for syn in synsets:
                sid = _sense_id(syn)
                vec = sense_vecs.get(sid)
                if vec is not None:
                    S_list.append(vec)       # [H]
                    kept_syns.append(syn)    # keep the Synset to return

            if not S_list:
                sent_preds.append(mfs(original, idx))
                continue

            S = torch.stack(S_list, dim=0)          # [k, H]
            t = tok_vec.unsqueeze(0)                # [1, H]

            # cosine via normalization + matmul (no Python loops)
            S = S / (S.norm(dim=1, keepdim=True) + 1e-12)
            t = t / (t.norm(dim=1, keepdim=True) + 1e-12)
            scores = S @ t.T                         # [k, 1]
            best = int(torch.argmax(scores, dim=0))  # index of best sense

            # append the best Synset object (not its vector)
            sent_preds.append(kept_syns[best])

        pred.append(sent_preds)

    return pred