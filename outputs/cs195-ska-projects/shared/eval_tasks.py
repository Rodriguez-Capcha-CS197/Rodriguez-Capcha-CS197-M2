import random
import torch
from torch.utils.data import Dataset


class MQARDataset(Dataset):
    """Multi-Query Associative Recall. M key-value pairs, distractor gap, query all M."""
    def __init__(self, n_examples=256, M=16, distractor_gap=None,
                 seq_len=4096, vocab_size=512):
        self.examples = []
        V = vocab_size
        q = V // 4
        keys_pool = list(range(0, q))
        vals_pool = list(range(q, 2 * q))
        noise_pool = list(range(2 * q, V))
        if distractor_gap is None:
            distractor_gap = seq_len - 4 * M
        for _ in range(n_examples):
            ks = random.sample(keys_pool, M)
            vs = [random.choice(vals_pool) for _ in range(M)]
            ns = [random.choice(noise_pool) for _ in range(distractor_gap)]
            perm = list(range(M))
            random.shuffle(perm)
            kv = []
            for i in range(M):
                kv.extend([ks[i], vs[i]])
            qr = []
            for i in perm:
                qr.extend([ks[i], vs[i]])
            full = kv + ns + qr
            T = len(full) - 1
            qs = 2 * M + distractor_gap
            loss_mask = torch.zeros(T, dtype=torch.float32)
            for i in range(M):
                pos = qs + 2 * i
                if pos < T:
                    loss_mask[pos] = 1.0
            prefix_mask = torch.zeros(T, dtype=torch.float32)
            prefix_mask[:qs] = 1.0
            self.examples.append({
                "input_ids": torch.tensor(full[:-1], dtype=torch.long),
                "labels": torch.tensor(full[1:], dtype=torch.long),
                "loss_mask": loss_mask,
                "prefix_mask": prefix_mask,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class PhonebookDataset(Dataset):
    """N name-number pairs, distractor gap, query one name."""
    def __init__(self, n_examples=256, n_entries=16, distractor_gap=None,
                 seq_len=4096, vocab_size=512):
        self.examples = []
        V = vocab_size
        name_pool = list(range(0, V // 4))
        num_pool = list(range(V // 4, V // 2))
        noise_pool = list(range(V // 2, V))
        tokens_per_entry = 5
        if distractor_gap is None:
            distractor_gap = seq_len - tokens_per_entry * n_entries - 8
        for _ in range(n_examples):
            names = [random.sample(name_pool, 2) for _ in range(n_entries)]
            numbers = [[random.choice(num_pool) for _ in range(3)] for _ in range(n_entries)]
            phonebook = []
            for name, num in zip(names, numbers):
                phonebook.extend(name + num)
            noise = [random.choice(noise_pool) for _ in range(distractor_gap)]
            qi = random.randint(0, n_entries - 1)
            query = names[qi]
            answer = numbers[qi]
            full = phonebook + noise + query + answer
            T = len(full) - 1
            ans_start = len(phonebook) + len(noise) + len(query)
            loss_mask = torch.zeros(T, dtype=torch.float32)
            for i in range(len(answer)):
                pos = ans_start + i - 1
                if 0 <= pos < T:
                    loss_mask[pos] = 1.0
            prefix_mask = torch.zeros(T, dtype=torch.float32)
            prefix_mask[:ans_start] = 1.0
            self.examples.append({
                "input_ids": torch.tensor(full[:-1], dtype=torch.long),
                "labels": torch.tensor(full[1:], dtype=torch.long),
                "loss_mask": loss_mask,
                "prefix_mask": prefix_mask,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class InductionDataset(Dataset):
    """Place [A][B] patterns early, query [A] after long gap, must predict [B]."""
    def __init__(self, n_examples=256, n_patterns=8, distractor_gap=None,
                 seq_len=4096, vocab_size=512):
        self.examples = []
        token_pool = list(range(0, vocab_size))
        if distractor_gap is None:
            distractor_gap = seq_len - 4 * n_patterns
        for _ in range(n_examples):
            a_tokens = random.sample(token_pool, n_patterns)
            b_tokens = [random.choice(token_pool) for _ in range(n_patterns)]
            patterns = []
            for a, b in zip(a_tokens, b_tokens):
                patterns.extend([a, b])
            noise = [random.choice(token_pool) for _ in range(distractor_gap)]
            perm = list(range(n_patterns))
            random.shuffle(perm)
            queries = []
            for i in perm:
                queries.extend([a_tokens[i], b_tokens[i]])
            full = patterns + noise + queries
            T = len(full) - 1
            qs = 2 * n_patterns + distractor_gap
            loss_mask = torch.zeros(T, dtype=torch.float32)
            for i in range(n_patterns):
                pos = qs + 2 * i
                if pos < T:
                    loss_mask[pos] = 1.0
            prefix_mask = torch.zeros(T, dtype=torch.float32)
            prefix_mask[:qs] = 1.0
            self.examples.append({
                "input_ids": torch.tensor(full[:-1], dtype=torch.long),
                "labels": torch.tensor(full[1:], dtype=torch.long),
                "loss_mask": loss_mask,
                "prefix_mask": prefix_mask,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class SelectiveCopyDataset(Dataset):
    """Marked tokens scattered across context, reproduce them at the end."""
    def __init__(self, n_examples=256, n_targets=16, seq_len=4096,
                 vocab_size=512, marker_token=0):
        self.examples = []
        content_pool = list(range(2, vocab_size))
        end_marker = 1
        for _ in range(n_examples):
            targets = [random.choice(content_pool) for _ in range(n_targets)]
            body_len = seq_len - 2 * n_targets - n_targets - 1
            positions = sorted(random.sample(range(body_len), n_targets))
            body = []
            ti = 0
            pos_set = set(positions)
            for i in range(body_len):
                if i in pos_set:
                    body.append(marker_token)
                    body.append(targets[ti])
                    ti += 1
                else:
                    body.append(random.choice(content_pool))
            body.append(end_marker)
            body.extend(targets)
            full = body
            T = len(full) - 1
            ans_start = len(body) - n_targets
            loss_mask = torch.zeros(T, dtype=torch.float32)
            for i in range(n_targets):
                pos = ans_start + i - 1
                if 0 <= pos < T:
                    loss_mask[pos] = 1.0
            prefix_mask = torch.zeros(T, dtype=torch.float32)
            prefix_mask[:ans_start] = 1.0
            self.examples.append({
                "input_ids": torch.tensor(full[:-1], dtype=torch.long),
                "labels": torch.tensor(full[1:], dtype=torch.long),
                "loss_mask": loss_mask,
                "prefix_mask": prefix_mask,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class MultiHopDataset(Dataset):
    """A->B->C chains separated by gaps. Query A, answer C."""
    def __init__(self, n_examples=256, n_chains=4, hops=2,
                 distractor_gap=None, seq_len=4096, vocab_size=512):
        self.examples = []
        token_pool = list(range(0, vocab_size))
        noise_pool = list(range(vocab_size // 2, vocab_size))
        tokens_per_chain = 2 * (hops + 1)
        if distractor_gap is None:
            gap_per_hop = (seq_len - n_chains * tokens_per_chain - 2 * n_chains) // max(n_chains * hops, 1)
            gap_per_hop = max(gap_per_hop, 10)
        else:
            gap_per_hop = distractor_gap
        for _ in range(n_examples):
            body = []
            chains = []
            for _ in range(n_chains):
                nodes = random.sample(token_pool, hops + 1)
                chains.append(nodes)
                for hi in range(hops):
                    body.extend([nodes[hi], nodes[hi + 1]])
                    body.extend([random.choice(noise_pool) for _ in range(gap_per_hop)])
            perm = list(range(n_chains))
            random.shuffle(perm)
            queries = []
            for i in perm:
                queries.extend([chains[i][0], chains[i][-1]])
            full = body + queries
            T = len(full) - 1
            qs = len(body)
            loss_mask = torch.zeros(T, dtype=torch.float32)
            for i in range(n_chains):
                pos = qs + 2 * i
                if pos < T:
                    loss_mask[pos] = 1.0
            prefix_mask = torch.zeros(T, dtype=torch.float32)
            prefix_mask[:qs] = 1.0
            self.examples.append({
                "input_ids": torch.tensor(full[:-1], dtype=torch.long),
                "labels": torch.tensor(full[1:], dtype=torch.long),
                "loss_mask": loss_mask,
                "prefix_mask": prefix_mask,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class SystemPromptDataset(Dataset):
    """Variable bindings in prefix, scratchpad, query one variable."""
    def __init__(self, n_examples=256, n_vars=8, distractor_gap=None,
                 seq_len=4096, vocab_size=512):
        self.examples = []
        var_name_pool = list(range(0, vocab_size // 8))
        var_val_pool = list(range(vocab_size // 8, vocab_size // 4))
        noise_pool = list(range(vocab_size // 4, vocab_size))
        separator = vocab_size // 4 - 1
        query_marker = vocab_size // 4 - 2
        if distractor_gap is None:
            distractor_gap = seq_len - 3 * n_vars - 4
        for _ in range(n_examples):
            var_names = random.sample(var_name_pool, n_vars)
            var_vals = [random.sample(var_val_pool, 2) for _ in range(n_vars)]
            header = []
            for name, val in zip(var_names, var_vals):
                header.extend([name] + val + [separator])
            noise = [random.choice(noise_pool) for _ in range(distractor_gap)]
            qi = random.randint(0, n_vars - 1)
            query = [query_marker, var_names[qi]]
            answer = var_vals[qi]
            full = header + noise + query + answer
            T = len(full) - 1
            ans_start = len(header) + len(noise) + len(query)
            loss_mask = torch.zeros(T, dtype=torch.float32)
            for i in range(len(answer)):
                pos = ans_start + i - 1
                if 0 <= pos < T:
                    loss_mask[pos] = 1.0
            prefix_mask = torch.zeros(T, dtype=torch.float32)
            prefix_mask[:ans_start] = 1.0
            self.examples.append({
                "input_ids": torch.tensor(full[:-1], dtype=torch.long),
                "labels": torch.tensor(full[1:], dtype=torch.long),
                "loss_mask": loss_mask,
                "prefix_mask": prefix_mask,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
