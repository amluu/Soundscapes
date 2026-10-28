"""all-MiniLM-L6-v2 sentence embeddings with ONNX Runtime."""
import os

import numpy as np


class MiniLM:
    def __init__(self, model_dir):
        import onnxruntime
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(model_dir, 'tokenizer.json'))
        self.tok.enable_truncation(256)
        self.tok.enable_padding()
        self.session = onnxruntime.InferenceSession(os.path.join(model_dir, 'model.onnx'),
                                                    providers=['CPUExecutionProvider'])
        self.inputs = {i.name for i in self.session.get_inputs()}

    def encode(self, texts, batch_size=64):
        out = []
        for b in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch([t.replace('\n', ' ') for t in texts[b:b + batch_size]])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': np.zeros_like(ids)}
            hidden = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
            m = mask[..., None]
            out.append(normalize((hidden * m).sum(1) / m.sum(1)))
        return np.vstack(out).astype(np.float32)


def normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, n, out=np.zeros_like(X), where=n > 0)
