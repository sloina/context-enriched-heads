# wekws patch: expose backbone features for extraction

`extract_features.py` calls `model.forward_with_features(...)`, a small
method that is **not** part of upstream
[wekws](https://github.com/wenet-e2e/wekws). It returns the backbone
hidden states alongside the per-frame posteriors, which is what the
linear decision head is trained on. Without this patch,
`extract_features.py` fails with:

```
AttributeError: 'KWSModel' object has no attribute 'forward_with_features'
```

## How to apply

Open `wekws/model/kws_model.py` in your wekws checkout and add the
following method to the `KWSModel` class, directly after the existing
`forward()` method (before `forward_softmax()`):

```python
    def forward_with_features(
            self,
            x: torch.Tensor,
            in_cache: torch.Tensor = torch.zeros(0, 0, 0, dtype=torch.float)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like forward(), but also returns backbone representations.

        Returns:
            probs: (B, T, odim) - post-activation scores (as in forward)
            hidden: (B, T, hdim) - backbone output, pre-classifier
            out_cache: streaming cache
        """
        if self.global_cmvn is not None:
            x = self.global_cmvn(x)
        x = self.preprocessing(x)
        hidden, out_cache = self.backbone(x, in_cache)
        probs = self.activation(self.classifier(hidden))
        return probs, hidden, out_cache
```

The computation is identical to `forward()`; the only difference is that
the backbone output is kept and returned before the classifier is
applied, so scores are unaffected.

Equivalently, as a unified diff (`git apply wekws_feature_extraction.patch`
from the wekws root, if you save the hunk below to a file):

```diff
--- a/wekws/model/kws_model.py
+++ b/wekws/model/kws_model.py
@@ -75,6 +75,25 @@
         x = self.activation(x)
         return x, out_cache
 
+    def forward_with_features(
+            self,
+            x: torch.Tensor,
+            in_cache: torch.Tensor = torch.zeros(0, 0, 0, dtype=torch.float)
+    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
+        """Like forward(), but also returns backbone representations.
+
+        Returns:
+            probs: (B, T, odim) - post-activation scores (as in forward)
+            hidden: (B, T, hdim) - backbone output, pre-classifier
+            out_cache: streaming cache
+        """
+        if self.global_cmvn is not None:
+            x = self.global_cmvn(x)
+        x = self.preprocessing(x)
+        hidden, out_cache = self.backbone(x, in_cache)
+        probs = self.activation(self.classifier(hidden))
+        return probs, hidden, out_cache
+
     def forward_softmax(
         self,
         x: torch.Tensor,
```

## Notes

- Line numbers may drift slightly across wekws versions; the anchor is
  the position **right after `forward()`** inside `KWSModel`. The method
  is self-contained and does not depend on any other change.
- No existing wekws behavior is modified; training, scoring, and DET
  computation are untouched.
- wekws is licensed under Apache-2.0; this addition is distributed under
  the same license.
