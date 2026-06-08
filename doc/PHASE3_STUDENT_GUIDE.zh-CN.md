# 第三阶段：自动化 LLM 推理运行时

在本阶段，你需要构建一个自动化 LLM 推理运行时。它应能够根据提供的配置和权重加载一个 decoder-only 模型，维护请求状态，并高效执行 prefill 和 decode。运行时会作为黑盒接受评测：我们会将它的 logits 与参考实现进行对比以测试正确性，然后使用类似服务场景的请求 trace 来测量吞吐量和内存行为。

评测系统会先运行你的 `run.sh`。在这个脚本中，你可以读取公开提供的模型结构和权重，并执行编译、准备、调试或自测。随后，评测系统会导入你的 `engine.py`，并通过固定接口测试正确性和吞吐量。

## 你需要读取的内容

模型配置文件为：

```text
/target/model_config.json
```

在公开示例中，对应文件为：

```text
target/model_config.json
```

它描述了层数、隐藏层维度、注意力头数、词表大小以及其他模型信息。你的 `engine.py` 不应硬编码这些值，而应使用传入 `create_engine(model_config, weight_dir, device)` 的 `model_config` 动态构建运行时。

模型权重目录为：

```text
/target/weights
```

在公开示例中，它包含：

```text
target/weights/model.pt
```

在隐藏评测中，隐藏权重会被放置在评测系统指定的位置。评测 trace 不会提前提供。你的运行时应能适应不同的 batch size、prompt 长度、decode 长度以及请求顺序。

## 你需要提供的内容

当你的 `run.sh` 执行结束后，以下文件必须存在：

```text
/workspace/engine.py
```

在公开示例中，对应文件为：

```text
workspace/engine.py
```

你的 agent 还应生成一份推理说明输出：

```text
/workspace/output3.*
```

## `engine.py` 中要求的接口

`engine.py` 必须包含：

```python
def create_engine(model_config: dict, weight_dir: str, device: str = "cuda"):
    return Engine(...)
```

返回的对象必须支持：

```python
class Engine:
    def prefill(self, request_ids, input_ids):
        ...

    def decode(self, request_ids, token_ids):
        ...

    def remove(self, request_ids):
        ...
```

`prefill()` 的输入：

- `request_ids: list(int)`：请求 ID 列表。
- `input_ids: list(torch.Tensor)`：token 序列列表，其中每个元素都是一维张量，且 `dtype=torch.long`。

`prefill()` 的返回值：

- `torch.Tensor`：形状为 `[batch_size, vocab_size]` 的 logits，其中 `batch_size = len(request_ids)`。第 `i` 行对应 `request_ids[i]` 的最后一个 token 的 logits。

`decode()` 的输入：

- `request_ids: list(int)`：已经完成 prefill 的请求 ID 列表。
- `token_ids: torch.Tensor`：形状为 `[batch_size]` 的一维张量，表示每个请求新追加的一个 token。

`decode()` 的返回值：

- `torch.Tensor`：形状为 `[batch_size, vocab_size]` 的 logits。第 `i` 行对应将 `token_ids[i]` 追加到 `request_ids[i]` 后，最后一个 token 的 logits。

`remove()` 的输入：

- `request_ids: list(int)`：需要终止的请求 ID 列表。

`remove()` 不需要返回任何内容，但必须释放或删除与这些请求关联的 KV cache / 请求状态。

## 正确性如何测试

评测系统会提供官方 PyTorch 参考模型。它会加载相同的隐藏权重，并针对同一批请求计算参考 logits。

我们不根据最终生成文本进行评分，因为采样策略会引入不必要的不确定性。相反，我们直接比较 logits。

比较规则为：

$$
|y_{\mathrm{student}} - y_{\mathrm{ref}}| \leq \mathrm{atol} + \mathrm{rtol} \cdot |y_{\mathrm{ref}}|
$$

在公开示例中，默认值为：

$$
\mathrm{atol}=10^{-2}, \quad \mathrm{rtol}=10^{-2}
$$

也就是说，我们使用：

```python
torch.allclose(student_logits, ref_logits, atol=1e-2, rtol=1e-2)
```

正确性测试覆盖：

- 单请求 prefill。
- 单请求 decode。
- 多请求 prefill。
- 多请求 decode。
- 插入新请求。
- 在移除部分请求后，继续 decode 其他请求。

如果某个用例的正确性失败，则该用例的性能分数为 0。

## 吞吐量如何测试

吞吐量测试由评测系统驱动。评测系统会导入 `engine.py`、构造 engine，然后运行固定 trace：

```python
engine = create_engine(model_config, weight_dir, device)
engine.prefill(...)
engine.decode(...)
engine.remove(...)
```

计时区域只包含 trace 中的 `prefill()`、`decode()` 和 `remove()` 调用，不包含 `create_engine()` 或权重加载时间。

吞吐量定义为：

$$
\mathrm{tokens/s}=\frac{\mathrm{prefill\ tokens}+\mathrm{decode\ tokens}}{\mathrm{elapsed\ seconds}}
$$

Decode 吞吐量定义为：

$$
\mathrm{decode\ tokens/s}=\frac{\mathrm{decode\ tokens}}{\mathrm{elapsed\ seconds}}
$$

公开示例提供三类 benchmark：

- `prefill`：长 prompt 的批量 prefill。
- `decode`：多个请求的连续 decode。
- `mixed`：包含 prefill、decode 和 remove 操作的混合 trace。

隐藏评测会使用相同的测试方法，但会替换模型大小、权重、batch size、prompt 长度、decode 步数和 trace。

## 如何运行公开示例

如果权重文件不存在，先生成 toy weights：

```bash
python3 scripts/generate_toy_weights.py \
  --config target/model_config.json \
  --output target/weights/model.pt
```

运行正确性测试：

```bash
python3 evaluator/test_correctness.py \
  --engine workspace/engine.py \
  --model-config target/model_config.json \
  --weight-dir target/weights \
  --device auto
```

运行吞吐量测试：

```bash
python3 evaluator/benchmark_throughput.py \
  --engine workspace/engine.py \
  --model-config target/model_config.json \
  --weight-dir target/weights \
  --device auto
```

你也可以直接运行：

```bash
bash scripts/run_public_tests.sh
```

如果默认的 `python3` 没有安装 PyTorch，可以指定解释器：

```bash
PYTHON=/path/to/python-with-torch bash scripts/run_public_tests.sh
```

## Baseline 说明

公开示例中的 `workspace/engine.py` 是一个最小 PyTorch baseline。它会为每个请求保存完整 token 序列，并在每次 `decode()` 调用时重新运行整个序列。因此它非常慢，但接口语义是正确的。
