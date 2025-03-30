```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制。

    Args:
        d_model: 输入和输出的维度。
        num_heads: 注意力头的数量。
        dropout: dropout 概率。
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        self.W_q = nn.Linear(d_model, d_model)  # Query 线性变换
        self.W_k = nn.Linear(d_model, d_model)  # Key 线性变换
        self.W_v = nn.Linear(d_model, d_model)  # Value 线性变换
        self.W_o = nn.Linear(d_model, d_model)  # 输出线性变换
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力。

        Args:
            Q: 查询矩阵 (batch_size, num_heads, seq_len, d_k)
            K: 键矩阵 (batch_size, num_heads, seq_len, d_k)
            V: 值矩阵 (batch_size, num_heads, seq_len, d_k)
            mask: 可选的掩码，用于屏蔽某些位置 (batch_size, 1, seq_len, seq_len)

        Returns:
            注意力输出和注意力权重。
        """

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # 计算注意力分数
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9) # 应用掩码, 使用极小值屏蔽
        attn_probs = torch.softmax(attn_scores, dim=-1)  # 应用 softmax 获得注意力权重
        attn_probs = self.dropout(attn_probs)
        output = torch.matmul(attn_probs, V)  # 将注意力权重与值相乘
        return output, attn_probs

    def split_heads(self, x):
        """
        将输入拆分为多个头。

        Args:
            x: 输入矩阵 (batch_size, seq_len, d_model)

        Returns:
            拆分后的矩阵 (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        将多个头合并为一个。

        Args:
            x: 输入矩阵 (batch_size, num_heads, seq_len, d_k)

        Returns:
            合并后的矩阵 (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        前向传播。

        Args:
            Q: 查询矩阵 (batch_size, seq_len, d_model)
            K: 键矩阵 (batch_size, seq_len, d_model)
            V: 值矩阵 (batch_size, seq_len, d_model)
            mask: 可选的掩码，用于屏蔽某些位置 (batch_size, 1, seq_len, seq_len)

        Returns:
            注意力输出和注意力权重。
        """

        Q = self.split_heads(self.W_q(Q))  # 线性变换并拆分头
        K = self.split_heads(self.W_k(K))  # 线性变换并拆分头
        V = self.split_heads(self.W_v(V))  # 线性变换并拆分头

        output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)  # 计算缩放点积注意力

        output = self.W_o(self.combine_heads(output))  # 合并头并进行线性变换

        return output, attn_probs


class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络。

    Args:
        d_model: 输入和输出的维度。
        d_ff: 隐藏层的维度。
        dropout: dropout 概率。
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播。

        Args:
            x: 输入矩阵 (batch_size, seq_len, d_model)

        Returns:
            输出矩阵 (batch_size, seq_len, d_model)
        """

        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    编码器层。

    Args:
        d_model: 输入和输出的维度。
        num_heads: 注意力头的数量。
        d_ff: 前馈网络的隐藏层维度。
        dropout: dropout 概率。
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播。

        Args:
            x: 输入矩阵 (batch_size, seq_len, d_model)
            mask: 可选的掩码，用于屏蔽某些位置 (batch_size, 1, seq_len, seq_len)

        Returns:
            输出矩阵 (batch_size, seq_len, d_model)
        """

        attention_output, _ = self.attention(x, x, x, mask)  # 计算多头注意力
        x = self.norm1(x + self.dropout(attention_output)) # 残差连接和层归一化
        feed_forward_output = self.feed_forward(x) # 前馈网络
        x = self.norm2(x + self.dropout(feed_forward_output)) # 残差连接和层归一化
        return x


class DecoderLayer(nn.Module):
    """
    解码器层。

    Args:
        d_model: 输入和输出的维度。
        num_heads: 注意力头的数量。
        d_ff: 前馈网络的隐藏层维度。
        dropout: dropout 概率。
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_attention = MultiHeadAttention(d_model, num_heads, dropout) # 用于 masked self-attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout) # 用于 encoder-decoder attention
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        前向传播。

        Args:
            x: 输入矩阵 (batch_size, seq_len, d_model)  解码器的输入
            encoder_output: 编码器的输出 (batch_size, seq_len, d_model)
            src_mask: 源掩码，用于屏蔽编码器的输入 (batch_size, 1, seq_len, seq_len)
            tgt_mask: 目标掩码，用于屏蔽解码器的输入 (batch_size, 1, seq_len, seq_len)

        Returns:
            输出矩阵 (batch_size, seq_len, d_model)
        """

        masked_attention_output, _ = self.masked_attention(x, x, x, tgt_mask) # Masked self-attention
        x = self.norm1(x + self.dropout(masked_attention_output)) # 残差连接和层归一化
        attention_output, _ = self.attention(x, encoder_output, encoder_output, src_mask) # Encoder-Decoder attention
        x = self.norm2(x + self.dropout(attention_output)) # 残差连接和层归一化
        feed_forward_output = self.feed_forward(x) # 前馈网络
        x = self.norm3(x + self.dropout(feed_forward_output)) # 残差连接和层归一化
        return x


class Encoder(nn.Module):
    """
    Transformer 编码器。

    Args:
        num_layers: 编码器层的数量。
        d_model: 输入和输出的维度。
        num_heads: 注意力头的数量。
        d_ff: 前馈网络的隐藏层维度。
        dropout: dropout 概率。
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        前向传播。

        Args:
            x: 输入矩阵 (batch_size, seq_len, d_model)
            mask: 可选的掩码，用于屏蔽某些位置 (batch_size, 1, seq_len, seq_len)

        Returns:
            输出矩阵 (batch_size, seq_len, d_model)
        """

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """
    Transformer 解码器。

    Args:
        num_layers: 解码器层的数量。
        d_model: 输入和输出的维度。
        num_heads: 注意力头的数量。
        d_ff: 前馈网络的隐藏层维度。
        dropout: dropout 概率。
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        前向传播。

        Args:
            x: 输入矩阵 (batch_size, seq_len, d_model)
            encoder_output: 编码器的输出 (batch_size, seq_len, d_model)
            src_mask: 源掩码，用于屏蔽编码器的输入 (batch_size, 1, seq_len, seq_len)
            tgt_mask: 目标掩码，用于屏蔽解码器的输入 (batch_size, 1, seq_len, seq_len)

        Returns:
            输出矩阵 (batch_size, seq_len, d_model)
        """

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """
    完整的 Transformer 模型。

    Args:
        src_vocab_size: 源词汇表大小。
        tgt_vocab_size: 目标词汇表大小。
        d_model: 输入和输出的维度。
        num_heads: 注意力头的数量。
        num_encoder_layers: 编码器层的数量。
        num_decoder_layers: 解码器层的数量。
        d_ff: 前馈网络的隐藏层维度。
        dropout: dropout 概率。
        max_seq_length: 最大序列长度。
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_encoder_layers=6,
                 num_decoder_layers=6, d_ff=2048, dropout=0.1, max_seq_length=512):
        super().__init__()

        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_length) # 添加位置编码
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        self.output_linear = nn.Linear(d_model, tgt_vocab_size) # 最终的线性层
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        # 初始化权重 (建议的做法)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def generate_mask(self, src, tgt):
        """
        生成源和目标掩码。

        Args:
            src: 源序列 (batch_size, src_seq_len)
            tgt: 目标序列 (batch_size, tgt_seq_len)

        Returns:
            源掩码 (batch_size, 1, src_seq_len, src_seq_len) 和 目标掩码 (batch_size, 1, tgt_seq_len, tgt_seq_len)
        """

        src_mask = (src != 0).unsqueeze(1).unsqueeze(2) # 0通常是padding token
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3) * torch.tril(torch.ones((tgt.size(1), tgt.size(1)), device=src.device)).unsqueeze(0).unsqueeze(1) # 结合 padding mask 和 subsequent mask (自回归mask)

        return src_mask, tgt_mask

    def forward(self, src, tgt):
        """
        前向传播。

        Args:
            src: 源序列 (batch_size, src_seq_len)
            tgt: 目标序列 (batch_size, tgt_seq_len)

        Returns:
            输出概率 (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        src_mask, tgt_mask = self.generate_mask(src, tgt)  # 生成掩码

        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src) * math.sqrt(self.d_model)))  # Embedding + Position Encoding + Dropout
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt) * math.sqrt(self.d_model)))  # Embedding + Position Encoding + Dropout


        encoder_output = self.encoder(src_embedded, src_mask) # 编码器
        decoder_output = self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)  # 解码器
        output = self.output_linear(decoder_output)  # 线性层
        return F.log_softmax(output, dim=-1) # 应用 LogSoftmax 获得概率


class PositionalEncoding(nn.Module):
    """
    位置编码。
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # 将 pe 作为 buffer, 不会被优化

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)




if __name__ == '__main__':
    # 超参数
    src_vocab_size = 100  # 源词汇表大小
    tgt_vocab_size = 100  # 目标词汇表大小
    d_model = 512  # Embedding 维度
    num_heads = 8  # 注意力头数
    num_encoder_layers = 6  # 编码器层数
    num_decoder_layers = 6  # 解码器层数
    d_ff = 2048  # 前馈网络中间层维度
    dropout = 0.1  # Dropout 概率
    max_seq_length = 64  # 最大序列长度
    batch_size = 32 # 批量大小

    # 创建 Transformer 模型实例
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_encoder_layers,
                              num_decoder_layers, d_ff, dropout, max_seq_length)

    # 创建示例输入数据
    src = torch.randint(1, src_vocab_size, (batch_size, max_seq_length))  # 源序列 (batch_size, seq_len)
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length))  # 目标序列 (batch_size, seq_len)

    # 前向传播
    output = transformer(src, tgt)

    # 打印输出形状
    print("Output shape:", output.shape) # (batch_size, seq_len, tgt_vocab_size)

    # 简单测试loss计算, 这里仅作演示,实际训练需要定义优化器等。
    criterion = nn.NLLLoss(ignore_index=0) # 使用负对数似然损失, 忽略 padding index
    loss = criterion(output.view(-1, tgt_vocab_size), tgt.view(-1)) #  计算loss
    print("Loss:", loss.item())
```

**代码解释和关键点:**

1.  **模块化设计:** 代码将 Transformer 分解为几个核心模块（MultiHeadAttention, PositionwiseFeedForward, EncoderLayer, DecoderLayer, Encoder, Decoder, Transformer），这使得代码更易于理解、维护和扩展。

2.  **多头注意力机制 (MultiHeadAttention):**
    *   `d_model`: 模型的维度 (例如 512).
    *   `num_heads`: 注意力头的数量 (例如 8).  `d_model` 必须能被 `num_heads` 整除.
    *   `d_k`: 每个头的维度, 等于 `d_model // num_heads`.
    *   它将输入 `Q` (Query), `K` (Key), `V` (Value) 线性变换到不同的空间, 并拆分成多个头.
    *   使用缩放点积注意力计算注意力权重 (`attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)`). 缩放是为了防止点积结果过大，导致 softmax 输出过于集中.
    *   `mask` 用于屏蔽某些位置，例如 padding 或未来信息 (在自回归解码中).  `mask` 的值通常是 0 表示屏蔽, 1 表示不屏蔽. 使用 `-1e9` 填充是为了使 softmax 输出在该位置接近于 0.
    *   将多个头的结果合并, 并再次线性变换.

3.  **位置前馈网络 (PositionwiseFeedForward):**
    *   这是一个简单的两层全连接网络, 用于对每个位置的信息进行更复杂的处理.
    *   通常 `d_ff` (隐藏层维度) 比 `d_model` 大 (例如, `d_ff = 4 * d_model`).

4.  **编码器和解码器层 (EncoderLayer, DecoderLayer):**
    *   每一层都包含多头注意力机制和位置前馈网络.
    *   使用残差连接 (`x + self.dropout(...)`) 和层归一化 (`nn.LayerNorm`).  残差连接有助于训练更深的网络.  层归一化加速训练, 并提高模型的稳定性.
    *   解码器层中的 `masked_attention` 使用 target mask 来防止模型看到未来的信息 (自回归).
    *   解码器层还包含一个额外的注意力机制，用于关注编码器的输出 (`self.attention(x, encoder_output, encoder_output, src_mask)`).

5.  **编码器和解码器 (Encoder, Decoder):**
    *   由多个编码器/解码器层堆叠而成.

6.  **Transformer (完整模型):**
    *   包含编码器和解码器.
    *   使用 `nn.Embedding` 将输入 token 转换为 embedding 向量.
    *   **位置编码 (Positional Encoding):**  Transformer 没有 RNN 的循环结构, 因此需要显式地添加位置信息.  位置编码将 token 在序列中的位置编码成向量, 并与 embedding 向量相加.  代码中使用的是正弦和余弦函数来生成位置编码。
    *   `generate_mask`:  生成源序列和目标序列的掩码.  目标序列的掩码需要同时考虑 padding 和 future information.  `torch.tril` 用于生成下三角矩阵, 用于屏蔽未来信息.
    *   `F.log_softmax`: 在输出层应用 `log_softmax` 函数, 获得每个 token 的概率.  `log_softmax` 提高了数值稳定性，并且与 `nn.NLLLoss` 配合使用。

7. **权重初始化:**  使用 `nn.init.xavier_uniform_` 初始化权重是一种常见的做法, 可以帮助模型更快地收敛。

8.  **Mask 的作用:**

    *   **Padding Mask:** 防止模型关注 padding 位置 (通常用 0 表示).  这对于处理不同长度的序列至关重要。
    *   **Subsequent Mask (也叫 causal mask):**  用于解码器, 防止模型在预测时看到未来的 token.  这对于自回归模型是必须的。

9.  **Positional Encoding:**  `d_model` 通常比较大，所以位置编码的范围也需要比较大。 除以 `10000**(2i/d_model)` 可以使波长随着embedding的维度i增加呈指数增长，保证了所有维度的数值都不会太大。 同时使用正弦和余弦函数编码位置信息， 这样可以方便模型学习到相对位置的关系。

**如何使用:**

1.  **准备数据:**  将文本数据转换为 token ID 序列.  需要一个词汇表 (`src_vocab_size`, `tgt_vocab_size`).
2.  **创建模型实例:**  根据需要调整超参数.
3.  **训练模型:**  使用 PyTorch 的标准训练循环.  需要定义损失函数 (例如 `nn.CrossEntropyLoss` 或 `nn.NLLLoss`) 和优化器 (例如 `torch.optim.Adam`).
4.  **推理:**  使用训练好的模型进行预测.  需要实现解码逻辑 (例如 beam search).

**关键改进点和注意事项:**

*   **优化器:** AdamW 通常比 Adam 效果更好。
*   **学习率调度器:** 使用学习率调度器 (例如 WarmupLinearSchedule) 可以提高训练效果。
*   **Beam Search:** 使用 beam search 解码可以提高生成序列的质量。
*   **梯度裁剪:** 使用梯度裁剪可以防止梯度爆炸。
*   **混合精度训练:** 使用混合精度训练 (AMP) 可以加速训练并减少内存消耗。
*   **数据并行:** 使用 `torch.nn.DataParallel` 或 `torch.distributed.launch` 可以进行数据并行训练.
*   **更复杂的 Mask:** 可以根据具体的任务定制更复杂的 Mask.
*   **BPE/WordPiece Tokenization:** 使用 BPE (Byte Pair Encoding) 或 WordPiece 等 subword tokenization 方法可以更好地处理未知词汇。
*   **Validation:** 使用验证集来监控训练过程，并防止过拟合。

这个代码提供了一个完整的 Transformer 神经网络框架模型，包括编码器、解码器、多头注意力机制、位置编码等核心组件，并附带详细的注释，方便理解和使用。 你可以根据自己的需求进行修改和扩展。 重要的是理解每个组件的作用,以及如何将它们组合在一起构建完整的 Transformer 模型。
