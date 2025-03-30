### 关键要点
- 研究表明，FLOPs 通常指浮点加法、减法、乘法和除法等基本操作。
- 证据显示，FLOPs 包括单精度（32 位）和双精度（64 位）浮点指令。
- 它似乎可能包括 SIMD 指令，每条指令可执行多个浮点操作。

---

### 直接回答

#### 什么是 FLOPs 所指的浮点运算指令？
FLOPs（每秒浮点运算次数）衡量的是处理器每秒能执行的浮点运算数量。从计算机体系结构的角度看，它通常包括加法、减法、乘法和除法这些基本浮点操作的指令。这些指令可以是单精度（32 位）或双精度（64 位）浮点数，具体取决于指令集。

例如，在 x86 架构中，可能会用到 ADDSS（单精度加法）或 MULPD（双精度乘法）这样的指令。研究表明，FLOPs 还可能包括 SIMD（单指令多数据）指令，比如 AVX 指令，一条指令可以同时对多个数据执行加法或乘法操作。

一个意外的细节是，在高性能计算中，FLOPs 常常更关注加法和乘法，因为它们在科学计算（如矩阵乘法）中最常见，但理论上所有浮点操作都应被计入。

---

### 调查报告：FLOPs 在指令集系统中的浮点运算指令详解

#### 引言
FLOPs（Floating Point Operations Per Second，每秒浮点运算次数）是衡量计算性能的核心指标，尤其在高性能计算（HPC）和人工智能领域。从计算机体系结构的角度看，FLOPs 的计量涉及指令集架构（ISA）中执行的浮点运算指令类型。本报告将详细探讨 FLOPs 所指的具体浮点运算指令，包括其定义、常见类型和实际应用。

[[FLOPS的计量单位]]

#### FLOPs 的定义与背景
FLOPs 是一种衡量计算机浮点运算速度的单位，反映处理器每秒能执行的浮点运算次数。在算法层面，每个浮点加法、减法、乘法或除法通常被计为一次 FLOP。而在计算机体系结构层面，FLOPs 的实现依赖于 ISA 中定义的浮点指令，这些指令由处理器执行。

研究表明，FLOPs 通常涵盖以下基本浮点操作：
- 加法（Addition）
- 减法（Subtraction）
- 乘法（Multiplication）
- 除法（Division）

此外，现代处理器还支持更复杂的浮点操作，如平方根（Square Root）、三角函数（Trigonometric Functions）等，但这些操作在 FLOPs 计数中通常较少被单独强调。

#### 浮点指令的类型
从指令集系统的角度看，浮点运算指令可以分为以下几类：
1. **标量浮点指令**：对单个浮点数执行操作。例如：
   - x86 中的 ADDSS（Add Scalar Single-Precision）：单精度浮点加法。
   - x86 中的 MULPD（Multiply Packed Double-Precision）：双精度浮点乘法。
2. **向量浮点指令（SIMD）**：利用单指令多数据技术，一条指令可对多个数据并行执行操作。例如：
   - AVX（Advanced Vector Extensions）指令，如 VADDPS（Vector Add Packed Single-Precision），可同时对 8 个单精度浮点数执行加法。
   - ARM 的 NEON 指令集也支持类似的功能。

证据显示，SIMD 指令在现代处理器中尤为重要，因为它们显著提高了浮点运算的并行性。例如，一条 AVX-512 指令可能执行 16 个单精度浮点加法，理论上贡献 16 次 FLOP。

#### 单精度与双精度
浮点指令通常分为单精度（32 位）和双精度（64 位）两种：
- **单精度**：32 位浮点数，适用于 AI 和机器学习等对精度要求较低的场景。
- **双精度**：64 位浮点数，常用在科学计算和工程模拟中。

研究表明，FLOPs 计数中两者都被包括，但具体性能可能因处理器设计而异。例如，Nvidia A100 GPU 的单精度峰值性能约为 19.5 TFLOPS，而双精度约为 9.7 TFLOPS，反映了硬件对不同精度的优化。

#### FLOPs 计数的实际实现
在性能测量中，FLOPs 的计数基于执行的浮点指令数量。例如：
- 如果一条指令执行一次单精度加法，则计为 1 次 FLOP。
- 如果一条 SIMD 指令（如 AVX-512 加法）同时对 16 个单精度数执行加法，则计为 16 次 FLOP。

这意味着，处理器的高峰 FLOPs 通常基于其 ALU（算术逻辑单元）数量和每周期可执行的操作数。例如，一个 GPU 可能有数千个 ALU，每个周期可执行多个浮点操作，从而达到 PFLOPS 或 EFLOPS 级别的性能。

#### 争议与挑战
FLOPs 的定义和计数存在一定争议。一方面，研究者普遍同意基本操作（如加法、乘法）应被计入；另一方面，复杂操作（如平方根）是否应计入 FLOPs 尚无统一标准。此外，SIMD 指令的贡献如何计算也可能因基准测试而异。例如，LINPACK 基准测试主要关注双精度加法和乘法，而 AI 工作负载更关注单精度操作。

#### 表格：典型浮点指令示例

| 指令类型       | 示例指令          | 描述                              | 贡献 FLOP 数量 |
|----------------|-------------------|-----------------------------------|---------------|
| 标量单精度加法 | ADDSS (x86)      | 对两个单精度浮点数执行加法        | 1             |
| 向量单精度加法 | VADDPS (AVX)     | 对 8 个单精度浮点数并行加法       | 8             |
| 双精度乘法     | MULPD (x86)      | 对两个双精度浮点数执行乘法        | 1             |
| SIMD 双精度加法| VADDPQ (AVX-512) | 对 8 个双精度浮点数并行加法       | 8             |

注：FLOP 数量基于每条指令执行的操作数，实际性能受硬件和优化影响。

#### 未来趋势
随着处理器架构的演进，FLOPs 的计算需求将继续增长。当前趋势包括：
- 更多支持 SIMD 和 VLIW（超长指令字）的指令集，以提高并行性。
- 混合精度计算的普及，结合单精度和双精度的优势。
- 可持续性关注，优化浮点运算的能源效率。

#### 结论
从计算机体系结构的角度看，FLOPs 所指的浮点运算指令包括加法、减法、乘法和除法等基本操作，支持单精度和双精度，并可能包括 SIMD 指令，每条指令可贡献多个 FLOP。理解这些指令的类型有助于评估处理器性能和优化计算任务。

---

### 关键引用
- [FLOP for Quantity, FLOP/s for Performance](https://blog.heim.xyz/flop-for-quantity-flop-s-for-performance/)
- [What is FLOP/s and is it a good measure of performance?](https://stackoverflow.com/questions/329174/what-is-flop-s-and-is-it-a-good-measure-of-performance)
- [What Is FLOPS & How Is It Used in Supercomputing?](https://biztechmagazine.com/article/2023/08/what-flops-and-how-does-it-help-supercomputer-performance-perfcon)
- [Floating-Point Operations Per Second (FLOPS)](https://en.wikichip.org/wiki/flops)