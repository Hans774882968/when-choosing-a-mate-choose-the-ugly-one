[TOC]

# 择偶只选丑的竟能匹配更优质配偶？让这个Python项目告诉你真相！

## 引言

在一个~~寂寞~~充实的夜晚，我刷到了[这个视频](https://www.bilibili.com/video/BV1pp421R7Ro)。视频大致的内容是，如果你不在意配偶的颜值，那么择偶时不妨只选丑的，这样你匹配到的配偶在其他你在意的方面会更优秀。这个说法直觉上就很对，但怎么稍微严谨点地证明呢？一开始我想建一个数学模型，但我数学太差了，建不出来，于是就有了这个Python项目和这篇~~水~~精华文章。

[项目GitHub传送门](https://github.com/Hans774882968/when-choosing-a-mate-choose-the-ugly-one)。**作者：[hans774882968](https://blog.csdn.net/hans774882968)以及[hans774882968](https://juejin.cn/user/1464964842528888)以及[hans774882968](https://www.52pojie.cn/home.php?mod=space&uid=1906177)**

本文52pojie：https://www.52pojie.cn/thread-1921724-1-1.html

本文juejin：https://juejin.cn/post/7366264344392581146

本文CSDN：https://blog.csdn.net/hans774882968/article/details/138572035

## 模型

设有N个维度用来衡量男性的优秀程度，组成一个向量`(a1, ..., an)`，`0 <= ai < 1`，其中`a1`是颜值，男性的整体分数`0 <= pt < 1`暂且定义为向量各维度的平均数。男性对女性有一个整体分数的要求`0 <= mn_girl_pt < 1`。生成`mn_girl_pt`的策略：有`MALE_LOW_STANDARD_P = 0.75`的概率生成小于自身整体分数的数，否则生成一个大于等于自身整体分数的数。

再假设女性的整体分数为`0 <= pt < 1`，有对男性的**整体分数**的要求`0 <= mn_boy_pt < 1`，还有对男性的颜值的期望，用一个函数表示：

```python
AP_THRESHOLD1 = 0.5
AP_THRESHOLD2 = 0.1

e1 = lambda x: x >= AP_THRESHOLD1
e2 = lambda x: x >= AP_THRESHOLD2
# 只选颜值低的
e3 = lambda x: x <= AP_THRESHOLD1
e4 = lambda x: x <= AP_THRESHOLD2
```

其实生成的男性可以改为按分数正态分布，以贴近现实。但有一定难度，就先留个TODO吧。

择偶过程：女性不断地在男性池子中随机抽取男性，若男性和女性互相满足对方的择偶标准，则女性以概率`1 - DISCARD_P = 0.5`和他成为情侣。女性选择`BF_NUM = 5`个npy后结束循环。npy中**除颜值外的整体分数**前`SPOUSE_NUM = 1`高的为最终的择偶结果，取上述分数的平均值作为最终的择偶分数。保证女性必定能找到配偶。

其实择偶过程可以设计得更复杂些，比如：

1. 增加女性自身分数随择偶时间增加而衰减的机制。
2. 改为择偶时间固定，并处理女性最终没有匹配到配偶的情况。

但这样的机制，相比于让女性必定能找到配偶的机制，要难实现、难分析得多。

**择偶时间**定义为女性选好`BF_NUM`个npy时抽取的男性总数。

本文只关注两个指标：匹配到的配偶**除颜值外的整体分数**，择偶时间。

## 核心代码讲解

1. `main.py`：懒得拆分逻辑了，全放一个文件了。
2. `main_test.py`：单测。

女性：

```python
class Female():
    def __init__(self, pt: float, mn_boy_pt: float, appearance_jdg) -> None:
        self.pt = pt
        self.mn_boy_pt = mn_boy_pt
        self.appearance_jdg = appearance_jdg

e1 = lambda x: x >= AP_THRESHOLD1
e2 = lambda x: x >= AP_THRESHOLD2
e3 = lambda x: x <= AP_THRESHOLD1
e4 = lambda x: x <= AP_THRESHOLD2
girl1 = Female(GIRL_PT1, GIRT_REQUIRE_MN_BOY_PT1, e1)
girl2 = Female(GIRL_PT1, GIRT_REQUIRE_MN_BOY_PT1, e2)
girl3 = Female(GIRL_PT1, GIRT_REQUIRE_MN_BOY_PT1, e3)
girl4 = Female(GIRL_PT1, GIRT_REQUIRE_MN_BOY_PT1, e4)
girl5 = Female(GIRL_PT2, GIRT_REQUIRE_MN_BOY_PT1, e1)
girl6 = Female(GIRL_PT2, GIRT_REQUIRE_MN_BOY_PT1, e2)
girl7 = Female(GIRL_PT2, GIRT_REQUIRE_MN_BOY_PT1, e3)
girl8 = Female(GIRL_PT2, GIRT_REQUIRE_MN_BOY_PT1, e4)
girl9 = Female(GIRL_PT2, GIRT_REQUIRE_MN_BOY_PT2, e1)
girl10 = Female(GIRL_PT2, GIRT_REQUIRE_MN_BOY_PT2, e2)
girl11 = Female(GIRL_PT2, GIRT_REQUIRE_MN_BOY_PT2, e3)
girl12 = Female(GIRL_PT2, GIRT_REQUIRE_MN_BOY_PT2, e4)
```

我希望同时展示女性自身分数和`mn_boy_pt`对择偶结果的影响，所以需要画并列的柱状图。所以我将数据结构组织为：

```python
girl_groups = [
    [girl1, girl5, girl9], [girl2, girl6, girl10], [girl3, girl7, girl11], [girl4, girl8, girl12]
]
```

遍历方式：

```python
for j, girl_group in enumerate(girl_groups):
    for k, girl in enumerate(girl_group):
        boyfriends, time_cost = choose_mate(men, girl)
        spouse_pt = calc_spouses_pt(boyfriends)
```

男性：

```python
class Male():
    def __init__(self, key, ap_le=None, ap_geq=None, self_pt_geq=None, girl_pt_le=None) -> None:
        self.key = key
        pt_vec = get_pt_vector_by_conditions(ap_le, ap_geq, self_pt_geq)
        self.appearance = pt_vec[0]
        self.other_dimensions = pt_vec[1:]
        self.pt = np.mean(pt_vec)
        self.pt_without_appearance = np.mean(self.other_dimensions)
        self.mn_girl_pt = self.get_mn_girl_pt(girl_pt_le)

    def get_mn_girl_pt(self, girl_pt_le=None):
        if girl_pt_le is not None:
            return random.uniform(0, girl_pt_le)
        p = random.random()
        if p < MALE_LOW_STANDARD_P:
            return random.uniform(0, self.pt)
        return random.uniform(self.pt, 1)
```

`key`参数仅用于择偶过程去重，其他4个参数都仅用于负责生成满足女性择偶标准的男性的函数`ensure_diversity`。其实也可以选择不写`ensure_diversity`，转而编写处理女性没有选到配偶的情况的逻辑。`ensure_diversity`不太重要但不太好写，放在本节最后介绍。

择偶逻辑：把《模型》一节的择偶过程描述翻译成代码就行，很好写。

```python
def choose_mate(men: List[Male], girl: Female) -> Tuple[List[Male], int]:
    BATCH_SIZE = max(1, math.floor(math.sqrt(len(men))))
    boyfriends = []
    bf_set = set()
    time_cost = 0
    while True:
        men_batch = random.sample(men, BATCH_SIZE)
        for man in men_batch:
            time_cost += 1
            if man.pt < girl.mn_boy_pt or man.mn_girl_pt > girl.pt or not girl.appearance_jdg(man.appearance):
                continue
            if bf_set.__contains__(man.key):
                continue
            p = random.random()
            if p < DISCARD_P:
                continue
            bf_set.add(man.key)
            boyfriends.append(man)
            if len(boyfriends) >= BF_NUM:
                return boyfriends, time_cost
```

`calc_spouses_pt`：计算匹配到的配偶的分数。

```python
def calc_spouses_pt(boyfriends: List[Male]) -> float:
    bf_pts = sorted([man.pt_without_appearance for man in boyfriends], reverse=True)
    return np.mean(bf_pts[:SPOUSE_NUM])
```

`data_analysis`：进行数据分析，画出柱状图。`matplotlib`画柱状图有点麻烦，我的代码思路参考了[参考链接1](https://blog.csdn.net/qq_44864262/article/details/108098227)。`ticks`表示刻度位置，`(-girl_num_in_group / 2 + 0.5) * BAR_WIDTH`表示最左侧柱子的左边界位置相比于`ticks[?]`的偏移。

```python
class ChooseMateResult():
    def __init__(self, spouse_pt: float, time_cost: float) -> None:
        self.spouse_pt = spouse_pt
        self.time_cost = time_cost

def data_analysis(choose_mate_results_groups: List[List[List[ChooseMateResult]]]):
    # ...
    def auto_label(rects, format_spec):
        for rect in rects:
            height = rect.get_height()  # type(height) = numpy.float64
            plt.text(rect.get_x(), 1.01 * height, format(float(height), format_spec), size=8)  # 让字小一点，不超过柱子宽度
    # 只展示画配偶分数柱状图的代码...
    plt.subplot(121)
    plt.title('Spouse Points')
    girl_num_in_group = len(choose_mate_mean_result_groups[0])
    BAR_WIDTH = 0.3
    group_num = len(choose_mate_results_groups)
    ticks = np.arange(group_num)
    girl_in_group_labels = [
        f'girls with point {GIRL_PT1}, require {GIRT_REQUIRE_MN_BOY_PT1}',
        f'girls with point {GIRL_PT2}, require {GIRT_REQUIRE_MN_BOY_PT1}',
        f'girls with point {GIRL_PT2}, require {GIRT_REQUIRE_MN_BOY_PT2}',
    ]
    for j in range(girl_num_in_group):
        spouse_pt_mean_bars = [cmg[j].spouse_pt for cmg in choose_mate_mean_result_groups]
        x = ticks + (-girl_num_in_group / 2 + 0.5 + j) * BAR_WIDTH
        rects = plt.bar(x, spouse_pt_mean_bars, BAR_WIDTH, label=girl_in_group_labels[j])
        auto_label(rects, '.3f')
    x_labels = [f'girl_group_{i}' for i in range(1, group_num + 1)]
    plt.xticks(ticks, x_labels)  # 图例
    plt.ylim((0, 1))  # 提高 y 范围防止图例遮挡数据
    plt.legend()
    # ...
```

`ensure_diversity`：保证女性能选到配偶的逻辑。

```python
def ensure_diversity(men: List[Male]):
    for _ in range(BF_NUM * 2):
        ky = len(men)
        man1 = Male(ky, ap_geq=AP_THRESHOLD1, self_pt_geq=GIRT_REQUIRE_MN_BOY_PT1, girl_pt_le=GIRL_PT1)
        man2 = Male(ky + 1, ap_geq=AP_THRESHOLD2, self_pt_geq=GIRT_REQUIRE_MN_BOY_PT1, girl_pt_le=GIRL_PT1)
        man3 = Male(ky + 2, ap_le=AP_THRESHOLD1, self_pt_geq=GIRT_REQUIRE_MN_BOY_PT1, girl_pt_le=GIRL_PT1)
        man4 = Male(ky + 3, ap_le=AP_THRESHOLD2, self_pt_geq=GIRT_REQUIRE_MN_BOY_PT1, girl_pt_le=GIRL_PT1)
        man5 = Male(ky + 4, ap_geq=AP_THRESHOLD1, self_pt_geq=GIRT_REQUIRE_MN_BOY_PT1, girl_pt_le=GIRL_PT2)
        man6 = Male(ky + 5, ap_geq=AP_THRESHOLD2, self_pt_geq=GIRT_REQUIRE_MN_BOY_PT1, girl_pt_le=GIRL_PT2)
        man7 = Male(ky + 6, ap_le=AP_THRESHOLD1, self_pt_geq=GIRT_REQUIRE_MN_BOY_PT1, girl_pt_le=GIRL_PT2)
        man8 = Male(ky + 7, ap_le=AP_THRESHOLD2, self_pt_geq=GIRT_REQUIRE_MN_BOY_PT1, girl_pt_le=GIRL_PT2)
        man9 = Male(ky + 8, ap_geq=AP_THRESHOLD1, self_pt_geq=GIRT_REQUIRE_MN_BOY_PT2, girl_pt_le=GIRL_PT2)
        man10 = Male(ky + 9, ap_geq=AP_THRESHOLD2, self_pt_geq=GIRT_REQUIRE_MN_BOY_PT2, girl_pt_le=GIRL_PT2)
        man11 = Male(ky + 10, ap_le=AP_THRESHOLD1, self_pt_geq=GIRT_REQUIRE_MN_BOY_PT2, girl_pt_le=GIRL_PT2)
        man12 = Male(ky + 11, ap_le=AP_THRESHOLD2, self_pt_geq=GIRT_REQUIRE_MN_BOY_PT2, girl_pt_le=GIRL_PT2)
        men.extend((man1, man2, man3, man4, man5, man6, man7, man8, man9, man10, man11, man12))
    return men
```

`get_pt_vector_by_conditions`不太重要写得也比较丑，不介绍了。只看下它用到的一个我自造的轮子：

```python
def gen_rands_sum_geq_s_range_0_1(n: int, s: float):
    if s >= n:
        raise ValueError('s is too big. Note: 0 <= s < n')
    if s < 0:
        raise ValueError('s is too small. Note: 0 <= s < n')
    while True:
        res = []
        tot = 0
        legal = True
        for i in range(n):
            v = random.random()
            tot += v
            if s - tot > n - i - 1:
                legal = False
                break
            res.append(v)
        if not legal or tot < s:
            continue
        return res
```

功能是生成`n`个`[0, 1)`的数，使得它们的和大于等于`s`。里面有一个简单的剪枝：`s - tot > n - i - 1`表示之前生成的数太小，本次生成过程已经失败。TODO: 是否存在一个库能做这件事？

## 运行结果分析

[这里](https://github.com/Hans774882968/when-choosing-a-mate-choose-the-ugly-one/blob/main/outp/choose_ugly_mates.md)展示了几个比较好的若干个结果。运行拿到比较好的结果还是很容易的，但其实也有不少结果并没有那么好。

输出示意：

```python
[0.532, 0.570, 0.618, 0.621, 0.626, 0.627, 0.656, 0.666, 0.667, 0.677, 0.678, 0.681, 0.709, 0.724, 0.729, 0.730, 0.732, 0.754, 0.770, 0.789] 0.678 44.2
[0.568, 0.605, 0.631, 0.634, 0.652, 0.667, 0.671, 0.680, 0.687, 0.701, 0.704, 0.711, 0.719, 0.732, 0.737, 0.737, 0.738, 0.773, 0.800, 0.823] 0.698 40.6
[0.611, 0.628, 0.648, 0.653, 0.658, 0.701, 0.708, 0.711, 0.719, 0.724, 0.728, 0.734, 0.736, 0.748, 0.748, 0.761, 0.775, 0.789, 0.813, 0.884] 0.724 84.2
[0.559, 0.560, 0.572, 0.610, 0.616, 0.620, 0.624, 0.626, 0.627, 0.676, 0.680, 0.682, 0.694, 0.716, 0.732, 0.738, 0.749, 0.772, 0.831, 0.831] 0.676 32.2
[0.549, 0.613, 0.643, 0.662, 0.664, 0.671, 0.679, 0.685, 0.687, 0.687, 0.700, 0.732, 0.738, 0.743, 0.749, 0.762, 0.777, 0.792, 0.794, 0.868] 0.710 32.0
[0.659, 0.683, 0.697, 0.704, 0.724, 0.725, 0.730, 0.731, 0.756, 0.764, 0.777, 0.781, 0.787, 0.792, 0.797, 0.829, 0.850, 0.859, 0.864, 0.891] 0.770 63.2
[0.663, 0.679, 0.688, 0.694, 0.706, 0.706, 0.709, 0.725, 0.733, 0.751, 0.756, 0.767, 0.774, 0.781, 0.781, 0.786, 0.787, 0.812, 0.817, 0.865] 0.749 95.6
[0.604, 0.604, 0.644, 0.645, 0.659, 0.669, 0.673, 0.687, 0.706, 0.729, 0.739, 0.744, 0.748, 0.760, 0.766, 0.768, 0.775, 0.825, 0.862, 0.865] 0.723 80.0
[0.683, 0.722, 0.740, 0.745, 0.750, 0.759, 0.762, 0.770, 0.787, 0.794, 0.800, 0.819, 0.819, 0.828, 0.854, 0.868, 0.871, 0.883, 0.888, 0.924] 0.803 242.2
[0.681, 0.699, 0.746, 0.757, 0.758, 0.762, 0.768, 0.776, 0.785, 0.790, 0.807, 0.815, 0.819, 0.829, 0.831, 0.833, 0.839, 0.840, 0.888, 0.888] 0.795 628.8
[0.639, 0.680, 0.705, 0.708, 0.712, 0.719, 0.720, 0.724, 0.735, 0.746, 0.759, 0.760, 0.760, 0.770, 0.807, 0.820, 0.836, 0.841, 0.842, 0.862] 0.757 458.8
[0.775, 0.787, 0.795, 0.797, 0.812, 0.816, 0.819, 0.827, 0.832, 0.840, 0.845, 0.845, 0.846, 0.850, 0.875, 0.879, 0.888, 0.896, 0.904, 0.904] 0.842 2397.6
```

柱状图示意：

![](.\outp\choose_ugly_mates4.png)

描述一下图中的信息：

1. 蓝色：女性分数0.5，要求男性整体分数不低于0.5。橙色：女性分数0.6，要求男性整体分数不低于0.5。绿色：女性分数0.6，要求男性整体分数不低于0.6。
2. 在配偶分数图中，同色柱子大概率是单增的。如果不单增，则择偶时间大概率会缩短。比如`(0.678, 44.2)`VS`(0.676, 32.2)`。
3. 在配偶分数图中，橙色柱子不一定会比同组的蓝色柱子高。如果橙色柱子比蓝色柱子矮，那么其择偶时间大概率会比蓝色柱子短。比如`(0.795, 628.8)`VS`(0.757, 458.8)`。
4. 在配偶分数图中，绿色柱子总是显著比蓝、橙色柱子高。但作为代价，其择偶时间也总是显著增加。

## 结论（TLDR）

一、只提高自身分数，不一定能提升匹配到的配偶的分数，提升的点大概率体现在择偶时间的缩短。

二、提高择偶标准，可以有效提升匹配到的配偶的分数，但会增加择偶时间，且可能增加到无穷。

三、若其他条件不变，对颜值的期望分别如下：

```python
AP_THRESHOLD1 = 0.5
AP_THRESHOLD2 = 0.1

e1 = lambda x: x >= AP_THRESHOLD1
e2 = lambda x: x >= AP_THRESHOLD2
# 只选颜值低的
e3 = lambda x: x <= AP_THRESHOLD1
e4 = lambda x: x <= AP_THRESHOLD2
```

则大概率匹配到的配偶的除颜值外的整体分数满足`e1 <= e2 <= e3 <= e4`。如果分数不单增，那么大概率提升的点体现在择偶时间的缩短。

四、只考虑选颜值低的为配偶的策略确实能提升除颜值外的整体分数，但也会显著增加择偶时间。具体牺牲哪个，优化哪个，看你自己咯。

## 参考资料

1. Python matplotlib实现 三条并列柱状图：https://blog.csdn.net/qq_44864262/article/details/108098227