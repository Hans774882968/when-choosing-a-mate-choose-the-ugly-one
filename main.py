import random
import math
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

N = 5  # 评价指标数目
MALE_NUM = 100000
MALE_LOW_STANDARD_P = 0.75
BF_NUM = 5
SPOUSE_NUM = 1  # 长期关系则等于1，短期关系则大于1
DISCARD_P = 0.5

GIRL_PT1 = 0.5
GIRL_PT2 = 0.6
GIRT_REQUIRE_MN_BOY_PT1 = 0.5
GIRT_REQUIRE_MN_BOY_PT2 = 0.6
AP_THRESHOLD1 = 0.5
AP_THRESHOLD2 = 0.1


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


def get_appearance(ap_le=None, ap_geq=None):
    if ap_le is not None and ap_geq is not None:
        raise ValueError('ap_le and ap_geq cannot both be non-None')
    if ap_le is not None:
        return random.uniform(0, ap_le)
    if ap_geq is not None:
        return random.uniform(ap_geq, 1)
    return random.random()


def get_pt_vector_by_conditions(ap_le=None, ap_geq=None, self_pt_geq=None) -> List[float]:
    if self_pt_geq is None:
        appearance = get_appearance(ap_le, ap_geq)
        other_dimensions = np.random.rand(N - 1).tolist()
        return [appearance] + other_dimensions
    if ap_le is None and ap_geq is None:
        actual_pt = self_pt_geq * N
        pt_vec = gen_rands_sum_geq_s_range_0_1(N, actual_pt)
        return pt_vec
    appearance = get_appearance(ap_le, ap_geq)
    other_dimensions_pt_upper = self_pt_geq * N - appearance
    other_dimensions_vec = gen_rands_sum_geq_s_range_0_1(N - 1, other_dimensions_pt_upper)
    pt_vec = [appearance] + other_dimensions_vec
    return pt_vec


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


class Female():
    def __init__(self, pt: float, mn_boy_pt: float, appearance_jdg) -> None:
        self.pt = pt
        self.mn_boy_pt = mn_boy_pt
        self.appearance_jdg = appearance_jdg


# 保证女性能选到配偶的逻辑
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


def calc_spouses_pt(boyfriends: List[Male]) -> float:
    bf_pts = sorted([man.pt_without_appearance for man in boyfriends], reverse=True)
    return np.mean(bf_pts[:SPOUSE_NUM])


class ChooseMateResult():
    def __init__(self, spouse_pt: float, time_cost: float) -> None:
        self.spouse_pt = spouse_pt
        self.time_cost = time_cost


def data_analysis(choose_mate_results_groups: List[List[List[ChooseMateResult]]]):
    def map_to_mean_result(choose_mate_results: List[ChooseMateResult]):
        spouse_pt_mean = np.mean([choose_mate_result.spouse_pt for choose_mate_result in choose_mate_results])
        time_cost_mean = np.mean([choose_mate_result.time_cost for choose_mate_result in choose_mate_results])
        res = ChooseMateResult(spouse_pt_mean, time_cost_mean)
        return res

    def format_float_array(a: List[float]):
        s = ', '.join([format(v, '.3f') for v in a])
        return f'[{s}]'

    def auto_label(rects, format_spec):
        for rect in rects:
            height = rect.get_height()  # type(height) = numpy.float64
            plt.text(rect.get_x(), 1.01 * height, format(float(height), format_spec), size=8)

    for choose_mate_results_group in choose_mate_results_groups:
        for choose_mate_results in choose_mate_results_group:
            choose_mate_results.sort(key=lambda x: x.spouse_pt)

    spouse_pts_groups = [[list(map(lambda c: c.spouse_pt, choose_mate_results)) for choose_mate_results in choose_mate_results_group]
                         for choose_mate_results_group in choose_mate_results_groups]
    choose_mate_mean_result_groups = [list(map(map_to_mean_result, choose_mate_results_group))
                                      for choose_mate_results_group in choose_mate_results_groups]
    for spouse_pts_group, choose_mate_mean_result_group in zip(spouse_pts_groups, choose_mate_mean_result_groups):
        for spouse_pts, choose_mate_mean_result in zip(spouse_pts_group, choose_mate_mean_result_group):
            spouse_pt_mean = choose_mate_mean_result.spouse_pt
            time_cost_mean = choose_mate_mean_result.time_cost
            print(format_float_array(spouse_pts), format(spouse_pt_mean, '.3f'), format(time_cost_mean, '.1f'))

    fig = plt.figure()
    fig.canvas.set_window_title('Choose Ugly Mates')

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
    plt.xticks(ticks, x_labels)
    plt.ylim((0, 1))  # 提高 y 范围防止图例遮挡数据
    plt.legend()

    plt.subplot(122)
    plt.title('Time Costs')
    for j in range(girl_num_in_group):
        time_cost_mean_bars = [cmg[j].time_cost for cmg in choose_mate_mean_result_groups]
        x = ticks + (-girl_num_in_group / 2 + 0.5 + j) * BAR_WIDTH
        rects = plt.bar(x, time_cost_mean_bars, BAR_WIDTH, label=girl_in_group_labels[j])
        auto_label(rects, '.1f')
    plt.xticks(ticks, x_labels)
    plt.legend()

    plt.show()


def main():
    men = [Male(i) for i in range(MALE_NUM)]
    ensure_diversity(men)
    def e1(x): return x >= AP_THRESHOLD1
    def e2(x): return x >= AP_THRESHOLD2
    def e3(x): return x <= AP_THRESHOLD1
    def e4(x): return x <= AP_THRESHOLD2
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
    girl_groups = [
        [girl1, girl5, girl9], [girl2, girl6, girl10], [girl3, girl7, girl11], [girl4, girl8, girl12]
    ]
    choose_mate_results_groups: List[List[List[ChooseMateResult]]] = [[[] for _ in range(len(girl_groups[i]))] for i in range(len(girl_groups))]
    for i in range(20):
        if i % 10 == 0:
            print(f'iter {i}...')
        for j, girl_group in enumerate(girl_groups):
            for k, girl in enumerate(girl_group):
                boyfriends, time_cost = choose_mate(men, girl)
                spouse_pt = calc_spouses_pt(boyfriends)
                choose_mate_result = ChooseMateResult(spouse_pt, time_cost)
                choose_mate_results_groups[j][k].append(choose_mate_result)
    data_analysis(choose_mate_results_groups)


if __name__ == '__main__':
    main()
