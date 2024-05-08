import unittest
import numpy as np
from main import GIRL_PT1, GIRL_PT2, GIRT_REQUIRE_MN_BOY_PT1, GIRT_REQUIRE_MN_BOY_PT2, AP_THRESHOLD1, AP_THRESHOLD2, gen_rands_sum_geq_s_range_0_1, Male, Female, get_pt_vector_by_conditions, ensure_diversity, choose_mate


def can_be_couple(man: Male, girl: Female):
    return man.pt >= girl.mn_boy_pt and man.mn_girl_pt <= girl.pt and girl.appearance_jdg(man.appearance)


class TestMain(unittest.TestCase):
    def test_gen_rands_sum_geq_s_range_0_1(self):
        for _ in range(514):
            a1 = gen_rands_sum_geq_s_range_0_1(3, 2.5)
            s1 = sum(a1)
            self.assertGreaterEqual(s1, 2.5)
            self.assertLess(s1, 3)
            a2 = gen_rands_sum_geq_s_range_0_1(5, 2)
            s2 = sum(a2)
            self.assertGreaterEqual(s2, 2)
            self.assertLess(s2, 5)
            arrs = [a1, a2]
            for arr in arrs:
                for v in arr:
                    self.assertGreaterEqual(v, 0)
                    self.assertLess(v, 1)

    def test_male_class(self):
        for _ in range(514):
            male1 = Male(1, girl_pt_le=GIRL_PT1)
            self.assertGreaterEqual(male1.mn_girl_pt, 0)
            self.assertLess(male1.mn_girl_pt, GIRL_PT1)
            male2 = Male(2)
            self.assertGreaterEqual(male2.mn_girl_pt, 0)
            self.assertLess(male2.mn_girl_pt, 1)

    def test_get_pt_vector_by_conditions(self):
        AP_THRESHOLD3 = 0.4
        for _ in range(514):
            pt_vec1 = get_pt_vector_by_conditions(ap_le=AP_THRESHOLD1)
            appearance1 = pt_vec1[0]
            self.assertGreaterEqual(appearance1, 0)
            self.assertLess(appearance1, AP_THRESHOLD1)

            pt_vec2 = get_pt_vector_by_conditions(ap_geq=AP_THRESHOLD3)
            appearance2 = pt_vec2[0]
            self.assertGreaterEqual(appearance2, AP_THRESHOLD3)
            self.assertLess(appearance2, 1)

            pt_vec3 = get_pt_vector_by_conditions(ap_le=AP_THRESHOLD1, self_pt_geq=GIRT_REQUIRE_MN_BOY_PT1)
            appearance3 = pt_vec3[0]
            pt3 = np.mean(pt_vec3)
            self.assertLess(appearance3, AP_THRESHOLD1)
            self.assertGreaterEqual(appearance3, 0)
            self.assertGreaterEqual(pt3, GIRT_REQUIRE_MN_BOY_PT1)
            self.assertLess(pt3, 1)

            pt_vec4 = get_pt_vector_by_conditions(ap_geq=AP_THRESHOLD3, self_pt_geq=GIRT_REQUIRE_MN_BOY_PT1)
            appearance4 = pt_vec4[0]
            pt4 = np.mean(pt_vec4)
            self.assertGreaterEqual(appearance4, AP_THRESHOLD3)
            self.assertLess(appearance4, 1)
            self.assertGreaterEqual(pt4, GIRT_REQUIRE_MN_BOY_PT1)
            self.assertLess(pt4, 1)

            pt_vec5 = get_pt_vector_by_conditions(self_pt_geq=GIRT_REQUIRE_MN_BOY_PT1)
            pt5 = np.mean(pt_vec5)
            self.assertGreaterEqual(pt5, GIRT_REQUIRE_MN_BOY_PT1)
            self.assertLess(pt5, 1)

            pt_vec6 = get_pt_vector_by_conditions()

            pt_vectors = [pt_vec1, pt_vec2, pt_vec3, pt_vec4, pt_vec5, pt_vec6]
            for pt_vector in pt_vectors:
                for v in pt_vector:
                    self.assertGreaterEqual(v, 0)
                    self.assertLess(v, 1)

    def test_ensure_diversity(self):
        for _ in range(114):
            men = ensure_diversity([])
            for i, man in enumerate(men):
                i_rem = i % 12
                if i_rem <= 3:
                    self.assertGreaterEqual(man.pt, GIRT_REQUIRE_MN_BOY_PT1)
                    self.assertGreaterEqual(GIRL_PT1, man.mn_girl_pt)
                elif i_rem <= 7:
                    self.assertGreaterEqual(man.pt, GIRT_REQUIRE_MN_BOY_PT1)
                    self.assertGreaterEqual(GIRL_PT2, man.mn_girl_pt)
                else:
                    self.assertGreaterEqual(man.pt, GIRT_REQUIRE_MN_BOY_PT2)
                    self.assertGreaterEqual(GIRL_PT2, man.mn_girl_pt)
                if i_rem == 0 or i_rem == 4 or i_rem == 8:
                    self.assertGreaterEqual(man.appearance, AP_THRESHOLD1)
                if i_rem == 1 or i_rem == 5 or i_rem == 9:
                    self.assertGreaterEqual(man.appearance, AP_THRESHOLD2)
                if i_rem == 2 or i_rem == 6 or i_rem == 10:
                    self.assertLess(man.appearance, AP_THRESHOLD1)
                if i_rem == 3 or i_rem == 7 or i_rem == 11:
                    self.assertLess(man.appearance, AP_THRESHOLD2)

    def test_choose_mate(self):
        for _ in range(114):
            men = ensure_diversity([])
            girl1 = Female(GIRL_PT1, GIRT_REQUIRE_MN_BOY_PT1, lambda x: x >= AP_THRESHOLD1)
            girl2 = Female(GIRL_PT1, GIRT_REQUIRE_MN_BOY_PT1, lambda x: x >= AP_THRESHOLD2)
            girl3 = Female(GIRL_PT1, GIRT_REQUIRE_MN_BOY_PT1, lambda x: x <= AP_THRESHOLD1)
            girl4 = Female(GIRL_PT1, GIRT_REQUIRE_MN_BOY_PT1, lambda x: x <= AP_THRESHOLD2)
            girls = [girl1, girl2, girl3, girl4]
            for girl in girls:
                boyfriends, _ = choose_mate(men, girl)
                bf_key_set = set([bf.key for bf in boyfriends])
                self.assertEqual(len(bf_key_set), len(boyfriends))
                for man in boyfriends:
                    self.assertTrue(can_be_couple(man, girl))


if __name__ == '__main__':
    unittest.main()
